#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

r"""
This module is a collection of grad samplers - methods to calculate per sample gradients
for a layer given two tensors: activations (module inputs) and
backpropagations (gradient values propagated from downstream layers).

Attributes:
    _supported_layers_grad_samplers (Dict[str, Callable]): Mapping
        from layer name to corresponding grad sampler
"""

from typing import Union

import numpy as np
from opacus.layers.dp_lstm import LSTMLinear
from opacus.layers.dp_multihead_attention import SequenceBias
import torch
from torch import nn
from torch.functional import F

from experimental2.privacy_utils.utils.module_inspection import get_layer_type
from experimental2.privacy_utils.utils.tensor_utils import sum_over_all_but_batch_and_last_n, unfold3d


def _create_or_extend_grad_sample(
    param: torch.Tensor, grad_sample: torch.Tensor, batch_dim: int, notes="",
) -> None:
    """
    Creates a ``grad_sample`` attribute in the given parameter, or appends to it
    if the ``grad_sample`` attribute already exists.

    Args:
        param: Parameter to which ``grad_sample`` will be added
        grad_sample: Per-sample gradients tensor. Must be of the same
            shape as ``param`` with extra batch dimension
        batch_dim: Position of the batch dimension in the shape of
            ``grad_sample``
        notes: Extra notes for debugging.
    """
    if hasattr(param, "requires_grad") and not param.requires_grad:
        return

    if hasattr(param, "grad_sample"):
        raise ValueError("Per-layer clipping breaks when there's parameter sharing.")
    else:
        param.grad_sample = grad_sample.detach()

    # Original opacus below.
    # if hasattr(param, "grad_sample"):
    #     param.grad_sample = torch.cat((param.grad_sample, grad_sample), batch_dim)
    # else:
    #     param.grad_sample = grad_sample


def _create_or_accumulate_grad_sample(
    param: torch.Tensor, grad_sample: torch.Tensor, batch_dim: int, layer: LSTMLinear
) -> None:
    """
    Creates a ``grad_sample`` attribute in the given parameter, or adds to it
    if the ``grad_sample`` attribute already exists.

    Args:
        param: Parameter to which ``grad_sample`` will be added
        grad_sample: Per-sample gradients tensor. Must be of the same
            shape as ``param`` with extra batch dimension
        batch_dim: Position of the batch dimension in the shape of
            ``grad_sample``
    """

    if hasattr(param, "grad_sample"):
        param.grad_sample[: grad_sample.shape[0]] += grad_sample
    else:
        max_batch_len = layer.max_batch_len
        param.grad_sample = torch.zeros(
            torch.Size([max_batch_len]) + grad_sample.shape[1:],
            device=grad_sample.device,
            dtype=grad_sample.dtype
        )
        param.grad_sample[: grad_sample.shape[0]] = grad_sample


@torch.no_grad()
def _clip_by_layer(layer, params, grad_samples):
    """Clip gradients on the layer level.

    Assumes `layer` has attribute `max_grad_norm`. By default, this is created in autograd_grad_sample.
    """
    norms = [grad_sample.flatten(start_dim=1).norm(2, dim=1) for grad_sample in grad_samples]
    norm = torch.stack(norms, dim=0).norm(2, dim=0)
    coef = torch.clamp_max(layer.max_grad_norm / (norm + 1e-6), 1)
    for param, grad_sample in zip(params, grad_samples):
        param.reservoir_grad = torch.einsum("i...,i->...", grad_sample, coef)


def _compute_linear_grad_sample(
    layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
) -> None:
    """
    Computes per sample gradients for ``nn.Linear`` layer
    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    params = [layer.weight]
    grad_samples = [torch.einsum("n...i,n...j->nij", B, A)]
    if layer.bias is not None:
        params.append(layer.bias)
        grad_samples.append(torch.einsum("n...k->nk", B))

    _clip_by_layer(layer, params=params, grad_samples=grad_samples)


def _compute_accumulate_linear_grad_sample(
    layer: LSTMLinear, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
) -> None:
    """
    Computes per sample gradients for ``LSTMLinear`` layer

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """

    gs = torch.einsum("n...i,n...j->nij", B, A)
    _create_or_accumulate_grad_sample(layer.weight, gs, batch_dim, layer)

    if layer.bias is not None:
        _create_or_accumulate_grad_sample(
            layer.bias,
            torch.einsum("n...k->nk", B),
            batch_dim,
            layer,
        )


def _compute_sequence_bias_grad_sample(
    layer: SequenceBias, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
) -> None:
    """
    Computes per sample gradients for ``SequenceBias`` layer

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    _create_or_extend_grad_sample(layer.bias, B[:, -1], batch_dim)


def _compute_norm_grad_sample(
    layer: Union[
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    ],
    A: torch.Tensor,
    B: torch.Tensor,
    batch_dim: int = 0,
) -> None:
    """
    Computes per sample gradients for normalization layers

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    layer_type = get_layer_type(layer)
    if layer_type == "LayerNorm":
        params = [layer.weight, layer.bias]
        grad_samples = [
            sum_over_all_but_batch_and_last_n(
                F.layer_norm(A, layer.normalized_shape, eps=layer.eps) * B,
                layer.weight.dim(),
            ),
            sum_over_all_but_batch_and_last_n(B, layer.bias.dim())
        ]
    else:
        raise NotImplementedError

    _clip_by_layer(layer, params=params, grad_samples=grad_samples)


def _compute_conv_grad_sample(
    layer: Union[nn.Conv2d, nn.Conv1d],
    A: torch.Tensor,
    B: torch.Tensor,
    batch_dim: int = 0,
) -> None:
    """
    Computes per sample gradients for convolutional layers

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    n = A.shape[0]
    layer_type = get_layer_type(layer)
    # get A and B in shape depending on the Conv layer
    if layer_type == "Conv2d":
        A = torch.nn.functional.unfold(
            A,
            layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
        B = B.reshape(n, -1, A.shape[-1])
    elif layer_type == "Conv1d":
        # unfold doesn't work for 3D tensors; so force it to be 4D
        A = A.unsqueeze(-2)  # add the H dimension
        # set arguments to tuples with appropriate second element
        A = torch.nn.functional.unfold(
            A,
            (1, layer.kernel_size[0]),
            padding=(0, layer.padding[0]),
            stride=(1, layer.stride[0]),
            dilation=(1, layer.dilation[0]),
        )
        B = B.reshape(n, -1, A.shape[-1])
    elif layer_type == "Conv3d":
        A = unfold3d(
            A,
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
        B = B.reshape(n, -1, A.shape[-1])

    # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
    grad_sample = torch.einsum("noq,npq->nop", B, A)
    # rearrange the above tensor and extract diagonals.
    grad_sample = grad_sample.view(
        n,
        layer.groups,
        -1,
        layer.groups,
        int(layer.in_channels / layer.groups),
        np.prod(layer.kernel_size),
    )
    grad_sample = torch.einsum("ngrg...->ngr...", grad_sample).contiguous()
    shape = [n] + list(layer.weight.shape)

    _create_or_extend_grad_sample(layer.weight, grad_sample.view(shape), batch_dim)

    if layer.bias is not None:
        _create_or_extend_grad_sample(layer.bias, torch.sum(B, dim=2), batch_dim)


def _compute_embedding_grad_sample(
    layer: nn.Embedding, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
) -> None:
    """
    Computes per sample gradients for ``nn.Embedding`` layer.

    Args:
        layer: Layer
        A: Activations
        B: Backpropagations
        batch_dim: Batch dimension position
    """
    saved = torch.backends.cudnn.deterministic
    torch.backends.cudnn.deterministic = True

    batch_size = A.shape[batch_dim]
    index = (
        A.unsqueeze(-1)
            .expand(*A.shape, layer.embedding_dim)
            .reshape(batch_size, -1, layer.embedding_dim)
    )
    grad_sample = torch.zeros(
        batch_size, *layer.weight.shape, device=layer.weight.device, dtype=layer.weight.dtype
    )
    grad_sample.scatter_add_(1, index, B.reshape(batch_size, -1, layer.embedding_dim))
    torch.backends.cudnn.deterministic = saved

    _clip_by_layer(layer, params=[layer.weight], grad_samples=[grad_sample])


def _custom_compute_conv1d_grad_sample(
    layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
):
    params = [layer.weight]
    grad_samples = [torch.einsum("n...i,n...j->nji", B, A)]
    if layer.bias is not None:
        params.append(layer.bias)
        grad_samples.append(torch.einsum("n...k->nk", B))

    _clip_by_layer(layer, params=params, grad_samples=grad_samples)


_supported_layers_grad_samplers = {
    "Embedding": _compute_embedding_grad_sample,
    "Linear": _compute_linear_grad_sample,
    "LSTMLinear": _compute_accumulate_linear_grad_sample,
    "Conv3d": _compute_conv_grad_sample,
    "Conv2d": _compute_conv_grad_sample,
    "Conv1d": _compute_conv_grad_sample,
    "LayerNorm": _compute_norm_grad_sample,
    "GroupNorm": _compute_norm_grad_sample,
    "InstanceNorm1d": _compute_norm_grad_sample,
    "InstanceNorm2d": _compute_norm_grad_sample,
    "InstanceNorm3d": _compute_norm_grad_sample,
    "SequenceBias": _compute_sequence_bias_grad_sample,

    # Open-AI GPT-2.
    "Conv1D": _custom_compute_conv1d_grad_sample,
    # Purely for debugging.
    "CounterEmbedding": _compute_embedding_grad_sample,
}  # Supported layer class types
