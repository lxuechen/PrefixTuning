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

import torch
from torch import nn
from torch.functional import F

from experimental.privacy_utils import autograd_grad_sample
from experimental.privacy_utils.utils.module_inspection import get_layer_type
from experimental.privacy_utils.utils.tensor_utils import sum_over_all_but_batch_and_last_n


# Warning: This does not support weight sharing, so don't tie weights!
def _create_or_extend_grad_sample(
    param: torch.Tensor, grad_sample: torch.Tensor,
    *args, **kwargs
) -> None:
    """
    Creates the sample gradient norm.

    Args:
        param: Parameter to which ``norm_sample`` will be added
        grad_sample: Per-sample gradients tensor. Must be of the same
            shape as ``param`` with extra batch dimension
    """
    if not hasattr(param, "requires_grad") and not param.requires_grad:
        return

    grad_sample = grad_sample.detach()
    assert grad_sample.shape[1:] == param.shape

    if autograd_grad_sample.get_hooks_mode() == "norm":
        param.norm_sample = grad_sample.flatten(start_dim=1).norm(2, dim=1)
    else:  # mode == "grad"; should not get here.
        raise ValueError


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
    _create_or_extend_grad_sample(
        layer.weight, torch.bmm(B.permute(0, 2, 1), A), batch_dim
    )
    if layer.bias is not None:
        _create_or_extend_grad_sample(
            layer.bias, B.sum(dim=1), batch_dim,
        )


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
        _create_or_extend_grad_sample(
            layer.weight,
            sum_over_all_but_batch_and_last_n(
                F.layer_norm(A, layer.normalized_shape, eps=layer.eps) * B,
                layer.weight.dim(),
            ),
            batch_dim,
        )
        _create_or_extend_grad_sample(
            layer.bias,
            sum_over_all_but_batch_and_last_n(B, layer.bias.dim()),
            batch_dim,
        )
    elif layer_type == "GroupNorm":
        gs = F.group_norm(A, layer.num_groups, eps=layer.eps) * B
        _create_or_extend_grad_sample(
            layer.weight, torch.einsum("ni...->ni", gs), batch_dim
        )
        if layer.bias is not None:
            _create_or_extend_grad_sample(
                layer.bias, torch.einsum("ni...->ni", B), batch_dim
            )
    elif layer_type in {"InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d"}:
        gs = F.instance_norm(A, eps=layer.eps) * B
        _create_or_extend_grad_sample(
            layer.weight, torch.einsum("ni...->ni", gs), batch_dim
        )
        if layer.bias is not None:
            _create_or_extend_grad_sample(
                layer.bias, torch.einsum("ni...->ni", B), batch_dim
            )


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

    _create_or_extend_grad_sample(layer.weight, grad_sample, batch_dim)


def _custom_compute_conv1d_grad_sample(
    layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
):
    _create_or_extend_grad_sample(
        layer.weight, torch.bmm(A.permute(0, 2, 1), B), batch_dim
    )
    if layer.bias is not None:
        _create_or_extend_grad_sample(
            layer.bias, B.sum(dim=1), batch_dim,
        )


# Supported layer class types
_supported_layers_grad_samplers = {
    "Embedding": _compute_embedding_grad_sample,
    "Linear": _compute_linear_grad_sample,
    "LayerNorm": _compute_norm_grad_sample,
    "Conv1D": _custom_compute_conv1d_grad_sample,
}
