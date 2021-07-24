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

from experimental3.privacy_utils import autograd_grad_sample
from experimental3.privacy_utils.utils.module_inspection import get_layer_type
from experimental3.privacy_utils.utils.tensor_utils import sum_over_all_but_batch_and_last_n


def _create_or_extend_norm_sample(param: torch.Tensor, norm_sample: torch.Tensor) -> None:
    if not hasattr(param, "requires_grad") or not param.requires_grad:
        return
    if autograd_grad_sample.get_hooks_mode() == "norm":
        param.norm_sample = norm_sample


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
    a = torch.bmm(A, A.permute(0, 2, 1))
    b = torch.bmm(B, B.permute(0, 2, 1))
    norm_sample = torch.sqrt((a * b).sum(dim=(1, 2)))
    _create_or_extend_norm_sample(layer.weight, norm_sample)
    if layer.bias is not None:
        norm_sample = B.sum(dim=1).norm(2, dim=1)
        _create_or_extend_norm_sample(layer.bias, norm_sample)


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
        grad_sample = sum_over_all_but_batch_and_last_n(
            F.layer_norm(A, layer.normalized_shape, eps=layer.eps) * B, layer.weight.dim(),
        )
        norm_sample = grad_sample.flatten(start_dim=1).norm(2, dim=1)
        _create_or_extend_norm_sample(layer.weight, norm_sample)

        grad_sample = sum_over_all_but_batch_and_last_n(B, layer.bias.dim())
        norm_sample = grad_sample.flatten(start_dim=1).norm(2, dim=1)
        _create_or_extend_norm_sample(layer.bias, norm_sample)


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

    # TODO: Make this efficient as well.
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

    norm_sample = grad_sample.flatten(start_dim=1).norm(2, dim=1)
    _create_or_extend_norm_sample(layer.weight, norm_sample)


def _custom_compute_conv1d_grad_sample(
    layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
):
    a = torch.bmm(A, A.permute(0, 2, 1))
    b = torch.bmm(B, B.permute(0, 2, 1))
    norm_sample = torch.sqrt((a * b).sum(dim=(1, 2)))
    _create_or_extend_norm_sample(layer.weight, norm_sample)
    if layer.bias is not None:
        norm_sample = B.sum(dim=1).norm(2, dim=1)
        _create_or_extend_norm_sample(layer.bias, norm_sample)


# Only support layers in HF Transformer.
_supported_layers_grad_samplers = {
    "Embedding": _compute_embedding_grad_sample,
    "Linear": _compute_linear_grad_sample,
    "LayerNorm": _compute_norm_grad_sample,
    "Conv1D": _custom_compute_conv1d_grad_sample,
}
