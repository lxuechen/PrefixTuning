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


def _fast_norm_sample(A, B):
    return torch.sqrt(
        (torch.bmm(A, A.permute(0, 2, 1)) * torch.bmm(B, B.permute(0, 2, 1))).sum(dim=(1, 2))
    )


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
    norm_sample = _fast_norm_sample(A, B)
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
    vocab_size = layer.weight.size(0)
    A = F.one_hot(A, num_classes=vocab_size).to(B.dtype)
    norm_sample = _fast_norm_sample(A, B)
    _create_or_extend_norm_sample(layer.weight, norm_sample)


def _custom_compute_conv1d_grad_sample(
    layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, batch_dim: int = 0
):
    norm_sample = _fast_norm_sample(A, B)
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
