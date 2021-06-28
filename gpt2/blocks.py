import abc

import torch
from torch import nn

from gpt2 import numerical


class Lrk(abc.ABC):
    """An abstract class used to check things."""

    @abc.abstractmethod
    def create_gradient(self):
        raise NotImplementedError


class LrkLinear(Lrk, nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LrkLinear, self).__init__()
        self.rank = rank

        # TODO: The bias here isn't updated, serious problem for my use case.
        # WARNING: You must explicitly track the params of `self.full` in the optimizer!
        # Also don't put the parameters of left or right for the optimizer to track!

        # Overall, the privacy engine tracks left, but not full or right.
        # The optimizer tracks full, but not full or right.
        self.full = nn.Linear(in_features, out_features, bias=bias).requires_grad_(False)
        self.left = nn.Linear(rank, out_features, bias=False)
        self.right = nn.Linear(in_features, rank, bias=False).requires_grad_(False)

        self.cached_weights = []

    @torch.no_grad()
    def restore_weight(self):
        self.full.weight.data.copy_(self.cached_weights.pop())

    @torch.no_grad()
    def decompose_weight(self):
        full_weight = self.full.weight.data
        self.cached_weights.append(full_weight)

        # TODO: ema should stabilize this a little.
        left_weight, right_weight, approx_error = numerical.weight_decomposition(full_weight, rank=self.rank)
        residual_weight = full_weight - torch.matmul(left_weight, right_weight)

        self.left.weight.data.copy_(left_weight.data)
        self.right.weight.data.copy_(right_weight.data)
        self.full.weight.data.copy_(residual_weight.data)

    @torch.no_grad()
    def create_gradient(self):
        partial_l_times_r = self.left.weight.grad @ self.right.weight
        # You don't have to delete this gradient, since it's refreshed and not accumulated.
        self.full.weight.grad = partial_l_times_r
        del self.left.weight.grad

    def forward(self, x):
        if self.training:
            net = self.left(self.right(x))
            net = net + self.full(x)
        else:
            net = self.full(x)
        return net
