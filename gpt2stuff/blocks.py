import abc
from typing import Callable

import torch
from torch import nn

from . import numerical


class Lrk(abc.ABC):
    """An abstract class used to check things."""

    @abc.abstractmethod
    def create_gradient(self):
        raise NotImplementedError

    @abc.abstractmethod
    def decompose_weight(self):
        raise NotImplementedError

    @abc.abstractmethod
    def restore_weight(self):
        raise NotImplementedError


# TODO: `full`, `left`, and `right` are names unique to low rank models. Don't use them elsewhere!
class LrkLinear(Lrk, nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LrkLinear, self).__init__()
        self.rank = rank

        # WARNING: You must explicitly track the params of `self.lrk_full` in the optimizer!
        # Also don't put the parameters of left or right for the optimizer to track!

        # Overall, the privacy engine tracks left, but not full or right.
        # The optimizer tracks full, but not full or right.
        self.lrk_full = nn.Linear(in_features, out_features, bias=bias).requires_grad_(False)
        self.lrk_left = nn.Linear(rank, out_features, bias=False)
        self.lrk_right = nn.Linear(in_features, rank, bias=False).requires_grad_(False)

        self.cached_weights = []

    # Overall rough logic of calling the following methods:
    #   decompose_weight -> forward + backward -> restore_weight + create_gradient -> optimizer.step

    # TODO: There should be an easier way of implementing all of this!
    @torch.no_grad()
    def decompose_weight(self):
        """Run this before *forward* pass."""
        full_weight = self.lrk_full.weight.data
        self.cached_weights.append(full_weight)

        # TODO: ema should stabilize this a little.
        left_weight, right_weight, approx_error = numerical.weight_decomposition(full_weight, rank=self.rank)
        residual_weight = full_weight - torch.matmul(left_weight, right_weight)

        self.lrk_left.weight.data.copy_(left_weight.data)
        self.lrk_right.weight.data.copy_(right_weight.data)
        self.lrk_full.weight.data.copy_(residual_weight.data)

    @torch.no_grad()
    def restore_weight(self):
        """Run this after backward pass and gradient accumulation but before optimizer.step."""
        self.lrk_full.weight.data.copy_(self.cached_weights.pop())

    @torch.no_grad()
    def create_gradient(self):
        """Run this before optimizer.step.

        Creates the gradient for the full matrix given the privatized gradient for the left and right weights.
        The relative order of this and restore_weight should not matter.
        """
        partial_l_times_r = self.lrk_left.weight.grad @ self.lrk_right.weight
        # You don't have to delete this gradient, since it's refreshed and not accumulated.
        self.lrk_full.weight.grad = partial_l_times_r
        del self.lrk_left.weight.grad

    def forward(self, x):
        if self.training:
            net = self.lrk_right(x)
            net = self.lrk_left(net)
            net = net + self.lrk_full(x)
        else:
            net = self.lrk_full(x)
        return net


# Only for low rank models.
def create_action(action_name):
    def recursive_action(module: nn.Module):
        for submodule in module.modules():
            if isinstance(submodule, Lrk):
                action_function = getattr(submodule, action_name)
                action_function()

    return recursive_action


def get_optimizer_params(module):
    params = []
    for name, param in module.named_parameters():
        if 'lrk_left' not in name and 'lrk_right' not in name:
            params.append(param)
    return params


decompose_weight: Callable = create_action("decompose_weight")
restore_weight: Callable = create_action("restore_weight")
create_gradient: Callable = create_action("create_gradient")
