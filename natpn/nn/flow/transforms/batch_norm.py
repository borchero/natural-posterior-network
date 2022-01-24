from typing import Tuple
import torch
from torch import nn
from ._base import Transform


class BatchNormTransform(Transform):
    r"""
    Batch Normalization layer for stabilizing deep normalizing flows. It was first introduced in
    `Density Estimation Using Real NVP <https://arxiv.org/pdf/1605.08803.pdf>`_ (Dinh et al.,
    2017).
    """

    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, dim: int, momentum: float = 0.5, eps: float = 1e-5):
        """
        Args:
            dim: The dimension of the inputs.
            momentum: Value used for calculating running average statistics.
            eps: A small value added in the denominator for numerical stability.
        """
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.empty(dim))
        self.beta = nn.Parameter(torch.empty(dim))

        self.register_buffer("running_mean", torch.empty(dim))
        self.register_buffer("running_var", torch.empty(dim))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets this module's parameters.
        """
        nn.init.zeros_(self.log_gamma)  # equal to `init.ones_(self.gamma)`
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.running_mean)
        nn.init.ones_(self.running_var)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            reduce = list(range(z.dim() - 1))
            mean = z.detach().mean(reduce)
            var = z.detach().var(reduce, unbiased=True)

            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(mean * (1 - self.momentum))
                self.running_var.mul_(self.momentum).add_(var * (1 - self.momentum))
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize input
        x = (z - mean) / (var + self.eps).sqrt()
        out = x * self.log_gamma.exp() + self.beta

        # Compute log-determinant
        log_det = self.log_gamma - 0.5 * (var + self.eps).log()
        # Do repeat instead of expand to allow fixing the log_det below
        log_det = log_det.sum(-1).repeat(z.size()[:-1])

        # Fix numerical issues during evaluation
        if not self.training:
            # Find all output rows where at least one value is not finite
            rows = (~torch.isfinite(out)).sum(-1) > 0
            # Fill these rows with 0 and set the log-determinant to -inf to indicate that they have
            # a density of exactly 0
            out[rows] = 0
            log_det[rows] = float("-inf")

        return out, log_det
