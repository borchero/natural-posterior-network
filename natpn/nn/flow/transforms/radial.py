import math
from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn
from ._base import Transform


class RadialTransform(Transform):
    r"""
    A radial transformation may be used to apply radial contractions and expansions around a
    reference point. It was introduced in "Variational Inference with Normalizing Flows" (Rezende
    and Mohamed, 2015).
    """

    def __init__(self, dim: int):
        r"""
        Args:
            dim: The dimension of the transform.
        """
        super().__init__()

        self.reference = nn.Parameter(torch.empty(dim))
        self.alpha_prime = nn.Parameter(torch.empty(1))
        self.beta_prime = nn.Parameter(torch.empty(1))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets this module's parameters. All parameters are sampled uniformly depending on this
        module's dimension.
        """
        std = 1 / math.sqrt(self.reference.size(0))
        nn.init.uniform_(self.reference, -std, std)
        nn.init.uniform_(self.alpha_prime, -std, std)
        nn.init.uniform_(self.beta_prime, -std, std)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = self.reference.size(0)
        alpha = F.softplus(self.alpha_prime)  # [1]
        beta = -alpha + F.softplus(self.beta_prime)  # [1]

        # Compute output
        diff = z - self.reference  # [*, D]
        r = diff.norm(dim=-1, keepdim=True)  # [*, 1]
        h = (alpha + r).reciprocal()  # [*]
        beta_h = beta * h  # [*]
        y = z + beta_h * diff  # [*, D]

        # Compute log-determinant of Jacobian
        h_d = -(h ** 2)  # [*]
        log_det_lhs = (dim - 1) * beta_h.log1p()  # [*]
        log_det_rhs = (beta_h + beta * h_d * r).log1p()  # [*, 1]
        log_det = (log_det_lhs + log_det_rhs).squeeze(-1)  # [*]

        return y, log_det
