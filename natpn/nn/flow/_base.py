import math
from typing import List, TypeVar
import torch
from torch import nn
from .transforms import Transform

T = TypeVar("T", bound=Transform, covariant=True)


class NormalizingFlow(nn.Module):
    """
    pass
    """

    def __init__(self, transforms: List[T]):
        """
        Args:
            transforms: The transforms to use in the normalizing flow.
        """
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-probability of observing the given input, transformed by the flow's
        transforms under the standard Normal distribution.

        Args:
            z: A tensor of shape ``[*, dim]`` with the inputs.

        Returns:
            A tensor of shape ``[*]`` including the log-probabilities.
        """
        batch_size = z.size()[:-1]
        dim = z.size(-1)

        log_det_sum = z.new_zeros(batch_size)
        for transform in self.transforms:
            z, log_det = transform.forward(z)
            log_det_sum += log_det

        # Compute log-probability
        const = dim * math.log(2 * math.pi)
        norm = torch.einsum("...ij,...ij->...i", z, z)
        normal_log_prob = -0.5 * (const + norm)
        return normal_log_prob + log_det_sum
