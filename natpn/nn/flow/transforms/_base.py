from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import nn


class Transform(nn.Module, ABC):
    """
    Base class for all normalizing flow transforms
    """

    @abstractmethod
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms the given input.

        Args:
            z: A tensor of shape ``[*, dim]`` where ``dim`` is the dimensionality of this module.

        Returns:
            The transformed inputs, a tensor of shape ``[*, dim]`` and the log-determinants of the
            Jacobian evaluated at the inputs, a tensor of shape ``[*]``.
        """
