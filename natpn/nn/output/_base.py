from abc import ABC, abstractmethod
import torch
from torch import nn
import natpn.distributions as D


class Output(nn.Module, ABC):
    """
    Base class for output distributions of NatPN.
    """

    prior: D.ConjugatePrior

    @abstractmethod
    def forward(self, x: torch.Tensor) -> D.Likelihood:
        """
        Derives the likelihood distribution from the latent representation via a linear mapping
        to the distribution parameters.

        Args:
            x: The inputs' latent representations.

        Returns:
            The distribution.
        """
