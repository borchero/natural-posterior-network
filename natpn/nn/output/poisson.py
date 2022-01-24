import torch
from torch import nn
import natpn.distributions as D
from ._base import Output


class PoissonOutput(Output):
    """
    Poisson output with Gamma prior. The prior yields a Poisson rate of 1e-3.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: The dimension of the latent space.
        """
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.prior = D.GammaPrior(rate=1, evidence=1)

    def forward(self, x: torch.Tensor) -> D.Likelihood:
        z = self.linear.forward(x)
        return D.Poisson(z.squeeze(-1).exp())
