import torch
from torch import nn
import natpn.distributions as D
from natpn.utils import chunk_squeeze_last
from ._base import Output


class NormalOutput(Output):
    """
    Normal output with Normal Gamma prior. The prior yields a mean of 0 and a scale of 10.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: The dimension of the latent space.
        """
        super().__init__()
        self.linear = nn.Linear(dim, 2)
        self.prior = D.NormalGammaPrior(mean=0, scale=10, evidence=1)

    def forward(self, x: torch.Tensor) -> D.Likelihood:
        z = self.linear.forward(x)
        loc, log_precision = chunk_squeeze_last(z)
        return D.Normal(loc, log_precision.exp() + 1e-10)
