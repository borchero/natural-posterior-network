from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from ._base import Transform


class MaskedAutoregressiveTransform(Transform):
    r"""
    Masked Autogressive Transform as introduced in `Masked Autoregressive Flow for Density
    Estimation <https://arxiv.org/abs/1705.07057>`_ (Papamakarios et al., 2018).
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: List[int],
    ):
        """
        Args:
            dim: The dimension of the inputs.
            hidden_dims: The hidden dimensions of the MADE model.
        """
        super().__init__()
        self.net = MADE(dim, hidden_dims)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logscale = self.net(z).chunk(2, dim=-1)
        logscale = logscale.tanh()
        out = (z - mean) * torch.exp(-logscale)
        log_det = -logscale.sum(-1)
        return out, log_det


class MADE(nn.Sequential):
    """
    Masked autoencoder for distribution estimation (MADE) as introduced in
    `MADE: Masked Autoencoder for Distribution Estimation <https://arxiv.org/abs/1502.03509>`_
    (Germain et al., 2015). In consists of a series of masked linear layers and a given
    non-linearity between them.
    """

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        """
        Initializes a new MADE model as a sequence of masked linear layers.

        Args:
            input_dim: The number of input dimensions.
            hidden_dims: The dimensions of the hidden layers.
        """
        assert len(hidden_dims) > 0, "MADE model must have at least one hidden layer."

        dims = [input_dim] + hidden_dims + [input_dim * 2]
        hidden_masks = _create_masks(input_dim, hidden_dims)

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            if i > 0:
                layers.append(nn.LeakyReLU())
            layers.append(_MaskedLinear(in_dim, out_dim, mask=hidden_masks[i]))
        super().__init__(*layers)


class _MaskedLinear(nn.Linear):
    mask: torch.Tensor

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-renamed
        return F.linear(x, self.weight * self.mask, self.bias)

    def __repr__(self):
        return f"MaskedLinear(in_features={self.in_features}, out_features={self.out_features})"


def _create_masks(input_dim: int, hidden_dims: List[int]) -> List[torch.Tensor]:
    permutation = torch.randperm(input_dim)

    input_degrees = permutation + 1
    hidden_degrees = [_sample_degrees(1, input_dim - 1, d) for d in hidden_dims]
    output_degrees = permutation.repeat(2)

    all_degrees = [input_degrees] + hidden_degrees + [output_degrees]
    hidden_masks = [
        _create_single_mask(in_deg, out_deg)
        for in_deg, out_deg in zip(all_degrees, all_degrees[1:])
    ]

    return hidden_masks


def _create_single_mask(in_degrees: torch.Tensor, out_degrees: torch.Tensor) -> torch.Tensor:
    return (out_degrees.unsqueeze(-1) >= in_degrees).float()


def _sample_degrees(minimum: int, maximum: int, num: int) -> torch.Tensor:
    return torch.linspace(minimum, maximum, steps=num).round()
