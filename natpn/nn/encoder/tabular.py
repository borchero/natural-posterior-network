from typing import List
from torch import nn


class TabularEncoder(nn.Sequential):
    """
    Encoder for tabular data. This encoder is a simple MLP with Leaky ReLU activations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        *,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_dim: The dimension of the inputs.
            hidden_dims: The dimensions of the hidden layers.
            output_dim: The dimension of the output, i.e. the latent space.
            dropout: The dropout probability. Dropout layers are added after every activation.
        """
        layers = []
        for i, (in_dim, out_dim) in enumerate(
            zip([input_dim] + hidden_dims, hidden_dims + [output_dim])
        ):
            if i > 0:
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_dim, out_dim))

        super().__init__(*layers)
