from typing import List
import torch
from torch import nn


class ShallowImageEncoder(nn.Sequential):
    """
    Encoder for image data. This encoder is relatively shallow, consisting of 6 small convolutions
    and 3 linear layers.
    """

    def __init__(self, input_size: torch.Size, out_dim: int, *, dropout: float = 0.0):
        conv_layers: List[nn.Module] = [
            SimpleConvSequence(input_size[0], 32, kernel_size=3, padding=1, dropout=dropout),
            SimpleConvSequence(32, 32, kernel_size=3, padding=1, dropout=dropout),
            SimpleConvSequence(32, 32, kernel_size=5, stride=2, dropout=dropout),
            SimpleConvSequence(32, 64, kernel_size=3, padding=1, dropout=dropout),
            SimpleConvSequence(64, 64, kernel_size=3, padding=1, dropout=dropout),
            SimpleConvSequence(64, 64, kernel_size=5, stride=2, dropout=dropout),
            nn.Flatten(),
        ]

        with torch.no_grad():
            linear_dim = nn.Sequential(*conv_layers)(torch.randn(1, *input_size)).size(-1)

        linear_layers: List[nn.Module] = [
            SimpleLinearSequence(linear_dim, 128, dropout=dropout),
            SimpleLinearSequence(128, 64, dropout=dropout),
            nn.Linear(64, out_dim),
        ]
        super().__init__(*(conv_layers + linear_layers))


class SimpleConvSequence(nn.Sequential):
    """
    2D convolution followed by LeakyReLU non-linearity and dropout layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
        )


class SimpleLinearSequence(nn.Sequential):
    """
    Linear layers with Leaky ReLU activations and dropout.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
        )
