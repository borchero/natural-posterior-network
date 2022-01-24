from typing import List
import torch
from torch import nn


class DeepImageEncoder(nn.Sequential):
    """
    Encoder for image data. This encoder is relatively shallow, consisting of 9 large convolutions
    and 3 linear layers. Batch normalization layers are used between convolutions and linear
    layers.
    """

    def __init__(self, input_size: torch.Size, out_dim: int, *, dropout: float = 0.0):
        conv_layers: List[nn.Module] = [
            NormedConvSequence(input_size[0], 32, kernel_size=3, padding=1, dropout=dropout),
            NormedConvSequence(32, 32, kernel_size=3, padding=1, dropout=dropout),
            NormedConvSequence(32, 32, kernel_size=5, stride=2, dropout=dropout),
            NormedConvSequence(32, 64, kernel_size=3, padding=1, dropout=dropout),
            NormedConvSequence(64, 64, kernel_size=3, padding=1, dropout=dropout),
            NormedConvSequence(64, 128, kernel_size=5, stride=2, dropout=dropout),
            NormedConvSequence(128, 128, kernel_size=3, padding=1, dropout=dropout),
            NormedConvSequence(128, 256, kernel_size=3, stride=2, dropout=dropout),
            nn.Flatten(),
        ]

        with torch.no_grad():
            linear_dim = nn.Sequential(*conv_layers)(torch.randn(1, *input_size)).size(-1)

        linear_layers: List[nn.Module] = [
            NormedLinearSequence(linear_dim, 128, dropout=dropout),
            NormedLinearSequence(128, 64, dropout=dropout),
            nn.Linear(64, out_dim),
        ]
        super().__init__(*(conv_layers + linear_layers))


class NormedConvSequence(nn.Sequential):
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
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=dropout),
        )


class NormedLinearSequence(nn.Sequential):
    """
    Linear layers with Leaky ReLU activations and dropout.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(p=dropout),
        )
