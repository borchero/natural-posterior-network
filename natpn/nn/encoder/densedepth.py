# The contents of this file are taken and modified from https://github.com/ialhashim/DenseDepth.
# pylint: disable=missing-function-docstring
from typing import Tuple
import torch
import torch.nn.functional as F
import torchvision.models as M  # type: ignore
from torch import nn


class DenseDepthEncoder(nn.Module):
    """
    The DenseDepth encoder uses DenseNet to obtain an encoding of shape [dim, 320, 240] from an
    input of size [3, 640, 480] for a given output dimension.
    """

    def __init__(self, out_dim: int, *, dropout: float = 0.0):
        super().__init__()
        self.feature_extractor = FeatureExtractor()

        in_features = 2208
        out_features = in_features // 2
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=1),
            nn.Dropout2d(dropout),
        )
        self.up1 = Upsample(out_features + 384, out_features // 2, dropout=dropout)
        self.up2 = Upsample(out_features // 2 + 192, out_features // 4, dropout=dropout)
        self.up3 = Upsample(out_features // 4 + 96, out_features // 8, dropout=dropout)
        self.up4 = Upsample(out_features // 8 + 96, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block0, block1, block2, block3, block4 = self.feature_extractor(x)
        x0 = self.conv_in(block4)
        x1 = self.up1(x0, block3)
        x2 = self.up2(x1, block2)
        x3 = self.up3(x2, block1)
        return self.up4(x3, block0)


class FeatureExtractor(nn.Module):
    """
    The feature extractor of the DenseDepth encoder, based on DenseNet.
    """

    def __init__(self):
        super().__init__()
        densenet = M.densenet161(pretrained=True)
        features = densenet.features
        self.encoder0 = nn.Sequential(
            features.conv0, features.norm0, features.relu0  # type: ignore
        )
        self.encoder1 = nn.Sequential(features.pool0)  # type: ignore
        self.encoder2 = nn.Sequential(features.denseblock1, features.transition1)  # type: ignore
        self.encoder3 = nn.Sequential(features.denseblock2, features.transition2)  # type: ignore
        self.encoder4 = nn.Sequential(
            features.denseblock3, features.transition3, features.denseblock4  # type: ignore
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        block0 = self.encoder0(x)
        block1 = self.encoder1(block0)
        block2 = self.encoder2(block1)
        block3 = self.encoder3(block2)
        block4 = self.encoder4(block3)
        return block0, block1, block2, block3, block4


class Upsample(nn.Module):
    """
    The upsample module increases the size of an image.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor, concat: torch.Tensor) -> torch.Tensor:
        z = F.interpolate(
            x, size=[concat.size(2), concat.size(3)], mode="bilinear", align_corners=True
        )
        z = torch.cat([z, concat], dim=1)
        return self.conv(z)
