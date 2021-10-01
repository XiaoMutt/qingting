from typing import Tuple

import torch.nn as nn
from torch import Tensor


class Block(nn.Module):
    def __init__(self, inplanes: int, outplanes: int) -> None:
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(outplanes),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class PrbCubeResNet(nn.Module):

    def __init__(self) -> None:
        super(PrbCubeResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)  # 2x16x16
        self.bn1 = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 2x16x16
        self.layers = nn.Sequential(
            Block(2, 4),
            Block(4, 8),
            Block(8, 16),
            Block(16, 32),
        )

        self.logits1x1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 32x1x1
            nn.Flatten(),
            nn.Linear(32, 11)  # 11
        )

        self.logits16x16 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1)),  # 1x16x16
            nn.Tanh()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # linear layer use default initialization

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, Block):
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def forward(self, x: Tensor) -> Tuple[Tuple, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layers(x)

        x1 = self.logits1x1(x)
        x2 = self.logits16x16(x)
        return x1, x2
