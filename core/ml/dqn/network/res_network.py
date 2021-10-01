import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from torch import Tensor

from .network import Network


class ResNetwork(Network):
    def __init__(self, n_channels: int, image_height: int, image_width: int, action_space_dim: int):
        super(ResNetwork, self).__init__(n_channels, image_height, image_width, action_space_dim)
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1

        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(n_channels, self.inplanes,
                               kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                               bias=False)  # 40
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 20
        self.layer1 = self._make_layer(64, 2)  # 20
        self.layer2 = self._make_layer(128, 2, stride=2)  # 10
        self.layer3 = self._make_layer(256, 2, stride=2)  # 5
        self.layer4 = self._make_layer(512, 2, stride=2)  # 3
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 1
        self.fc = nn.Linear(512 * resnet.BasicBlock.expansion, action_space_dim)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, resnet.BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        block = resnet.BasicBlock
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
