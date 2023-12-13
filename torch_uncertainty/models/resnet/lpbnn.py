import torch.nn.functional as F
from torch import Tensor, nn

from torch_uncertainty.layers.lpbnn import LpbnnConv2d, LpbnnLinear

__all__ = [
    "lpbnn_resnet18",
    "lpbnn_resnet34",
    "lpbnn_resnet50",
    "lpbnn_resnet101",
    "lpbnn_resnet152",
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        num_estimators: int = 4,
        groups: int = 1,
    ) -> None:
        super().__init__()

        # No subgroups for the first layer
        self.conv1 = LpbnnConv2d(
            in_planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            groups=groups,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = LpbnnConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            groups=groups,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                LpbnnConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    num_estimators=num_estimators,
                    groups=groups,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        alpha: float = 2,
        num_estimators: int = 4,
        gamma: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()

        # No subgroups for the first layer
        self.conv1 = LpbnnConv2d(
            in_planes,
            planes,
            kernel_size=1,
            num_estimators=num_estimators,
            groups=groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = LpbnnConv2d(
            planes,
            planes,
            kernel_size=3,
            num_estimators=num_estimators,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = LpbnnConv2d(
            planes,
            self.expansion * planes,
            kernel_size=1,
            num_estimators=num_estimators,
            groups=groups,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                LpbnnConv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    num_estimators=num_estimators,
                    groups=groups,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class _LpbnnResNet(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock | Bottleneck],
        num_blocks: list[int],
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        groups: int = 1,
        style: str = "imagenet",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.groups = groups
        self.num_estimators = num_estimators
        self.in_planes = 64
        block_planes = self.in_planes

        if style == "imagenet":
            self.conv1 = LpbnnConv2d(
                self.in_channels,
                block_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                num_estimators=num_estimators,
                groups=groups,
                bias=False,
            )
        else:
            self.conv1 = LpbnnConv2d(
                self.in_channels,
                block_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                num_estimators=num_estimators,
                groups=groups,
                bias=False,
            )

        self.bn1 = nn.BatchNorm2d(block_planes)

        if style == "imagenet":
            self.optional_pool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1
            )
        else:
            self.optional_pool = nn.Identity()

        self.layer1 = self._make_layer(
            block,
            block_planes,
            num_blocks[0],
            stride=1,
            num_estimators=num_estimators,
            groups=groups,
        )
        self.layer2 = self._make_layer(
            block,
            block_planes * 2,
            num_blocks[1],
            stride=2,
            num_estimators=num_estimators,
            groups=groups,
        )
        self.layer3 = self._make_layer(
            block,
            block_planes * 4,
            num_blocks[2],
            stride=2,
            num_estimators=num_estimators,
            groups=groups,
        )
        self.layer4 = self._make_layer(
            block,
            block_planes * 8,
            num_blocks[3],
            stride=2,
            num_estimators=num_estimators,
            groups=groups,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten(1)

        self.linear = LpbnnLinear(
            block_planes * 8 * block.expansion,
            num_classes,
            num_estimators=num_estimators,
        )

    def _make_layer(
        self,
        block: type[BasicBlock | Bottleneck],
        planes: int,
        num_blocks: int,
        stride: int,
        num_estimators: int,
        groups: int,
    ) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    num_estimators=num_estimators,
                    groups=groups,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = x.repeat(self.num_estimators, 1, 1, 1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.optional_pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool(out)
        out = self.flatten(out)
        return self.linear(out)


def lpbnn_resnet18(
    in_channels: int,
    num_estimators: int,
    num_classes: int,
    groups: int,
    style: str = "imagenet",
) -> _LpbnnResNet:
    return _LpbnnResNet(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        in_channels=in_channels,
        num_estimators=num_estimators,
        groups=groups,
        num_classes=num_classes,
        style=style,
    )


def lpbnn_resnet34(
    in_channels: int,
    num_estimators: int,
    num_classes: int,
    groups: int,
    style: str = "imagenet",
) -> _LpbnnResNet:
    return _LpbnnResNet(
        block=BasicBlock,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        groups=groups,
        num_classes=num_classes,
        style=style,
    )


def lpbnn_resnet50(
    in_channels: int,
    num_estimators: int,
    num_classes: int,
    groups: int,
    style: str = "imagenet",
) -> _LpbnnResNet:
    return _LpbnnResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        groups=groups,
        num_classes=num_classes,
        style=style,
    )


def lpbnn_resnet101(
    in_channels: int,
    num_estimators: int,
    num_classes: int,
    groups: int,
    style: str = "imagenet",
) -> _LpbnnResNet:
    return _LpbnnResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 23, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        groups=groups,
        num_classes=num_classes,
        style=style,
    )


def lpbnn_resnet152(
    in_channels: int,
    num_estimators: int,
    num_classes: int,
    groups: int,
    style: str = "imagenet",
) -> _LpbnnResNet:
    return _LpbnnResNet(
        block=Bottleneck,
        num_blocks=[3, 8, 36, 3],
        in_channels=in_channels,
        num_estimators=num_estimators,
        groups=groups,
        num_classes=num_classes,
        style=style,
    )
