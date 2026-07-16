import copy
import torch
import torch.nn as nn
import torchvision

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.InstanceNorm3d(out_channels, affine=True),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    """
    3D ResNet-style model for multi-label CT volume classification.
    """

    def __init__(self, n_outputs: int = 14, base_channels: int = 32, dropout: float = 0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(
                1,
                base_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.InstanceNorm3d(base_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.in_channels = base_channels

        self.layer1 = self._make_layer(base_channels, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_channels * 8, n_outputs),
        )

    def _make_layer(self, out_channels, blocks, stride):
        layers = []

        layers.append(
            BasicBlock3D(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
            )
        )
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                BasicBlock3D(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.flatten(1)

        return self.head(x)


def _inflate_conv(conv2d: nn.Conv2d) -> nn.Conv3d:
    """
    Inflate a pretrained Conv2d into a Conv3d using the "bootstrapped" weight
    inflation trick from I3D (Carreira & Zisserman, 2017, "Quo Vadis, Action
    Recognition?"). The 2D filter is repeated along a new temporal/depth axis
    and rescaled by 1/T, so a constant signal across that axis reproduces the
    original 2D response.

    Assumes square spatial kernels/strides/paddings, which holds for all
    torchvision ResNet variants.
    """
    kh, kw = conv2d.kernel_size
    sh, sw = conv2d.stride
    ph, pw = conv2d.padding

    t = kh  # inflate the depth kernel to match the spatial kernel size

    conv3d = nn.Conv3d(
        in_channels=conv2d.in_channels,
        out_channels=conv2d.out_channels,
        kernel_size=(t, kh, kw),
        stride=(sh, sh, sw),
        padding=(ph, ph, pw),
        groups=conv2d.groups,
        bias=(conv2d.bias is not None),
    )

    with torch.no_grad():
        weight_3d = conv2d.weight.data.unsqueeze(2).repeat(1, 1, t, 1, 1) / t
        conv3d.weight.copy_(weight_3d)

        if conv2d.bias is not None:
            conv3d.bias.copy_(conv2d.bias.data)

    return conv3d


def _inflate_stem_conv(conv2d: nn.Conv2d, in_channels: int = 1) -> nn.Conv3d:
    """
    Inflate the first (stem) conv, additionally collapsing the pretrained
    3-channel RGB input weights down to `in_channels` (1 for grayscale CT),
    by averaging across the pretrained input channels.
    """
    inflated = _inflate_conv(conv2d)

    with torch.no_grad():
        averaged = inflated.weight.data.mean(dim=1, keepdim=True)
        new_weight = averaged.repeat(1, in_channels, 1, 1, 1)

    new_conv3d = nn.Conv3d(
        in_channels=in_channels,
        out_channels=inflated.out_channels,
        kernel_size=inflated.kernel_size,
        stride=inflated.stride,
        padding=inflated.padding,
        bias=(inflated.bias is not None),
    )

    with torch.no_grad():
        new_conv3d.weight.copy_(new_weight)
        if inflated.bias is not None:
            new_conv3d.bias.copy_(inflated.bias.data)

    return new_conv3d


def _bn2d_to_bn3d(bn2d: nn.BatchNorm2d) -> nn.BatchNorm3d:
    """
    BatchNorm parameters (weight, bias, running_mean, running_var) are
    per-channel, so a BatchNorm2d's state dict loads directly into a
    BatchNorm3d of the same width with no reshaping needed.
    """
    bn3d = nn.BatchNorm3d(
        bn2d.num_features,
        eps=bn2d.eps,
        momentum=bn2d.momentum,
        affine=bn2d.affine,
        track_running_stats=bn2d.track_running_stats,
    )
    bn3d.load_state_dict(bn2d.state_dict())
    return bn3d


def _inflate_basic_block(block: nn.Module) -> None:
    """In-place inflation of a torchvision ResNet BasicBlock's submodules."""
    block.conv1 = _inflate_conv(block.conv1)
    block.bn1 = _bn2d_to_bn3d(block.bn1)
    block.conv2 = _inflate_conv(block.conv2)
    block.bn2 = _bn2d_to_bn3d(block.bn2)

    if block.downsample is not None:
        block.downsample[0] = _inflate_conv(block.downsample[0])
        block.downsample[1] = _bn2d_to_bn3d(block.downsample[1])


class InflatedResNet3D(nn.Module):
    """
    3D ResNet initialized from an ImageNet-pretrained 2D torchvision ResNet
    via I3D-style weight inflation.

    Rationale: labeled 3D medical volumes are scarce compared to ImageNet, so
    starting from pretrained 2D edge/texture filters (inflated into 3D) gives
    the network a substantial head start over training a 3D ResNet from
    random initialization, which is what the from-scratch `ResNet3D` above
    does.

    Only supports torchvision's BasicBlock-based backbones (resnet18,
    resnet34) since `_inflate_basic_block` assumes that block structure.
    """

    SUPPORTED_BACKBONES = ("resnet18", "resnet34")

    def __init__(
        self,
        n_outputs: int = 14,
        backbone: str = "resnet34",
        dropout: float = 0.3,
        pretrained: bool = True,
        in_channels: int = 1,
    ):
        super().__init__()

        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"backbone must be one of {self.SUPPORTED_BACKBONES}, got {backbone}"
            )

        weights = "DEFAULT" if pretrained else None
        base = getattr(torchvision.models, backbone)(weights=weights)

        self.stem = nn.Sequential(
            _inflate_stem_conv(base.conv1, in_channels=in_channels),
            _bn2d_to_bn3d(base.bn1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = copy.deepcopy(base.layer1)
        self.layer2 = copy.deepcopy(base.layer2)
        self.layer3 = copy.deepcopy(base.layer3)
        self.layer4 = copy.deepcopy(base.layer4)

        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                _inflate_basic_block(block)

        in_features = base.fc.in_features

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, n_outputs),
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.flatten(1)

        return self.head(x)


def build_model(
    model_name: str,
    n_outputs: int,
    backbone: str = "resnet34",
    dropout: float = 0.3,
    pretrained: bool = True,
):
    if model_name == "resnet3d18":
        return ResNet3D(n_outputs=n_outputs, dropout=dropout)

    if model_name == "resnet3d_inflated":
        return InflatedResNet3D(
            n_outputs=n_outputs,
            backbone=backbone,
            dropout=dropout,
            pretrained=pretrained,
        )

    raise ValueError(f"Unknown model_name: {model_name}")

