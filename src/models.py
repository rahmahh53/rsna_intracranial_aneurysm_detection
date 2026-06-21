import torch
import torch.nn as nn


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


def build_model(model_name: str, n_outputs: int):
    if model_name == "resnet3d18":
        return ResNet3D(n_outputs=n_outputs)

    raise ValueError(f"Unknown model_name: {model_name}")
