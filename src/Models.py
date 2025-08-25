import torch
import torch.nn as nn


class ResBottleNeck(nn.Module):
    """Bottleneck residual block (1x1 → 3x3 → 1x1) with optional downsampling."""

    def __init__(self, intermediate_channels: int, downsample: bool, is_first_block: bool) -> None:
        super().__init__()
        self.relu = nn.ReLU()

        if is_first_block:
            if not downsample:
                in_channels = 64
            else:
                in_channels = intermediate_channels * 2
            out_channels = intermediate_channels * 4
            first_stride, first_padding = 2, 0
        else:
            out_channels = in_channels = intermediate_channels * 4
            first_stride, first_padding = 1, 0

        if not downsample:  # conv2_x
            self.inner_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=1, stride=1, padding=0)
            self.inner_bn1 = nn.BatchNorm2d(num_features=intermediate_channels)
            self.inner_conv2 = nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels, kernel_size=3, stride=1, padding=1)
            self.inner_bn2 = nn.BatchNorm2d(num_features=intermediate_channels)
            self.inner_conv3 = nn.Conv2d(in_channels=intermediate_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.inner_bn3 = nn.BatchNorm2d(num_features=out_channels)
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(num_features=out_channels),
            )
        else:  # conv3_x, conv4_x, conv5_x
            self.inner_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=1, stride=first_stride, padding=first_padding)
            self.inner_bn1 = nn.BatchNorm2d(num_features=intermediate_channels)
            self.inner_conv2 = nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels, kernel_size=3, stride=1, padding=1)
            self.inner_bn2 = nn.BatchNorm2d(num_features=intermediate_channels)
            self.inner_conv3 = nn.Conv2d(in_channels=intermediate_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.inner_bn3 = nn.BatchNorm2d(num_features=out_channels)
            if is_first_block:
                self.skip_connection = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0),
                    nn.BatchNorm2d(num_features=out_channels),
                )
            else:
                self.skip_connection = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(num_features=out_channels),
                )

    def forward(self, x):
        y = x
        x = self.inner_conv1(x)
        x = self.inner_bn1(x)
        x = self.relu(x)
        x = self.inner_conv2(x)
        x = self.inner_bn2(x)
        x = self.relu(x)
        x = self.inner_conv3(x)
        x = self.inner_bn3(x)
        x = x + self.skip_connection(y)
        x = self.relu(x)
        return x


class ResNet50(nn.Module):
    """ResNet-50 classifier; input [B, C, 224, 224] → logits [B, num_classes]."""

    def __init__(self, input_channel: int, label_num: int) -> None:
        super().__init__()
        self.input_channel = input_channel
        self.label_num = label_num
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=self.input_channel, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = nn.Sequential(
            ResBottleNeck(intermediate_channels=64, downsample=False, is_first_block=True),
            ResBottleNeck(intermediate_channels=64, downsample=False, is_first_block=False),
            ResBottleNeck(intermediate_channels=64, downsample=False, is_first_block=False),
        )
        self.conv3_x = nn.Sequential(
            ResBottleNeck(intermediate_channels=128, downsample=True, is_first_block=True),
            ResBottleNeck(intermediate_channels=128, downsample=True, is_first_block=False),
            ResBottleNeck(intermediate_channels=128, downsample=True, is_first_block=False),
        )
        self.conv4_x = nn.Sequential(
            ResBottleNeck(intermediate_channels=256, downsample=True, is_first_block=True),
            ResBottleNeck(intermediate_channels=256, downsample=True, is_first_block=False),
            ResBottleNeck(intermediate_channels=256, downsample=True, is_first_block=False),
        )
        self.conv5_x = nn.Sequential(
            ResBottleNeck(intermediate_channels=512, downsample=True, is_first_block=True),
            ResBottleNeck(intermediate_channels=512, downsample=True, is_first_block=False),
            ResBottleNeck(intermediate_channels=512, downsample=True, is_first_block=False),
        )
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(in_features=2048, out_features=self.label_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.pool2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x


class ResNet101(nn.Module):
    """ResNet-101 classifier; input [B, C, 224, 224] → logits [B, num_classes]."""

    def __init__(self, input_channel: int, label_num: int) -> None:
        super().__init__()
        self.input_channel = input_channel
        self.label_num = label_num
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=self.input_channel, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = nn.Sequential(
            ResBottleNeck(64, downsample=False, is_first_block=True),
            ResBottleNeck(64, downsample=False, is_first_block=False),
            ResBottleNeck(64, downsample=False, is_first_block=False),
        )
        self.conv3_x = nn.Sequential(
            ResBottleNeck(128, downsample=True, is_first_block=True),
            *[ResBottleNeck(128, downsample=True, is_first_block=False) for _ in range(3)],
        )
        self.conv4_x = nn.Sequential(
            ResBottleNeck(256, downsample=True, is_first_block=True),
            *[ResBottleNeck(256, downsample=True, is_first_block=False) for _ in range(22)],
        )
        self.conv5_x = nn.Sequential(
            ResBottleNeck(512, downsample=True, is_first_block=True),
            ResBottleNeck(512, downsample=True, is_first_block=False),
            ResBottleNeck(512, downsample=True, is_first_block=False),
        )
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear = nn.Linear(in_features=2048, out_features=self.label_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ResNet50_MCDropout(nn.Module):
    """ResNet-50 with MC Dropout (p=0.3) after GAP for predictive uncertainty."""

    def __init__(self, input_channel: int, label_num: int) -> None:
        super().__init__()
        self.input_channel = input_channel
        self.label_num = label_num
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=self.input_channel, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2_x = nn.Sequential(
            ResBottleNeck(intermediate_channels=64, downsample=False, is_first_block=True),
            ResBottleNeck(intermediate_channels=64, downsample=False, is_first_block=False),
            ResBottleNeck(intermediate_channels=64, downsample=False, is_first_block=False),
        )
        self.conv3_x = nn.Sequential(
            ResBottleNeck(intermediate_channels=128, downsample=True, is_first_block=True),
            ResBottleNeck(intermediate_channels=128, downsample=True, is_first_block=False),
            ResBottleNeck(intermediate_channels=128, downsample=True, is_first_block=False),
        )
        self.conv4_x = nn.Sequential(
            ResBottleNeck(intermediate_channels=256, downsample=True, is_first_block=True),
            ResBottleNeck(intermediate_channels=256, downsample=True, is_first_block=False),
            ResBottleNeck(intermediate_channels=256, downsample=True, is_first_block=False),
        )
        self.conv5_x = nn.Sequential(
            ResBottleNeck(intermediate_channels=512, downsample=True, is_first_block=True),
            ResBottleNeck(intermediate_channels=512, downsample=True, is_first_block=False),
            ResBottleNeck(intermediate_channels=512, downsample=True, is_first_block=False),
        )
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(in_features=2048, out_features=self.label_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x
