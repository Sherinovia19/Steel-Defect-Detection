"""
Multi-Scale Convolution (MSC) module used in the proposed YOLO26m model.

The module replaces one intermediate 3x3 convolution in YOLO26m
with two parallel convolution branches:
    - 3x3 convolution for local features
    - 5x5 convolution for broader spatial features

The outputs are concatenated to preserve the original channel count.
"""

import torch
import torch.nn as nn


class MSC(nn.Module):
    """
    Multi-Scale Convolution module.

    Args:
        channels (int): Number of input and output channels.
    """

    def __init__(self, channels: int = 128):
        super().__init__()

        branch_channels = channels // 2

        self.branch3 = nn.Conv2d(
            channels,
            branch_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.branch5 = nn.Conv2d(
            channels,
            branch_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply parallel 3x3 and 5x5 convolutions and concatenate outputs.
        """
        x3 = self.branch3(x)
        x5 = self.branch5(x)

        return torch.cat((x3, x5), dim=1)


if __name__ == "__main__":
    # Simple shape test
    x = torch.randn(1, 128, 25, 25)

    msc = MSC(channels=128)
    output = msc(x)

    print("Input shape :", x.shape)
    print("Output shape:", output.shape)
