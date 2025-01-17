import torch
import torch.nn as nn


# Kalasa bazująca na teorii przedstawionej w: https://www.youtube.com/watch?v=KOF38xAvo8I&t=574s
class SpatialAttention(nn.Module):
    def __init__(self, in_channels_g, in_channels_x, intermediate_channels):
        super(SpatialAttention, self).__init__()
        self.W_g = nn.Conv2d(
            in_channels_g,
            intermediate_channels,
            kernel_size=1,
            stride=(1, 1),
            padding=0,
            bias=True,
        )
        self.W_x = nn.Conv2d(
            in_channels_x,
            intermediate_channels,
            kernel_size=1,
            stride=(2, 2),
            padding=0,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Conv2d(
            intermediate_channels, 1, kernel_size=1, stride=(1, 1), padding=0, bias=True
        )
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        combined = g1 + x1
        combined = self.relu(combined)
        psi = self.sigmoid(self.psi(combined))
        psi = self.upsample(psi)
        out = x * psi
        return out
