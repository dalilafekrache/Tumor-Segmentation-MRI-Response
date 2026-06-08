import torch
import torch.nn as nn
import torch.nn.functional as F

class TinySwinBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(hidden_dim)

        # Adapter residual si nécessaire
        self.residual_conv = None
        if in_channels != hidden_dim:
            self.residual_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

    def forward(self, x):
        residual = x
        if self.residual_conv:
            residual = self.residual_conv(residual)
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)


class MiniSwinUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder1 = TinySwinBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = TinySwinBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = TinySwinBlock(64, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = TinySwinBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = TinySwinBlock(64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.decoder1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)

        return torch.sigmoid(self.final(d2))
