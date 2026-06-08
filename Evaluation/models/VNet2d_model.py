import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(DownBlock, self).__init__()
        self.block = ConvBlock(in_channels, out_channels, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.block(x)
        x_pooled = self.pool(x)
        return x, x_pooled

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class VNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(VNet2D, self).__init__()

        # Encoder
        self.enc1 = DownBlock(in_channels, 16)
        self.enc2 = DownBlock(16, 32)
        self.enc3 = DownBlock(32, 64)
        self.enc4 = DownBlock(64, 128)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.dec4 = UpBlock(256, 128)
        self.dec3 = UpBlock(128, 64)
        self.dec2 = UpBlock(64, 32)
        self.dec1 = UpBlock(32, 16)

        # Output
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x1, x = self.enc1(x)
        x2, x = self.enc2(x)
        x3, x = self.enc3(x)
        x4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)

        return self.final(x)
