import torch
import torch.nn as nn
import torch.nn.functional as F

# Bloc de convolution simple
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# Bloc d’attention inspiré de UCTransNet
class UCA_Block(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# UCTransNet 2D
class UCTransNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck + attention
        self.bottleneck = ConvBlock(512, 1024)
        self.attn = UCA_Block(1024)

        # Decoder
        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(512, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(256, 128)
        self.up0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(128, 64)

        # Output
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)               # (B, 64, 256, 256)
        e2 = self.enc2(self.pool1(e1)) # (B, 128, 128, 128)
        e3 = self.enc3(self.pool2(e2)) # (B, 256, 64, 64)
        e4 = self.enc4(self.pool3(e3)) # (B, 512, 32, 32)

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))  # (B, 1024, 16, 16)
        b = self.attn(b)

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(b), e4], dim=1))  # (B, 512, 32, 32)
        d2 = self.dec2(torch.cat([self.up2(d3), e3], dim=1)) # (B, 256, 64, 64)
        d1 = self.dec1(torch.cat([self.up1(d2), e2], dim=1)) # (B, 128, 128, 128)
        d0 = self.dec0(torch.cat([self.up0(d1), e1], dim=1)) # (B, 64, 256, 256)

        # Final output
        out = self.final(d0)
        return out
