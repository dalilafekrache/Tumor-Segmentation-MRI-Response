import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B x N x C
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x) + residual
        x = x.transpose(1, 2).reshape(B, C, H, W)  # reshape back to B x C x H x W
        return x

class MedFormer2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.trans = TransformerBlock(dim=128, heads=4, mlp_dim=256)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(192, 64)  # 64 from up2 + 128 from e2

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(96, 32)  # 32 from up1 + 64 from e1

        self.out = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)  # [B, 64, H, W]
        e2 = self.enc2(self.pool1(e1))  # [B, 128, H/2, W/2]
        t = self.trans(self.pool2(e2))  # [B, 128, H/4, W/4]

        d2 = self.up2(t)  # [B, 64, H/2, W/2]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # [B, 64, H/2, W/2]

        d1 = self.up1(d2)  # [B, 32, H, W]
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # [B, 32, H, W]

        return self.out(d1)
