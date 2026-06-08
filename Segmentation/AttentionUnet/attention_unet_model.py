import torch
import torch.nn as nn

# 📦 Bloc de Convolution
class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# 📦 Bloc d'Attention
class AttentionBlock2D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock2D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# 🧠 Modèle Attention UNet (2D)
class AttentionUNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super(AttentionUNet2D, self).__init__()

        # Encoder
        self.enc1 = ConvBlock2D(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock2D(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock2D(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock2D(base_ch * 4, base_ch * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.att3 = AttentionBlock2D(F_g=base_ch * 4, F_l=base_ch * 4, F_int=base_ch * 2)
        self.dec3 = ConvBlock2D(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.att2 = AttentionBlock2D(F_g=base_ch * 2, F_l=base_ch * 2, F_int=base_ch)
        self.dec2 = ConvBlock2D(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.att1 = AttentionBlock2D(F_g=base_ch, F_l=base_ch, F_int=base_ch // 2)
        self.dec1 = ConvBlock2D(base_ch * 2, base_ch)

        # Final output
        self.final = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        x3 = self.enc3(p2)
        p3 = self.pool3(x3)

        # Bottleneck
        x4 = self.bottleneck(p3)

        # Decoder avec Attention
        up3 = self.up3(x4)
        att3 = self.att3(g=up3, x=x3)
        cat3 = torch.cat([up3, att3], dim=1)
        x3d = self.dec3(cat3)

        up2 = self.up2(x3d)
        att2 = self.att2(g=up2, x=x2)
        cat2 = torch.cat([up2, att2], dim=1)
        x2d = self.dec2(cat2)

        up1 = self.up1(x2d)
        att1 = self.att1(g=up1, x=x1)
        cat1 = torch.cat([up1, att1], dim=1)
        x1d = self.dec1(cat1)

        out = self.final(x1d)
        return out
