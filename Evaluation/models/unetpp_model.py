import torch
import torch.nn as nn

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

class UNetPP(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=32):
        super(UNetPP, self).__init__()
        # Encoder
        self.enc1 = ConvBlock2D(in_ch, base_ch)
        self.enc2 = ConvBlock2D(base_ch, base_ch * 2)
        # Correction: enc3 input channels = base_ch*2
        self.enc3 = ConvBlock2D(base_ch * 2, base_ch * 4)

        self.pool = nn.MaxPool2d(2)

        # Decoder paths
        self.up2_1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.conv2_1 = ConvBlock2D(base_ch * 2, base_ch)

        self.up3_1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.conv3_1 = ConvBlock2D(base_ch * 4, base_ch * 2)

        self.up3_2 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.conv3_2 = ConvBlock2D(base_ch * 3, base_ch)

        # Final 1x1 convolution
        self.final = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0_0 = self.enc1(x)
        x1_0 = self.enc2(self.pool(x0_0))
        x2_0 = self.enc3(self.pool(x1_0))

        # Decoder level 1
        x0_1 = self.conv2_1(torch.cat([x0_0, self.up2_1(x1_0)], dim=1))
        x1_1 = self.conv3_1(torch.cat([x1_0, self.up3_1(x2_0)], dim=1))

        # Decoder level 2 (nested skip)
        x0_2 = self.conv3_2(torch.cat([x0_0, x0_1, self.up3_2(x1_1)], dim=1))

        out = self.final(x0_2)
        return out