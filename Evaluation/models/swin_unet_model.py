import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import swin_tiny_patch4_window7_224

class SwinUnet(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        super(SwinUnet, self).__init__()
        self.encoder = swin_tiny_patch4_window7_224(pretrained=pretrained)
        self.encoder.head = nn.Identity()
        self.hidden_dim = self.encoder.num_features  # 768

        # Décodeur UNet
        self.up1 = nn.ConvTranspose2d(self.hidden_dim, 384, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(48, num_classes, kernel_size=1)

    def forward(self, x):
        B = x.size(0)

        # Convertir 1 canal → 3 canaux
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        features = self.encoder.forward_features(x)  # (B, 768)

        # Pour une image 224x224, swin donne (B, 768)
        # donc spatial_dim doit être 7
        spatial_dim = 7
        try:
            x = features.view(B, self.hidden_dim, spatial_dim, spatial_dim)
        except Exception as e:
            print(f"[ERREUR] Mauvais reshape. Feature shape: {features.shape}")
            raise e

        x = F.relu(self.up1(x))  # (B, 384, 14, 14)
        x = F.relu(self.up2(x))  # (B, 192, 28, 28)
        x = F.relu(self.up3(x))  # (B, 96, 56, 56)
        x = F.relu(self.up4(x))  # (B, 48, 112, 112)

        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        out = self.final_conv(x)
        return out
