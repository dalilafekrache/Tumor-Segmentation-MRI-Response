import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Patch Embedding (Divise l’image en patches)
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_size=768, img_size=256):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, emb_size))

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)                        # [B, emb_size, H/patch, W/patch]
        x = rearrange(x, 'b c h w -> b (h w) c')  # [B, N_patches, emb_size]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_size]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N_patches+1, emb_size]
        x = x + self.pos_embed
        return x

# Transformer Encoder (simplifié)
class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=768, depth=4, heads=8, dim_feedforward=2048):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=heads, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)

# Decoder block CNN
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# 🧠 TransUNet 2D
class TransUNet2D(nn.Module):
    def __init__(self, img_size=256, in_channels=1, out_channels=1, patch_size=16, emb_size=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.emb_size = emb_size

        # CNN Encoder
        self.enc1 = DecoderBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)  # 128x128
        self.enc2 = DecoderBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)  # 64x64

        # Transformer
        self.patch_embed = PatchEmbedding(in_channels=128, patch_size=patch_size,
                                          emb_size=emb_size, img_size=img_size // 4)
        self.transformer = TransformerEncoder(emb_size=emb_size)

        # Reconstruction: projection linéaire vers patch puis reshape
        self.reconstruct = nn.Linear(emb_size, patch_size * patch_size)

        # CNN Decoder
        self.up1 = nn.ConvTranspose2d(128 + 1, 64, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(64 + 128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(32 + in_channels, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder CNN ---
        x1 = self.enc1(x)       # (B, 64, 256, 256)
        p1 = self.pool1(x1)     # (B, 64, 128, 128)

        x2 = self.enc2(p1)      # (B, 128, 128, 128)
        p2 = self.pool2(x2)     # (B, 128, 64, 64)

        # --- Transformer ---
        patches = self.patch_embed(p2)           # (B, N+1, emb_size)
        t_out = self.transformer(patches)        # (B, N+1, emb_size)

        # Enlever le CLS token et projeter chaque token en patch_size²
        t_feat = self.reconstruct(t_out[:, 1:])  # (B, N, patch_size²)

        B, N, D = t_feat.shape                   # D = patch_size²
        h = w = int(N**0.5)                      # N = h*w

        t_feat = t_feat.view(B, h, w, self.patch_size, self.patch_size)  # (B, h, w, ph, pw)
        t_feat = t_feat.permute(0, 3, 4, 1, 2).contiguous()               # (B, ph, pw, h, w)
        t_feat = t_feat.view(B, 1, self.img_size // 4, self.img_size // 4)  # (B, 1, 64, 64)

        # --- Fusion CNN + Transformer ---
        t_cat = torch.cat([p2, t_feat], dim=1)  # (B, 128+1, 64, 64)

        # --- Decoder ---
        up1 = self.up1(t_cat)               # (B, 64, 128, 128)
        cat1 = torch.cat([up1, x2], dim=1)  # (B, 128, 128, 128)
        d1 = self.dec1(cat1)

        up2 = self.up2(d1)                  # (B, 32, 256, 256)
        cat2 = torch.cat([up2, x], dim=1)   # (B, 32+1, 256, 256)
        d2 = self.dec2(cat2)

        out = self.final(d2)                # (B, 1, 256, 256)
        return out
