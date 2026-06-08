import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from einops import rearrange

# =====================
# Dataset
# =====================
from dataset import MedicalDataset2D  # ton dataset.py avec target_size


# =====================
# Patch Embedding avec interpolation pos_embed
# =====================
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=128, patch_size=16, emb_size=768, img_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, emb_size))

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)                           # [B, emb_size, H/patch, W/patch]
        x = rearrange(x, 'b c h w -> b (h w) c')   # [B, N_patches, emb_size]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_size]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, N+1, emb_size]

        # Interpolation du pos_embed si nécessaire
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed[:, 1:]
            N = x.size(1) - 1
            h = w = int(N ** 0.5)
            pos_embed = pos_embed.transpose(1, 2).view(
                1, self.emb_size, int(self.n_patches ** 0.5), int(self.n_patches ** 0.5)
            )
            pos_embed = F.interpolate(pos_embed, size=(h, w), mode="bilinear", align_corners=False)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            x = x + torch.cat([self.pos_embed[:, :1], pos_embed], dim=1)
        else:
            x = x + self.pos_embed

        return x


# =====================
# Transformer Encoder
# =====================
class TransformerEncoder(nn.Module):
    def __init__(self, emb_size=768, depth=4, heads=8, dim_feedforward=2048):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=heads, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)


# =====================
# CNN Decoder Block
# =====================
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


# =====================
# TransUNet2D corrigé
# =====================
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

        # Reconstruction
        self.reconstruct = nn.Linear(emb_size, patch_size * patch_size)

        # Decoder
        self.up1 = nn.ConvTranspose2d(128 + 1, 64, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(64 + 128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(32 + in_channels, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x1 = self.enc1(x)       # (B, 64, 256, 256)
        p1 = self.pool1(x1)     # (B, 64, 128, 128)

        x2 = self.enc2(p1)      # (B, 128, 128, 128)
        p2 = self.pool2(x2)     # (B, 128, 64, 64)

        # --- Transformer ---
        patches = self.patch_embed(p2)           # (B, N+1, emb_size)
        t_out = self.transformer(patches)        # (B, N+1, emb_size)

        t_feat = self.reconstruct(t_out[:, 1:])  # (B, N, patch_size²)
        B, N, D = t_feat.shape
        h = w = int(N ** 0.5)

        t_feat = t_feat.view(B, h, w, self.patch_size, self.patch_size)
        t_feat = t_feat.permute(0, 3, 4, 1, 2).contiguous()
        t_feat = t_feat.view(B, 1, self.img_size // 4, self.img_size // 4)

        # --- Fusion + Decoder ---
        t_cat = torch.cat([p2, t_feat], dim=1)
        up1 = self.up1(t_cat)
        cat1 = torch.cat([up1, x2], dim=1)
        d1 = self.dec1(cat1)

        up2 = self.up2(d1)
        cat2 = torch.cat([up2, x], dim=1)
        d2 = self.dec2(cat2)

        out = self.final(d2)
        return out


# =====================
# Métriques
# =====================
def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f, y_pred_f = y_true.view(-1), y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def iou_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f, y_pred_f = y_true.view(-1), y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def precision_score(y_true, y_pred):
    y_true_f, y_pred_f = y_true.view(-1), y_pred.view(-1)
    tp = (y_true_f * y_pred_f).sum()
    fp = ((1 - y_true_f) * y_pred_f).sum()
    return tp / (tp + fp + 1e-6)


def recall_score(y_true, y_pred):
    y_true_f, y_pred_f = y_true.view(-1), y_pred.view(-1)
    tp = (y_true_f * y_pred_f).sum()
    fn = (y_true_f * (1 - y_pred_f)).sum()
    return tp / (tp + fn + 1e-6)


# =====================
# Sauvegarde overlay
# =====================
def save_prediction_overlay(image, mask, pred, save_path):
    img, mask, pred = image.squeeze().cpu().numpy(), mask.squeeze().cpu().numpy(), pred.squeeze().cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img, cmap="gray"); axs[0].set_title("Image"); axs[0].axis("off")
    axs[1].imshow(mask, cmap="gray"); axs[1].set_title("Mask réel"); axs[1].axis("off")
    axs[2].imshow(img, cmap="gray"); axs[2].imshow(pred, cmap="Reds", alpha=0.5)
    axs[2].set_title("Prédiction (rouge)"); axs[2].axis("off")
    plt.tight_layout(); plt.savefig(save_path, bbox_inches="tight"); plt.close()


# =====================
# Test TransUNet
# =====================
def test_transunet(model, test_loader, device, output_dir):
    model.eval()
    patients, slices = [], []
    dice_scores, iou_scores, precision_scores, recall_scores = [], [], [], []

    seg_dir = os.path.join(output_dir, "prediction", "transunet_model")
    os.makedirs(seg_dir, exist_ok=True)

    with torch.no_grad():
        for images, masks, patient_ids, slice_idxs in test_loader:
            print(f"[TransUNet] Batch shape : {images.shape}")
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            for b in range(preds.size(0)):
                patient_id, slice_idx = patient_ids[b], slice_idxs[b].item()
                save_path = os.path.join(seg_dir, f"{patient_id}_slice{slice_idx}_overlay.png")
                save_prediction_overlay(images[b], masks[b], preds[b], save_path)

                patients.append(patient_id); slices.append(slice_idx)
                dice_scores.append(dice_coef(masks[b], preds[b]).item())
                iou_scores.append(iou_coef(masks[b], preds[b]).item())
                precision_scores.append(precision_score(masks[b], preds[b]).item())
                recall_scores.append(recall_score(masks[b], preds[b]).item())

    metrics_df = pd.DataFrame({
        "Patient": patients, "Slice": slices,
        "Dice": dice_scores, "IoU": iou_scores,
        "Precision": precision_scores, "Recall": recall_scores
    })
    metrics_df.to_csv(os.path.join(output_dir, "metrics_transunet.csv"), index=False)

    summary = metrics_df[["Dice", "IoU", "Precision", "Recall"]].mean().to_dict()
    summary.update({k+"_std": metrics_df[k].std() for k in ["Dice","IoU","Precision","Recall"]})
    summary.update({k+"_median": metrics_df[k].median() for k in ["Dice","IoU","Precision","Recall"]})
    pd.DataFrame([summary], index=["transunet"]).to_csv(os.path.join(output_dir, "summary_transunet.csv"))

    print("\n✅ TransUNet terminé. Résumé :")
    print(pd.DataFrame([summary], index=["transunet"]).round(4))
    return summary


# =====================
# Main
# =====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = r"C:\Users\Admin\PycharmProjects\pythonProjectTest"

    image_dir, mask_dir = os.path.join(base_dir, "dataset", "images"), os.path.join(base_dir, "dataset", "masks")
    test_dataset = MedicalDataset2D(image_dir, mask_dir, empty_ratio=0.2, target_size=(256, 256))
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    ckpt_path = os.path.join(base_dir, "checkpoints", "best_model_Transunet.pth")
    model = TransUNet2D(img_size=256, in_channels=1, out_channels=1).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # strict=False pour pos_embed

    output_dir = os.path.join(base_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    test_transunet(model, test_loader, device, output_dir)
