import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import shutil
from datetime import datetime

from unet_model import UNet
from dataset import NiftiSegmentationDataset
from sklearn.metrics import precision_score, recall_score, jaccard_score

# 🔁 Nettoyage des anciens dossiers
for folder in ["segmentations", "metrics", "debug"]:
    if os.path.exists(folder):
        shutil.rmtree(folder)

# 📁 Recréation des dossiers
os.makedirs("segmentations", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("debug", exist_ok=True)

def dice_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + eps)

def compute_metrics(pred, target):
    pred = torch.sigmoid(pred)
    pred_bin = (pred > 0.5).cpu().numpy().astype(np.uint8).flatten()
    target_bin = (target > 0.5).cpu().numpy().astype(np.uint8).flatten()
    return {
        'dice': dice_score(torch.tensor(pred_bin), torch.tensor(target_bin)).item(),
        'iou': jaccard_score(target_bin, pred_bin, zero_division=0),
        'precision': precision_score(target_bin, pred_bin, zero_division=0),
        'recall': recall_score(target_bin, pred_bin, zero_division=0)
    }

def show_segmentation(img, pred, mask, index, epoch, debug=False):
    pred = torch.sigmoid(pred)
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
    plt.title("Image")

    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    plt.title("Mask réel")

    plt.subplot(1, 3, 3)
    plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
    plt.imshow((pred.squeeze().detach().cpu().numpy() > 0.5), cmap='Reds', alpha=0.5)
    plt.title("Prédiction (rouge)")

    plt.tight_layout()
    fname = f"debug/epoch_{epoch+1}_img_{index}.png" if debug else f"segmentations/seg_epoch_{epoch+1}_img_{index}.png"
    plt.savefig(fname)
    plt.close()

transform = Resize((256, 256))
dataset = NiftiSegmentationDataset("dataset/images", "dataset/masks", transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = UNet(in_channels=1, out_channels=1)
device = torch.device("cpu")
model.to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 🔖 Ajout de timestamp dans le nom du fichier CSV
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
metrics_filename = f"metrics/training_metrics_{timestamp}.csv"
metrics_file = open(metrics_filename, mode="w", newline="")
csv_writer = csv.writer(metrics_file)
csv_writer.writerow(["Epoch", "Loss", "Dice", "IoU", "Precision", "Recall"])

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    metrics_sum = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0}
    n_samples = 0

    print(f"\n💝 Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (imgs, masks) in enumerate(tqdm(dataloader, desc="  Batches")):
        imgs, masks = imgs.to(device), masks.to(device)

        non_empty = (masks.view(masks.size(0), -1).sum(dim=1) > 0)
        if non_empty.sum() == 0:
            continue

        imgs = imgs[non_empty]
        masks = masks[non_empty]

        preds = model(imgs)
        masks = (masks > 0).float()

        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if batch_idx < 3 and epoch < 3:
            for i in range(imgs.size(0)):
                pred_bin = (torch.sigmoid(preds[i]) > 0.5).float()
                print(f"[DEBUG] Epoch {epoch+1} - Batch {batch_idx+1} - Slice {i}: mask_sum={masks[i].sum().item():.1f}, pred_sum={pred_bin.sum().item():.1f}")
                show_segmentation(imgs[i], preds[i], masks[i], f"{batch_idx}_{i}", epoch, debug=True)

        for i in range(imgs.size(0)):
            met = compute_metrics(preds[i], masks[i])
            for key in metrics_sum:
                metrics_sum[key] += met[key]
            n_samples += 1

            if epoch % 10 == 0 and i == 0:
                show_segmentation(imgs[i], preds[i], masks[i], batch_idx, epoch)

        tqdm.write(f"    Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(dataloader)
    avg_metrics = {key: metrics_sum[key] / n_samples for key in metrics_sum}

    print(f"✅ Epoch {epoch+1} terminé - Avg Loss: {avg_loss:.4f}")
    print("📊 Moyennes des métriques:")
    for key, value in avg_metrics.items():
        print(f"  {key.capitalize()} : {value:.4f}")

    csv_writer.writerow([epoch+1, avg_loss, avg_metrics['dice'], avg_metrics['iou'], avg_metrics['precision'], avg_metrics['recall']])
    metrics_file.flush()  # ✅ Force l'écriture dans le fichier à chaque époque

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

torch.save(model.state_dict(), "unet_final.pth")
metrics_file.close()
print("✅ Modèle sauvegardé - Métriques enregistrées")
