import os
import shutil
import csv
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.metrics import precision_score, recall_score, jaccard_score

from VNet2d_model import VNet2D
from dataset import NiftiSegmentationDataset2D

# 🔁 Nettoyage des anciens dossiers (conserver les résultats précédents si désiré)
for folder in ["segmentations", "metrics", "debug"]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
for folder in ["segmentations", "metrics", "debug"]:
    os.makedirs(folder, exist_ok=True)

# ⚙️ Fonctions utilitaires
def dice_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + eps)

def compute_metrics(pred, mask):
    probs = torch.sigmoid(pred)
    pred_bin = (probs > 0.5).cpu().numpy().astype(np.uint8).flatten()
    target_bin = (mask > 0.5).cpu().numpy().astype(np.uint8).flatten()
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
    folder = "debug" if debug else "segmentations"
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, f"seg_epoch_{epoch+1}_img_{index}.png")
    plt.savefig(fname)
    plt.close()

# 📊 Initialisation du dataset et DataLoader
transform = Resize((256, 256))

image_dir = r"C:\Users\Admin\PycharmProjects2\VNet2d\dataset\images"
mask_dir = r"C:\Users\Admin\PycharmProjects2\VNet2d\dataset\masks"

if not os.path.isdir(image_dir) or not os.listdir(image_dir):
    raise FileNotFoundError(f"Aucun fichier image dans {image_dir}")
if not os.path.isdir(mask_dir) or not os.listdir(mask_dir):
    raise FileNotFoundError(f"Aucun fichier masque dans {mask_dir}")

dataset = NiftiSegmentationDataset2D(image_dir, mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# ⚙️ Configuration de l'entraînement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VNet2D(in_channels=1, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 🧾 Enregistrement des métriques
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
metrics_filename = f"metrics/vnet_training_{timestamp}.csv"
metrics_file = open(metrics_filename, mode="w", newline="")
csv_writer = csv.writer(metrics_file)
csv_writer.writerow(["Epoch", "Loss", "Dice", "IoU", "Precision", "Recall"])

# 🛑 Early stopping
best_loss = float('inf')
patience = 10
counter = 0
best_model_path = "vnet_best.pth"
num_epochs = 250

# 🚀 Entraînement
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    epoch_loss = 0.0
    metrics_sum = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
    n_samples = 0

    print(f"\n🎯 Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (imgs, masks) in enumerate(tqdm(dataloader, desc="Batches")):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).float()
        for i in range(imgs.size(0)):
            show_segmentation(imgs[i], outputs[i], masks[i], f"{batch_idx}_{i}", epoch)
            m = compute_metrics(outputs[i], masks[i])
            for k in metrics_sum:
                metrics_sum[k] += m[k]
            n_samples += 1

    avg_loss = epoch_loss / len(dataloader)
    avg_metrics = {k: metrics_sum[k]/n_samples for k in metrics_sum}

    elapsed = time.time() - start_time
    print(f"✅ Epoch {epoch+1} - Loss: {avg_loss:.4f} | Dice: {avg_metrics['dice']:.4f} | IoU: {avg_metrics['iou']:.4f} | Prec: {avg_metrics['precision']:.4f} | Rec: {avg_metrics['recall']:.4f} | Time: {elapsed:.1f}s")

    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"✔️ Meilleur modèle sauvegardé (loss={best_loss:.4f})")
    else:
        counter += 1
        print(f"⏳ Pas d'amélioration du loss: {counter}/{patience}")
        if counter >= patience:
            print("🛑 Early stopping activé.")
            break

    csv_writer.writerow([epoch+1, avg_loss,
                         avg_metrics['dice'], avg_metrics['iou'],
                         avg_metrics['precision'], avg_metrics['recall']])
    metrics_file.flush()

metrics_file.close()
print("✅ Entraînement terminé. Best model saved.")

