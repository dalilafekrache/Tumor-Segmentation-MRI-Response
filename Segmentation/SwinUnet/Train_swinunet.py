import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time, csv, shutil
from datetime import datetime
from sklearn.metrics import jaccard_score, precision_score, recall_score

from dataset_swinunet import NiftiSegmentationDataset
from swin_unet_model import SwinUnet


# 🧹 Nettoyage des dossiers
for folder in ["segmentations", "metrics"]:
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)

# ⚙️ Configuration
image_dir = "dataset/images"
mask_dir = "dataset/masks"
batch_size = 2
num_epochs = 250
learning_rate = 1e-4
patience = 10
max_debug_images = 5  # ✅ Nombre max d’images sauvegardées par epoch

# 📦 Dataset & DataLoader
dataset = NiftiSegmentationDataset(image_dir, mask_dir, transform_size=(224, 224))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 🧠 Modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SwinUnet(num_classes=1).to(device)

# 🎯 Optimisation
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 📊 Fichier CSV pour métriques
csv_path = f"metrics/swinunet_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Epoch", "Loss", "Dice", "IoU", "Precision", "Recall"])

best_loss = float("inf")
counter = 0

# 📷 Fonction de visualisation
def show_segmentation(img, pred, mask, name, epoch):
    pred = torch.sigmoid(pred).detach()
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
    plt.title("Image")
    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
    plt.title("Mask réel")
    plt.subplot(1, 3, 3)
    plt.imshow(img.squeeze().cpu().numpy(), cmap='gray')
    plt.imshow((pred.squeeze().cpu().numpy() > 0.5), cmap='Reds', alpha=0.5)
    plt.title("Prédiction")
    plt.tight_layout()
    plt.savefig(f"segmentations/epoch{epoch+1}_{name}.png")
    plt.close()

# 🎓 Fonction métriques
def compute_metrics(pred, target):
    probs = torch.sigmoid(pred)
    pred_bin = (probs > 0.5).cpu().numpy().astype(np.uint8).flatten()
    target_bin = (target > 0.5).cpu().numpy().astype(np.uint8).flatten()

    if pred_bin.sum() == 0 and target_bin.sum() == 0:
        dice = 1.0
    else:
        intersection = (pred_bin * target_bin).sum()
        dice = 2. * intersection / (pred_bin.sum() + target_bin.sum() + 1e-7)

    return {
        "dice": dice,
        "iou": jaccard_score(target_bin, pred_bin, zero_division=0),
        "precision": precision_score(target_bin, pred_bin, zero_division=0),
        "recall": recall_score(target_bin, pred_bin, zero_division=0)
    }

# 🔁 Entraînement
for epoch in range(num_epochs):
    start = time.time()
    model.train()
    epoch_loss = 0
    metric_sum = {"dice": 0, "iou": 0, "precision": 0, "recall": 0}
    n_samples = 0
    debug_image_count = 0  # ✅ compteur image debug

    print(f"\n🎯 Epoch {epoch+1}/{num_epochs}")

    for batch_idx, (imgs, masks) in enumerate(tqdm(dataloader, desc="  Batches", leave=False)):
        imgs, masks = imgs.to(device), masks.to(device)

        non_empty = (masks.view(masks.size(0), -1).sum(dim=1) > 0)
        if non_empty.sum() == 0:
            continue

        imgs, masks = imgs[non_empty], masks[non_empty]

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

        # ✅ Limiter le nombre d’images enregistrées par epoch
        if debug_image_count < max_debug_images:
            for i in range(imgs.size(0)):
                if debug_image_count >= max_debug_images:
                    break
                show_segmentation(imgs[i], preds[i], masks[i], f"{batch_idx}_{i}", epoch)
                debug_image_count += 1

        with torch.no_grad():
            for i in range(imgs.size(0)):
                metrics = compute_metrics(preds[i], masks[i])
                for k in metric_sum:
                    metric_sum[k] += metrics[k]
                n_samples += 1

    avg_loss = epoch_loss / len(dataloader)
    avg_metrics = {k: metric_sum[k] / n_samples for k in metric_sum}

    print(f"✅ Epoch {epoch+1} terminé - ⏱️ Temps: {time.time() - start:.1f}s - 📉 Loss: {avg_loss:.4f}")
    print(f"📊 Dice: {avg_metrics['dice']:.4f} | IoU: {avg_metrics['iou']:.4f} | Precision: {avg_metrics['precision']:.4f} | Recall: {avg_metrics['recall']:.4f}")

    csv_writer.writerow([epoch+1, avg_loss, avg_metrics["dice"], avg_metrics["iou"], avg_metrics["precision"], avg_metrics["recall"]])
    csv_file.flush()

    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
        torch.save(model.state_dict(), "swinunet_best.pth")
        print("🎉 Nouveau meilleur modèle sauvegardé !")
    else:
        counter += 1
        print(f"⏳ EarlyStopping {counter}/{patience} (best loss: {best_loss:.4f})")
        if counter >= patience:
            print("🛑 Early stopping déclenché.")
            break

csv_file.close()
print("🏁 Entraînement terminé.")