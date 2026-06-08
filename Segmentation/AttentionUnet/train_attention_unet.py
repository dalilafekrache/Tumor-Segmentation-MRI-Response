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
from attention_unet_model import AttentionUNet2D
from dataset import NiftiSegmentationDataset2D
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
    probs = torch.sigmoid(pred)
    pred_bin = (probs > 0.5).cpu().numpy().astype(np.uint8).flatten()
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

# Dataset et DataLoader
transform = Resize((256, 256))
dataset = NiftiSegmentationDataset2D("dataset/images", "dataset/masks", transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Modèle Attention U-Net 2D
model = AttentionUNet2D(in_ch=1, out_ch=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Critère, optimisateur
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Fichier métriques
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
metrics_filename = f"metrics/training_metrics_{timestamp}.csv"
metrics_file = open(metrics_filename, mode="w", newline="")
csv_writer = csv.writer(metrics_file)
csv_writer.writerow(["Epoch", "Loss", "Dice", "IoU", "Precision", "Recall"])

# 🌟 Early Stopping
patience = 10
best_loss = float('inf')
counter = 0
best_model_path = "best_model.pth"

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    metrics_sum = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0}
    n_samples = 0

    print(f"\n💝 Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (imgs, masks) in enumerate(tqdm(dataloader, desc="  Batches")):
        imgs, masks = imgs.to(device), masks.to(device)

        # Filtrer les images sans annotation
        non_empty = (masks.view(masks.size(0), -1).sum(dim=1) > 0)
        if non_empty.sum() == 0:
            continue
        imgs, masks = imgs[non_empty], masks[non_empty]

        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # Debug visuel pour les premiers epochs
        if batch_idx < 3 and epoch < 3:
            for i in range(imgs.size(0)):
                tqdm.write(f"[DEBUG] Epoch {epoch+1} - Batch {batch_idx+1} - Slice {i}: mask_sum={masks[i].sum().item():.1f}, pred_sum={(torch.sigmoid(preds[i])>0.5).sum().item():.1f}")
                show_segmentation(imgs[i], preds[i], masks[i], f"{batch_idx}_{i}", epoch, debug=True)

        # 🔥 Sauvegarde visuelle à CHAQUE epoch
        for i in range(imgs.size(0)):
            if i == 0:
                show_segmentation(imgs[i], preds[i], masks[i], batch_idx, epoch)

        # Écriture de la progression batch par batch
        tqdm.write(f"    Batch {batch_idx+1}/{len(dataloader)} - Loss: {loss.item():.4f}")

        # Calcul des métriques
        for i in range(imgs.size(0)):
            m = compute_metrics(preds[i], masks[i])
            for k in metrics_sum:
                metrics_sum[k] += m[k]
            n_samples += 1

    # Moyennes et affichage en fin d'epoch
    avg_loss = epoch_loss / len(dataloader)
    avg_metrics = {k: metrics_sum[k] / n_samples for k in metrics_sum}
    print(f"✅ Epoch {epoch+1} terminé - Avg Loss: {avg_loss:.4f}")
    print(f"📊 Dice: {avg_metrics['dice']:.4f} | IoU: {avg_metrics['iou']:.4f} | Precision: {avg_metrics['precision']:.4f} | Recall: {avg_metrics['recall']:.4f}")

    # Enregistrement dans le CSV
    csv_writer.writerow([epoch+1, avg_loss, avg_metrics['dice'], avg_metrics['iou'], avg_metrics['precision'], avg_metrics['recall']])
    metrics_file.flush()

    # 🎯 Early Stopping Check
    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"🌟 Nouveau meilleur modèle sauvegardé avec loss={best_loss:.4f}")
    else:
        counter += 1
        print(f"⏳ EarlyStopping: {counter}/{patience} (meilleure loss {best_loss:.4f})")
        if counter >= patience:
            print("⛔ Early stopping déclenché.")
            break

# Sauvegarde du modèle final
torch.save(model.state_dict(), "attention_unet_final.pth")
metrics_file.close()
print("✅ Modèle final sauvegardé et métriques enregistrées")

# Génération des visualisations finales
print("📸 Génération des visualisations finales...")
model.eval()
with torch.no_grad():
    for i in range(5):
        img, mask = dataset[i]
        img, mask = img.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)
        pred = model(img)
        show_segmentation(img.squeeze(0), pred.squeeze(0), mask.squeeze(0), f"final_{i}", epoch)

print("📁 Tous les résultats sont dans 'segmentations'")
