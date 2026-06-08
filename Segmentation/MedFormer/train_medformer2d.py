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
from sklearn.metrics import precision_score, recall_score, jaccard_score
import time

from medformer_model import MedFormer2D
from dataset import NiftiSegmentationDataset2D

# ✅ Métriques
def dice_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + eps)

def compute_metrics(pred, target):
    with torch.no_grad():
        probs = torch.sigmoid(pred)
        pred_bin = (probs > 0.5).cpu().numpy().astype(np.uint8).flatten()
        target_bin = (target > 0.5).cpu().numpy().astype(np.uint8).flatten()
        return {
            'dice': dice_score(torch.tensor(pred_bin), torch.tensor(target_bin)).item(),
            'iou': jaccard_score(target_bin, pred_bin, zero_division=0),
            'precision': precision_score(target_bin, pred_bin, zero_division=0),
            'recall': recall_score(target_bin, pred_bin, zero_division=0)
        }

def show_segmentation(img, pred, mask, index, epoch):
    with torch.no_grad():
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
        plt.imshow((pred.squeeze().cpu().numpy() > 0.5), cmap='Reds', alpha=0.5)
        plt.title("Prédiction (rouge)")

        plt.tight_layout()
        fname = f"segmentations/epoch_{epoch + 1}_img_{index}.png"
        plt.savefig(fname)
        plt.close('all')

# ✅ Point d'entrée principal (Windows-safe)
if __name__ == "__main__":
    # 🔄 Nettoyage
    shutil.rmtree("segmentations", ignore_errors=True)
    os.makedirs("segmentations", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)

    # 📁 Données
    transform = Resize((256, 256))
    image_dir = r"C:\Users\Admin\PycharmProjects\MedFormer\dataset\images"
    mask_dir = r"C:\Users\Admin\PycharmProjects\MedFormer\dataset\masks"

    if len(os.listdir(image_dir)) == 0 or len(os.listdir(mask_dir)) == 0:
        raise FileNotFoundError("⚠️ Vérifie le dossier des images/masques")

    dataset = NiftiSegmentationDataset2D(image_dir, mask_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # ⚙️ Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedFormer2D(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"metrics/medformer_training_{timestamp}.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "Loss", "Dice", "IoU", "Precision", "Recall"])

    best_loss = float("inf")
    patience = 10
    counter = 0
    best_model_path = "medformer_best.pth"
    num_epochs = 250
    max_debug_images = 5

    # 🚀 Entraînement
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0
        metrics_sum = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0}
        n_samples = 0
        debug_image_count = 0

        print(f"\n🎯 Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (imgs, masks) in enumerate(tqdm(dataloader, desc="  Batches", leave=False, dynamic_ncols=True)):
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

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

            # 🔎 Évaluation
            model.eval()
            with torch.no_grad():
                for i in range(imgs.size(0)):
                    if debug_image_count < max_debug_images:
                        show_segmentation(imgs[i], preds[i], masks[i], f"{batch_idx}_{i}", epoch)
                        debug_image_count += 1

                    m = compute_metrics(preds[i], masks[i])
                    for k in metrics_sum:
                        metrics_sum[k] += m[k]
                    n_samples += 1
            model.train()

        avg_loss = epoch_loss / len(dataloader)
        avg_metrics = {k: metrics_sum[k] / n_samples for k in metrics_sum}
        epoch_time = time.time() - start_time

        print(f"✅ Epoch {epoch + 1} terminé - Temps: {epoch_time:.1f}s - Loss: {avg_loss:.4f}")
        print(f"📊 Dice: {avg_metrics['dice']:.4f} | IoU: {avg_metrics['iou']:.4f} | Precision: {avg_metrics['precision']:.4f} | Recall: {avg_metrics['recall']:.4f}")

        if avg_loss < best_loss:
            print("🎉 Nouveau meilleur modèle (loss diminué) !")
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            counter = 0
        else:
            counter += 1
            print(f"⏳ EarlyStopping: {counter}/{patience} (meilleur loss {best_loss:.4f})")
            if counter >= patience:
                print("🛑 Early stopping activé.")
                break

        csv_writer.writerow([epoch + 1, avg_loss, avg_metrics['dice'], avg_metrics['iou'],
                             avg_metrics['precision'], avg_metrics['recall']])
        csv_file.flush()

    csv_file.close()
    print(f"📁 Modèle avec meilleure loss sauvegardé sous : {best_model_path}")
