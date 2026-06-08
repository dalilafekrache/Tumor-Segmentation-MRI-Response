import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os, csv, shutil, time
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, jaccard_score

from resunetpp_model import ResUNetPP
from dataset_resunetpp import NiftiSegmentationDataset2D

# 🔬 Fonctions métriques
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

# 🖼️ Visualisation
def show_segmentation(img, pred, mask, name, epoch, debug=False):
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
    folder = "debug" if debug else "segmentations"
    plt.savefig(f"{folder}/epoch_{epoch+1}_{name}.png")
    plt.close()

def main():
    # 🧹 Nettoyage des anciens dossiers
    for folder in ["segmentations", "metrics", "debug"]:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

    # 📦 Dataset
    dataset = NiftiSegmentationDataset2D("dataset/images", "dataset/masks")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)

    # 🧠 Modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNetPP(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 📊 CSV
    csv_path = f"metrics/resunetpp_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "Loss", "Dice", "IoU", "Precision", "Recall"])

    num_epochs = 250
    best_loss = float("inf")
    best_model_path = "resunetpp_best.pth"
    patience, counter = 10, 0

    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        epoch_loss = 0
        metrics_sum = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0}
        n_samples = 0

        # ✅ Limite debug
        debug_image_count = 0
        max_debug_images = 5

        print(f"\n🎯 Epoch {epoch + 1}/{num_epochs}")
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

            # ✅ Sauvegarde limitée à 5 images
            for i in range(imgs.size(0)):
                if debug_image_count < max_debug_images:
                    show_segmentation(imgs[i], preds[i], masks[i], f"{batch_idx}_{i}", epoch)
                    debug_image_count += 1

            with torch.no_grad():
                for i in range(imgs.size(0)):
                    m = compute_metrics(preds[i], masks[i])
                    for k in metrics_sum:
                        metrics_sum[k] += m[k]
                    n_samples += 1

        avg_loss = epoch_loss / len(dataloader)
        avg_metrics = {k: metrics_sum[k] / n_samples for k in metrics_sum}

        print(f"✅ Epoch {epoch+1} terminé - ⏱️ Temps: {time.time() - start:.1f}s - 📉 Loss: {avg_loss:.4f}")
        print(f"📊 Dice: {avg_metrics['dice']:.4f} | IoU: {avg_metrics['iou']:.4f} | Precision: {avg_metrics['precision']:.4f} | Recall: {avg_metrics['recall']:.4f}")

        if avg_loss < best_loss:
            print("🎉 Nouveau meilleur modèle sauvegardé !")
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            counter = 0
        else:
            counter += 1
            print(f"⏳ EarlyStopping {counter}/{patience} (best loss: {best_loss:.4f})")
            if counter >= patience:
                print("🛑 Early stopping déclenché.")
                break

        csv_writer.writerow([epoch + 1, avg_loss, avg_metrics['dice'], avg_metrics['iou'],
                             avg_metrics['precision'], avg_metrics['recall']])
        csv_file.flush()

    csv_file.close()
    print(f"📁 Meilleur modèle sauvegardé dans : {best_model_path}")

if __name__ == '__main__':
    main()
