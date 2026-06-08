import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, precision_score, recall_score
import os


def dice_score(pred, target, eps=1e-7):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + eps)


def compute_metrics(pred, target):
    """
    Calcule Dice, IoU, précision et rappel pour deux tenseurs binaires.
    """
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
    """
    Sauvegarde une image d'entrée avec masque réel et prédiction en rouge.
    """
    pred = torch.sigmoid(pred)
    pred_np = pred.squeeze().detach().cpu().numpy() > 0.5
    img_np = img.squeeze().cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(img_np, cmap='gray')
    plt.title("Image")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title("Masque réel")

    plt.subplot(1, 3, 3)
    plt.imshow(img_np, cmap='gray')
    plt.imshow(pred_np, cmap='Reds', alpha=0.5)
    plt.title("Prédiction (rouge)")

    plt.tight_layout()
    folder = "debug" if debug else "segmentations"
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/epoch_{epoch + 1}_img_{index}.png"
    plt.savefig(filename)
    plt.close()
