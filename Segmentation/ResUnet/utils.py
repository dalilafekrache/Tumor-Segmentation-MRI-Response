import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def dice_coef(preds, targets, smooth=1e-6):
    """
    Calcule le coefficient de Dice entre les prédictions et les cibles.
    """
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

def save_predictions(model, dataloader, device, output_folder, max_batches=1):
    """
    Sauvegarde les prédictions du modèle pour un maximum de `max_batches` lots.
    """
    model.eval()
    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            images = images.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            for i in range(images.size(0)):
                img = images[i].cpu().squeeze().numpy() * 255
                mask = masks[i].cpu().squeeze().numpy() * 255
                pred = preds[i].cpu().squeeze().numpy() * 255

                img_path = os.path.join(output_folder, f"image_{batch_idx}_{i}.png")
                mask_path = os.path.join(output_folder, f"mask_{batch_idx}_{i}.png")
                pred_path = os.path.join(output_folder, f"pred_{batch_idx}_{i}.png")

                Image.fromarray(img.astype(np.uint8)).save(img_path)
                Image.fromarray(mask.astype(np.uint8)).save(mask_path)
                Image.fromarray(pred.astype(np.uint8)).save(pred_path)

def save_segmentation_visualization(images, masks, preds, output_folder, batch_idx):
    """
    Sauvegarde une visualisation composite des images, des masques réels et des prédictions.
    """
    os.makedirs(output_folder, exist_ok=True)

    for i in range(images.size(0)):
        img = images[i].cpu().squeeze().numpy() * 255
        mask = masks[i].cpu().squeeze().numpy() * 255
        pred = preds[i].cpu().squeeze().numpy() * 255

        # Crée une image composite pour visualiser ensemble l'image, le masque et la prédiction
        visualization = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        visualization[..., 0] = img  # Rouge: image
        visualization[..., 1] = mask  # Vert: masque réel
        visualization[..., 2] = pred  # Bleu: prédiction

        # Sauvegarde l'image composite
        vis_path = os.path.join(output_folder, f"visualization_{batch_idx}_{i}.png")
        Image.fromarray(visualization).save(vis_path)

def calculate_metrics(model, dataloader, device):
    """
    Calcule les métriques de performance pour un modèle donné sur un dataloader.
    Retourne le Dice coefficient moyen pour tous les échantillons du dataloader.
    """
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Calcul du Dice coefficient pour chaque lot
            for i in range(images.size(0)):
                dice = dice_coef(preds[i], masks[i])
                dice_scores.append(dice.item())

    mean_dice = np.mean(dice_scores)
    return mean_dice

def save_metrics_plot(metrics, output_path):
    """
    Sauvegarde un graphique des métriques, comme le Dice coefficient.
    """
    plt.figure()
    plt.plot(metrics, label='Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.title('Metric Plot')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def save_model_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    Sauvegarde un modèle et son optimiseur à un instant donné (checkpoint).
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, save_path)

def load_model_checkpoint(model, optimizer, checkpoint_path):
    """
    Charge un modèle et son optimiseur depuis un checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss
