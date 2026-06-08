import os
import torch
import torchvision
import numpy as np
from PIL import Image

def dice_coef(preds, targets, smooth=1e-6):
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

def save_predictions(model, dataloader, device, output_folder, max_batches=1):
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

                Image.fromarray(img.astype(np.uint8)).save(os.path.join(output_folder, f"image_{batch_idx}_{i}.png"))
                Image.fromarray(mask.astype(np.uint8)).save(os.path.join(output_folder, f"mask_{batch_idx}_{i}.png"))
                Image.fromarray(pred.astype(np.uint8)).save(os.path.join(output_folder, f"pred_{batch_idx}_{i}.png"))
