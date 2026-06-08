import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def load_nifti(path):
    """Charge un fichier NIfTI et retourne un array 2D normalisé."""
    nifti_img = nib.load(path)
    data = nifti_img.get_fdata()
    data = np.squeeze(data)

    # Normalisation
    if np.max(data) != 0:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    return data.astype(np.float32)

def resize_image(image, size=(256, 256)):
    """Redimensionne une image 2D à la taille donnée."""
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    image_resized = F.interpolate(image_tensor, size=size, mode='bilinear', align_corners=False)
    return image_resized.squeeze().numpy()

class BreastMRIDataset(Dataset):
    """Dataset personnalisé pour les images IRM du sein."""
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = load_nifti(self.image_paths[idx])
        mask = load_nifti(self.mask_paths[idx])

        image = resize_image(image)
        mask = resize_image(mask)

        image = torch.tensor(image).unsqueeze(0).float()  # [1, H, W]
        mask = torch.tensor(mask).unsqueeze(0).float()    # [1, H, W]

        return image, mask

def dice_score(pred, target, threshold=0.5):
    """Calcule le Dice score entre deux tensors."""
    pred_bin = (pred > threshold).float()
    target = target.float()
    smooth = 1e-6
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.item()
