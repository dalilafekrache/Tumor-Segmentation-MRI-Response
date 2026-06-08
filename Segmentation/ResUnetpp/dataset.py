import os
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class NiftiSegmentationDataset2D(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])

        if len(self.image_files) != len(self.mask_files):
            raise ValueError("Le nombre d'images et de masques ne correspond pas.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)

        image = image_nii.get_fdata()
        mask = mask_nii.get_fdata()

        # ✅ Extraction d'une seule slice centrale (2D)
        image = image[:, :, image.shape[2] // 2]
        mask = mask[:, :, mask.shape[2] // 2]

        # ✅ Conversion en Tensor et ajout d'une dimension canal [1, H, W]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # ✅ Normalisation de l’image dans l’intervalle [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # ✅ Binarisation du masque : 0 ou 1
        mask = (mask > 0).float()

        # ✅ Resize en 256x256 (bilinear pour image, nearest pour masque)
        image = F.interpolate(image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)

        return image, mask
