import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

class NiftiSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform_size=(224, 224), slice_step=1):
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".nii") or f.endswith(".nii.gz")
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".nii") or f.endswith(".nii.gz")
        ])
        self.slice_step = slice_step
        self.transform_size = transform_size
        self.slices = self._generate_slices()
        print(f"[INFO] Nombre total de slices valides : {len(self.slices)}")

    def _generate_slices(self):
        slices = []
        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            img = nib.load(img_path).get_fdata()
            num_slices = img.shape[2]
            for i in range(0, num_slices, self.slice_step):
                slices.append((img_path, mask_path, i))
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_path, mask_path, slice_idx = self.slices[idx]

        img_3d = nib.load(img_path).get_fdata()
        mask_3d = nib.load(mask_path).get_fdata()

        img_slice = img_3d[:, :, slice_idx]
        mask_slice = mask_3d[:, :, slice_idx]

        # Convert to torch tensors
        img = torch.from_numpy(img_slice).float().unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask_slice).float().unsqueeze(0)

        # Min-max normalization
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        # Binarize mask
        mask = (mask > 0).float()

        # Resize to 224x224 (for SwinUnet compatibility)
        if self.transform_size:
            img = F.interpolate(img.unsqueeze(0), size=self.transform_size, mode='bilinear', align_corners=False).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0), size=self.transform_size, mode='nearest').squeeze(0)

        return img, mask
