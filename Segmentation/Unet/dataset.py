import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import warnings

class NiftiSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.image_size = image_size
        self.slices = []

        for img_file, mask_file in zip(self.image_files, self.mask_files):
            img_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)

            try:
                img_nii = nib.load(img_path)
                mask_nii = nib.load(mask_path)
                img_data = img_nii.get_fdata()
                mask_data = mask_nii.get_fdata()

                if img_data.shape != mask_data.shape:
                    warnings.warn(f"Shapes do not match for {img_file} and {mask_file}")
                    continue

                for i in range(img_data.shape[2]):
                    self.slices.append((img_path, mask_path, i))

            except Exception as e:
                warnings.warn(f"Skipping {img_file} due to error: {e}")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_path, mask_path, slice_idx = self.slices[idx]

        img = nib.load(img_path).get_fdata()[:, :, slice_idx]
        mask = nib.load(mask_path).get_fdata()[:, :, slice_idx]

        # Normalisation image et binarisation masque
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        mask = (mask > 0).astype(np.float32)

        # Redimensionnement
        resize = T.Resize(self.image_size)
        img = torch.tensor(img).unsqueeze(0).float()  # (1, H, W)
        mask = torch.tensor(mask).unsqueeze(0).float()

        img = resize(img)
        mask = resize(mask)

        if self.transform:
            img = self.transform(img)

        return img, mask
