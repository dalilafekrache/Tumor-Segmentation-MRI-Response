import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import Resize

class NiftiSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, slice_step=1):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".nii.gz")])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".nii.gz")])
        self.transform = transform
        self.slice_step = slice_step
        self.slices = self._generate_slices()

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

        img = torch.from_numpy(img_slice).float().unsqueeze(0)
        mask = torch.from_numpy(mask_slice).float().unsqueeze(0)

        # Binariser le masque : 0 ou 1
        mask = (mask > 0).float()

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask
