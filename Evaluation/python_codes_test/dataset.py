import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T


class NiftiSegmentationDataset2D(Dataset):

    def __init__(self, images_dir, masks_dir, target_size=(416,416)):

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.target_size = target_size

        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith(".nii") or f.endswith(".nii.gz")
        ])

        self.slices = []

        for fname in self.image_files:

            img_path = os.path.join(images_dir, fname)
            img = nib.load(img_path).get_fdata()

            num_slices = img.shape[2]

            for i in range(num_slices):
                self.slices.append((fname, i))

        self.resize = T.Resize(target_size)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        fname, slice_idx = self.slices[idx]

        image_path = os.path.join(self.images_dir, fname)
        mask_path  = os.path.join(self.masks_dir, fname)

        image = nib.load(image_path).get_fdata()
        mask  = nib.load(mask_path).get_fdata()

        image = image[:,:,slice_idx]
        mask  = mask[:,:,slice_idx]

        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask  = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        image = self.resize(image)
        mask  = self.resize(mask)

        return image, mask, fname, slice_idx