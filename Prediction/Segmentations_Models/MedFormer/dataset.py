import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class NiftiSegmentationDataset2D(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

        if len(self.image_filenames) != len(self.mask_filenames):
            raise ValueError("Le nombre d’images et de masques ne correspond pas.")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Si les images sont 3D, on prend une seule slice (au milieu)
        if image.ndim == 3:
            image = image[:, :, image.shape[2] // 2]
        if mask.ndim == 3:
            mask = mask[:, :, mask.shape[2] // 2]

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)   # [1, H, W]

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
