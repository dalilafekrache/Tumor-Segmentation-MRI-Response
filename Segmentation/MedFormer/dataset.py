import os
import nibabel as nib
import torch
from torch.utils.data import Dataset

class NiftiSegmentationDataset2D(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

        if len(self.image_filenames) != len(self.mask_filenames):
            raise ValueError("❌ Le nombre d’images et de masques ne correspond pas.")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Charger les volumes NIfTI
        image = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Si les images/masques sont 3D → on prend la slice du milieu
        if image.ndim == 3:
            image = image[:, :, image.shape[2] // 2]
        if mask.ndim == 3:
            mask = mask[:, :, mask.shape[2] // 2]

        # ✅ Normalisation image en [0,1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # ✅ Binarisation masque (0 ou 1)
        mask = (mask > 0).astype("float32")

        # Conversion en tenseurs avec canal
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)    # [1, H, W]

        # Appliquer une éventuelle transformation (resize, augmentation, etc.)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
