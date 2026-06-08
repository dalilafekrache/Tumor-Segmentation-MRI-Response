import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T

class NiftiSegmentationDataset2D(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # Liste des fichiers d'images
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])

        self.slices = []  # Liste des tuples (nom_fichier, index_slice)

        # Création de la liste des slices
        for fname in self.image_files:
            img_path = os.path.join(images_dir, fname)
            img = nib.load(img_path).get_fdata()
            num_slices = img.shape[2]  # Nombre de slices dans l'axe axial
            for i in range(num_slices):
                self.slices.append((fname, i))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        fname, slice_idx = self.slices[idx]
        image_path = os.path.join(self.images_dir, fname)
        mask_path = os.path.join(self.masks_dir, fname)

        # Vérifier l'existence du masque
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Le masque pour l'image {fname} est manquant")

        # Chargement des données
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Sélection de la slice 2D
        image = image[:, :, slice_idx]
        mask = mask[:, :, slice_idx]

        # Normalisation de l'image
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        # Convertir en tenseurs PyTorch (1 canal, H, W)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        # Appliquer les transformations (resize, etc.)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
