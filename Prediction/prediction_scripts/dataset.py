import os
import torch
import nibabel as nib
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import re

def extract_patient_id(filename: str):
    """
    Extrait l'identifiant patient à partir du nom de fichier.
    Exemple : '975_RN_slice072.nii.gz' → '975_RN'
    """
    base = os.path.splitext(os.path.splitext(filename)[0])[0]
    parts = base.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return parts[0]

class MedicalDataset2D(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, empty_ratio=0.0, img_size=256):
        """
        Dataset pour données médicales 2D/3D (.nii.gz)
        - Associe les masques aux images par nom (correspondance tolérante)
        - Gère automatiquement les volumes 2D et 3D
        - empty_ratio : proportion de slices vides à garder (0.0 = ignorer toutes les slices vides)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = (img_size, img_size)
        self.slices = []

        # --- lister et normaliser ---
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])

        def normalize_name(name):
            base = os.path.splitext(os.path.splitext(name)[0])[0]
            b = base.lower()
            for suf in ["_pred", "_prediction", "_predicted", "_mask", "_seg", "_segmentation",
                        "_unet", "_resunet", "_resunetpp", "_swinunet", "_nifti", "_final"]:
                b = b.replace(suf, "")
            b = b.replace("--", "-").replace("__", "_").strip("_-")
            return b

        norm_to_masks = {}
        for m in mask_files:
            key = normalize_name(m)
            norm_to_masks.setdefault(key, []).append(m)

        # construire liste d'appariements image->masque
        self.image_files = []
        unmatched_images = []
        for img in image_files:
            key = normalize_name(img)
            candidates = norm_to_masks.get(key)
            if not candidates:
                looser = re.sub(r"_slice\d+$", "", key)
                looser = re.sub(r"_s\d+$", "", looser)
                candidates = norm_to_masks.get(looser, None)
            if not candidates:
                unmatched_images.append(img)
                continue

            chosen = None
            base_img = os.path.splitext(os.path.splitext(img)[0])[0].lower()
            for c in candidates:
                if os.path.splitext(os.path.splitext(c)[0])[0].lower() == base_img:
                    chosen = c
                    break
            if chosen is None:
                for c in candidates:
                    low = c.lower()
                    if not any(tag in low for tag in ["pred", "prediction", "predicted", "seg", "unet"]):
                        chosen = c
                        break
            if chosen is None:
                chosen = candidates[0]
            self.image_files.append((img, chosen))

        matched_masks = set([m for _, m in self.image_files])
        unmatched_masks = [m for m in mask_files if m not in matched_masks]

        print(f"🧩 {len(self.image_files)} paires image/masque trouvées dans {image_dir} (sur {len(image_files)} images, {len(mask_files)} masques)")
        if unmatched_images:
            print(f"⚠️ {len(unmatched_images)} images sans masque : exemple(s) {unmatched_images[:10]}")
        if unmatched_masks:
            print(f"⚠️ {len(unmatched_masks)} masques non appariés : exemple(s) {unmatched_masks[:10]}")

        # construire index des slices
        for i, (img_file, mask_file) in enumerate(self.image_files):
            try:
                mask = nib.load(os.path.join(mask_dir, mask_file)).get_fdata()
            except Exception as e:
                print(f"⚠️ Impossible de charger le masque {mask_file} : {e} — ignoré")
                continue

            if getattr(mask, "ndim", 2) == 2:
                mask_slices = [mask]
            else:
                if mask.shape[-1] == 1:
                    mask_slices = [mask[:, :, 0]]
                else:
                    mask_slices = [mask[:, :, s] for s in range(mask.shape[-1])]

            for s_idx, mask_slice in enumerate(mask_slices):
                if mask_slice.sum() == 0:
                    if random.random() > empty_ratio:
                        continue
                self.slices.append((i, s_idx))

        print(f"📊 Dataset construit : {len(self.slices)} slices (empty_ratio={empty_ratio}, img_size={img_size})")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_idx, slice_idx = self.slices[idx]
        img_file, mask_file = self.image_files[img_idx]
        patient_id = extract_patient_id(img_file)  # ✅ CORRECTION ICI

        image = nib.load(os.path.join(self.image_dir, img_file)).get_fdata()
        mask = nib.load(os.path.join(self.mask_dir, mask_file)).get_fdata()

        if getattr(image, "ndim", 2) == 3:
            z = image.shape[-1]
            s_idx = min(slice_idx, z - 1)
            img_slice = image[:, :, s_idx]
        else:
            img_slice = image

        if getattr(mask, "ndim", 2) == 3:
            z = mask.shape[-1]
            s_idx = min(slice_idx, z - 1)
            mask_slice = mask[:, :, s_idx]
        else:
            mask_slice = mask

        img_min = float(img_slice.min()) if img_slice.size else 0.0
        img_max = float(img_slice.max()) if img_slice.size else 1.0
        denom = (img_max - img_min) if (img_max - img_min) != 0 else 1.0
        img_slice = (img_slice - img_min) / denom

        img_slice = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0)
        mask_slice = torch.tensor(mask_slice, dtype=torch.float32).unsqueeze(0)

        img_slice = TF.resize(img_slice, self.target_size, antialias=True)
        mask_slice = TF.resize(mask_slice, self.target_size, antialias=True)

        if self.transform:
            img_slice, mask_slice = self.transform(img_slice, mask_slice)

        return img_slice, mask_slice, patient_id, slice_idx
