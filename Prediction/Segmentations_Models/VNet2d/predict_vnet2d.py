import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Resize, ToTensor
from VNet2d_model import VNet2D

# === Chemins ===
image_dir = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\VNet2D\images"
model_path = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\VNet2D\best_model_vnet2d.pth"

output_png_dir = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\VNet2D\prediction_vnet2d_png"
output_nii_dir = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\VNet2D\prediction_vnet2d_nifti"

os.makedirs(output_png_dir, exist_ok=True)
os.makedirs(output_nii_dir, exist_ok=True)

# === Prétraitements ===
resize = Resize((256, 256))
to_tensor = ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Chargement du modèle ===
model = VNet2D(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Prédiction volume par volume ===
for filename in sorted(os.listdir(image_dir)):
    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        continue

    print(f"🔍 Prédiction : {filename}")
    img_path = os.path.join(image_dir, filename)
    img_nifti = nib.load(img_path)
    img_data = img_nifti.get_fdata()
    affine = img_nifti.affine

    pred_volume = np.zeros_like(img_data)

    for i in range(img_data.shape[2]):
        img_slice = img_data[:, :, i]
        img_norm = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-8)

        # Conversion image PIL → Resize → Tensor
        img_pil = Image.fromarray((img_norm * 255).astype(np.uint8)).convert("L")
        img_resized = resize(img_pil)
        input_tensor = to_tensor(img_resized).unsqueeze(0).to(device)  # [1, 1, 256, 256]

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_bin = (pred > 0.5).astype(np.uint8)

        # Resize inverse vers shape d’origine
        pred_resized = Resize(img_slice.shape, interpolation=Image.NEAREST)(
            Image.fromarray(pred_bin * 255)
        )
        pred_np = np.array(pred_resized) // 255
        pred_volume[:, :, i] = pred_np

        # Overlay PNG slice
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_slice, cmap='gray')
        ax.imshow(pred_np, cmap='Reds', alpha=0.4)
        ax.axis('off')
        ax.set_title(f"{filename}_slice{i}")

        png_path = os.path.join(output_png_dir, f"{os.path.splitext(filename)[0]}_slice{i}.png")
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Sauvegarde du volume NIfTI
    pred_nifti = nib.Nifti1Image(pred_volume.astype(np.uint8), affine)
    nifti_path = os.path.join(output_nii_dir, f"{os.path.splitext(filename)[0]}_prediction.nii.gz")
    nib.save(pred_nifti, nifti_path)

print("✅ Prédiction VNet2D terminée.")
print("📁 PNG : ", output_png_dir)
print("📁 NIfTI : ", output_nii_dir)
