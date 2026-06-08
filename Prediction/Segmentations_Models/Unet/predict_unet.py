import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Resize, ToTensor
from unet_model import UNet

# === Chemins ===
image_dir = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\UNet\images"
model_path = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\UNet\best_model_unet.pth"

output_png_dir = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\UNet\prediction_unet_png"
output_nii_dir = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\UNet\prediction_unet_nifti"

os.makedirs(output_png_dir, exist_ok=True)
os.makedirs(output_nii_dir, exist_ok=True)

# === Prétraitements ===
resize = Resize((256, 256))
to_tensor = ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Chargement modèle ===
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Prédiction ===
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])

for filename in image_files:
    print(f"🔍 Traitement : {filename}")
    image_path = os.path.join(image_dir, filename)
    image_nii = nib.load(image_path)
    image_data = image_nii.get_fdata()
    affine = image_nii.affine

    pred_volume = np.zeros_like(image_data)

    for slice_idx in range(image_data.shape[2]):
        image_slice = image_data[:, :, slice_idx]

        # Normalisation
        image_norm = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice) + 1e-8)
        image_pil = Image.fromarray((image_norm * 255).astype(np.uint8)).convert("L")
        image_resized = resize(image_pil)
        input_tensor = to_tensor(image_resized).unsqueeze(0).to(device)  # [1, 1, 256, 256]

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_binary = (pred > 0.5).astype(np.uint8)

        # Resize à la taille originale
        pred_resized = Resize(image_slice.shape, interpolation=Image.NEAREST)(
            Image.fromarray(pred_binary * 255)
        )
        pred_resized_np = np.array(pred_resized) // 255

        pred_volume[:, :, slice_idx] = pred_resized_np

        # Sauvegarde PNG
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_slice, cmap='gray')
        ax.imshow(pred_resized_np, cmap='Reds', alpha=0.4)
        ax.set_title(f"{filename}_slice{slice_idx}")
        ax.axis('off')

        png_path = os.path.join(output_png_dir, f"{os.path.splitext(filename)[0]}_slice{slice_idx}.png")
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Sauvegarde NIfTI
    pred_nii = nib.Nifti1Image(pred_volume.astype(np.uint8), affine)
    pred_path = os.path.join(output_nii_dir, f"{os.path.splitext(filename)[0]}_prediction.nii.gz")
    nib.save(pred_nii, pred_path)

print("✅ Prédiction UNet terminée.")
print("📁 PNG : ", output_png_dir)
print("📁 NIfTI : ", output_nii_dir)
