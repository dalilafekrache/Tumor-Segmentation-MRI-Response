import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import Resize, ToTensor
from unetpp_model import UNetPP

# === Paramètres ===
image_dir = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\Unetpp\images"
model_path = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\Unetpp\best_model_unetpp.pth"

output_png_dir = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\Unetpp\prediction_unetpp_png"
output_nii_dir = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\Unetpp\prediction_unetpp_nifti"

os.makedirs(output_png_dir, exist_ok=True)
os.makedirs(output_nii_dir, exist_ok=True)

resize = Resize((256, 256))
to_tensor = ToTensor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Chargement modèle ===
model = UNetPP(in_ch=1, out_ch=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Prédiction pour chaque volume ===
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

        # Mise en forme pour le modèle
        img_pil = Image.fromarray((img_norm * 255).astype(np.uint8)).convert("L")
        img_resized = resize(img_pil)
        input_tensor = to_tensor(img_resized).unsqueeze(0).to(device)  # [1, 1, 256, 256]

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_bin = (pred > 0.5).astype(np.uint8)

        # Resize à taille originale
        pred_resized = Resize(img_slice.shape, interpolation=Image.NEAREST)(
            Image.fromarray(pred_bin * 255)
        )
        pred_np = np.array(pred_resized) // 255
        pred_volume[:, :, i] = pred_np

        # PNG overlay
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_slice, cmap='gray')
        ax.imshow(pred_np, cmap='Reds', alpha=0.4)
        ax.axis('off')
        ax.set_title(f"{filename}_slice{i}")

        png_path = os.path.join(output_png_dir, f"{os.path.splitext(filename)[0]}_slice{i}.png")
        plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    # Sauvegarde NIfTI 3D
    pred_nifti = nib.Nifti1Image(pred_volume.astype(np.uint8), affine)
    nifti_path = os.path.join(output_nii_dir, f"{os.path.splitext(filename)[0]}_prediction.nii.gz")
    nib.save(pred_nifti, nifti_path)

print("✅ Prédiction UNet++ terminée.")
print("📁 Résultats PNG : ", output_png_dir)
print("📁 Résultats NIfTI : ", output_nii_dir)
