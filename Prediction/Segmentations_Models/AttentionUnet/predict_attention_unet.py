import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from attention_unet_model import AttentionUNet2D
from torchvision import transforms
from torch.nn.functional import sigmoid
from PIL import Image

# === Chemins ===
image_dir = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\AttentionUnet\images"
mask_dir = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\AttentionUnet\masks"
model_path = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\AttentionUnet\best_model_attentionunet.pth"
output_png_dir = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\AttentionUnet\predictionattentionunet"
output_nii_dir = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\SegmentationsParModeles\AttentionUnet\predictionattentionunet_nifti"

os.makedirs(output_png_dir, exist_ok=True)
os.makedirs(output_nii_dir, exist_ok=True)

# === Prétraitement ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Chargement du modèle ===
model = AttentionUNet2D(in_ch=1, out_ch=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Prédiction ===
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])

for filename in image_files:
    print(f"🔍 Traitement de : {filename}")

    # Chargement image/mask NIfTI
    image_path = os.path.join(image_dir, filename)
    image_nii = nib.load(image_path)
    image_data = image_nii.get_fdata()
    affine = image_nii.affine

    pred_volume = np.zeros_like(image_data)

    for slice_idx in range(image_data.shape[2]):
        image_slice = image_data[:, :, slice_idx]

        # Skip if image slice is not 2D
        if image_slice.ndim != 2:
            print(f"❌ Image non 2D ignorée : {filename}_slice{slice_idx}")
            continue

        # Normalisation [0,1]
        image_slice_norm = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice) + 1e-8)

        # Redimensionnement
        image_resized = transforms.Resize((256, 256))(Image.fromarray((image_slice_norm * 255).astype(np.uint8)))
        input_tensor = transform(image_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred = sigmoid(output).cpu().squeeze().numpy()
            pred_binary = (pred > 0.5).astype(np.uint8)

        # Retour à la taille originale
        pred_resized = transforms.Resize(image_slice.shape, interpolation=transforms.InterpolationMode.NEAREST)(
            Image.fromarray(pred_binary * 255)
        )
        pred_resized_np = np.array(pred_resized) // 255

        # Ajout à la prédiction 3D
        pred_volume[:, :, slice_idx] = pred_resized_np

        # 🔸 Sauvegarde PNG
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_slice, cmap='gray')
        ax.imshow(pred_resized_np, cmap='Reds', alpha=0.4)
        ax.set_title(f"{filename}_slice{slice_idx}")
        ax.axis('off')

        output_path = os.path.join(output_png_dir, f"{os.path.splitext(filename)[0]}_slice{slice_idx}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    # 🔸 Sauvegarde NIfTI
    pred_nii = nib.Nifti1Image(pred_volume.astype(np.uint8), affine)
    output_nii_path = os.path.join(output_nii_dir, f"{os.path.splitext(filename)[0]}_prediction.nii.gz")
    nib.save(pred_nii, output_nii_path)

print("✅ Tous les résultats ont été générés !")
print("📁 PNG sauvegardés dans :", output_png_dir)
print("📁 NIfTI sauvegardés dans :", output_nii_dir)
