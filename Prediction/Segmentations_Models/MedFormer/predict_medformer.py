import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from medformer_model import MedFormer2D

# ----- 1. Chargement du modèle -----
model = MedFormer2D(in_channels=1, out_channels=1)
model.load_state_dict(torch.load("best_model_medformer.pth", map_location="cpu"))
model.eval()

# ----- 2. Chargement de l’image test -----
# Remplace par ton image
img_path = "mon_image_test.png"
img = Image.open(img_path).convert("L")  # Niveaux de gris

# Redimensionner si nécessaire (si entraînement fait en 256x256 par ex.)
img = img.resize((256, 256))

transform = T.ToTensor()  # [0,1] et forme [1, H, W]
input_tensor = transform(img).unsqueeze(0)  # [1, 1, H, W]

# ----- 3. Prédiction -----
with torch.no_grad():
    output = model(input_tensor)
    output = torch.sigmoid(output)
    mask = (output > 0.5).float()

# ----- 4. Visualisation -----
# Image en numpy
image_np = input_tensor.squeeze().numpy()
mask_np = mask.squeeze().numpy()

plt.figure(figsize=(6, 6))
plt.imshow(image_np, cmap='gray')
plt.imshow(mask_np, cmap='Reds', alpha=0.5)  # Superposition rouge
plt.axis('off')
plt.title("Masque de prédiction (tumeur en rouge)")
plt.show()
