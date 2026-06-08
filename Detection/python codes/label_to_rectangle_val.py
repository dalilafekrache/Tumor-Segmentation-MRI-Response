import os
import cv2
import numpy as np

# Dossiers source et destination
images_path = r"C:\Users\Admin\PycharmProjects\pythonProjectEssaie\yolov8_preparation\dataset\images\val"
labels_path = r"C:\Users\Admin\PycharmProjects\pythonProjectEssaie\yolov8_preparation\dataset\labels\val"
output_masks_path = r"C:\Users\Admin\PycharmProjects\pythonProjectEssaie\dataset\images\val\masks"

os.makedirs(output_masks_path, exist_ok=True)

# Récupérer toutes les images
image_files = [f for f in os.listdir(images_path) if f.endswith('.png')]

# Dimensions des images (à adapter si nécessaire)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640

for image_file in image_files:
    # Charger l'image pour récupérer ses dimensions réelles
    img = cv2.imread(os.path.join(images_path, image_file))
    height, width, _ = img.shape

    # Créer un masque vide
    mask = np.zeros((height, width), dtype=np.uint8)

    # Trouver le fichier d'annotation correspondant
    label_file = os.path.join(labels_path, image_file.replace('.png', '.txt'))
    if not os.path.exists(label_file):
        print(f"⚠️ Pas d'annotation pour {image_file}, masque vide.")
        continue

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # Ignorer lignes invalides

            # YOLO format : classe, x_centre, y_centre, largeur, hauteur (normalisés 0-1)
            _, x_center, y_center, w, h = map(float, parts)

            # Convertir en coordonnées pixels
            x1 = int((x_center - w / 2) * width)
            y1 = int((y_center - h / 2) * height)
            x2 = int((x_center + w / 2) * width)
            y2 = int((y_center + h / 2) * height)

            # Dessiner un rectangle blanc sur le masque
            cv2.rectangle(mask, (x1, y1), (x2, y2), (255), thickness=-1)

    # Sauvegarder le masque
    mask_filename = os.path.join(output_masks_path, image_file)
    cv2.imwrite(mask_filename, mask)

print("✅ Conversion YOLO → Masques terminée !")