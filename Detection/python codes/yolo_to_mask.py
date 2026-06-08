import os
import cv2
import numpy as np

# Dossiers
image_folder = r"C:\Users\Admin\PycharmProjects\pythonProjectEssaie\runs\detect\predict"
label_folder = os.path.join(image_folder, "labels")
mask_folder = os.path.join(image_folder, "masks")

# Créer le dossier des masques s'il n'existe pas
os.makedirs(mask_folder, exist_ok=True)

# Obtenir la liste des images
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    label_path = os.path.join(label_folder, image_file.replace(".png", ".txt").replace(".jpg", ".txt"))

    # Charger l'image originale pour récupérer ses dimensions
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Image introuvable ou corrompue : {image_path}")
        continue
    height, width = image.shape[:2]

    # Créer un masque vide (noir)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Vérifier si un fichier de label existe
    if os.path.exists(label_path):
        with open(label_path, "r") as file:
            lines = file.readlines()
            if not lines:
                print(f"⚠️ Fichier label vide : {label_path}")
                continue

            for line in lines:
                values = line.strip().split()
                if len(values) < 5:
                    print(f"⚠️ Format incorrect dans {label_path}: {line}")
                    continue

                class_id = int(values[0])  # Classe (0 si une seule classe)
                x_center, y_center, w, h = map(float, values[1:])

                # Convertir les coordonnées normalisées en pixels
                x_min = int((x_center - w / 2) * width)
                y_min = int((y_center - h / 2) * height)
                x_max = int((x_center + w / 2) * width)
                y_max = int((y_center + h / 2) * height)

                # Vérification des coordonnées
                if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
                    print(f"⚠️ BBox hors limites pour {label_path}: {x_min}, {y_min}, {x_max}, {y_max}")
                    continue

                # Dessiner le masque en blanc (255)
                mask[y_min:y_max, x_min:x_max] = 255

        # Sauvegarder le masque seulement s'il y a eu des annotations
        mask_path = os.path.join(mask_folder, image_file)
        cv2.imwrite(mask_path, mask)
        print(f"✅ Masque généré pour : {mask_path}")

    else:
        print(f"❌ Aucun fichier label trouvé pour : {image_file}")

print("🎭 Conversion terminée ! Vérifie le dossier :", mask_folder)