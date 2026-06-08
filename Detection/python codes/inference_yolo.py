from ultralytics import YOLO
import os

# Charger ton modèle entraîné
model = YOLO(r"C:\Users\Admin\PycharmProjects\pythonProjectEssaie\yolov8_preparation\runs\detect\train3\weights\best.pt")  # Remplace par le chemin de ton modèle

# Chemin vers le dossier des images
image_folder = r"C:\Users\Admin\PycharmProjects\pythonProjectEssaie\yolov8_preparation\dataset\images\mes_images"

# Chemin vers le dossier de sauvegarde des résultats
save_folder = r"C:\Users\Admin\PycharmProjects\pythonProjectEssaie\yolov8_preparation\runs\detect"

# Créer le dossier de sauvegarde s'il n'existe pas déjà
os.makedirs(save_folder, exist_ok=True)

# Liste des fichiers image dans le dossier
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Faire l'inférence pour chaque image
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    results = model(image_path)

    # Vérifier si des détections ont eu lieu
    if len(results) > 0 and len(results[0].boxes) > 0:
        # Sauvegarder l'image avec les boîtes englobantes dans le dossier de sortie
        save_path = os.path.join(save_folder, image_file)
        results[0].save(save_path)  # Sauvegarder l'image
        print(f"Image avec détections sauvegardée pour {image_file} à {save_path}")
    else:
        print(f"Aucune détection pour {image_file}")

    # Extraire les coordonnées des boîtes englobantes
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xywh.cpu().numpy()  # Format [x_center, y_center, width, height]
        print(f"Boîtes englobantes pour {image_file} : {boxes}")
    else:
        print(f"Aucune boîte détectée pour {image_file}")
