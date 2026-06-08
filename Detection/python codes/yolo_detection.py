from ultralytics import YOLO
import glob
import os

# Charger le modèle YOLOv8
model = YOLO(r"C:/Users/Admin/PycharmProjects/pythonProjectEssaie/yolov8_preparation/runs/detect/train3/weights/best.pt")

# Dossier contenant les images de validation
image_folder = r"C:/Users/Admin/PycharmProjects/pythonProjectEssaie/yolov8_preparation/dataset/images/val"

# Chercher les images .png
image_paths = glob.glob(os.path.join(image_folder, "*.png"))

if not image_paths:
    raise FileNotFoundError(f"Aucune image trouvée dans {image_folder} ! Vérifie le chemin et l'extension des fichiers.")

# Traiter chaque image une par une pour éviter l'explosion de RAM
for image_path in image_paths:
    results = model(image_path, save=True)  # On traite une seule image à la fois
    print(f"✅ Détection terminée pour : {image_path}")

print("🎉 Détection complète ! Les résultats sont enregistrés dans 'C:/Users/Admin/PycharmProjects/pythonProjectEssaie/yolov8_preparation/runs/detect/predict2'")