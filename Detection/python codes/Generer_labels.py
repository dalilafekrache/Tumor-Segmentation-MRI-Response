from ultralytics import YOLO

# Charger le modèle entraîné
model = YOLO(r"C:/Users/Admin/PycharmProjects/pythonProjectEssaie/yolov8_preparation/runs/detect/train3/weights/best.pt")

# Définir le dossier des images de validation
image_folder = r"C:/Users/Admin/PycharmProjects/pythonProjectEssaie/yolov8_preparation/dataset/images/val/*.png"

# Effectuer la détection et sauvegarder les résultats avec les labels
results = model(image_folder, save=True, save_txt=True)

print("Détection terminée ! Les résultats sont enregistrés dans 'C:/Users/Admin/PycharmProjects/pythonProjectEssaie/yolov8_preparation/runs/detect/predict2'")