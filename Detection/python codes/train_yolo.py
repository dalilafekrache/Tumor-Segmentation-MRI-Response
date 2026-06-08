import os
if not os.path.exists("config.yaml"):
    print("❌ ERREUR: Le fichier config.yaml n'existe pas !")
else:
    print("✅ config.yaml trouvé, lancement de l'entraînement...")

from ultralytics import YOLO
model = YOLO("yolov8n.yaml")
model.train(data="config.yaml", epochs=100, imgsz=640, verbose=True)