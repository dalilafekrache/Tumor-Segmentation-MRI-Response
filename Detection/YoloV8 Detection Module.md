YOLOv8 Detection Module
Overview
This module performs the object detection stage of the proposed framework. YOLOv8 is first trained on the annotated dataset. The trained detector is then used to localize target regions. Finally, the detected bounding boxes are converted into binary masks that are used as inputs for the subsequent nnU-Net segmentation stage.
Workflow
Step 1 – Train the YOLOv8 detector
Script: train_yolo.py
Description: Trains a YOLOv8n detector using the dataset configuration file (config.yaml).
Input: Training images, YOLO annotations, config.yaml
Output: Trained model weights (best.pt)
Step 2 – Generate binary masks from ground-truth annotations
Scripts: label_to_rectangle_train.py, label_to_rectangle_val.py
Description: Converts YOLO bounding-box annotations into binary rectangular masks for the training and validation datasets.
Input: Images, YOLO label files (.txt)
Output: Binary masks
Step 3 – Perform object detection
Scripts: yolo_detection.py, Generer_labels.py
Description: Runs inference using the trained YOLOv8 model and generates predicted bounding boxes and corresponding YOLO label files.
Input: Trained model (best.pt), Validation images
Output: Detection results, Predicted label files
Step 4 – Convert predicted bounding boxes into masks
Script: yolo_to_mask.py
Description: Converts predicted YOLO bounding boxes into binary masks that are subsequently used by the segmentation module.
Input: Predicted YOLO labels
Output: Binary masks
Additional Utility Script: inference_yolo.py
This script is provided as a standalone inference utility for testing the trained detector on individual image collections. It is not required for the main experimental pipeline.
Requirements
•	Python 3.10
•	ultralytics 8.3.0
•	numpy 1.26.4
•	opencv-python 4.10.0
•	torch 2.2.1
•	torchvision 0.17.1
•	nibabel 5.2.1
•	scikit-learn 1.5.0
Installation
pip install ultralytics numpy opencv-python
Output
The final output of this module consists of binary masks generated from YOLO detections. These masks are used as inputs for the segmentation stage of the proposed framework.

