Segmentation Models
The repository contains implementations of the following segmentation architectures:
•	U-Net
•	U-Net++
•	Attention U-Net
•	ResUNet
•	ResUNet++
•	TransUNet
•	Swin-Unet
•	UCTransNet
•	MedFormer
•	V-Net
•	nnU-Net
Each model is stored in a dedicated directory and follows the same structure:
ModelName/
model_definition.py
dataset.py
train_model.py
utils.py
best_model.pth
File Description
•	model_definition.py : implementation of the segmentation architecture.
•	dataset.py : loading and preprocessing of NIfTI images and masks.
•	train_model.py : training procedure, optimization, metrics computation, and checkpoint saving.
•	utils.py : Dice score, IoU, precision, recall, and visualization utilities.
•	best_model.pth : best-performing trained model checkpoint.
Training
Each model can be trained independently using its corresponding training script.
Outputs include:
•	trained weights
•	Dice score
•	IoU score
•	precision
•	recall
•	segmentation visualizations
•	training logs
