Evaluation Module
This module evaluates the performance of the trained segmentation models on the test dataset.
Available Models
•	Attention U-Net
•	MedFormer
•	ResUNet
•	ResUNet++
•	Swin-Unet
•	TransUNet
•	UCTransNet
•	U-Net
•	U-Net++
•	V-Net
•	nnU-Net
Evaluation Procedure
For each model:
1.	Load the trained checkpoint.
2.	Load the test dataset.
3.	Generate segmentation predictions.
4.	Compute quantitative metrics.
5.	Save prediction visualizations.
6.	Export detailed CSV reports.
Evaluation Metrics
•	Dice Similarity Coefficient (DSC)
•	Intersection over Union (IoU)
•	Precision
•	Recall


Outputs
results/
│
├── prediction/
│ └── model_name/
│
├── metrics_model.csv
│
└── summary_model.csv
Statistical Analysis
For each model, the following statistics are reported:
•	Mean
•	Standard deviation
•	Median
for Dice, IoU, Precision and Recall.

