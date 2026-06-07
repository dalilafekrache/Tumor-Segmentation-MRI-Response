# High-Quality Tumor Segmentation Boosts AI-Based MRI Response Assessment of Breast Cancer

Official implementation of the manuscript:
**"High-Quality Tumor Segmentation Boosts AI-Based MRI Response Assessment of Breast Cancer"**, submitted to *The Visual Computer*.

---

## Overview

This repository contains the complete experimental framework developed for **breast tumor segmentation** and **AI-based MRI response assessment** in breast cancer patients.

The proposed framework investigates the impact of **tumor segmentation quality** on downstream clinical response prediction. It combines **object detection**, **medical image segmentation**, **quantitative evaluation**, and **predictive analysis pipelines**.

The framework includes:

* YOLOv8-based tumor detection
* Automatic binary mask generation
* Comparative evaluation of multiple segmentation architectures
* nnU-Net integration for large-scale medical segmentation
* Segmentation quality assessment
* AI-based MRI response prediction pipelines

This repository provides all scripts required to reproduce the experiments presented in the manuscript.

---

## Repository Structure

```text
detection_module/       # YOLOv8 detection pipeline
segmentation_models/    # Segmentation architectures
evaluation_module/      # Quantitative evaluation
prediction_module/      # Clinical prediction pipelines
nnUNet_module/          # nnU-Net experimental setup
requirements.txt
README.md
```

---

## Detection Module

The detection module performs **tumor localization using YOLOv8**.

### Features

* Train the YOLOv8 detector
* Generate binary masks from bounding-box annotations
* Perform object detection
* Convert predicted bounding boxes into binary masks

### Outputs

* Tumor detections
* YOLO label files
* Binary masks for segmentation

---

## Segmentation Models

The repository includes implementations of the following segmentation architectures:

* U-Net
* U-Net++
* Attention U-Net
* ResUNet
* ResUNet++
* TransUNet
* Swin-Unet
* UCTransNet
* MedFormer
* V-Net
* nnU-Net

Each model is trained independently under the same experimental protocol to ensure **fair comparison**.

### Training Outputs

* Trained model weights
* Dice score
* IoU score
* Precision
* Recall
* Segmentation visualizations
* Training logs

---

## Evaluation Module

The evaluation module quantitatively assesses segmentation performance.

### Evaluation Metrics

* Dice Similarity Coefficient (DSC)
* Intersection over Union (IoU)
* Precision
* Recall

For each model, prediction visualizations and CSV reports are automatically generated.

### Statistical Summaries

* Mean
* Standard deviation
* Median

---

## Prediction Module

The prediction module investigates the impact of segmentation quality on downstream clinical prediction tasks.

Generated segmentations are used as inputs for predictive pipelines in order to assess their influence on **MRI-based response assessment**.

### Available Pipelines

#### Pipeline A – 2D CNN classification with patient-level aggregation

Slice-level CNN classification followed by patient-level prediction aggregation.

#### Pipeline B – Radiomics + Random Forest classification

Extraction of radiomic features from segmented images with Random Forest classification.

#### Pipeline C – Deep features + Random Forest classification

CNN-based deep feature extraction followed by Random Forest classification.

### Outputs

* Prediction results
* Performance summaries
* Comparative analyses
* CSV reports

---

## nnU-Net Module

Due to the high computational requirements of **nnU-Net**, experiments were conducted on the **CCUB HPC cluster (University of Burgundy)**.

The nnU-Net workflow includes:

* Dataset preparation
* Automatic preprocessing
* Model training
* Prediction generation
* Quantitative evaluation

nnU-Net experiments are fully integrated into the comparative study framework.

---

## Installation

### Clone the repository

```bash
git clone https://github.com/dalilafekrache/Tumor-Segmentation-MRI-Response.git
cd Tumor-Segmentation-MRI-Response
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

### Recommended Environment

* Python 3.10+
* PyTorch
* ultralytics
* OpenCV
* NumPy
* nibabel
* SimpleITK
* scikit-learn
* pandas
* matplotlib

Detailed package versions are provided in:

```text
requirements.txt
```

---

## Dataset Availability

The datasets generated and analysed during this study are **not publicly available** due to patient confidentiality and institutional restrictions.

**De-identified data** will be available from the corresponding author upon reasonable request and subject to applicable ethical approval.
