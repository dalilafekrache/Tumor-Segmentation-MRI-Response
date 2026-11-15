# Impact of Tumor Segmentation Quality on MRI-Based Prediction of Breast Cancer Treatment Response: Propagation of Detection Errors in an Automated Pipeline

Official implementation of the manuscript *"Impact of Tumor Segmentation Quality on MRI-Based Prediction of Breast Cancer Treatment Response: Propagation of Detection Errors in an Automated Pipeline"*, submitted to the *Journal of Radiation Research and Applied Sciences* (manuscript JRRAS-2026-1469R1).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20593550.svg)](https://doi.org/10.5281/zenodo.20593550)

## Overview

This repository contains the complete experimental framework used to study how the quality of automated breast tumor segmentation on MRI affects the downstream prediction of pathological complete response (pCR) to neoadjuvant chemotherapy.

The framework is an automated detection–segmentation–prediction pipeline:

1. **Detection**: a YOLOv8 detector selects the MRI slices that contain a tumor.
2. **Segmentation**: eleven segmentation architectures delineate the tumor on the selected slices.
3. **Prediction**: three complementary pipelines predict pCR from the manual masks and from the masks produced by each of the eleven models.

In addition to segmentation quality (Dice, IoU, precision, recall), the framework quantifies how **false-positive detections of the detector propagate** through segmentation to response prediction.

Segmentation models were developed on 99 manually annotated breast MRI examinations. Response prediction was evaluated on 72 patients (23 pCR), with the expert manual masks as a reference condition.

## Repository structure

```
detection_module/       # YOLOv8 detection pipeline
segmentation_models/    # Eleven segmentation architectures
evaluation_module/      # Segmentation evaluation, false-positive propagation, figures
prediction_module/      # pCR prediction pipelines and cross-validation
nnUNet_module/          # nnU-Net experimental setup
requirements.txt
README.md
```

## Detection module

Tumor localization with YOLOv8 at a single high-recall operating point (confidence threshold 0.25).

- Training of the YOLOv8 detector
- Conversion of bounding-box annotations into YOLO label files
- Slice-level detection and conversion of predicted boxes into binary masks for the segmentation stage

## Segmentation models

The repository includes the following architectures, all trained under the same experimental protocol:

U-Net, U-Net++, Attention U-Net, ResUNet, ResUNet++, TransUNet, Swin-Unet, UCTransNet, MedFormer, V-Net 2D and nnU-Net.

Training outputs: model weights (not distributed), Dice, IoU, precision and recall, training logs and segmentation visualizations.

## Evaluation module

- **Segmentation metrics** at slice level and patient level: Dice similarity coefficient, IoU, precision and recall, with mean, standard deviation and median.
- **Behavior on tumor-free slices**: proportion of detector-selected slices without tumor in the manual reference on which each model segmented tissue.
- **False-positive propagation analysis** (`cv_etape6_propagation_faux_positifs.py`): tracking of the slices selected by YOLOv8 without tumor in the manual reference, and repetition of the radiomics cross-validation after removing these slices, which simulates a detector without false positives.
- **Qualitative figure** (`figure_qualitative_v2.py`): masks of the eleven models on a tumor-containing slice and on a false-positive slice.

## Prediction module

Three prediction pipelines are evaluated with every mask set (manual masks and the eleven automated mask sets):

- **Pipeline A**: fine-tuned 2D CNN with slice-level classification and patient-level aggregation.
- **Pipeline B**: radiomic features with a Random Forest classifier.
- **Pipeline C**: deep features from an ImageNet-pretrained ResNet-18, either frozen or fine-tuned, with a Random Forest classifier.

### Validation protocol

- Repeated stratified 5-fold cross-validation on the 72 patients (10 repetitions, random seed 42) for Pipeline B and frozen Pipeline C. All mask sets are evaluated on identical folds.
- Pipeline A and fine-tuned Pipeline C are evaluated on the five folds of the first repetition, which are identical to those used for the other pipelines, because fine-tuning a CNN within each fold is computationally demanding.
- All data-driven steps (imputation, standardization, classifier training and decision-threshold selection) are performed within the training folds only. The decision threshold is selected with the Youden index by inner cross-validation.
- AUC with stratified bootstrap 95% confidence intervals (2,000 resamples); sensitivity and specificity with Wilson confidence intervals.
- Significance: label-permutation tests (1,000 permutations) for Pipelines B and C (frozen); Mann–Whitney test for Pipeline A and fine-tuned Pipeline C.
- Exploratory Spearman correlations across the eleven models between prediction AUC and segmentation characteristics.

### Main scripts

| Script | Purpose |
|---|---|
| `cv_etape5_pipelineA_et_C_finetune_v2.py` | Cross-validation of Pipeline A and fine-tuned Pipeline C |
| `[nom du script de validation croisée du pipeline B]` | Repeated cross-validation of Pipeline B (radiomics) |
| `[nom du script de validation croisée du pipeline C gelé]` | Repeated cross-validation of frozen Pipeline C |
| `cv_etape6_propagation_faux_positifs.py` | False-positive propagation and sensitivity analysis |
| `figure_qualitative_v2.py` | Qualitative comparison figure |

Outputs: per-fold and per-repetition predictions, AUC, sensitivity, specificity, confidence intervals, p-values and CSV summary tables.

## nnU-Net module

Because of its computational requirements, nnU-Net was trained on the CCUB HPC cluster (University of Burgundy). The workflow includes dataset preparation, automatic preprocessing, training, prediction and evaluation, and its outputs are integrated into the same comparative framework as the other architectures.

## Installation

```bash
git clone https://github.com/dalilafekrache/Tumor-Segmentation-MRI-Response.git
cd Tumor-Segmentation-MRI-Response
pip install -r requirements.txt
```

Recommended environment: Python 3.10+, PyTorch, ultralytics, OpenCV, NumPy, nibabel, SimpleITK, PyRadiomics, scikit-learn, SciPy, pandas and matplotlib. Package versions are listed in `requirements.txt`.

Data paths are defined at the top of each script and must be adapted to the local directory structure.

## Versions

- **v1.0**: version associated with the initial submission.
- **v2**: metadata update of the initial version (June 2026).
- **v3.0**: revised version accompanying the revision of the manuscript (repeated stratified cross-validation, permutation tests, false-positive propagation analysis, corrected Pipeline C).

## Data availability

The datasets generated and analysed during this study are not publicly available due to patient confidentiality and institutional restrictions. De-identified data are available from the corresponding author upon reasonable request and subject to applicable ethical approval.

## Citation

If you use this code, please cite the archived software (all versions):

Fekrache, D. *Tumor-Segmentation-MRI-Response*. Zenodo. https://doi.org/10.5281/zenodo.20593550
