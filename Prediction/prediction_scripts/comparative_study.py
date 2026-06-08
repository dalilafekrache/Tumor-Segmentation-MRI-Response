# comparative_study.py
"""
comparative_study.py
Version avec seuil optimal (Youden) intégré pour Pipeline A, B et C.
- Utilise directement les masques dans:
    F:/SEGMENTATION IRM/pythonProjetPrediction/data/masks_pred/<MODEL>/
- Applique le masque uniquement s'il existe (les images négatives restent inchangées)
- Recherche intelligente des noms de masques (suffixes _mask, _overlay, _slice0_overlay ...)
- Affiche correctement les epochs / tqdm
- Charge dynamiquement les modèles et checkpoints si nécessaire pour générer les masques
"""

import os
import shutil
import random
import logging
from pathlib import Path
from datetime import datetime
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import importlib.util

# ML & DL
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import nibabel as nib
from PIL import Image

# =======================
# CONFIGURATION (adapte si besoin)
# =======================
CONFIG = {
    'PROJECT_ROOT': r'F:\SEGMENTATION IRM\pythonProjetPrediction',
    'MODELS_DIR': r'F:\SEGMENTATION IRM\pythonProjetPrediction\models',
    'CHECKPOINTS_DIR': r'F:\SEGMENTATION IRM\pythonProjetPrediction\checkpoints',
    'IMAGES_DIR': r'F:\SEGMENTATION IRM\pythonProjetPrediction\data\images',
    'MASKS_GT_DIR': r'F:\SEGMENTATION IRM\pythonProjetPrediction\data\masks_gt',
    'MASKS_PRED_ROOT': r'F:\SEGMENTATION IRM\pythonProjetPrediction\data\masks_pred',
    'LABELS_CSV': r'F:\SEGMENTATION IRM\pythonProjetPrediction\data\labels.csv',
    'OUTPUT_DIR': r'F:\SEGMENTATION IRM\pythonProjetPrediction\results',
    'IMG_SIZE': (256, 256),
    'BATCH_SIZE': 16,
    'NUM_EPOCHS_MAX': 50,
    'EARLY_STOPPING_PATIENCE': 5,
    'MIN_DELTA': 1e-4,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'TEST_SIZE': 0.2,
    'SEED': 42,
    'SEG_MODELS': ['Unet', 'Unetpp', 'AttentionUnet', 'Medformer',
                   'Resunet', 'Resunetpp', 'Swinunet', 'Transunet', 'Uctransnet', 'Vnet']
}

os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
os.makedirs(CONFIG['MASKS_PRED_ROOT'], exist_ok=True)

# =======================
# LOGGING
# =======================
LOG_FILE = os.path.join(CONFIG['OUTPUT_DIR'], 'comparative_study.log')
logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S'))
logging.getLogger('').addHandler(console)
logging.info("=== Nouvelle exécution du script comparative_study.py ===")

# =======================
# RANDOM SEED
# =======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG['SEED'])

# =======================
# MODEL REGISTRY (mapping fichiers/classes/checkpoints)
# Ajuste les noms de classes si nécessaire
# =======================
MODEL_REGISTRY = {
    "Unet": {"file": "unet_model.py", "class": "UNet", "checkpoint": "best_model_Unet.pth"},
    "Unetpp": {"file": "unetpp_model.py", "class": "UNetPP", "checkpoint": "best_model_Unetpp.pth"},
    "AttentionUnet": {"file": "attention_unet_model.py", "class": "AttentionUNet2D", "checkpoint": "best_model_AttentionUnet.pth"},
    "Medformer": {"file": "medformer_model.py", "class": "MedFormer2D", "checkpoint": "best_model_Medformer.pth"},
    "Resunet": {"file": "resunet_model.py", "class": "ResUNet", "checkpoint": "best_model_Resunet.pth"},
    "Resunetpp": {"file": "resunetpp_model.py", "class": "ResUNetPP", "checkpoint": "best_model_Resunetpp.pth"},
    "Swinunet": {"file": "swin_unet_model.py", "class": "SwinUnet", "checkpoint": "best_model_Swinunet.pth"},
    "Transunet": {"file": "transunet_model.py", "class": "TransUNet2D", "checkpoint": "best_model_Transunet.pth"},
    "Uctransnet": {"file": "uctransnet_model.py", "class": "UCTransNet2D", "checkpoint": "best_model_Uctransnet.pth"},
    "Vnet": {"file": "VNet2d_model.py", "class": "VNet2D", "checkpoint": "best_model_Vnet.pth"},
}

# =======================
# UTILITAIRES I/O & LABELS
# =======================
def load_labels(path):
    df = pd.read_csv(path)
    if 'patient_id' not in df.columns or 'response' not in df.columns:
        raise ValueError("labels.csv must contain 'patient_id' and 'response' columns")
    return df

def split_train_test(df, test_size=0.2, seed=42):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['response'], random_state=seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# =======================
# Dynamic import util
# =======================
def _dynamic_import_class(module_path: Path, class_name: str):
    """
    Importe dynamiquement un module Python depuis module_path et retourne la classe class_name.
    """
    spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    return cls

# =======================
# Generate masks if needed (optional)
# =======================
def generate_masks_for_model(seg_model_name, images_dir=None, models_dir=None, checkpoints_dir=None, out_root=None, device=None, img_size=(256,256), threshold=0.5):
    """
    Génère des masques .nii.gz pour toutes les images dans images_dir à l'aide du modèle de segmentation.
    Retourne le dossier contenant les masques générés (Path).
    """
    device = device or CONFIG['DEVICE']
    images_dir = Path(images_dir or CONFIG['IMAGES_DIR'])
    models_dir = Path(models_dir or CONFIG['MODELS_DIR'])
    checkpoints_dir = Path(checkpoints_dir or CONFIG['CHECKPOINTS_DIR'])
    out_root = Path(out_root or CONFIG['MASKS_PRED_ROOT'])
    out_dir = out_root / seg_model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if seg_model_name not in MODEL_REGISTRY:
        raise ValueError(f"{seg_model_name} not found in MODEL_REGISTRY")

    entry = MODEL_REGISTRY[seg_model_name]
    model_file = models_dir / entry['file']
    checkpoint_file = checkpoints_dir / entry['checkpoint']

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    logging.info(f"[generate_masks] Loading model class {entry['class']} from {model_file}")
    try:
        ModelClass = _dynamic_import_class(model_file, entry['class'])
    except Exception as e:
        logging.error(f"Failed to import class {entry['class']} from {model_file}: {e}")
        raise

    # Instantiate model
    model = ModelClass()
    model = model.to(device)
    logging.info(f"[generate_masks] Loading checkpoint {checkpoint_file}")
    try:
        state = torch.load(str(checkpoint_file), map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        elif isinstance(state, dict):
            try:
                model.load_state_dict(state)
            except RuntimeError:
                new_state = {k.replace('module.', ''): v for k, v in state.items()}
                model.load_state_dict(new_state)
        else:
            model.load_state_dict(state)
    except Exception as e:
        logging.error(f"[generate_masks] Error loading checkpoint: {e}")
        raise

    model.eval()
    logging.info(f"[generate_masks] Model loaded on {device}. Starting inference on images in {images_dir}")

    image_files = sorted(list(Path(images_dir).glob("*.nii*")))
    if len(image_files) == 0:
        logging.warning(f"No images found in {images_dir}")
        return out_dir

    for img_path in tqdm(image_files, desc=f"Segmentation ({seg_model_name})", leave=False):
        try:
            nii = nib.load(str(img_path))
            arr = nii.get_fdata()
            affine = nii.affine if hasattr(nii, 'affine') else np.eye(4)

            if arr.ndim == 2:
                arr = arr[:, :, None]

            mask_out = np.zeros_like(arr, dtype=np.uint8)
            zdim = arr.shape[2]
            for z in range(zdim):
                slice_img = arr[:, :, z].astype(np.float32)
                if slice_img.max() - slice_img.min() > 1e-8:
                    norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
                else:
                    norm = slice_img * 0.0
                pil = Image.fromarray((norm * 255).astype(np.uint8)).convert("L")
                pil_resized = pil.resize(img_size, resample=Image.BILINEAR)
                x = np.array(pil_resized, dtype=np.float32) / 255.0
                x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x)
                    prob = torch.sigmoid(logits).cpu().numpy()
                    if prob.ndim == 3:
                        prob_slice = prob[0, :, :]
                    elif prob.ndim == 2:
                        prob_slice = prob
                    else:
                        prob_slice = prob.squeeze()
                prob_img = Image.fromarray((prob_slice * 255).astype(np.uint8)).convert("L")
                prob_orig_size = prob_img.resize((slice_img.shape[1], slice_img.shape[0]), resample=Image.BILINEAR)
                prob_arr = np.array(prob_orig_size, dtype=np.float32) / 255.0
                mask_bin = (prob_arr >= threshold).astype(np.uint8)
                mask_out[:, :, z] = mask_bin

            base_name = Path(img_path).stem
            out_path = out_dir / (base_name + '_mask.nii.gz')
            nib.save(nib.Nifti1Image(mask_out.astype(np.uint8), affine), str(out_path))
        except Exception as e:
            logging.warning(f"[generate_masks] Failed to process {img_path}: {e}")
            continue

    logging.info(f"[generate_masks] Masks saved to {out_dir}")
    return out_dir

# =======================
# Find mask for an image (intelligent matching)
# =======================
def find_mask_for_image(masks_dir: Path, img_path: Path):
    """
    Trouve le masque STRICT correspondant à l'image.
    Ne permet PAS de matching large.
    """
    img_name = img_path.name        # 1003_GE_slice086.nii.gz
    img_stem = img_path.stem        # 1003_GE_slice086

    # 1) Exact filename
    cand = masks_dir / img_name
    if cand.exists():
        return cand

    # 2) Exact stem with possible extensions
    for ext in [".nii.gz", ".nii"]:
        cand2 = masks_dir / f"{img_stem}_mask{ext}"
        if cand2.exists():
            return cand2

        cand3 = masks_dir / f"{img_stem}{ext}"
        if cand3.exists():
            return cand3

    # 3) No partial matching – safer
    return None


# =======================
# Dataset slice-level
# =======================
class MRISliceDataset(Dataset):
    def __init__(self, df_labels, images_dir, masks_dir=None, img_size=(256, 256)):
        self.df_labels = df_labels.set_index('patient_id')
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.img_size = img_size
        files = sorted(list(self.images_dir.glob('*.nii*')))
        valid_pids = set(self.df_labels.index)
        # keep only image files containing patient ids
        self.image_files = [f for f in files if any(pid in f.name for pid in valid_pids)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        parts = img_path.stem.split('_')
        patient_id = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
        y = int(self.df_labels.loc[patient_id, 'response'])
        arr = nib.load(str(img_path)).get_fdata()
        # if 3D, take central slice (consistent with prior behavior)
        if arr.ndim == 3:
            arr = arr[:, :, arr.shape[2] // 2]
        arr = arr.astype(np.float32)
        if arr.max() - arr.min() > 1e-8:
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        else:
            arr = arr * 0.0
        pil = Image.fromarray((arr * 255).astype(np.uint8)).resize(self.img_size)
        img = np.array(pil, dtype=np.float32) / 255.0

        # Apply mask only if we find it
        if self.masks_dir:
            mask_path = find_mask_for_image(self.masks_dir, img_path)
            if mask_path and mask_path.exists():
                try:
                    m = nib.load(str(mask_path)).get_fdata()
                    if m.ndim == 3:
                        m = m[:, :, m.shape[2] // 2]
                    m = (m > 0).astype(np.float32)
                    pilm = Image.fromarray((m * 255).astype(np.uint8)).resize(self.img_size)
                    m_arr = np.array(pilm, dtype=np.float32) / 255.0
                    img = img * m_arr
                except Exception as e:
                    logging.warning(f"Failed to load/apply mask {mask_path}: {e}")
            else:
                # no mask found -> do NOT multiply by zeros. Keep image as is.
                pass

        return torch.from_numpy(img[None, :, :]).float(), torch.tensor(y, dtype=torch.long)

# =======================
# Model (ResNet18) for slice-level classification
# =======================
class ResNetBinary(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        try:
            self.model = models.resnet18(pretrained=pretrained)
        except Exception:
            self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)

# =======================
# Training / Eval (with clearer prints & tqdm leave=True)
# =======================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch=None):
    model.train()
    running_loss = 0.0
    n = 0

    print(f"\n--- Training Epoch {epoch} ---")
    loop = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=True)
    for x, y in loop:
        x = x.to(device)
        y = y.to(device).float()
        optimizer.zero_grad()
        logits = model(x).squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        n += x.size(0)
        loop.set_postfix(loss=running_loss / (n if n > 0 else 1))
    return running_loss / (n if n > 0 else 1)

def evaluate_logits(model, dataloader, device, desc="Eval"):
    model.eval()
    ys, probs = [], []
    print(f"\n--- Evaluation: {desc} ---")
    loop = tqdm(dataloader, desc=desc, leave=True)
    with torch.no_grad():
        for x, y in loop:
            x = x.to(device)
            logits = model(x).squeeze()
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p.tolist())
            ys.extend(y.numpy().tolist())
    return np.array(ys), np.array(probs)

# =======================
# Metrics & plots (unchanged essentially)
# =======================
def compute_metrics(y_true, y_pred, y_prob=None):
    if len(np.unique(y_true)) == 1:
        tn = fp = fn = tp = 0
        acc = float((y_true == y_pred).mean())
        prec = recall = f1 = np.nan
        aucroc = np.nan
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        aucroc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    return {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'accuracy': float(acc), 'precision': float(prec),
            'recall': float(recall), 'f1': float(f1), 'auc': float(aucroc)}

def plot_confusion_matrix(y_true, y_pred, save_path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], ['0', '1'])
    plt.yticks([0, 1], ['0', '1'])
    plt.ylabel('True')
    plt.xlabel('Pred')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_pr(y_true, y_prob, out_dir, prefix=""):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc_score(y_true, y_prob):.3f}')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend()
        plt.savefig(os.path.join(out_dir, f"{prefix}_ROC.png")); plt.close()
        plt.figure()
        plt.plot(recall, precision, label=f'PR AUC={np.trapz(precision, recall):.3f}')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curve'); plt.legend()
        plt.savefig(os.path.join(out_dir, f"{prefix}_PR.png")); plt.close()
    except Exception as e:
        logging.warning(f"Failed to create ROC/PR plots: {e}")

def aggregate_slice_probs_to_patient(slice_filenames, slice_probs):
    rows = {}
    for fn, p in zip(slice_filenames, slice_probs):
        name = Path(fn).stem
        parts = name.split('_')
        pid = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
        rows.setdefault(pid, []).append(float(p))
    out = []
    for pid, probs in rows.items():
        out.append({'patient_id': pid, 'prob_mean': float(np.mean(probs)),
                    'prob_median': float(np.median(probs)),
                    'n_slices': len(probs)})
    return pd.DataFrame(out)

# =======================
# Helper: prepare masks dir (direct, no tmp) — generate if empty (optional)
# =======================
def prepare_masks_dir_for_experiment_direct(seg_model_name):
    """
    Return a Path to masks directory for given seg_model_name.
    If seg_model_name == 'GT' -> require masks in MASKS_GT_DIR.
    Otherwise, expect masks in MASKS_PRED_ROOT/seg_model_name.
    If not present or empty, try to call generate_masks_for_model() (optional).
    """
    if seg_model_name == 'GT':
        src = Path(CONFIG['MASKS_GT_DIR'])
        if not src.exists() or len(list(src.glob("*.nii*"))) == 0:
            raise FileNotFoundError(f"GT masks directory is missing or empty: {src}. Please provide GT masks.")
        return src

    src = Path(CONFIG['MASKS_PRED_ROOT']) / seg_model_name
    # if missing or empty -> try generate automatic (if MODEL_REGISTRY has entry)
    if not src.exists() or len(list(src.glob("*.nii*"))) == 0:
        # try to auto-generate if model exists in registry
        if seg_model_name in MODEL_REGISTRY:
            logging.info(f"Masks folder {src} not found or empty. Attempting to generate masks for {seg_model_name}...")
            try:
                generate_masks_for_model(seg_model_name,
                                         images_dir=CONFIG['IMAGES_DIR'],
                                         models_dir=CONFIG['MODELS_DIR'],
                                         checkpoints_dir=CONFIG['CHECKPOINTS_DIR'],
                                         out_root=CONFIG['MASKS_PRED_ROOT'],
                                         device=CONFIG['DEVICE'],
                                         img_size=CONFIG['IMG_SIZE'],
                                         threshold=0.5)
            except Exception as e:
                logging.error(f"Automatic generation of masks for {seg_model_name} failed: {e}", exc_info=True)
        # refresh src and check again
        if not src.exists() or len(list(src.glob("*.nii*"))) == 0:
            raise FileNotFoundError(f"Masks for {seg_model_name} not found in {src}. Provide masks manually or check generation.")
    return src

# =======================
# Pipeline A (uses direct masks) - with Youden threshold
# =======================
def run_pipeline_A(train_df, test_df, images_dir, masks_dir, out_dir, config):
    device = config['DEVICE']
    logging.info("Pipeline A: building slice datasets")
    train_ds = MRISliceDataset(train_df, images_dir, masks_dir, img_size=config['IMG_SIZE'])
    test_ds = MRISliceDataset(test_df, images_dir, masks_dir, img_size=config['IMG_SIZE'])
    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False)
    model = ResNetBinary(pretrained=False).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_state, best_auc = None, -1.0
    epochs_no_improve = 0
    logging.info(f"Pipeline A: training on {len(train_ds)} slices, validating on {len(test_ds)} slices")

    for epoch in range(1, config['NUM_EPOCHS_MAX'] + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        ys_slice, probs_slice = evaluate_logits(model, test_loader, device, desc=f"Eval Epoch {epoch}")
        try:
            auc_score = roc_auc_score(ys_slice, probs_slice) if len(np.unique(ys_slice)) > 1 else float('nan')
        except Exception:
            auc_score = float('nan')
        logging.info(f"[A] Epoch {epoch} loss={loss:.4f} slice_auc={auc_score:.4f}")

        # Early stopping
        if not np.isnan(auc_score) and auc_score > best_auc + config['MIN_DELTA']:
            best_auc = auc_score
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config['EARLY_STOPPING_PATIENCE']:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break

    if best_state is not None:
        model_path = os.path.join(out_dir, 'pipelineA_best.pth')
        torch.save(best_state, model_path)
        model.load_state_dict(best_state)
        logging.info(f"Saved best model to {model_path}")

    # Slice-level predictions
    slice_filenames, slice_probs, slice_labels = [], [], []
    model.eval()
    with torch.no_grad():
        test_files = test_ds.image_files
        test_loader_single = DataLoader(test_ds, batch_size=1, shuffle=False)
        for (x, y), fpath in zip(test_loader_single, test_files):
            x = x.to(device)
            p = torch.sigmoid(model(x)).cpu().numpy().item()
            slice_probs.append(float(p))
            slice_labels.append(int(y.item()))
            slice_filenames.append(str(fpath.name))

    slice_df = pd.DataFrame({'slice_file': slice_filenames, 'slice_prob': slice_probs, 'slice_label': slice_labels})
    slice_df.to_csv(os.path.join(out_dir, 'predictions_slices.csv'), index=False)
    logging.info(f"Saved slice-level predictions to {os.path.join(out_dir, 'predictions_slices.csv')}")

    # Slice-level plots
    y_true_slice = np.array(slice_labels)
    y_pred_slice = (np.array(slice_probs) >= 0.5).astype(int)
    plot_confusion_matrix(y_true_slice, y_pred_slice, os.path.join(out_dir, 'confusion_matrix_slices.png'),
                          title="Slice-level Confusion Matrix")
    plot_roc_pr(y_true_slice, np.array(slice_probs), out_dir, prefix="slice_level")

    # Patient-level aggregation
    patient_df = aggregate_slice_probs_to_patient(slice_filenames, slice_probs)
    patient_df = patient_df.merge(test_df[['patient_id', 'response']].drop_duplicates(), on='patient_id', how='left')

    # --- Youden's J threshold selection (patient-level) ---
    y_true = patient_df['response'].fillna(0).astype(int).values
    y_prob = patient_df['prob_mean'].values if 'prob_mean' in patient_df.columns else patient_df['prob_mean'].values

    if len(np.unique(y_true)) > 1:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden = tpr - fpr
        best_thresh = thresholds[np.argmax(youden)]
    else:
        best_thresh = 0.5

    patient_df['pred'] = (patient_df['prob_mean'] >= best_thresh).astype(int)
    logging.info(f"[A] Youden threshold for patient-level: {best_thresh:.4f}")
    print(f"\n=== Pipeline A: Seuil optimal (Youden) trouvé : {best_thresh:.4f} ===")

    y_pred = patient_df['pred'].values
    metrics_patient = compute_metrics(y_true, y_pred, y_prob)
    metrics_patient['n_patients'] = int(len(patient_df))
    metrics_patient['model'] = 'A_CNN_slices_patientlevel'
    metrics_patient['pipeline'] = 'A'

    # Patient-level plots
    plot_confusion_matrix(y_true, y_pred, os.path.join(out_dir, 'confusion_matrix_patient.png'))
    plot_roc_pr(y_true, y_prob, out_dir, prefix="patient_level")

    patient_df.to_csv(os.path.join(out_dir, 'predictions_patients.csv'), index=False)
    logging.info(f"Saved patient-level predictions to {os.path.join(out_dir, 'predictions_patients.csv')}")
    return metrics_patient

# =======================
# Pipeline B (radiomics + RF) - with Youden threshold
# =======================
def run_pipeline_B(train_df, test_df, masks_dir, out_dir, config):
    """
    Pipeline B : Extraction de features radiomiques (2D)
    + Classification par RandomForest sur les features patient-level.
    Compatible ResUNet / ResUNet++.
    """
    import numpy as np
    import pandas as pd
    import nibabel as nib
    import logging
    from pathlib import Path
    from tqdm import tqdm
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        confusion_matrix, accuracy_score, precision_score,
        recall_score, f1_score, roc_auc_score
    )
    import seaborn as sns
    import matplotlib.pyplot as plt

    logging.info("Pipeline B (2D): début de l'extraction radiomique")

    masks_path = Path(masks_dir)
    mask_files = sorted(list(masks_path.glob("*.nii*")))
    if len(mask_files) == 0:
        logging.warning(f"Aucun masque trouvé dans {masks_dir}")
        return None

    # 🔍 Récupération des images correspondantes
    all_images = list(Path(config['IMAGES_DIR']).rglob("*.nii*"))
    image_dict = {Path(img).name: img for img in all_images}
    logging.info(f"[Pipeline B] {len(all_images)} images détectées dans {config['IMAGES_DIR']}.")

    # --- Initialisation PyRadiomics ---
    try:
        from radiomics import featureextractor
        import SimpleITK as sitk
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllImageTypes()
        extractor.enableAllFeatures()
        USE_PYRADIOMICS = True
        logging.info("✅ PyRadiomics détecté — extraction complète activée (mode 2D).")
    except Exception as e:
        USE_PYRADIOMICS = False
        logging.warning(f"⚠️ PyRadiomics non disponible ({e}) — fallback sur features basiques.")

    features_list = []
    ignored_masks = []
    valid_pairs = 0

    for f in tqdm(mask_files, desc="Extraction features (PyRadiomics activé)", leave=False):
        img_name = Path(f.name).name
        img_path = image_dict.get(img_name)

        # 🧩 Tentatives de correspondance souples entre noms d’image et masque
        if img_path is None:
            alt_name = img_name.replace("_nifti", "")
            img_path = image_dict.get(alt_name)
        if img_path is None:
            alt_name2 = img_name.replace(".nii.gz", "_nifti.nii.gz")
            img_path = image_dict.get(alt_name2)
        if img_path is None:
            alt_name3 = img_name.replace("_mask", "")
            img_path = image_dict.get(alt_name3)
        if img_path is None:
            alt_name4 = img_name.replace("_seg", "")
            img_path = image_dict.get(alt_name4)

        if img_path is None:
            logging.warning(f"[Pipeline B] Image manquante pour {img_name}")
            continue

        pid = "_".join(Path(f.stem).stem.split('_')[:2])

        try:
            if USE_PYRADIOMICS:
                # === Lecture 2D ===
                img_itk = sitk.ReadImage(str(img_path))
                mask_itk = sitk.ReadImage(str(f))

                # Forcer 2D pur
                if img_itk.GetDimension() > 2:
                    img_itk = sitk.Extract(img_itk, (img_itk.GetWidth(), img_itk.GetHeight(), 0))
                if mask_itk.GetDimension() > 2:
                    mask_itk = sitk.Extract(mask_itk, (mask_itk.GetWidth(), mask_itk.GetHeight(), 0))

                img_arr = sitk.GetArrayFromImage(img_itk)
                mask_arr = sitk.GetArrayFromImage(mask_itk)

                # Vérifier si masque vide
                if np.sum(mask_arr > 0) == 0:
                    ignored_masks.append(f.name)
                    continue

                # 🧩 Si tailles différentes → resize avec interpolation simple 2D
                if img_arr.shape != mask_arr.shape:
                    from skimage.transform import resize
                    mask_arr = resize(mask_arr, img_arr.shape, order=0, preserve_range=True, anti_aliasing=False)
                    mask_arr = (mask_arr > 0.5).astype(np.uint8)
                    logging.warning(f"[Pipeline B] Redimensionnement 2D du masque : {f.name}")

                # Reconstruire images SimpleITK 2D cohérentes
                img_itk = sitk.GetImageFromArray(img_arr.astype(np.float32))
                mask_itk = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
                mask_itk.CopyInformation(img_itk)

                # Extraction PyRadiomics (2D)
                extractor.settings['force2D'] = True
                feats = extractor.execute(img_itk, mask_itk)
                feats = {k: float(v) for k, v in feats.items() if isinstance(v, (int, float))}

                if feats:
                    feats["patient_id"] = pid
                    features_list.append(feats)
                    valid_pairs += 1

            else:
                # Fallback simple (si PyRadiomics non dispo)
                arr = nib.load(str(f)).get_fdata()
                if np.sum(arr > 0) == 0:
                    ignored_masks.append(f.name)
                    continue
                feats = {
                    "mask_ratio": float(np.sum(arr > 0) / arr.size),
                    "mask_mean": float(arr.mean()),
                    "mask_std": float(arr.std()),
                }
                feats["patient_id"] = pid
                features_list.append(feats)
                valid_pairs += 1

        except Exception as e:
            logging.warning(f"[Pipeline B] Erreur d’extraction pour {f.name}: {e}")
            ignored_masks.append(f.name)
            continue

    # Sauvegarde des masques ignorés
    ignored_path = Path(out_dir) / "ignored_masks.txt"
    with open(ignored_path, "w") as f_out:
        f_out.write("\n".join(ignored_masks))
    logging.info(f"🧹 {len(ignored_masks)} masques vides/ignorés — liste sauvegardée dans {ignored_path}")
    logging.info(f"{valid_pairs} paires image/masque valides détectées sur {len(mask_files)} masques.")

    if len(features_list) == 0:
        logging.error("[Pipeline B] Aucune paire image/masque valide trouvée.")
        return None

    # === Construction du DataFrame ===
    df_features = pd.DataFrame(features_list)
    logging.info(f"Extraction terminée — shape: {df_features.shape}")

    # Fusion labels
    df_features = df_features.merge(train_df[['patient_id', 'response']], on='patient_id', how='left', indicator=True)
    n_train = df_features['_merge'].eq('both').sum()
    df_features.drop(columns=['_merge'], inplace=True)
    df_features = df_features.merge(test_df[['patient_id', 'response']], on='patient_id',
                                    how='left', suffixes=('_train', '_test'), indicator=True)
    n_test = df_features['_merge'].eq('both').sum()
    df_features.drop(columns=['_merge'], inplace=True)
    logging.info(f"{n_train} patients train trouvés, {n_test} patients test trouvés dans les features.")

    # Split train/test
    train_X = df_features[df_features['response_train'].notna()].drop(columns=['patient_id', 'response_train', 'response_test'])
    test_X = df_features[df_features['response_test'].notna()].drop(columns=['patient_id', 'response_train', 'response_test'])
    y_train = df_features[df_features['response_train'].notna()]['response_train'].astype(int).values
    y_test = df_features[df_features['response_test'].notna()]['response_test'].astype(int).values

    if len(train_X) == 0 or len(test_X) == 0:
        logging.warning("Aucun patient train/test trouvé dans les features extraites.")
        return None

    # Normalisation + RandomForest
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_X)
    X_test_scaled = scaler.transform(test_X)
    rf = RandomForestClassifier(n_estimators=200, random_state=config['SEED'])
    rf.fit(X_train_scaled, y_train)
    probs = rf.predict_proba(X_test_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else float('nan')
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    metrics_patient = {
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'accuracy': float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds, zero_division=0)),
        'recall': float(recall_score(y_test, preds, zero_division=0)),
        'f1': float(f1_score(y_test, preds, zero_division=0)),
        'auc': float(auc),
        'n_patients': int(len(y_test)),
        'model': 'B_radiomics_RF_2D',
        'pipeline': 'B'
    }

    # Sauvegarde résultats
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(out_dir / "radiomics_features.csv", index=False)

    try:
        importances = rf.feature_importances_
        imp_df = pd.DataFrame({'feature': train_X.columns, 'importance': importances})
        imp_df = imp_df.sort_values('importance', ascending=False).head(20)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=imp_df, x='importance', y='feature', palette='viridis')
        plt.title("Top 20 des features (Pipeline B - 2D)")
        plt.tight_layout()
        plt.savefig(out_dir / "feature_importance.png")
        plt.close()
    except Exception as e:
        logging.warning(f"Impossible de tracer les importances de features: {e}")

    logging.info(f"Pipeline B (2D) terminé — résultats sauvegardés dans {out_dir/'radiomics_features.csv'}")
    return metrics_patient


# =======================
# Orchestration
# =======================
def run_all_experiments(config):
    labels_df = load_labels(config['LABELS_CSV'])
    summary_rows = []
    train_df, test_df = split_train_test(labels_df, test_size=config['TEST_SIZE'], seed=config['SEED'])

    for seg in config['SEG_MODELS']:
        logging.info(f"=== Running experiments for segmentation model: {seg} ===")
        masks_dir = prepare_masks_dir_for_experiment(seg)
        exp_out = Path(config['OUTPUT_DIR']) / f"exp_{seg}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_out.mkdir(parents=True, exist_ok=True)
        out_summary_rows = []

        try:
            resA = run_pipeline_A(train_df, test_df, config['IMAGES_DIR'], masks_dir, exp_out, config)
            out_summary_rows.append(resA)
            logging.info(f"Pipeline A finished for {seg}: {resA}")
        except Exception as e:
            logging.error(f"Pipeline A failed for {seg}: {e}", exc_info=True)

        try:
            resB = run_pipeline_B(train_df, test_df, masks_dir, exp_out, config)
            out_summary_rows.append(resB)
            logging.info(f"Pipeline B finished for {seg}: {resB}")
        except Exception as e:
            logging.error(f"Pipeline B failed for {seg}: {e}", exc_info=True)

        summary_rows.extend(out_summary_rows)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = Path(config['OUTPUT_DIR']) / "summary_all_exps.csv"
    summary_df.to_csv(summary_csv, index=False)
    logging.info(f"All experiments finished. Summary saved to {summary_csv}")
    return summary_df


# =======================

# =======================
# Pipeline C (deep features + RF) - with Youden threshold
# =======================
# ============================================================
# 🔥 PIPELINE C — DEEP FEATURES + RANDOM FOREST (clean version)
# ============================================================

# ============================================================
# ✔ PIPELINE C — Deep Features + RandomForest (VERSION FINALE)
# ============================================================

def run_pipeline_C(train_df, test_df, images_dir, masks_dir, out_dir, config):
    """
    Pipeline C : deep features (ResNet18) + RandomForest.
    Compatible avec test_pipeline_C.py
    """
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_curve, roc_auc_score, confusion_matrix,
        accuracy_score, precision_score, recall_score, f1_score
    )
    from tqdm.auto import tqdm
    import nibabel as nib
    from PIL import Image
    import pandas as pd
    import os
    from pathlib import Path

    device = config["DEVICE"]
    IMG_SIZE = config["IMG_SIZE"]

    # ============================================================
    # --- HELPERS ---
    # ============================================================

    def load_and_resize(img_path):
        arr = nib.load(str(img_path)).get_fdata()
        if arr.ndim == 3:
            arr = arr[:, :, arr.shape[2] // 2]

        arr = arr.astype(np.float32)
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())

        pil = Image.fromarray((arr * 255).astype(np.uint8)).resize(IMG_SIZE)
        return np.array(pil, dtype=np.float32) / 255.0

    def load_and_resize_mask(mask_path):
        m = nib.load(str(mask_path)).get_fdata()
        if m.ndim == 3:
            m = m[:, :, m.shape[2] // 2]

        m = (m > 0).astype(np.float32)
        pil = Image.fromarray((m * 255).astype(np.uint8)).resize(IMG_SIZE)
        return np.array(pil, dtype=np.float32) / 255.0

    # ============================================================
    # --- DATASET ---
    # ============================================================

    class FeatureDataset(Dataset):
        def __init__(self, df_labels, images_dir, masks_dir):
            self.df = df_labels.set_index("patient_id")
            self.imdir = Path(images_dir)
            self.maskdir = Path(masks_dir)
            all_imgs = sorted(list(self.imdir.glob("*.nii*")))
            valid_ids = set(self.df.index)
            self.image_files = [f for f in all_imgs if any(pid in f.name for pid in valid_ids)]

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            f = self.image_files[idx]
            parts = f.stem.split("_")
            pid = "_".join(parts[:2])

            img = load_and_resize(f)
            mask_path = find_mask_for_image(self.maskdir, f)

            if mask_path and mask_path.exists():
                m = load_and_resize_mask(mask_path)
                img = img * m

            return torch.from_numpy(img[None]).float(), pid

    # ============================================================
    # --- FEATURE EXTRACTION BACKBONE ---
    # ============================================================

    backbone = models.resnet18(weights=None)
    backbone.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device)
    backbone.eval()

    def extract(loader, desc=""):
        feats, pids = [], []
        with torch.no_grad():
            for x, pid in tqdm(loader, desc=desc):
                x = x.to(device)
                v = backbone(x).cpu().numpy()
                feats.append(v)
                pids.extend(pid)
        return np.concatenate(feats, axis=0), pids

    # ============================================================
    # --- DATALOADERS ---
    # ============================================================

    train_ds = FeatureDataset(train_df, images_dir, masks_dir)
    test_ds = FeatureDataset(test_df, images_dir, masks_dir)

    train_dl = DataLoader(train_ds, batch_size=config["BATCH_SIZE"], shuffle=False)
    test_dl  = DataLoader(test_ds, batch_size=config["BATCH_SIZE"], shuffle=False)

    # ============================================================
    # --- EXTRACT FEATURES ---
    # ============================================================

    Xtrain_slices, Ptrain = extract(train_dl, "Deep features C (train)")
    Xtest_slices,  Ptest  = extract(test_dl,  "Deep features C (test)")

    # ============================================================
    # --- AGGREGATE patient-level ---
    # ============================================================

    def aggregate(X, P):
        d = {}
        for feat, pid in zip(X, P):
            d.setdefault(pid, []).append(feat)
        ids = list(d.keys())
        Xagg = np.array([np.mean(np.vstack(d[p]), axis=0) for p in ids])
        return Xagg, ids

    Xtrain, Ptrain2 = aggregate(Xtrain_slices, Ptrain)
    Xtest,  Ptest2  = aggregate(Xtest_slices, Ptest)

    # ============================================================
    # --- FILTER USING LABELS ---
    # ============================================================

    train_idx = train_df.set_index("patient_id")
    test_idx  = test_df.set_index("patient_id")

    y_train = np.array([int(train_idx.loc[p, "response"]) for p in Ptrain2])
    y_test  = np.array([int(test_idx.loc[p, "response"]) for p in Ptest2])

    # ============================================================
    # --- RANDOM FOREST ---
    # ============================================================

    scaler = StandardScaler()
    Xtrain_s = scaler.fit_transform(Xtrain)
    Xtest_s  = scaler.transform(Xtest)

    clf = RandomForestClassifier(n_estimators=200, random_state=config["SEED"])
    clf.fit(Xtrain_s, y_train)

    probs = clf.predict_proba(Xtest_s)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # ============================================================
    # --- SAVE RESULTS ---
    # ============================================================

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "patient_id": Ptest2,
        "y_true": y_test,
        "y_pred": preds,
        "y_prob": probs
    }).to_csv(out_dir / "predictions_patients_C.csv", index=False)

    # ============================================================
    # --- METRICS ---
    # ============================================================

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    auc = roc_auc_score(y_test, probs)
    # ============================================================
    # --- SAVE FIGURES (Confusion matrix, ROC, PR curves)
    # ============================================================

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title("Confusion Matrix (Pipeline C)")
    plt.colorbar()
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="red", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_C.png")
    plt.close()

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Pipeline C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "C_patient_level_ROC.png")
    plt.close()

    # --- Precision–Recall Curve ---
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.figure()
    plt.plot(recall, precision, label=f"PR AUC={np.trapz(precision, recall):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve (Pipeline C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "C_patient_level_PR.png")
    plt.close()

    return {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "auc": float(auc),
        "n_patients": len(y_test),
        "model": "C_deep_RF",
        "pipeline": "C"
    }


# =======================
# Orchestration (uses direct masks)
# =======================
def run_all_experiments(config):
    labels_df = load_labels(config['LABELS_CSV'])
    summary_rows = []
    train_df, test_df = split_train_test(labels_df, test_size=config['TEST_SIZE'], seed=config['SEED'])

    for seg in config['SEG_MODELS']:
        logging.info(f"=== Running experiments for segmentation model: {seg} ===")
        try:
            masks_dir = prepare_masks_dir_for_experiment_direct(seg)
        except Exception as e:
            logging.error(f"Skipping {seg} because prepare_masks_dir_for_experiment_direct failed: {e}")
            continue

        exp_out = Path(config['OUTPUT_DIR']) / f"testA_{seg}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_out.mkdir(parents=True, exist_ok=True)
        out_summary_rows = []

        # ---------- Pipeline A ----------
        try:
            resA = run_pipeline_A(train_df, test_df, config['IMAGES_DIR'], masks_dir, exp_out, config)
            out_summary_rows.append(resA)
            logging.info(f"Pipeline A finished for {seg}: {resA}")
        except Exception as e:
            logging.error(f"Pipeline A failed for {seg}: {e}", exc_info=True)

        # ---------- Pipeline B ----------
        try:
            resB = run_pipeline_B(train_df, test_df, masks_dir, exp_out, config)
            if resB is not None:
                out_summary_rows.append(resB)
                logging.info(f"Pipeline B finished for {seg}: {resB}")
        except Exception as e:
            logging.error(f"Pipeline B failed for {seg}: {e}", exc_info=True)

        # ---------- Pipeline C ----------
        try:
            resC = run_pipeline_C(train_df, test_df, config['IMAGES_DIR'], masks_dir, exp_out, config)
            if resC is not None:
                out_summary_rows.append(resC)
                logging.info(f"Pipeline C finished for {seg}: {resC}")
        except Exception as e:
            logging.error(f"Pipeline C failed for {seg}: {e}", exc_info=True)

        summary_rows.extend(out_summary_rows)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = Path(config['OUTPUT_DIR']) / "summary_all_exps.csv"
    summary_df.to_csv(summary_csv, index=False)
    logging.info(f"All experiments finished. Summary saved to {summary_csv}")
    return summary_df

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    summary_df = run_all_experiments(CONFIG)
    logging.info("=== Script finished successfully ===")
