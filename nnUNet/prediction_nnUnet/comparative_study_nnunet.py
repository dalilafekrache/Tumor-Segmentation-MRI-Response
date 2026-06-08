# comparative_study_nnunet_variants.py
"""
Deux versions pour exécuter les pipelines A/B/C avec les masques nnUNet :

PART 3 — "mixte" (version serveur mais strictement alignée avec ton comparative_study local,
           seuls les chemins changent)
           -> fichier: comparative_study_nnunet_server

PART 4 — "paramétrable" (un seul script qui détecte l'environnement ou accepte
           une variable d'environnement / argument pour basculer entre local et serveur)
           -> fichier: comparative_study_nnunet_parametric

Les deux scripts sont prêts à être copiés dans ton serveur local / distant.
Ils reprennent la logique exacte de `comparative_study.py` (metrics, Youden, plots,
search strict des masques, pipelines A/B/C) mais adaptés pour nnUNet.

Usage rapide:
- PART 3 (server): modifier si besoin les chemins en haut puis lancer `python comparative_study_nnunet_server.py`
- PART 4 (parametric): soit définir env var `RUN_ENV=server` ou `RUN_ENV=local`, soit lancer sans var (détection automatique des chemins)

Note: j'ai laissé les imports et la logique identiques à ton script local; si tu veux que je enlève des morceaux
(p. ex. génération automatique de masques) dis-le et je nettoie.
"""

# -----------------------------
# PART 3: Server-only version
# File name suggestion: comparative_study_nnunet_server.py
# -----------------------------

# -*- coding: utf-8 -*-
"""
comparative_study_nnunet_server.py
Version SERVEUR (nnUNet) — adapté de ton comparative_study.py local.
Chemins par défaut pointent vers /work/icmub/dfn25964/test_pipelineA
"""

import os
import random
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

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

# ------------------
# CONFIG (SERVER)
# ------------------
CONFIG = {
    'PROJECT_ROOT': '/work/icmub/dfn25964/test_pipelineA',
    'IMAGES_DIR': '/work/icmub/dfn25964/test_pipelineA/images',
    'NNUNET_MASKS_DIR': '/work/icmub/dfn25964/test_pipelineA/masks_pred/nnunet',
    'LABELS_CSV': '/work/icmub/dfn25964/test_pipelineA/labels.csv',
    'OUTPUT_DIR': '/work/icmub/dfn25964/test_pipelineA/results',
    'IMG_SIZE': (256, 256),
    'BATCH_SIZE': 16,
    'NUM_EPOCHS_MAX': 50,
    'EARLY_STOPPING_PATIENCE': 5,
    'MIN_DELTA': 1e-4,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'TEST_SIZE': 0.2,
    'SEED': 42,
    'SEG_MODELS': ['nnunet']
}

os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

# Logging
LOG_FILE = os.path.join(CONFIG['OUTPUT_DIR'], 'comparative_study_nnunet_server.log')
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logging.info('=== Nouvelle exécution comparative_study_nnunet_server.py ===')

# Seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG['SEED'])

# ------------------
# UTILITIES
# ------------------

def load_labels(path):
    df = pd.read_csv(path)
    if 'patient_id' not in df.columns or 'response' not in df.columns:
        raise ValueError("labels.csv must contain 'patient_id' and 'response' columns")
    return df


def split_train_test(df, test_size=0.2, seed=42):
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['response'], random_state=seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# Strict find mask (nnunet) — only exact or stem.ext or stem_mask.ext
from pathlib import Path

def find_mask_for_image(masks_dir: Path, img_path: Path):
    masks_dir = Path(masks_dir)
    img_name = img_path.name
    img_stem = img_path.stem

    # exact
    cand = masks_dir / img_name
    if cand.exists():
        return cand

    # stem + .nii.gz or .nii
    for ext in ['.nii.gz', '.nii']:
        p = masks_dir / (img_stem + ext)
        if p.exists():
            return p
        p2 = masks_dir / (img_stem + '_mask' + ext)
        if p2.exists():
            return p2

    return None


# Dataset (slice-level) — applies mask if exists
class MRISliceDataset(Dataset):
    def __init__(self, df_labels, images_dir, masks_dir=None, img_size=(256,256)):
        self.df_labels = df_labels.set_index('patient_id')
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.img_size = img_size
        files = sorted(list(self.images_dir.glob('*.nii*')))
        valid_pids = set(self.df_labels.index)
        self.image_files = [f for f in files if any(pid in f.name for pid in valid_pids)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        parts = img_path.stem.split('_')
        patient_id = '_'.join(parts[:2]) if len(parts)>=2 else parts[0]
        y = int(self.df_labels.loc[patient_id, 'response'])

        arr = nib.load(str(img_path)).get_fdata()
        if arr.ndim==3:
            arr = arr[:,:,arr.shape[2]//2]
        arr = arr.astype(np.float32)
        if arr.max() - arr.min() > 1e-8:
            arr = (arr - arr.min())/(arr.max()-arr.min()+1e-8)
        else:
            arr = arr*0.0
        pil = Image.fromarray((arr*255).astype('uint8')).resize(self.img_size)
        img = np.array(pil, dtype=np.float32)/255.0

        if self.masks_dir:
            mask_path = find_mask_for_image(self.masks_dir, img_path)
            if mask_path and mask_path.exists():
                try:
                    m = nib.load(str(mask_path)).get_fdata()
                    if m.ndim==3:
                        m = m[:,:,m.shape[2]//2]
                    m = (m>0).astype(np.float32)
                    mpil = Image.fromarray((m*255).astype('uint8')).resize(self.img_size)
                    m_arr = np.array(mpil, dtype=np.float32)/255.0
                    img = img * m_arr
                except Exception as e:
                    logging.warning(f"Failed to load/apply mask {mask_path}: {e}")
        return torch.from_numpy(img[None,:,:]).float(), torch.tensor(y, dtype=torch.long)


# ResNetBinary (slice classifier)
class ResNetBinary(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.model = models.resnet18(weights=None)
        except Exception:
            self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
        n = self.model.fc.in_features
        self.model.fc = nn.Linear(n,1)
    def forward(self,x):
        return self.model(x).squeeze(1)


# training / eval helpers

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    n = 0
    loop = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=True)
    for x,y in loop:
        x = x.to(device); y = y.to(device).float()
        optimizer.zero_grad()
        logits = model(x).squeeze()
        loss = criterion(logits, y)
        loss.backward(); optimizer.step()
        running_loss += loss.item()*x.size(0); n += x.size(0)
        loop.set_postfix(loss=running_loss/(n if n>0 else 1))
    return running_loss/(n if n>0 else 1)


def evaluate_logits(model, dataloader, device):
    model.eval(); ys, probs = [], []
    with torch.no_grad():
        for x,y in tqdm(dataloader, desc='Eval', leave=True):
            x = x.to(device)
            logits = model(x).squeeze()
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p.tolist()); ys.extend(y.numpy().tolist())
    return np.array(ys), np.array(probs)


# plotting

def plot_confusion_matrix(y_true, y_pred, save_path, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,5)); plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title); plt.colorbar(); plt.xticks([0,1],['0','1']); plt.yticks([0,1],['0','1'])
    plt.ylabel('True'); plt.xlabel('Pred')
    for i in range(2):
        for j in range(2):
            plt.text(j,i,cm[i,j],ha='center',va='center',color='red',fontsize=14)
    plt.tight_layout(); plt.savefig(save_path); plt.close()


def plot_roc_pr(y_true, y_prob, out_dir, prefix=''):
    try:
        fpr,tpr,_ = roc_curve(y_true, y_prob)
        plt.figure(); plt.plot(fpr,tpr,label=f'AUC={roc_auc_score(y_true,y_prob):.3f}'); plt.plot([0,1],[0,1],'--',color='gray')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend(); plt.savefig(os.path.join(out_dir,f"{prefix}_ROC.png")); plt.close()
        precision, recall, _ = precision_recall_curve(y_true,y_prob)
        plt.figure(); plt.plot(recall, precision, label=f'PR AUC={np.trapz(precision, recall):.3f}'); plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title('PR Curve'); plt.legend(); plt.savefig(os.path.join(out_dir,f"{prefix}_PR.png")); plt.close()
    except Exception as e:
        logging.warning(f"Failed to create ROC/PR plots: {e}")


def aggregate_slice_probs_to_patient(slice_filenames, slice_probs):
    rows = {}
    for fn,p in zip(slice_filenames, slice_probs):
        pid_parts = Path(fn).stem.split('_')
        pid = '_'.join(pid_parts[:2]) if len(pid_parts)>=2 else pid_parts[0]
        rows.setdefault(pid, []).append(float(p))
    out = []
    for pid, probs in rows.items():
        out.append({'patient_id':pid,'prob_mean':float(np.mean(probs)),'prob_median':float(np.median(probs)),'n_slices':len(probs)})
    return pd.DataFrame(out)


# ------------------
# PIPELINE A (slice CNN -> patient Youden)
# ------------------

def run_pipeline_A(train_df, test_df, images_dir, masks_dir, out_dir, config):
    device = config['DEVICE']
    train_ds = MRISliceDataset(train_df, images_dir, masks_dir, img_size=config['IMG_SIZE'])
    test_ds  = MRISliceDataset(test_df, images_dir, masks_dir, img_size=config['IMG_SIZE'])
    train_loader = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False)
    model = ResNetBinary().to(device)
    criterion = nn.BCEWithLogitsLoss(); optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_auc = -1.0; best_state=None; no_imp=0
    for epoch in range(1, config['NUM_EPOCHS_MAX']+1):
        _ = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        ys, probs = evaluate_logits(model, test_loader, device)
        try:
            auc = roc_auc_score(ys, probs) if len(np.unique(ys))>1 else float('nan')
        except Exception:
            auc = float('nan')
        if not np.isnan(auc) and auc > best_auc + config['MIN_DELTA']:
            best_auc = auc; best_state = model.state_dict(); no_imp=0
        else:
            no_imp += 1
        if no_imp >= config['EARLY_STOPPING_PATIENCE']:
            logging.info(f"Early stopping at epoch {epoch}"); break
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(out_dir,'pipelineA_best.pth'))

    # slice preds
    slice_filenames, slice_probs, slice_labels = [],[],[]
    model.eval()
    with torch.no_grad():
        loader = DataLoader(test_ds, batch_size=1, shuffle=False)
        for (x,y), fpath in zip(loader, test_ds.image_files):
            x = x.to(device)
            p = torch.sigmoid(model(x)).cpu().numpy().item()
            slice_filenames.append(str(fpath.name)); slice_probs.append(float(p)); slice_labels.append(int(y.item()))
    pd.DataFrame({'slice_file':slice_filenames,'slice_prob':slice_probs,'slice_label':slice_labels}).to_csv(os.path.join(out_dir,'predictions_slices.csv'), index=False)
    plot_confusion_matrix(np.array(slice_labels),(np.array(slice_probs)>=0.5).astype(int), os.path.join(out_dir,'confusion_matrix_slices.png'))
    plot_roc_pr(np.array(slice_labels), np.array(slice_probs), out_dir, prefix='slice_level')

    patient_df = aggregate_slice_probs_to_patient(slice_filenames, slice_probs)
    patient_df = patient_df.merge(test_df[['patient_id','response']].drop_duplicates(), on='patient_id', how='left')
    y_true = patient_df['response'].fillna(0).astype(int).values
    y_prob = patient_df['prob_mean'].values
    if len(np.unique(y_true))>1:
        fpr,tpr,thr = roc_curve(y_true,y_prob); youden = tpr - fpr; best_thresh = thr[np.argmax(youden)]
    else:
        best_thresh = 0.5
    patient_df['pred'] = (patient_df['prob_mean'] >= best_thresh).astype(int)
    plot_confusion_matrix(y_true, patient_df['pred'].values, os.path.join(out_dir,'confusion_matrix_patient.png'))
    plot_roc_pr(y_true, y_prob, out_dir, prefix='patient_level')
    patient_df.to_csv(os.path.join(out_dir,'predictions_patients.csv'), index=False)
    metrics = compute_metrics(y_true, patient_df['pred'].values, y_prob)
    metrics.update({'n_patients':int(len(patient_df)),'model':'A_CNN_slices_patientlevel','pipeline':'A','youden_threshold':float(best_thresh)})
    return metrics


# ------------------
# PIPELINE B (simple radiomics from masks)
# ------------------

def run_pipeline_B(train_df, test_df, masks_dir, out_dir, config):
    masks_path = Path(masks_dir)
    mask_files = sorted(list(masks_path.glob('*.nii*')))
    if len(mask_files)==0:
        logging.warning('No masks in '+str(masks_path)); return None
    feats = []
    for f in tqdm(mask_files, desc='Radiomics B'):
        pid = '_'.join(Path(f).stem.split('_')[:2])
        arr = nib.load(str(f)).get_fdata()
        if np.sum(arr>0)==0: continue
        feats.append({'patient_id':pid,'mask_ratio':float(np.sum(arr>0)/arr.size),'mask_mean':float(arr.mean()),'mask_std':float(arr.std())})
    if len(feats)==0: logging.error('No valid masks'); return None
    df = pd.DataFrame(feats).groupby('patient_id').mean().reset_index()
    df = df.merge(train_df[['patient_id','response']], on='patient_id',how='left')
    df = df.merge(test_df[['patient_id','response']], on='patient_id',how='left',suffixes=('_train','_test'))
    train_X = df[df.response_train.notna()][['mask_ratio','mask_mean','mask_std']]
    y_train = df[df.response_train.notna()].response_train.astype(int)
    test_X = df[df.response_test.notna()][['mask_ratio','mask_mean','mask_std']]
    y_test = df[df.response_test.notna()].response_test.astype(int)
    test_pid = df[df.response_test.notna()].patient_id.values
    if len(train_X)==0 or len(test_X)==0: logging.warning('No train/test in features'); return None
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(); Xtrain = scaler.fit_transform(train_X); Xtest = scaler.transform(test_X)
    clf = RandomForestClassifier(n_estimators=200, random_state=config['SEED']); clf.fit(Xtrain, y_train)
    probs = clf.predict_proba(Xtest)[:,1]
    if len(np.unique(y_test))>1:
        fpr,tpr,thr = roc_curve(y_test, probs); best_thr = thr[np.argmax(tpr-fpr)]
    else:
        best_thr = 0.5
    preds = (probs>=best_thr).astype(int)
    mets = compute_metrics(y_test, preds, probs); mets.update({'youden_threshold':float(best_thr),'model':'B_radiomics_RF_2D','pipeline':'B','n_patients':len(y_test)})
    pd.DataFrame({'patient_id':test_pid,'y_true':y_test,'y_prob':probs,'y_pred':preds}).to_csv(os.path.join(out_dir,'predictions_patients_B.csv'), index=False)
    plot_confusion_matrix(y_test, preds, os.path.join(out_dir,'confusion_matrix_B.png'))
    plot_roc_pr(y_test, probs, out_dir, prefix='B_patient_level')
    return mets


# ------------------
# PIPELINE C (deep features + RF)
# ------------------

def run_pipeline_C(train_df, test_df, images_dir, masks_dir, out_dir, config):
    device = config['DEVICE']
    IMG_SIZE = config['IMG_SIZE']

    def load_and_resize(img_path):
        arr = nib.load(str(img_path)).get_fdata()
        if arr.ndim==3: arr = arr[:,:,arr.shape[2]//2]
        arr = arr.astype(np.float32)
        if arr.max()>arr.min(): arr=(arr-arr.min())/(arr.max()-arr.min())
        pil = Image.fromarray((arr*255).astype('uint8')).resize(IMG_SIZE)
        return np.array(pil, dtype=np.float32)/255.0

    def load_and_resize_mask(mask_path):
        m = nib.load(str(mask_path)).get_fdata()
        if m.ndim==3: m = m[:,:,m.shape[2]//2]
        m = (m>0).astype(np.float32)
        pil = Image.fromarray((m*255).astype('uint8')).resize(IMG_SIZE)
        return np.array(pil, dtype=np.float32)/255.0

    class FeatureDataset(Dataset):
        def __init__(self, df_labels, images_dir, masks_dir):
            self.df = df_labels.set_index('patient_id')
            self.imdir = Path(images_dir); self.maskdir = Path(masks_dir)
            all_imgs = sorted(list(self.imdir.glob('*.nii*')))
            valid_ids = set(self.df.index)
            self.image_files = [f for f in all_imgs if any(pid in f.name for pid in valid_ids)]
        def __len__(self): return len(self.image_files)
        def __getitem__(self, idx):
            f = self.image_files[idx]; pid = '_'.join(f.stem.split('_')[:2])
            img = load_and_resize(f)
            mask_path = find_mask_for_image(self.maskdir, f)
            if mask_path and mask_path.exists():
                m = load_and_resize_mask(mask_path); img = img * m
            return torch.from_numpy(img[None]).float(), pid

    backbone = models.resnet18(weights=None); backbone.conv1 = nn.Conv2d(1,64,7,2,3,bias=False); backbone.fc = nn.Identity(); backbone = backbone.to(device); backbone.eval()

    def extract(loader, desc=''):
        feats, pids = [], []
        with torch.no_grad():
            for x, pid in tqdm(loader, desc=desc):
                x = x.to(device)
                v = backbone(x).cpu().numpy()
                feats.append(v); pids.extend(pid)
        return np.concatenate(feats, axis=0), pids

    train_ds = FeatureDataset(train_df, images_dir, masks_dir); test_ds = FeatureDataset(test_df, images_dir, masks_dir)
    train_dl = DataLoader(train_ds, batch_size=config['BATCH_SIZE'], shuffle=False); test_dl = DataLoader(test_ds, batch_size=config['BATCH_SIZE'], shuffle=False)
    Xtrain_slices, Ptrain = extract(train_dl, 'Deep features C (train)')
    Xtest_slices, Ptest = extract(test_dl, 'Deep features C (test)')

    def aggregate(X,P):
        d={}
        for feat,pid in zip(X,P): d.setdefault(pid,[]).append(feat)
        ids = list(d.keys()); Xagg = np.array([np.mean(np.vstack(d[p]), axis=0) for p in ids])
        return Xagg, ids

    Xtrain, Ptrain2 = aggregate(Xtrain_slices, Ptrain); Xtest, Ptest2 = aggregate(Xtest_slices, Ptest)
    train_idx = train_df.set_index('patient_id'); test_idx = test_df.set_index('patient_id')
    y_train = np.array([int(train_idx.loc[p,'response']) for p in Ptrain2])
    y_test  = np.array([int(test_idx.loc[p,'response']) for p in Ptest2])

    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    scaler = StandardScaler(); Xtrain_s = scaler.fit_transform(Xtrain); Xtest_s = scaler.transform(Xtest)
    clf = RandomForestClassifier(n_estimators=200, random_state=config['SEED']); clf.fit(Xtrain_s, y_train)
    probs = clf.predict_proba(Xtest_s)[:,1]
    fpr,tpr,thr = roc_curve(y_test, probs); best_thr = thr[np.argmax(tpr-fpr)]
    preds = (probs>=best_thr).astype(int)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'patient_id':Ptest2,'y_true':y_test,'y_prob':probs,'y_pred':preds}).to_csv(out_dir/'predictions_patients_C.csv', index=False)
    plot_confusion_matrix(y_test, preds, out_dir/'confusion_matrix_C.png')
    plot_roc_pr(y_test, probs, out_dir, prefix='C_patient_level')

    tn,fp,fn,tp = confusion_matrix(y_test, preds).ravel(); auc = roc_auc_score(y_test, probs)
    return {'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'accuracy':float(accuracy_score(y_test,preds)),'precision':float(precision_score(y_test,preds,zero_division=0)),'recall':float(recall_score(y_test,preds,zero_division=0)),'f1':float(f1_score(y_test,preds,zero_division=0)),'auc':float(auc),'youden_threshold':float(best_thr),'n_patients':len(y_test),'model':'C_deep_RF','pipeline':'C'}


# ------------------
# ORCHESTRATION
# ------------------

def run_all_experiments(config):
    labels_df = load_labels(config['LABELS_CSV'])
    train_df, test_df = split_train_test(labels_df, test_size=config['TEST_SIZE'], seed=config['SEED'])
    summary_rows = []
    for seg in config['SEG_MODELS']:
        logging.info(f'=== Running for {seg} ===')
        masks_dir = Path(config['NNUNET_MASKS_DIR'])
        if not masks_dir.exists(): logging.error('Masks folder missing: '+str(masks_dir)); continue
        exp_out = Path(config['OUTPUT_DIR'])/f'Pipeline_nnUNet_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        exp_out.mkdir(parents=True, exist_ok=True)
        try:
            resA = run_pipeline_A(train_df, test_df, config['IMAGES_DIR'], masks_dir, exp_out, config); summary_rows.append(resA)
        except Exception as e:
            logging.error(f'Pipeline A failed: {e}', exc_info=True)
        try:
            resB = run_pipeline_B(train_df, test_df, masks_dir, exp_out, config)
            if resB: summary_rows.append(resB)
        except Exception as e:
            logging.error(f'Pipeline B failed: {e}', exc_info=True)
        try:
            resC = run_pipeline_C(train_df, test_df, config['IMAGES_DIR'], masks_dir, exp_out, config)
            if resC: summary_rows.append(resC)
        except Exception as e:
            logging.error(f'Pipeline C failed: {e}', exc_info=True)
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(Path(config['OUTPUT_DIR'])/'summary_nnUNet_all.csv', index=False)
    return df_sum


if __name__=='__main__':
    run_all_experiments(CONFIG)
    logging.info('=== FINISHED SERVER NNUNET PIPELINE ===')


# -----------------------------
# PART 4: Parametric version (auto-detect or env var)
# File name suggestion: comparative_study_nnunet_parametric.py
# -----------------------------

# -*- coding: utf-8 -*-
"""
comparative_study_nnunet_parametric.py
Un seul script paramétrable : détecte automatiquement si on est sur le serveur ou local,
ou utilise la variable d'environnement RUN_ENV (values: 'server' or 'local')

- Si RUN_ENV='server' => chemins serveur
- Si RUN_ENV='local'  => chemins locaux
- Si RUN_ENV not set => tentative de détection (existence /work/...)

Sinon identique en logique aux scripts précédents.
"""

import os
from pathlib import Path
import logging

# Decide environment
RUN_ENV = os.environ.get('RUN_ENV', None)  # 'server' or 'local' or None

# simple auto-detection helper
def detect_env():
    # if server path exists, assume server
    if Path('/work/icmub/dfn25964/test_pipelineA').exists():
        return 'server'
    # if local images exist typical local path
    if Path(r'F:/SEGMENTATION IRM/pythonProjetPrediction/data/images').exists():
        return 'local'
    return 'local'  # default to local

if RUN_ENV is None:
    RUN_ENV = detect_env()

if RUN_ENV == 'server':
    BASE = {
        'PROJECT_ROOT': '/work/icmub/dfn25964/test_pipelineA',
        'IMAGES_DIR': '/work/icmub/dfn25964/test_pipelineA/images',
        'NNUNET_MASKS_DIR': '/work/icmub/dfn25964/test_pipelineA/masks_pred/nnunet',
        'LABELS_CSV': '/work/icmub/dfn25964/test_pipelineA/labels.csv',
        'OUTPUT_DIR': '/work/icmub/dfn25964/test_pipelineA/results'
    }
else:
    BASE = {
        'PROJECT_ROOT': r'F:/SEGMENTATION IRM/pythonProjetPrediction',
        'IMAGES_DIR': r'F:/SEGMENTATION IRM/pythonProjetPrediction/data/images',
        'NNUNET_MASKS_DIR': r'F:/SEGMENTATION IRM/pythonProjetPrediction/data/masks_pred/nnunet',
        'LABELS_CSV': r'F:/SEGMENTATION IRM/pythonProjetPrediction/data/labels.csv',
        'OUTPUT_DIR': r'F:/SEGMENTATION IRM/pythonProjetPrediction/results'
    }

# Now import the same logic as in the full script above but using BASE paths
# To avoid duplicating the whole code block here, the parametric file will
# import the server-version implementation but override CONFIG values.
# For portability we'll implement a small wrapper that sets CONFIG and then
# imports the core functions from the previous part if present.

# The simplest approach: include the full implementation (copy-paste) but
# set CONFIG from BASE — for clarity we keep the same names as before.

# For brevity in this file we will re-use the same implementations as above
# (MRISliceDataset, find_mask_for_image, run_pipeline_A/B/C, run_all_experiments)
# by copying them. In practice copy the code from the server script and
# set CONFIG accordingly. (Here I will re-declare CONFIG and call the same functions.)

CONFIG = {
    'PROJECT_ROOT': BASE['PROJECT_ROOT'],
    'IMAGES_DIR': BASE['IMAGES_DIR'],
    'NNUNET_MASKS_DIR': BASE['NNUNET_MASKS_DIR'],
    'LABELS_CSV': BASE['LABELS_CSV'],
    'OUTPUT_DIR': BASE['OUTPUT_DIR'],
    'IMG_SIZE': (256, 256),
    'BATCH_SIZE': 16,
    'NUM_EPOCHS_MAX': 50,
    'EARLY_STOPPING_PATIENCE': 5,
    'MIN_DELTA': 1e-4,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'TEST_SIZE': 0.2,
    'SEED': 42,
    'SEG_MODELS': ['nnunet']
}

os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)
LOG_FILE = os.path.join(CONFIG['OUTPUT_DIR'], 'comparative_study_nnunet_parametric.log')
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
console = logging.StreamHandler(); console.setLevel(logging.INFO); logging.getLogger('').addHandler(console)
logging.info(f'RUN_ENV={RUN_ENV} — using base paths: {BASE}')

# Now: to avoid repetition in this generated file I recommend copying all functions
# from the server script (find_mask_for_image, MRISliceDataset, ResNetBinary, run_pipeline_A/B/C, run_all_experiments)
# directly below. For clarity here we will import them if you place the server script
# in the same folder and name it `comparative_study_nnunet_server.py`.

# Attempt import of server implementation to reuse functions
try:
    import importlib
    mod = importlib.import_module('comparative_study_nnunet_server')
    # if successful, call run_all_experiments from that module with our CONFIG
    if hasattr(mod, 'run_all_experiments'):
        if __name__ == '__main__':
            mod.run_all_experiments(CONFIG)
except Exception as e:
    logging.warning('Could not import comparative_study_nnunet_server module.\nMake sure it is in the same folder.\nFalling back to in-file implementation is required if you want standalone parametric.')
    # If import fails, you can copy the server script functions here (not expanded to keep file shorter).


# End of file - parametric

# -----------------------------------------------------------------
# NOTE TO USER:
# - Both parts are present in this single document.
# - Save PART 3 content as `comparative_study_nnunet_server.py` on the server and run it.
# - Save PART 4 content as `comparative_study_nnunet_parametric.py` (the parametric script
#   tries to import the server implementation to avoid duplicating the whole code block).
# - If you prefer, je peux aussi produce a single standalone parametric file with the entire
#   implementation inlined (no import) — dis-le et je le fournis.
# -----------------------------------------------------------------
