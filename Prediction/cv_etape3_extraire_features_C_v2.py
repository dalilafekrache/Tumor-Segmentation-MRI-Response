# -*- coding: utf-8 -*-
"""
cv_etape3_extraire_features_C_v2.py

VERSION 2 : utilise les VRAIS masques prédits (data/masks_pred_v2).
Un modèle dont les masques ne sont pas terminés (_TERMINE.txt absent) est
sauté : il suffit de relancer le script plus tard pour l'ajouter.

RÉVISION JRRAS - ÉTAPE 3 : extraction des caractéristiques profondes du
Pipeline C (ResNet-18 + moyenne par patiente) pour les 72 patientes et
les 12 jeux de masques (GT + 11 architectures).

Deux versions du ResNet-18 (sans couche finale, vecteur de 512 valeurs) :
  - "random"   : poids aléatoires (graine fixe), 1 canal en entrée.
                 = ce que faisait le code d'origine (weights=None).
  - "imagenet" : poids pré-entraînés sur ImageNet (IMAGENET1K_V1), image
                 répétée sur 3 canaux + normalisation ImageNet.
                 = le pipeline C "pré-entraîné" décrit dans le manuscrit.
Le réseau n'est PAS entraîné ici : il sert uniquement d'extracteur.

Prétraitement de chaque coupe (comme dans l'article) :
  normalisation min-max, redimensionnement 256 x 256, multiplication par
  le masque tumoral (binaire).
Règle commune à tous les pipelines (correction de la révision) :
  une coupe dont le masque est absent, vide ou < 2 pixels est EXCLUE.

Sorties : results/CV_revision/features_C_v2/
  features_C_<random|imagenet>_<MODELE>.csv : une ligne par patiente
  bilan_extraction_C.csv
Le script peut être relancé : ce qui est déjà extrait est sauté.
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models

# ---------------------------------------------------------------------------
# PARAMÈTRES
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"F:\SEGMENTATION IRM\pythonProjetPrediction")
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
MASKS_GT_DIR = PROJECT_ROOT / "data" / "masks_gt"
MASKS_PRED_ROOT = PROJECT_ROOT / "data" / "masks_pred_v2"
LABELS_CSV = PROJECT_ROOT / "data" / "labels.csv"
OUT_DIR = PROJECT_ROOT / "results" / "CV_revision" / "features_C_v2"

MODELES = {
    "GT": MASKS_GT_DIR,
    "AttentionUnet": MASKS_PRED_ROOT / "AttentionUnet",
    "MedFormer": MASKS_PRED_ROOT / "MedFormer",
    "ResUnet": MASKS_PRED_ROOT / "ResUnet",
    "ResUnetpp": MASKS_PRED_ROOT / "ResUnetpp",
    "SwinUnet": MASKS_PRED_ROOT / "SwinUnet",
    "TransUnet": MASKS_PRED_ROOT / "TransUnet",
    "UCTransnet": MASKS_PRED_ROOT / "UCTransnet",
    "UNet": MASKS_PRED_ROOT / "UNet",
    "UNetpp": MASKS_PRED_ROOT / "UNetpp",
    "nnunet": MASKS_PRED_ROOT / "nnunet",
    "vnet": MASKS_PRED_ROOT / "vnet",
}
BACKBONES = ["random", "imagenet"]
IMG_SIZE = (256, 256)
MIN_ROI = 2          # même règle que le Pipeline B
BATCH = 32
SEED = 42

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(OUT_DIR / "extraction_C.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("etape3")


def patient_id(nom):
    base = nom.replace(".nii.gz", "").replace(".nii", "")
    return "_".join(base.split("_")[:2])


def lire_2d(chemin):
    arr = np.asarray(nib.load(str(chemin)).dataobj, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, arr.shape[2] // 2]
    return arr


def preparer_coupe(f_img, f_mask):
    """Renvoie l'image masquée 256x256 (float 0-1) ou None si coupe exclue."""
    m = lire_2d(f_mask) > 0
    if m.sum() < MIN_ROI:
        return None
    img = lire_2d(f_img)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = img * 0.0
    img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize(
        IMG_SIZE, Image.BILINEAR), dtype=np.float32) / 255.0
    m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(
        IMG_SIZE, Image.NEAREST)) > 0
    return img * m


def creer_backbone(nom):
    torch.manual_seed(SEED)
    if nom == "random":
        net = models.resnet18(weights=None)
        net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    else:
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    return net.eval()


@torch.no_grad()
def encoder(net, nom_backbone, lot):
    x = torch.from_numpy(np.stack(lot)[:, None])          # (N,1,256,256)
    if nom_backbone == "imagenet":
        x = (x.repeat(1, 3, 1, 1) - IMAGENET_MEAN) / IMAGENET_STD
    return net(x).numpy()


def trouver_masque(dossier, nom_img, index_masques):
    for cand in (nom_img,
                 nom_img.replace(".nii.gz", "_mask.nii.gz"),
                 nom_img.replace(".nii", "_mask.nii")):
        if cand in index_masques:
            return index_masques[cand]
    return None


def main():
    log.info("=== ÉTAPE 3 : extraction des features profondes (Pipeline C) ===")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    labels = pd.read_csv(LABELS_CSV)
    p72 = set(labels["patient_id"].astype(str))
    images = sorted(f for f in IMAGES_DIR.glob("*.nii*") if patient_id(f.name) in p72)
    log.info(f"{len(images)} coupes pour {len(p72)} patientes")

    nets = {b: creer_backbone(b) for b in BACKBONES}
    log.info("ResNet-18 chargés (random + imagenet)")

    bilan_path = OUT_DIR / "bilan_extraction_C.csv"
    bilan = pd.read_csv(bilan_path).to_dict("records") if bilan_path.exists() else []

    for nom, dossier in MODELES.items():
        sorties = {b: OUT_DIR / f"features_C_{b}_{nom}.csv" for b in BACKBONES}
        if all(s.exists() for s in sorties.values()):
            log.info(f"[{nom}] déjà extrait -> on passe")
            continue
        if not dossier.exists():
            log.error(f"[{nom}] dossier introuvable : {dossier}")
            continue
        if nom != "GT" and not (dossier / "_TERMINE.txt").exists():
            log.warning(f"[{nom}] masques v2 pas encore terminés -> on passe (relancer plus tard)")
            continue
        t0 = time.time()
        index_masques = {f.name: f for f in dossier.glob("*.nii*")}
        feats = {b: [] for b in BACKBONES}
        pids, lot, lot_pid = [], [], []
        n_exclues = 0

        def vider_lot():
            for b in BACKBONES:
                feats[b].append(encoder(nets[b], b, lot))
            pids.extend(lot_pid)
            lot.clear()
            lot_pid.clear()

        for f_img in images:
            f_mask = trouver_masque(dossier, f_img.name, index_masques)
            x = preparer_coupe(f_img, f_mask) if f_mask is not None else None
            if x is None:
                n_exclues += 1
                continue
            lot.append(x)
            lot_pid.append(patient_id(f_img.name))
            if len(lot) == BATCH:
                vider_lot()
        if lot:
            vider_lot()

        pids_arr = np.array(pids)
        for b in BACKBONES:
            X = np.concatenate(feats[b], axis=0)
            df = pd.DataFrame(X, columns=[f"deep_{i:03d}" for i in range(X.shape[1])])
            df.insert(0, "patient_id", pids_arr)
            agg = df.groupby("patient_id").mean()
            agg.insert(0, "n_slices_valides", df.groupby("patient_id").size())
            agg.reset_index().to_csv(sorties[b], index=False)

        n_pat = len(set(pids))
        manquantes = sorted(p72 - set(pids))
        log.info(f"[{nom}] {len(pids)} coupes utilisées, {n_exclues} exclues, "
                 f"{n_pat} patientes ({time.time() - t0:.0f} s)")
        if manquantes:
            log.warning(f"[{nom}] patientes sans coupe valide : {manquantes}")
        bilan = [r for r in bilan if r["modele"] != nom] + [{
            "modele": nom, "n_coupes_utilisees": len(pids),
            "n_coupes_exclues": n_exclues, "n_patientes": n_pat,
            "patientes_sans_coupe": " ".join(manquantes)}]
        pd.DataFrame(bilan).to_csv(bilan_path, index=False)

    log.info(f"=== Terminé. Bilan : {bilan_path} ===")


if __name__ == "__main__":
    main()