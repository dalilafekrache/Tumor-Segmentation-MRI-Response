# -*- coding: utf-8 -*-
"""
cv_etape1_extraire_features_B_v2.py

VERSION 2 : utilise les VRAIS masques prédits (data/masks_pred_v2).
Un modèle dont les masques ne sont pas terminés (_TERMINE.txt absent) est
sauté : il suffit de relancer le script plus tard pour l'ajouter.

RÉVISION JRRAS - ÉTAPE 1 : extraction des caractéristiques radiomiques
(Pipeline B) pour les 72 patientes, une fois pour chaque jeu de masques :
  - GT (masques manuels)
  - les 11 architectures de segmentation

Les réglages d'extraction sont EXACTEMENT ceux du Pipeline B de l'article :
  PyRadiomics, image "Original" uniquement, 2D (force2D=True,
  force2Ddimension=0), binWidth=25, minimumROIDimensions=1,
  minimumROISize=2, familles firstorder, shape2D, GLCM, GLRLM, GLSZM,
  GLDM, NGTDM ; moyenne des coupes valides pour obtenir une ligne par patiente.

Pourquoi extraire sur les 72 patientes d'un coup ?
  L'extraction et la moyenne par patiente n'utilisent PAS la pCR : elles ne
  peuvent donc pas créer de fuite d'information. Tout ce qui « apprend » à
  partir des données (imputation, standardisation, Random Forest, seuil)
  sera fait plus tard, à l'intérieur de chaque pli de la validation croisée
  (étape 2).

Sorties (dans results/CV_revision/features_v2/) :
  features_slices_<MODELE>.csv    : une ligne par coupe
  features_patients_<MODELE>.csv  : une ligne par patiente (moyenne)
  ignored_<MODELE>.txt            : coupes ignorées et raison
  bilan_extraction.csv            : résumé pour tous les modèles

Le script peut être relancé : un modèle déjà extrait est sauté.
"""

import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PARAMÈTRES
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"F:\SEGMENTATION IRM\pythonProjetPrediction")
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
MASKS_GT_DIR = PROJECT_ROOT / "data" / "masks_gt"
MASKS_PRED_ROOT = PROJECT_ROOT / "data" / "masks_pred_v2"
LABELS_CSV = PROJECT_ROOT / "data" / "labels.csv"
OUT_DIR = PROJECT_ROOT / "results" / "CV_revision" / "features_v2"

# Nom affiché -> dossier des masques
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

# Réglages PyRadiomics (identiques à l'article)
RADIOMICS_SETTINGS = dict(
    binWidth=25,
    force2D=True,
    force2Ddimension=0,
    minimumROIDimensions=1,
    minimumROISize=2,
)
FAMILLES = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape2D"]

# ---------------------------------------------------------------------------
# JOURNAL
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(OUT_DIR / "extraction.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("etape1")


def patient_id_depuis_nom(nom_fichier):
    """'1003_GE_slice058.nii.gz' -> '1003_GE'."""
    base = nom_fichier
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    parts = base.split("_")
    return "_".join(parts[:2])


def nom_image_pour_masque(nom_masque):
    """Liste des noms d'image possibles pour un fichier masque."""
    candidats = [nom_masque]
    for motif in ("_mask", "_seg", "_nifti"):
        if motif in nom_masque:
            candidats.append(nom_masque.replace(motif, ""))
    return candidats


def creer_extracteur():
    from radiomics import featureextractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**RADIOMICS_SETTINGS)
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.disableAllFeatures()
    disponibles = set(getattr(extractor, "featureClassNames", []))
    manquantes = [c for c in FAMILLES if c not in disponibles]
    if manquantes:
        raise RuntimeError(f"Familles PyRadiomics non reconnues : {manquantes}")
    for fam in FAMILLES:
        extractor.enableFeatureClassByName(fam)
    # Moins de messages de PyRadiomics dans la console
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    return extractor


def lire_en_2d(sitk, chemin):
    """Lit une image NIfTI et renvoie un tableau numpy 2D."""
    img = sitk.ReadImage(str(chemin))
    if img.GetDimension() > 2:
        img = sitk.Extract(img, (img.GetWidth(), img.GetHeight(), 0))
    return sitk.GetArrayFromImage(img)


def extraire_un_modele(nom_modele, dossier_masques, patients_72, extractor):
    import SimpleITK as sitk

    images = {p.name: p for p in IMAGES_DIR.glob("*.nii*")}
    masques = sorted(dossier_masques.glob("*.nii*"))
    log.info(f"[{nom_modele}] {len(masques)} masques trouvés dans {dossier_masques}")

    lignes, ignores = [], []
    t0 = time.time()
    for i, f_mask in enumerate(masques, 1):
        pid = patient_id_depuis_nom(f_mask.name)
        if pid not in patients_72:
            ignores.append((f_mask.name, "patiente hors des 72"))
            continue

        f_img = None
        for cand in nom_image_pour_masque(f_mask.name):
            if cand in images:
                f_img = images[cand]
                break
        if f_img is None:
            ignores.append((f_mask.name, "image correspondante introuvable"))
            continue

        try:
            img_arr = lire_en_2d(sitk, f_img).astype(np.float32)
            mask_arr = lire_en_2d(sitk, f_mask)

            if img_arr.shape != mask_arr.shape:
                from skimage.transform import resize
                mask_arr = resize(mask_arr, img_arr.shape, order=0,
                                  preserve_range=True, anti_aliasing=False)
            mask_arr = (mask_arr > 0).astype(np.uint8)

            n_vox = int(mask_arr.sum())
            if n_vox == 0:
                ignores.append((f_mask.name, "masque vide"))
                continue
            if n_vox < RADIOMICS_SETTINGS["minimumROISize"]:
                ignores.append((f_mask.name, f"ROI trop petite ({n_vox})"))
                continue

            img_itk = sitk.GetImageFromArray(img_arr)
            mask_itk = sitk.GetImageFromArray(mask_arr)
            mask_itk.CopyInformation(img_itk)

            feats = extractor.execute(img_itk, mask_itk)
            ligne = {}
            for k, v in feats.items():
                if k.startswith("diagnostics_"):
                    continue
                try:
                    ligne[k] = float(v)
                except (TypeError, ValueError):
                    pass
            if not ligne:
                ignores.append((f_mask.name, "aucune feature produite"))
                continue
            ligne["patient_id"] = pid
            ligne["slice_file"] = f_mask.name
            ligne["roi_pixels"] = n_vox
            lignes.append(ligne)
        except Exception as e:  # on note l'erreur et on continue
            ignores.append((f_mask.name, f"erreur : {e}"))

        if i % 250 == 0:
            log.info(f"[{nom_modele}] {i}/{len(masques)} masques traités "
                     f"({time.time() - t0:.0f} s)")

    # --- sauvegardes ---
    with open(OUT_DIR / f"ignored_{nom_modele}.txt", "w", encoding="utf-8") as fo:
        for nom, raison in ignores:
            fo.write(f"{nom}\t{raison}\n")

    if not lignes:
        log.error(f"[{nom_modele}] aucune coupe valide !")
        return None

    df_slices = pd.DataFrame(lignes)
    feat_cols = [c for c in df_slices.columns
                 if c not in ("patient_id", "slice_file", "roi_pixels")]
    df_slices[feat_cols] = df_slices[feat_cols].replace([np.inf, -np.inf], np.nan)
    df_slices.to_csv(OUT_DIR / f"features_slices_{nom_modele}.csv", index=False)

    # Moyenne par patiente (comme dans l'article)
    df_pat = df_slices.groupby("patient_id")[feat_cols].mean()
    df_pat.insert(0, "n_slices_valides", df_slices.groupby("patient_id").size())
    df_pat = df_pat.reset_index()
    df_pat.to_csv(OUT_DIR / f"features_patients_{nom_modele}.csv", index=False)

    manquantes = sorted(set(patients_72) - set(df_pat["patient_id"]))
    log.info(f"[{nom_modele}] terminé en {time.time() - t0:.0f} s : "
             f"{len(df_slices)} coupes valides, {len(df_pat)} patientes, "
             f"{len(feat_cols)} features, {len(ignores)} coupes ignorées")
    if manquantes:
        log.warning(f"[{nom_modele}] patientes SANS aucune coupe valide : {manquantes}")

    return {
        "modele": nom_modele,
        "n_masques": len(masques),
        "n_coupes_valides": len(df_slices),
        "n_coupes_ignorees": len(ignores),
        "n_patientes": len(df_pat),
        "patientes_sans_coupe": " ".join(manquantes),
        "n_features": len(feat_cols),
    }


def main():
    log.info("=== ÉTAPE 1 : extraction des features radiomiques (Pipeline B) ===")
    for chemin in (IMAGES_DIR, LABELS_CSV):
        if not chemin.exists():
            log.error(f"Introuvable : {chemin}")
            return

    labels = pd.read_csv(LABELS_CSV)
    patients_72 = set(labels["patient_id"].astype(str))
    log.info(f"{len(patients_72)} patientes dans labels.csv")

    extractor = creer_extracteur()

    bilan_path = OUT_DIR / "bilan_extraction.csv"
    bilan = pd.read_csv(bilan_path).to_dict("records") if bilan_path.exists() else []
    deja = {b["modele"] for b in bilan}

    for nom, dossier in MODELES.items():
        sortie = OUT_DIR / f"features_patients_{nom}.csv"
        if sortie.exists() and nom in deja:
            log.info(f"[{nom}] déjà extrait -> on passe")
            continue
        if not dossier.exists():
            log.error(f"[{nom}] dossier de masques introuvable : {dossier}")
            continue
        if nom != "GT" and not (dossier / "_TERMINE.txt").exists():
            log.warning(f"[{nom}] masques v2 pas encore terminés -> on passe (relancer plus tard)")
            continue
        res = extraire_un_modele(nom, dossier, patients_72, extractor)
        if res:
            bilan = [b for b in bilan if b["modele"] != nom] + [res]
            pd.DataFrame(bilan).to_csv(bilan_path, index=False)

    log.info(f"=== Terminé. Bilan : {bilan_path} ===")


if __name__ == "__main__":
    main()