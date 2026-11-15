# -*- coding: utf-8 -*-
"""
recalcul_segmentation_tableau6_v2.py

VERSION 2 : utilise les VRAIS masques prédits (data/masks_pred_v2), générés
le 13/11/2026 avec les modèles entraînés (et nnU-Net v1 pour nnU-Net).

RÉVISION JRRAS - commentaire 3 : recalcul complet des performances de
segmentation (Tableau 6) à partir des masques enregistrés sur le disque,
avec UNE définition claire et identique pour toutes les métriques.

Coupes évaluées : les 2 998 coupes sélectionnées par YOLOv8 (data/images)
pour les 72 patientes. Pour chaque coupe :
  - masque manuel (data/masks_gt), éventuellement vide (faux positif YOLO) ;
  - masque prédit par chaque modèle (data/masks_pred/<modèle>) ; un fichier
    absent = masque prédit vide (les masques vides n'ont pas été enregistrés).

Métriques (définitions écrites dans le rapport) :
  A) PAR PATIENTE (métrique principale) : les pixels VP, FP, FN de toutes les
     coupes de la patiente sont additionnés, puis
       Dice = 2VP/(2VP+FP+FN), IoU = VP/(VP+FP+FN),
       précision = VP/(VP+FP), rappel = VP/(VP+FN).
     -> moyenne, écart-type et médiane sur les patientes.
     Cette définition n'a pas d'ambiguïté avec les coupes vides.
  B) PAR COUPE, uniquement sur les coupes contenant une tumeur manuelle
     (masque manuel non vide) : mêmes formules coupe par coupe.
     Si le modèle ne prédit rien sur une coupe tumorale : Dice = IoU =
     rappel = 0, et précision = 0 (convention indiquée).
  C) COUPES SANS TUMEUR (faux positifs YOLO) : proportion de ces coupes sur
     lesquelles le modèle segmente au moins un pixel (taux de coupes faussement
     segmentées).

Groupes de patientes :
  - validation_9 : les 9 patientes de validation de la segmentation présentes
                   dans la cohorte de prédiction (non utilisées pour
                   l'apprentissage des poids, mais utilisées pour l'arrêt
                   anticipé) ;
  - entrainement_63 : patientes ayant servi à entraîner les modèles ;
  - cohorte_72 : les deux ensemble ;
  - ancien_test_15 : les 15 patientes de l'ancien Tableau 6 (pour comparaison).

Contrôle supplémentaire : nombre de masques prédits ENREGISTRÉS sur les
coupes sans tumeur (pour vérifier comment les masques nnU-Net ont été générés).

Le script ne modifie rien. Sorties : results/CV_revision/segmentation_v2/
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image

# ---------------------------------------------------------------------------
# PARAMÈTRES
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"F:\SEGMENTATION IRM\pythonProjetPrediction")
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
MASKS_GT_DIR = PROJECT_ROOT / "data" / "masks_gt"
MASKS_PRED_ROOT = PROJECT_ROOT / "data" / "masks_pred_v2"
LABELS_CSV = PROJECT_ROOT / "data" / "labels.csv"
OUT_DIR = PROJECT_ROOT / "results" / "CV_revision" / "segmentation_v2"

MODELES = {  # nom affiché dans le tableau -> dossier
    "U-Net 2D": "UNet", "U-Net++": "UNetpp", "Attention U-Net": "AttentionUnet",
    "ResUNet": "ResUnet", "ResUNet++": "ResUnetpp", "VNet 2D": "vnet",
    "nnU-Net": "nnunet", "TransUNet": "TransUnet", "UCTransNet": "UCTransnet",
    "SwinUNet": "SwinUnet", "MedFormer": "MedFormer",
}

VALIDATION = ["1021_VC", "1023_TN", "1028_RI", "1033_TM", "1034_TM",
              "1041_CM", "1047_VS", "1054_MO", "1059_GC"]
ANCIEN_TEST_15 = ["1028_RI", "1059_GC", "784_DC", "817_LC", "829_CS", "841_CC",
                  "853_DM", "881_MM", "906_GS", "954_FT", "975_RN", "824_BL",
                  "861_MM", "945_BV", "966_DM"]


# ---------------------------------------------------------------------------
def patient_id(nom):
    base = nom.replace(".nii.gz", "").replace(".nii", "")
    return "_".join(base.split("_")[:2])


def lire_masque(chemin, forme=None):
    a = np.squeeze(np.asarray(nib.load(str(chemin)).dataobj)) > 0
    if a.ndim == 3:                       # volume d'une seule coupe mal orienté
        a = a[:, :, a.shape[2] // 2]
    if forme is not None and a.shape != forme:
        a = np.asarray(Image.fromarray(a.astype(np.uint8) * 255).resize(
            (forme[1], forme[0]), Image.NEAREST)) > 0
    return a


def ratio(n, d):
    return n / d if d > 0 else np.nan


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    labels = pd.read_csv(LABELS_CSV)
    p72 = set(labels["patient_id"].astype(str))
    images = sorted(f.name for f in IMAGES_DIR.glob("*.nii*") if patient_id(f.name) in p72)
    print(f"{len(images)} coupes, {len(p72)} patientes")

    index_pred = {m: {f.name: f for f in (MASKS_PRED_ROOT / d).glob("*.nii*")}
                  for m, d in MODELES.items()}

    lignes = []          # une ligne par (modèle, coupe)
    for k, nom in enumerate(images, 1):
        gt = lire_masque(MASKS_GT_DIR / nom)
        pid = patient_id(nom)
        for m in MODELES:
            f = index_pred[m].get(nom) or index_pred[m].get(nom.replace(".nii.gz", "_mask.nii.gz"))
            enregistre = f is not None
            pr = lire_masque(f, gt.shape) if enregistre else np.zeros_like(gt)
            vp = int(np.logical_and(gt, pr).sum())
            fp = int(np.logical_and(~gt, pr).sum())
            fn = int(np.logical_and(gt, ~pr).sum())
            lignes.append((m, pid, nom, int(gt.sum()), int(pr.sum()), vp, fp, fn, enregistre))
        if k % 250 == 0:
            print(f"  {k}/{len(images)} coupes lues ({time.time() - t0:.0f} s)")

    df = pd.DataFrame(lignes, columns=["modele", "patient_id", "coupe", "pixels_gt",
                                       "pixels_pred", "VP", "FP", "FN", "masque_enregistre"])
    df.to_csv(OUT_DIR / "detail_par_coupe.csv", index=False)

    # ---------------- contrôle : masques enregistrés sur coupes sans tumeur ----
    sans_tumeur = df[df.pixels_gt == 0]
    ctrl = sans_tumeur.groupby("modele").agg(
        coupes_sans_tumeur=("coupe", "size"),
        masques_enregistres=("masque_enregistre", "sum"),
        masques_non_vides=("pixels_pred", lambda s: int((s > 0).sum())))
    ctrl.to_csv(OUT_DIR / "controle_coupes_sans_tumeur.csv")

    # ---------------- métriques par patiente ----------------------------------
    pp = df.groupby(["modele", "patient_id"])[["VP", "FP", "FN"]].sum().reset_index()
    pp["Dice"] = [ratio(2 * a, 2 * a + b + c) for a, b, c in zip(pp.VP, pp.FP, pp.FN)]
    pp["IoU"] = [ratio(a, a + b + c) for a, b, c in zip(pp.VP, pp.FP, pp.FN)]
    pp["Precision"] = [ratio(a, a + b) if a + b > 0 else 0.0 for a, b in zip(pp.VP, pp.FP)]
    pp["Recall"] = [ratio(a, a + c) for a, c in zip(pp.VP, pp.FN)]
    pp.to_csv(OUT_DIR / "detail_par_patiente.csv", index=False)

    # ---------------- métriques par coupe tumorale ----------------------------
    ct = df[df.pixels_gt > 0].copy()
    ct["Dice"] = 2 * ct.VP / (2 * ct.VP + ct.FP + ct.FN)
    ct["IoU"] = ct.VP / (ct.VP + ct.FP + ct.FN)
    ct["Precision"] = np.where(ct.VP + ct.FP > 0, ct.VP / (ct.VP + ct.FP).replace(0, 1), 0.0)
    ct["Recall"] = ct.VP / (ct.VP + ct.FN)

    groupes = {
        "validation_9": set(VALIDATION),
        "entrainement_63": p72 - set(VALIDATION),
        "cohorte_72": p72,
        "ancien_test_15": set(ANCIEN_TEST_15),
    }
    for g, pats in groupes.items():
        res = []
        for m in MODELES:
            a = pp[(pp.modele == m) & pp.patient_id.isin(pats)]
            b = ct[(ct.modele == m) & ct.patient_id.isin(pats)]
            c = df[(df.modele == m) & df.patient_id.isin(pats) & (df.pixels_gt == 0)]
            r = {"modele": m, "n_patientes": a.patient_id.nunique(),
                 "n_coupes_tumorales": len(b), "n_coupes_sans_tumeur": len(c)}
            for met in ["Dice", "IoU", "Precision", "Recall"]:
                r[f"{met}_patiente_moyenne"] = a[met].mean()
                r[f"{met}_patiente_ecart_type"] = a[met].std(ddof=1)
                r[f"{met}_patiente_mediane"] = a[met].median()
            for met in ["Dice", "IoU", "Precision", "Recall"]:
                r[f"{met}_coupe_moyenne"] = b[met].mean()
                r[f"{met}_coupe_mediane"] = b[met].median()
            r["coupes_tumorales_non_segmentees"] = int((b.pixels_pred == 0).sum())
            r["taux_coupes_sans_tumeur_segmentees"] = (c.pixels_pred > 0).mean() if len(c) else np.nan
            res.append(r)
        out = pd.DataFrame(res)
        out.to_csv(OUT_DIR / f"tableau6_recalcule_{g}.csv", index=False)
        # contrôles de cohérence (doivent toujours être vrais)
        ok = (out.IoU_patiente_moyenne <= out.Dice_patiente_moyenne + 1e-9).all()
        print(f"\n=== {g} ({len(pats)} patientes) — contrôle IoU <= Dice : {'OK' if ok else 'PROBLÈME'} ===")
        cols = ["modele", "Dice_patiente_moyenne", "Dice_patiente_ecart_type",
                "IoU_patiente_moyenne", "Precision_patiente_moyenne",
                "Recall_patiente_moyenne", "Dice_coupe_moyenne",
                "taux_coupes_sans_tumeur_segmentees"]
        pd.set_option("display.width", 220)
        print(out[cols].round(3).to_string(index=False))

    print("\n=== Contrôle : masques enregistrés sur les coupes SANS tumeur ===")
    print(ctrl.to_string())
    print(f"\nTerminé en {time.time() - t0:.0f} s. Résultats : {OUT_DIR}")


if __name__ == "__main__":
    main()