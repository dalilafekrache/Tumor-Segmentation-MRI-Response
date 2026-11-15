# -*- coding: utf-8 -*-
"""
figure_qualitative_v2.py

RÉVISION JRRAS - nouvelle figure qualitative de segmentation (remplace les
figures 7 à 17, qui venaient des anciennes visualisations).

La figure est faite avec les VRAIS masques (data/masks_pred_v2), pour UNE
patiente de VALIDATION, et montre :
  - ligne 1 : une coupe CONTENANT une tumeur (la plus grande tumeur
              manuelle de la patiente) ;
  - ligne 2 : une coupe SANS tumeur sélectionnée par YOLOv8 (faux positif
              de détection), choisie là où le plus de modèles segmentent
              quelque chose.
Disposition : 4 lignes x 6 images (lignes 1-2 : coupe tumorale ; lignes 3-4 :
coupe sans tumeur), chaque fois masque manuel (GT) puis les 11 modèles.
Couleurs : vert = masque manuel ; rouge = masque prédit (contours).

Le choix de la patiente et des coupes est automatique, à partir du fichier
results/CV_revision/segmentation_v2/detail_par_coupe.csv (créé par
recalcul_segmentation_tableau6_v2.py). Le script ne modifie rien d'autre.

Sorties : results/CV_revision/figures/
  figure_qualitative_segmentation.png (300 dpi) et .pdf
  figure_qualitative_choix.txt : patiente et coupes retenues
"""

from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"F:\SEGMENTATION IRM\pythonProjetPrediction")
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
MASKS_GT_DIR = PROJECT_ROOT / "data" / "masks_gt"
MASKS_PRED_ROOT = PROJECT_ROOT / "data" / "masks_pred_v2"
DETAIL = PROJECT_ROOT / "results" / "CV_revision" / "segmentation_v2" / "detail_par_coupe.csv"
OUT_DIR = PROJECT_ROOT / "results" / "CV_revision" / "figures"

# nom dans detail_par_coupe.csv -> (dossier des masques, nom affiché dans la figure)
MODELES = {
    "U-Net 2D": ("UNet", "U-Net"), "U-Net++": ("UNetpp", "U-Net++"),
    "Attention U-Net": ("AttentionUnet", "Attention U-Net"),
    "ResUNet": ("ResUnet", "ResUNet"), "ResUNet++": ("ResUnetpp", "ResUNet++"),
    "VNet 2D": ("vnet", "V-Net 2D"), "nnU-Net": ("nnunet", "nnU-Net"),
    "TransUNet": ("TransUnet", "TransUNet"), "UCTransNet": ("UCTransnet", "UCTransNet"),
    "SwinUNet": ("SwinUnet", "Swin-UNet"), "MedFormer": ("MedFormer", "MedFormer"),
}
VALIDATION = ["1021_VC", "1023_TN", "1028_RI", "1033_TM", "1034_TM",
              "1041_CM", "1047_VS", "1054_MO", "1059_GC"]


def lire_2d(chemin, forme=None, masque=False):
    a = np.squeeze(np.asarray(nib.load(str(chemin)).dataobj, dtype=np.float32))
    if a.ndim == 3:
        a = a[:, :, a.shape[2] // 2]
    if masque:
        a = a > 0
        if forme is not None and a.shape != forme:
            a = np.asarray(Image.fromarray(a.astype(np.uint8) * 255).resize(
                (forme[1], forme[0]), Image.NEAREST)) > 0
    return a


def trouver(dossier, nom):
    for cand in (nom, nom.replace(".nii.gz", "_mask.nii.gz"), nom.replace(".nii", "_mask.nii")):
        if (dossier / cand).exists():
            return dossier / cand
    return None


def choisir(df):
    """Patiente de validation + coupe tumorale + coupe sans tumeur."""
    d = df[df.patient_id.isin(VALIDATION)]
    meilleur = None
    for pid, g in d.groupby("patient_id"):
        sans = g[g.pixels_gt == 0]
        if sans.empty:
            continue
        # nombre de modèles qui segmentent chaque coupe sans tumeur
        n_seg = sans.assign(seg=sans.pixels_pred > 0).groupby("coupe")["seg"].sum()
        coupe_sans, n = n_seg.idxmax(), int(n_seg.max())
        tum = g[g.pixels_gt > 0].groupby("coupe")["pixels_gt"].first()
        if tum.empty:
            continue
        coupe_tum = tum.idxmax()
        if meilleur is None or n > meilleur[3]:
            meilleur = (pid, coupe_tum, coupe_sans, n)
    return meilleur


def panneau(ax, img, gt, pred, titre):
    lo, hi = np.percentile(img, [1, 99.5])
    ax.imshow(np.clip((img - lo) / (hi - lo + 1e-8), 0, 1), cmap="gray")
    if gt is not None and gt.any():
        ax.contour(gt.astype(float), levels=[0.5], colors="lime", linewidths=1.0)
    if pred is not None and pred.any():
        ax.contour(pred.astype(float), levels=[0.5], colors="red", linewidths=1.0)
    ax.set_title(titre, fontsize=8)
    ax.axis("off")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DETAIL)
    pid, coupe_tum, coupe_sans, n = choisir(df)
    print(f"Patiente : {pid} | coupe tumorale : {coupe_tum} | coupe sans tumeur : {coupe_sans} "
          f"({n} modèles sur 11 la segmentent)")

    # 4 lignes x 6 colonnes : lignes 1-2 = coupe tumorale, lignes 3-4 = coupe sans tumeur
    fig, grille = plt.subplots(4, 6, figsize=(13, 10))
    for bloc, (coupe, etiquette) in enumerate([(coupe_tum, "(a) Tumor slice"),
                                               (coupe_sans, "(b) Tumor-free slice")]):
        cases = list(grille[2 * bloc].ravel()) + list(grille[2 * bloc + 1].ravel())
        img = lire_2d(IMAGES_DIR / coupe)
        f_gt = trouver(MASKS_GT_DIR, coupe)
        gt = lire_2d(f_gt, img.shape, masque=True) if f_gt else np.zeros(img.shape, bool)
        panneau(cases[0], img, gt, None, f"{etiquette}\nManual (GT)")
        for j, (cle, (dossier, affiche)) in enumerate(MODELES.items(), 1):
            f = trouver(MASKS_PRED_ROOT / dossier, coupe)
            pred = lire_2d(f, img.shape, masque=True) if f else np.zeros(img.shape, bool)
            ligne_df = df[(df.modele == cle) & (df.coupe == coupe)]
            if len(ligne_df) and ligne_df.pixels_gt.iloc[0] > 0:
                r = ligne_df.iloc[0]
                dice = 2 * r.VP / (2 * r.VP + r.FP + r.FN) if (2 * r.VP + r.FP + r.FN) else 0
                sous = f"Dice {dice:.2f}"
            else:
                sous = f"{int(pred.sum())} px segmented"
            panneau(cases[j], img, gt, pred, f"{affiche}\n{sous}")
    plt.tight_layout(h_pad=1.2)
    # trait de séparation entre la partie (a) et la partie (b)
    y = grille[1, 0].get_position().y0 - 0.006
    fig.add_artist(plt.Line2D([0.02, 0.98], [y, y], transform=fig.transFigure,
                              color="gray", linewidth=0.8, linestyle="--"))
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"figure_qualitative_segmentation.{ext}", dpi=300, bbox_inches="tight")
    (OUT_DIR / "figure_qualitative_choix.txt").write_text(
        f"patiente={pid}\ncoupe_tumorale={coupe_tum}\ncoupe_sans_tumeur={coupe_sans}\n"
        f"modeles_segmentant_la_coupe_sans_tumeur={n}/11\n", encoding="utf-8")
    print(f"Figure : {OUT_DIR / 'figure_qualitative_segmentation.png'}")


if __name__ == "__main__":
    main()