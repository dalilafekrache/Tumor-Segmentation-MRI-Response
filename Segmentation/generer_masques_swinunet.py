# -*- coding: utf-8 -*-
"""
generer_masques_swinunet.py

RÉVISION JRRAS : génère les VRAIS masques de SwinUnet UNIQUEMENT
(entrée 224 x 224, exigée par ce modèle) sur les 2 998 coupes de data/images.
Les 9 autres modèles ont déjà été traités par generer_vrais_masques.py.

Pourquoi : les fichiers de data/masks_pred ne sont pas des masques (ils ont
la taille d'images de visualisation, 407 x 1174). Ce script les remplace par
de vrais masques, SANS toucher aux anciens fichiers.

Pour chaque coupe (même procédure que generate_masks_for_model du code
d'origine) :
  normalisation min-max -> 256 x 256 (224 x 224 pour SwinUnet) -> modèle -> probabilité ->
  redimensionnement à la taille d'origine -> seuil 0,5 -> masque 0/1
  enregistré en NIfTI avec la MÊME forme et la MÊME géométrie que l'image.
Tous les masques sont enregistrés, y compris les masques vides.

Sortie de chaque modèle :
  - si les valeurs sont toutes entre 0 et 1, elles sont considérées comme
    des probabilités (le modèle applique déjà une sigmoïde) ;
  - sinon, ce sont des logits et une sigmoïde est appliquée.
  La règle utilisée est indiquée dans le journal pour chaque modèle.

Contrôle immédiat : à la fin de chaque modèle, le Dice moyen par rapport au
masque manuel est calculé sur les coupes tumorales des 9 patientes de
validation. Il doit être nettement supérieur à 0 (attendu : ~0,6 à 0,9).

Sorties : data/masks_pred_v2/<modèle>/   (mêmes noms de fichiers que data/images)
          results/CV_revision/segmentation/generation_masques_v2.log
Le script peut être relancé : un modèle déjà terminé est sauté.
"""

import sys
import time
import logging
import importlib.util
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# PARAMÈTRES
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"F:\SEGMENTATION IRM\pythonProjetPrediction")
IMAGES_DIR = PROJECT_ROOT / "data" / "images"
MASKS_GT_DIR = PROJECT_ROOT / "data" / "masks_gt"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUT_ROOT = PROJECT_ROOT / "data" / "masks_pred_v2"
LOG_DIR = PROJECT_ROOT / "results" / "CV_revision" / "segmentation"

IMG_SIZE = (256, 256)
# Taille d'entrée particulière pour certains modèles (SwinUnet exige 224 x 224)
TAILLE_PAR_MODELE = {"SwinUnet": (224, 224)}
SEUIL = 0.5

# nom du dossier de sortie (identique aux anciens dossiers) -> fichiers
MODELES = {
    "SwinUnet":      ("swin_unet_model.py",      "SwinUnet",        "best_model_Swinunet.pth"),
}

VALIDATION = ["1021_VC", "1023_TN", "1028_RI", "1033_TM", "1034_TM",
              "1041_CM", "1047_VS", "1054_MO", "1059_GC"]

# ---------------------------------------------------------------------------
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_DIR / "generation_masques_swinunet.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("masques")


def importer_classe(fichier, nom_classe):
    sys.path.insert(0, str(fichier.parent))
    spec = importlib.util.spec_from_file_location(fichier.stem, str(fichier))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, nom_classe)


def charger_modele(fichier_py, classe, fichier_poids):
    Classe = importer_classe(MODELS_DIR / fichier_py, classe)
    modele = Classe()
    etat = torch.load(str(CHECKPOINTS_DIR / fichier_poids), map_location="cpu")
    if isinstance(etat, dict):
        for cle in ("state_dict", "model_state_dict", "model"):
            if cle in etat and isinstance(etat[cle], dict):
                etat = etat[cle]
                break
        etat = {k.replace("module.", "", 1): v for k, v in etat.items()}
        modele.load_state_dict(etat, strict=True)
    else:                      # modèle complet enregistré
        modele = etat
    return modele.eval()


def premier_tenseur(sortie):
    while isinstance(sortie, (list, tuple)):
        sortie = sortie[0]
    if isinstance(sortie, dict):
        sortie = next(iter(sortie.values()))
    return sortie


@torch.no_grad()
def predire_coupe(modele, coupe, taille=IMG_SIZE):
    """coupe : tableau 2D float -> (probabilité, min brut, max brut, règle)."""
    if coupe.max() - coupe.min() > 1e-8:
        norm = (coupe - coupe.min()) / (coupe.max() - coupe.min())
    else:
        norm = coupe * 0.0
    x = np.asarray(Image.fromarray((norm * 255).astype(np.uint8)).resize(
        taille, Image.BILINEAR), dtype=np.float32) / 255.0
    out = premier_tenseur(modele(torch.from_numpy(x)[None, None]))
    out = out.float()
    if out.ndim == 4 and out.shape[1] == 2:          # 2 canaux : fond / tumeur
        prob = torch.softmax(out, dim=1)[0, 1]
        return prob.numpy(), float(out.min()), float(out.max()), "softmax"
    out = out.squeeze()
    mn, mx = float(out.min()), float(out.max())
    if mn >= 0.0 and mx <= 1.0:
        return out.numpy(), mn, mx, "probabilite"
    return torch.sigmoid(out).numpy(), mn, mx, "sigmoide"


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s > 0 else np.nan


def traiter_modele(nom, fichier_py, classe, fichier_poids, images):
    out_dir = OUT_ROOT / nom
    fait = out_dir / "_TERMINE.txt"
    if fait.exists():
        log.info(f"[{nom}] déjà terminé -> on passe")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    modele = charger_modele(fichier_py, classe, fichier_poids)
    log.info(f"[{nom}] modèle et poids chargés")

    regles, dices_val, n_vides = set(), [], 0
    for k, f_img in enumerate(images, 1):
        nii = nib.load(str(f_img))
        arr = np.asarray(nii.dataobj, dtype=np.float32)
        a3 = arr[:, :, None] if arr.ndim == 2 else arr
        masque = np.zeros(a3.shape, dtype=np.uint8)
        for z in range(a3.shape[2]):
            coupe = a3[:, :, z]
            prob, mn, mx, regle = predire_coupe(modele, coupe, TAILLE_PAR_MODELE.get(nom, IMG_SIZE))
            regles.add(regle)
            if k <= 3 and z == 0:
                log.info(f"[{nom}] sortie brute du modèle : min {mn:.3f}, max {mx:.3f} -> {regle}")
            prob_img = Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8))
            prob_orig = np.asarray(prob_img.resize((coupe.shape[1], coupe.shape[0]),
                                                  Image.BILINEAR), dtype=np.float32) / 255.0
            masque[:, :, z] = (prob_orig >= SEUIL).astype(np.uint8)
        masque = masque.reshape(arr.shape)
        if masque.sum() == 0:
            n_vides += 1
        entete = nii.header.copy()
        entete.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(masque, nii.affine, entete), str(out_dir / f_img.name))

        # contrôle sur les patientes de validation
        pid = "_".join(f_img.name.split("_")[:2])
        if pid in VALIDATION:
            gt = np.squeeze(np.asarray(nib.load(str(MASKS_GT_DIR / f_img.name)).dataobj)) > 0
            if gt.sum() > 0:
                dices_val.append(dice(gt, np.squeeze(masque) > 0))
        if k % 500 == 0:
            log.info(f"[{nom}] {k}/{len(images)} coupes ({time.time() - t0:.0f} s)")

    d = np.nanmean(dices_val) if dices_val else np.nan
    msg = (f"[{nom}] terminé en {(time.time() - t0) / 60:.1f} min | règle de sortie : "
           f"{', '.join(sorted(regles))} | masques vides : {n_vides}/{len(images)} | "
           f"Dice moyen (coupes tumorales, 9 patientes de validation) : {d:.3f}")
    log.info(msg)
    if not (d > 0.2):
        log.warning(f"[{nom}] ATTENTION : Dice très bas, les masques sont probablement incorrects !")
    fait.write_text(msg, encoding="utf-8")


def main():
    torch.set_grad_enabled(False)
    images = sorted(IMAGES_DIR.glob("*.nii*"))
    log.info(f"=== Génération des vrais masques : {len(images)} coupes, {len(MODELES)} modèles ===")
    for nom, (fpy, classe, fpoids) in MODELES.items():
        try:
            traiter_modele(nom, fpy, classe, fpoids, images)
        except Exception as e:
            log.error(f"[{nom}] ÉCHEC : {type(e).__name__} : {e}")
    log.info("=== Terminé ===")


if __name__ == "__main__":
    main()