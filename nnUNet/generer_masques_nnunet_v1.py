# -*- coding: utf-8 -*-
"""
generer_masques_nnunet_v1.py

RÉVISION JRRAS : génère les VRAIS masques nnU-Net avec nnU-Net VERSION 1
(version avec laquelle le modèle a été entraîné), configuration 2D, fold 0.

À LANCER AVEC L'ENVIRONNEMENT nnU-Net v1 :
  F:\\SEGMENTATION IRM\\venv_nnunet_v1\\Scripts\\python.exe

Mode test (recommandé d'abord) : ajoute  --test  à la fin de la commande.
  -> traite seulement 20 coupes tumorales de patientes de validation et
     affiche le Dice. Rien n'est écrit dans le dossier final.

Procédure (standard nnU-Net v1) : prétraitement défini par les plans du
modèle, prédiction par fenêtre glissante avec miroirs, retour à la géométrie
d'origine de la coupe, masque 0/1 enregistré avec la même forme que l'image.

Poids utilisés : le script essaie, dans cet ordre,
  1) fold_0/model_best.model            (format nnU-Net v1, clé 'state_dict')
  2) fold_0/checkpoint_best.pth         (poids v1, clé 'network_weights')
  3) fold_0/checkpoint_final.pth        (poids v1, clé 'state_dict')
et indique lequel a été chargé.

Sorties : pythonProjetPrediction/data/masks_pred_v2/nnunet/ (mêmes noms de
fichiers que data/images). Le script peut être relancé : les coupes déjà
faites sont sautées.
"""

import sys
import time
import shutil
from pathlib import Path

import numpy as np
import torch

# Les versions récentes de PyTorch refusent par défaut de charger les
# checkpoints nnU-Net v1 (weights_only=True). Ce sont tes propres fichiers :
# on autorise leur chargement complet. (À faire AVANT d'importer nnunet.)
_torch_load = torch.load


def _load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load(*args, **kwargs)


torch.load = _load_compat

import nibabel as nib  # noqa: E402
from nnunet.training.model_restore import load_model_and_checkpoint_files  # noqa: E402
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax  # noqa: E402

# ---------------------------------------------------------------------------
# PARAMÈTRES
# ---------------------------------------------------------------------------
TRAINER_FOLDER = Path(r"F:\SEGMENTATION IRM\pythonProjectNNunet\nnUNet_results"
                      r"\Dataset001_TumorSegmentation\nnUNetTrainer__nnUNetPlans__2d")
FOLD_DIR = TRAINER_FOLDER / "fold_0"

PRED_ROOT = Path(r"F:\SEGMENTATION IRM\pythonProjetPrediction")
IMAGES_DIR = PRED_ROOT / "data" / "images"
MASKS_GT_DIR = PRED_ROOT / "data" / "masks_gt"
OUT_DIR = PRED_ROOT / "data" / "masks_pred_v2" / "nnunet"
TEST_DIR = PRED_ROOT / "results" / "CV_revision" / "segmentation" / "test_nnunet_v1"

VALIDATION = ["1021_VC", "1023_TN", "1028_RI", "1033_TM", "1034_TM",
              "1041_CM", "1047_VS", "1054_MO", "1059_GC"]
MODE_TEST = "--test" in sys.argv
N_TEST = 20


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * np.logical_and(a, b).sum() / s if s > 0 else np.nan


def charger_trainer():
    print(f"Dossier du modèle : {TRAINER_FOLDER}")
    trainer, params = load_model_and_checkpoint_files(
        str(TRAINER_FOLDER), folds=[0], mixed_precision=False, checkpoint_name="model_best")
    print(f"Classe du trainer : {type(trainer).__name__}")

    candidats = [
        ("model_best.model", params[0] if params else None, "state_dict"),
        ("checkpoint_best.pth", None, "network_weights"),
        ("checkpoint_final.pth", None, "state_dict"),
    ]
    for nom, ck, cle in candidats:
        try:
            if ck is None:
                f = FOLD_DIR / nom
                if not f.exists():
                    continue
                ck = torch.load(str(f), map_location="cpu")
            poids = ck[cle] if isinstance(ck, dict) and cle in ck else None
            if poids is None:
                print(f"  {nom} : pas de clé '{cle}', on essaie le suivant")
                continue
            poids = {k.replace("module.", "", 1): v for k, v in poids.items()}
            trainer.network.load_state_dict(poids, strict=True)
            print(f"  Poids chargés depuis : {nom} (époque {ck.get('epoch', '?')})")
            break
        except Exception as e:
            print(f"  {nom} : échec du chargement ({type(e).__name__} : {str(e)[:150]})")
    else:
        raise RuntimeError("Aucun fichier de poids n'a pu être chargé.")

    trainer.network.eval()
    return trainer


def predire_coupe(trainer, f_img, f_out):
    data, _, props = trainer.preprocess_patient([str(f_img)])
    res = trainer.predict_preprocessed_data_return_seg_and_softmax(
        data, do_mirroring=True, mirror_axes=trainer.data_aug_params["mirror_axes"],
        use_sliding_window=True, step_size=0.5, use_gaussian=True,
        all_in_gpu=False, mixed_precision=False)
    softmax = res[1]
    save_segmentation_nifti_from_softmax(softmax, str(f_out), props, 1, None, None, None,
                                         None, None, None, 0)
    # on s'assure d'un masque binaire 0/1 (tumeur = toute classe > 0)
    nii = nib.load(str(f_out))
    m = (np.asarray(nii.dataobj) > 0).astype(np.uint8)
    h = nii.header.copy()
    h.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(m, nii.affine, h), str(f_out))
    return np.squeeze(m) > 0


def main():
    t0 = time.time()
    trainer = charger_trainer()
    images = sorted(IMAGES_DIR.glob("*.nii*"))

    if MODE_TEST:
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        choix = []
        for f in images:
            pid = "_".join(f.name.split("_")[:2])
            if pid in VALIDATION:
                gt = np.squeeze(np.asarray(nib.load(str(MASKS_GT_DIR / f.name)).dataobj)) > 0
                if gt.sum() > 0:
                    choix.append((f, gt))
        choix = choix[::max(1, len(choix) // N_TEST)][:N_TEST]
        print(f"MODE TEST : {len(choix)} coupes tumorales de validation")
        ds = []
        for k, (f, gt) in enumerate(choix, 1):
            t1 = time.time()
            m = predire_coupe(trainer, f, TEST_DIR / f.name)
            d = dice(gt, m) if gt.shape == m.shape else np.nan
            ds.append(d)
            print(f"  {k:2d}. {f.name} : forme {m.shape} (GT {gt.shape}), "
                  f"Dice {d:.3f}, {time.time() - t1:.1f} s")
        print(f"\nDice moyen sur le test : {np.nanmean(ds):.3f}")
        print("Si le Dice est bon (> 0,6), relance SANS --test pour traiter les 2 998 coupes.")
        shutil.rmtree(TEST_DIR, ignore_errors=True)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(images)} coupes à traiter")
    dices_val, n_vides = [], 0
    for k, f_img in enumerate(images, 1):
        f_out = OUT_DIR / f_img.name
        if f_out.exists():
            m = np.squeeze(np.asarray(nib.load(str(f_out)).dataobj)) > 0
        else:
            m = predire_coupe(trainer, f_img, f_out)
        if m.sum() == 0:
            n_vides += 1
        pid = "_".join(f_img.name.split("_")[:2])
        if pid in VALIDATION:
            gt = np.squeeze(np.asarray(nib.load(str(MASKS_GT_DIR / f_img.name)).dataobj)) > 0
            if gt.sum() > 0 and gt.shape == m.shape:
                dices_val.append(dice(gt, m))
        if k % 100 == 0:
            e = time.time() - t0
            print(f"  {k}/{len(images)} coupes ({e / 60:.1f} min, fin estimée dans "
                  f"{e / k * (len(images) - k) / 60:.0f} min)")

    d = np.nanmean(dices_val) if dices_val else float("nan")
    msg = (f"[nnunet] terminé en {(time.time() - t0) / 60:.1f} min | masques vides : "
           f"{n_vides}/{len(images)} | Dice moyen (coupes tumorales, 9 patientes de "
           f"validation) : {d:.3f}")
    print(msg)
    (OUT_DIR / "_TERMINE.txt").write_text(msg, encoding="utf-8")


if __name__ == "__main__":
    main()