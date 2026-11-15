# -*- coding: utf-8 -*-
"""
cv_etape6_propagation_faux_positifs.py

RÉVISION JRRAS - commentaire 7 : analyse de PROPAGATION des faux positifs
de YOLOv8 jusqu'à la prédiction (Pipeline B, radiomique + Random Forest).

Une coupe "faux positif" (FP) = coupe sélectionnée par YOLOv8 dont le masque
manuel est vide (pas de tumeur) : 489 coupes sur 2 998.

Pour chaque modèle de segmentation, le script mesure :
  1. PROPAGATION : combien de coupes FP ont produit un masque non vide ET des
     features radiomiques valides (elles entrent donc dans la moyenne par
     patiente) ; combien de patientes sont touchées ; quelle part des coupes
     valides d'une patiente vient de coupes FP ; combien de patientes
     n'auraient plus aucune coupe valide sans les coupes FP.
  2. EFFET SUR LA PRÉDICTION : la validation croisée du Pipeline B est
     refaite EXACTEMENT comme à l'étape 2 (mêmes plis : 5 plis x 10
     répétitions, graine 42 ; imputation médiane, standardisation, Random
     Forest 200 arbres, tout appris dans le pli), mais en retirant les coupes
     FP avant la moyenne par patiente (scénario "détecteur parfait").
     On compare l'AUC avec et sans coupes FP, répétition par répétition.

Entrées (déjà créées) :
  results/CV_revision/features_v2/features_slices_<MODELE>.csv   (étape 1 v2)
  results/CV_revision/segmentation_v2/detail_par_coupe.csv        (tableau 6 v2)
  data/labels.csv
Le script ne modifie rien. Il est léger (quelques minutes) et n'utilise que
2 cœurs : il peut tourner en même temps que l'étape 5.

Sorties : results/CV_revision/propagation_FP/
  propagation_FP_resume.csv  : une ligne par modèle (le futur tableau 10)
  propagation_FP_par_patiente_<MODELE>.csv
  propagation_FP_auc_par_repetition.csv
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(r"F:\SEGMENTATION IRM\pythonProjetPrediction")
FEATURES_DIR = PROJECT_ROOT / "results" / "CV_revision" / "features_v2"
DETAIL = PROJECT_ROOT / "results" / "CV_revision" / "segmentation_v2" / "detail_par_coupe.csv"
LABELS_CSV = PROJECT_ROOT / "data" / "labels.csv"
OUT_DIR = PROJECT_ROOT / "results" / "CV_revision" / "propagation_FP"

MODELES = ["GT", "AttentionUnet", "MedFormer", "ResUnet", "ResUnetpp",
           "SwinUnet", "TransUnet", "UCTransnet", "UNet", "UNetpp", "nnunet", "vnet"]
SEED, N_SPLITS, N_REPEATS, N_TREES = 42, 5, 10, 200
N_JOBS = 2
NON_FEATURES = {"patient_id", "slice_file", "roi_pixels"}

OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S",
                    handlers=[logging.FileHandler(OUT_DIR / "propagation_FP.log", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("etape6")


def nom_image(nom_masque):
    """Nom du fichier masque -> nom de la coupe image (retire '_mask')."""
    return nom_masque.replace("_mask.nii.gz", ".nii.gz").replace("_mask.nii", ".nii")


def creer_modele():
    return Pipeline([("imputation", SimpleImputer(strategy="median")),
                     ("standardisation", StandardScaler()),
                     ("rf", RandomForestClassifier(n_estimators=N_TREES, random_state=SEED, n_jobs=1))])


def auc_une_repetition(X, y, plis):
    p = np.full(len(y), np.nan)
    for tr, te in plis:
        m = creer_modele().fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p)


def matrice_patients(df_slices, feat_cols, ordre):
    pat = df_slices.groupby("patient_id")[feat_cols].mean()
    return pat.reindex(ordre).to_numpy(dtype=float)


def main():
    labels = pd.read_csv(LABELS_CSV)
    labels["patient_id"] = labels["patient_id"].astype(str)
    labels = labels.sort_values("patient_id").reset_index(drop=True)   # même ordre que l'étape 2
    ordre = labels["patient_id"].tolist()
    y = labels["response"].astype(int).to_numpy()

    detail = pd.read_csv(DETAIL)
    coupes_fp = set(detail.loc[detail.pixels_gt == 0, "coupe"].unique())
    coupes_toutes = detail.drop_duplicates("coupe")
    fp_par_patiente = coupes_toutes[coupes_toutes.pixels_gt == 0].groupby("patient_id").size()
    log.info(f"{len(coupes_fp)} coupes FP (masque manuel vide), "
             f"chez {len(fp_par_patiente)} patientes")

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    tous_plis = list(rskf.split(np.zeros(len(y)), y))
    reps = [tous_plis[r * N_SPLITS:(r + 1) * N_SPLITS] for r in range(N_REPEATS)]

    resume, auc_rows = [], []
    for nom in MODELES:
        f = FEATURES_DIR / f"features_slices_{nom}.csv"
        if not f.exists():
            log.warning(f"[{nom}] {f.name} absent -> on passe")
            continue
        s = pd.read_csv(f)
        s["patient_id"] = s["patient_id"].astype(str)
        s["coupe"] = s["slice_file"].map(nom_image)
        s["fp"] = s["coupe"].isin(coupes_fp)
        feat_cols = [c for c in s.columns if c not in NON_FEATURES | {"coupe", "fp"}]

        # ---------------- 1. propagation -------------------------------------
        par_pat = s.groupby("patient_id").agg(coupes_valides=("fp", "size"),
                                              coupes_valides_FP=("fp", "sum"))
        par_pat = par_pat.reindex(ordre).fillna(0).astype(int)
        par_pat["coupes_FP_selectionnees"] = fp_par_patiente.reindex(ordre).fillna(0).astype(int)
        par_pat["part_FP_parmi_valides"] = np.where(
            par_pat.coupes_valides > 0, par_pat.coupes_valides_FP / par_pat.coupes_valides.replace(0, 1), np.nan)
        par_pat["sans_coupe_si_FP_retirees"] = ((par_pat.coupes_valides > 0) &
                                               (par_pat.coupes_valides == par_pat.coupes_valides_FP))
        par_pat.to_csv(OUT_DIR / f"propagation_FP_par_patiente_{nom}.csv")
        touchees = par_pat[par_pat.coupes_valides_FP > 0]

        # ---------------- 2. effet sur la prédiction ---------------------------
        X_avec = matrice_patients(s, feat_cols, ordre)
        X_sans = matrice_patients(s[~s.fp], feat_cols, ordre)
        auc_avec = np.array(Parallel(n_jobs=N_JOBS)(delayed(auc_une_repetition)(X_avec, y, p) for p in reps))
        auc_sans = np.array(Parallel(n_jobs=N_JOBS)(delayed(auc_une_repetition)(X_sans, y, p) for p in reps))
        delta = auc_sans - auc_avec
        for r in range(N_REPEATS):
            auc_rows.append({"modele": nom, "repetition": r + 1, "AUC_avec_FP": auc_avec[r],
                             "AUC_sans_FP": auc_sans[r], "delta": delta[r]})

        ligne = {
            "modele": nom,
            "coupes_FP_selectionnees": len(coupes_fp),
            "coupes_FP_valides_radiomique": int(s.fp.sum()),
            "pct_coupes_FP_propagees": 100 * s.fp.sum() / len(coupes_fp),
            "patientes_touchees": len(touchees),
            "part_FP_parmi_coupes_valides_mediane_touchees": touchees.part_FP_parmi_valides.median() if len(touchees) else np.nan,
            "part_FP_parmi_coupes_valides_max": touchees.part_FP_parmi_valides.max() if len(touchees) else np.nan,
            "patientes_sans_coupe_si_FP_retirees": int(par_pat.sans_coupe_si_FP_retirees.sum()),
            "AUC_avec_FP_moyenne": auc_avec.mean(), "AUC_avec_FP_ET": auc_avec.std(ddof=1),
            "AUC_sans_FP_moyenne": auc_sans.mean(), "AUC_sans_FP_ET": auc_sans.std(ddof=1),
            "delta_AUC_moyen": delta.mean(), "delta_AUC_min": delta.min(), "delta_AUC_max": delta.max(),
            "repetitions_AUC_baisse": int((delta < 0).sum()),
        }
        resume.append(ligne)
        log.info(f"[{nom}] FP propagées {ligne['coupes_FP_valides_radiomique']}/{len(coupes_fp)} | "
                 f"patientes touchées {ligne['patientes_touchees']} | "
                 f"AUC avec FP {auc_avec.mean():.3f} -> sans FP {auc_sans.mean():.3f} "
                 f"(delta {delta.mean():+.3f}, de {delta.min():+.3f} à {delta.max():+.3f})")

    res = pd.DataFrame(resume)
    res.to_csv(OUT_DIR / "propagation_FP_resume.csv", index=False)
    pd.DataFrame(auc_rows).to_csv(OUT_DIR / "propagation_FP_auc_par_repetition.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n" + res[["modele", "coupes_FP_valides_radiomique", "patientes_touchees",
                      "part_FP_parmi_coupes_valides_mediane_touchees",
                      "patientes_sans_coupe_si_FP_retirees", "AUC_avec_FP_moyenne",
                      "AUC_sans_FP_moyenne", "delta_AUC_moyen", "delta_AUC_min",
                      "delta_AUC_max"]].round(3).to_string(index=False))
    log.info(f"=== Terminé. Résultats : {OUT_DIR} ===")


if __name__ == "__main__":
    main()