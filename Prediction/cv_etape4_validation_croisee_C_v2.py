# -*- coding: utf-8 -*-
"""
cv_etape4_validation_croisee_C_v2.py

VERSION 2 : caractéristiques extraites avec les VRAIS masques prédits
(results/CV_revision/features_C_v2). Les modèles pas encore extraits sont sautés.

RÉVISION JRRAS - ÉTAPE 4 : validation croisée stratifiée répétée du
Pipeline C (features profondes ResNet-18 + Random Forest) sur les 72
patientes, pour les deux versions du ResNet-18 :
  - "imagenet" : pré-entraîné sur ImageNet (pipeline C corrigé)
  - "random"   : poids aléatoires (comme le code d'origine)

MÉTHODE STRICTEMENT IDENTIQUE à l'étape 2 (Pipeline B) : mêmes plis,
même modèle, mêmes seuils, même test de permutation.

Réponses aux demandes du Reviewer #1 :
  - résultats "out-of-fold" : chaque patiente est prédite par un modèle qui
    ne l'a jamais vue (validation croisée 5 plis, répétée 10 fois) ;
  - plus d'intervalles de confiance dégénérés à 1.000 : IC bootstrap pour
    l'AUC et IC de Wilson pour sensibilité, spécificité, etc. ;
  - test de permutation des labels (hypothèse nulle : pas de lien entre les
    features et la pCR) ;
  - seuils : (a) seuil fixe 0,5, non optimisé ; (b) seuil de Youden choisi
    UNIQUEMENT sur les données d'entraînement de chaque pli (validation
    croisée interne), puis appliqué au pli de test -> pas de biais ;
  - aucune sélection de features : tout ce qui apprend des données
    (imputation par la médiane, standardisation, Random Forest, seuil) est
    refait à l'intérieur de chaque pli.

Entrées : sorties de l'étape 3 (results/CV_revision/features_C/)
Sorties : results/CV_revision/cv_C/
  resultats_cv_C.csv                 : un résumé par (réseau, jeu de masques)
  predictions_oof_<MODELE>.csv       : probabilités out-of-fold par patiente
  permutations_<MODELE>.csv          : AUC sous hypothèse nulle
  details_seuils_<MODELE>.csv        : seuil choisi dans chaque pli
  parametres_cv_C.txt                : paramètres utilisés
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# PARAMÈTRES
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"F:\SEGMENTATION IRM\pythonProjetPrediction")
LABELS_CSV = PROJECT_ROOT / "data" / "labels.csv"
FEATURES_DIR = PROJECT_ROOT / "results" / "CV_revision" / "features_C_v2"
OUT_DIR = PROJECT_ROOT / "results" / "CV_revision" / "cv_C_v2"

# Ordre : la version ImageNet (pipeline C corrigé) d'abord
BACKBONES = ["imagenet", "random"]

MODELES = ["GT", "AttentionUnet", "MedFormer", "ResUnet", "ResUnetpp",
           "SwinUnet", "TransUnet", "UCTransnet", "UNet", "UNetpp",
           "nnunet", "vnet"]

SEED = 42
N_SPLITS = 5          # nombre de plis
N_REPEATS = 10        # nombre de répétitions de la validation croisée
N_INNER = 5           # plis internes pour choisir le seuil de Youden
N_BOOT = 2000         # rééchantillonnages bootstrap pour l'IC de l'AUC
N_PERM = 1000         # permutations des labels
N_TREES = 200         # comme dans l'article
N_JOBS = -1           # -1 = tous les cœurs du processeur

# Colonnes du fichier de features qui ne sont PAS des features radiomiques
COLONNES_NON_FEATURES = {"patient_id", "n_slices_valides"}

# ---------------------------------------------------------------------------
# JOURNAL
# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(OUT_DIR / "cv_C.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("etape4")


# ---------------------------------------------------------------------------
# OUTILS
# ---------------------------------------------------------------------------
def creer_modele():
    """Imputation (médiane) -> standardisation -> Random Forest."""
    return Pipeline([
        ("imputation", SimpleImputer(strategy="median")),
        ("standardisation", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=N_TREES,
                                      random_state=SEED, n_jobs=1)),
    ])


def wilson(k, n, alpha=0.05):
    """Intervalle de confiance de Wilson pour une proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    demi = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - demi), min(1.0, centre + demi))


def auc_bootstrap_ic(y, p, n_boot=N_BOOT, seed=SEED):
    """IC à 95 % de l'AUC par bootstrap stratifié (classes rééchantillonnées
    séparément pour garder 23 pCR / 49 non-pCR)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def seuil_youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    t = thr[np.argmax(j)]
    return float(min(t, 1.0))   # roc_curve peut renvoyer inf pour le 1er seuil


def un_pli(X, y, idx_train, idx_test):
    """Entraîne sur idx_train, prédit idx_test.
    Le seuil de Youden est choisi par validation croisée INTERNE sur
    idx_train uniquement."""
    Xtr, ytr = X[idx_train], y[idx_train]
    # --- seuil de Youden interne (sans jamais voir le pli de test) ---
    inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=SEED)
    p_inner = np.zeros(len(ytr))
    for itr, ite in inner.split(Xtr, ytr):
        m = clone(creer_modele()).fit(Xtr[itr], ytr[itr])
        p_inner[ite] = m.predict_proba(Xtr[ite])[:, 1]
    seuil = seuil_youden(ytr, p_inner)
    # --- modèle final du pli ---
    m = clone(creer_modele()).fit(Xtr, ytr)
    p_test = m.predict_proba(X[idx_test])[:, 1]
    return idx_test, p_test, seuil


def auc_cv_simple(X, y, seed):
    """AUC out-of-fold d'une validation croisée 5 plis (sans répétition)."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    p = np.zeros(len(y))
    for itr, ite in skf.split(X, y):
        m = clone(creer_modele()).fit(X[itr], y[itr])
        p[ite] = m.predict_proba(X[ite])[:, 1]
    return roc_auc_score(y, p)


def metriques_binaires(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                sensibilite=sens, specificite=spec,
                vpp=tp / (tp + fp) if tp + fp else np.nan,
                vpn=tn / (tn + fn) if tn + fn else np.nan,
                exactitude=(tp + tn) / len(y),
                exactitude_equilibree=np.nanmean([sens, spec]),
                f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)


def resumer_seuil(y, preds_par_rep, prefixe):
    """Moyenne des métriques sur les répétitions + IC de Wilson.
    Pour l'IC de Wilson, la proportion moyenne est rapportée au nombre
    réel de patientes concernées (ex. 23 pCR pour la sensibilité)."""
    lignes = [metriques_binaires(y, pr) for pr in preds_par_rep]
    df = pd.DataFrame(lignes)
    moy = df.mean()
    n_pos, n_neg, n = int((y == 1).sum()), int((y == 0).sum()), len(y)
    res = {}
    for nom in ["sensibilite", "specificite", "vpp", "vpn", "exactitude",
                "exactitude_equilibree", "f1"]:
        res[f"{prefixe}_{nom}"] = moy[nom]
    # Wilson
    for nom, denom in [("sensibilite", n_pos), ("specificite", n_neg),
                       ("exactitude", n)]:
        lo, hi = wilson(moy[nom] * denom, denom)
        res[f"{prefixe}_{nom}_IC95"] = f"[{lo:.3f}-{hi:.3f}]"
    for nom in ["tp", "tn", "fp", "fn"]:
        res[f"{prefixe}_{nom}_moyen"] = moy[nom]
    return res


# ---------------------------------------------------------------------------
# ANALYSE D'UN JEU DE MASQUES
# ---------------------------------------------------------------------------
def analyser(nom_complet, labels):
    # nom_complet = "<backbone>_<modele>", ex. "imagenet_GT"
    nom = nom_complet
    f = FEATURES_DIR / f"features_C_{nom}.csv"
    if not f.exists():
        log.error(f"[{nom}] fichier introuvable : {f}")
        return None
    feats = pd.read_csv(f)
    feat_cols = [c for c in feats.columns if c not in COLONNES_NON_FEATURES]

    # On garde TOUJOURS les 72 patientes, dans le même ordre pour tous les
    # modèles (mêmes plis pour tous). Une patiente sans aucune coupe valide
    # garde des features manquantes, imputées par la médiane du pli
    # d'entraînement.
    df = labels.merge(feats, on="patient_id", how="left")
    sans_features = df.loc[df[feat_cols].isna().all(axis=1), "patient_id"].tolist()
    if sans_features:
        log.warning(f"[{nom}] patientes sans features (imputées) : {sans_features}")

    X = df[feat_cols].to_numpy(dtype=float)
    X[~np.isfinite(X)] = np.nan
    y = df["response"].astype(int).to_numpy()
    n = len(y)
    log.info(f"[{nom}] {n} patientes ({y.sum()} pCR), {len(feat_cols)} features")

    # ----- validation croisée répétée -----
    t0 = time.time()
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=SEED)
    plis = list(rskf.split(X, y))
    sorties = Parallel(n_jobs=N_JOBS)(
        delayed(un_pli)(X, y, tr, te) for tr, te in plis)

    probas = np.full((N_REPEATS, n), np.nan)
    pred_youden = np.zeros((N_REPEATS, n), dtype=int)
    seuils = []
    for k, (idx_test, p_test, seuil) in enumerate(sorties):
        r = k // N_SPLITS
        probas[r, idx_test] = p_test
        pred_youden[r, idx_test] = (p_test >= seuil).astype(int)
        seuils.append(dict(repetition=r + 1, pli=k % N_SPLITS + 1, seuil=seuil))

    auc_rep = np.array([roc_auc_score(y, probas[r]) for r in range(N_REPEATS)])
    p_moy = probas.mean(axis=0)
    auc_moy_proba = roc_auc_score(y, p_moy)
    ic_lo, ic_hi = auc_bootstrap_ic(y, p_moy)
    log.info(f"[{nom}] AUC = {auc_rep.mean():.3f} ± {auc_rep.std(ddof=1):.3f} "
             f"(validation croisée en {time.time() - t0:.0f} s)")

    # ----- test de permutation -----
    t0 = time.time()
    auc_obs_simple = auc_cv_simple(X, y, SEED)
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(y) for _ in range(N_PERM)]
    auc_null = np.array(Parallel(n_jobs=N_JOBS)(
        delayed(auc_cv_simple)(X, yp, SEED) for yp in perms))
    p_perm = (1 + np.sum(auc_null >= auc_obs_simple)) / (N_PERM + 1)
    log.info(f"[{nom}] permutation : AUC observée (CV simple) = "
             f"{auc_obs_simple:.3f}, p = {p_perm:.4f} "
             f"({time.time() - t0:.0f} s)")

    # ----- sauvegardes détaillées -----
    oof = pd.DataFrame({"patient_id": df["patient_id"], "pCR": y,
                        "proba_moyenne": p_moy})
    for r in range(N_REPEATS):
        oof[f"proba_rep{r + 1}"] = probas[r]
    oof.to_csv(OUT_DIR / f"predictions_oof_{nom}.csv", index=False)
    pd.DataFrame({"auc_permutee": auc_null}).to_csv(
        OUT_DIR / f"permutations_{nom}.csv", index=False)
    df_seuils = pd.DataFrame(seuils)
    df_seuils.to_csv(OUT_DIR / f"details_seuils_{nom}.csv", index=False)

    # ----- résumé -----
    backbone, modele = nom.split("_", 1)
    res = {
        "reseau": backbone,
        "modele": modele,
        "n_patientes": n,
        "n_pCR": int(y.sum()),
        "n_features": len(feat_cols),
        "patientes_sans_features": " ".join(sans_features),
        "AUC_moyenne_repetitions": auc_rep.mean(),
        "AUC_ecart_type_repetitions": auc_rep.std(ddof=1),
        "AUC_min_repetitions": auc_rep.min(),
        "AUC_max_repetitions": auc_rep.max(),
        "AUC_proba_moyenne": auc_moy_proba,
        "AUC_IC95_bootstrap": f"[{ic_lo:.3f}-{ic_hi:.3f}]",
        "AUC_CV_simple_pour_permutation": auc_obs_simple,
        "AUC_nulle_moyenne": auc_null.mean(),
        "AUC_nulle_percentile95": np.percentile(auc_null, 95),
        "p_permutation": p_perm,
        "seuil_youden_mediane": df_seuils["seuil"].median(),
        "seuil_youden_min": df_seuils["seuil"].min(),
        "seuil_youden_max": df_seuils["seuil"].max(),
    }
    pred05 = [(probas[r] >= 0.5).astype(int) for r in range(N_REPEATS)]
    res.update(resumer_seuil(y, pred05, "s05"))
    res.update(resumer_seuil(y, list(pred_youden), "youden"))
    return res


def main():
    log.info("=== ÉTAPE 4 : validation croisée répétée du Pipeline C ===")
    labels = pd.read_csv(LABELS_CSV)
    labels["patient_id"] = labels["patient_id"].astype(str)
    labels = labels.sort_values("patient_id").reset_index(drop=True)
    log.info(f"{len(labels)} patientes, {labels['response'].sum()} pCR")

    with open(OUT_DIR / "parametres_cv_C.txt", "w", encoding="utf-8") as fo:
        fo.write(f"SEED={SEED}\nN_SPLITS={N_SPLITS}\nN_REPEATS={N_REPEATS}\n"
                 f"N_INNER={N_INNER}\nN_BOOT={N_BOOT}\nN_PERM={N_PERM}\n"
                 f"N_TREES={N_TREES}\n"
                 "Modèle : SimpleImputer(median) -> StandardScaler -> "
                 "RandomForestClassifier, tout ajusté dans chaque pli.\n"
                 "Aucune sélection de features.\n")

    resultats = []
    chemin = OUT_DIR / "resultats_cv_C.csv"
    for backbone in BACKBONES:
        for nom in MODELES:
            res = analyser(f"{backbone}_{nom}", labels)
            if res:
                resultats.append(res)
                pd.DataFrame(resultats).to_csv(chemin, index=False)

    # Affichage court dans la console
    if resultats:
        df = pd.DataFrame(resultats)
        cols = ["reseau", "modele", "AUC_moyenne_repetitions", "AUC_ecart_type_repetitions",
                "AUC_IC95_bootstrap", "p_permutation",
                "s05_sensibilite", "s05_specificite",
                "youden_sensibilite", "youden_specificite"]
        pd.set_option("display.width", 200)
        print("\n" + df[cols].round(3).to_string(index=False))
    log.info(f"=== Terminé. Résultats : {chemin} ===")


if __name__ == "__main__":
    main()