# -*- coding: utf-8 -*-
"""
cv_etape5_pipelineA_et_C_finetune_v2.py

VERSION 2 : utilise les VRAIS masques prédits (data/masks_pred_v2).
  - Les résultats du GT (masques manuels) déjà obtenus dans cv_A sont
    valables : ils sont recopiés automatiquement dans cv_A_v2.
  - Un modèle dont les masques ne sont pas terminés (_TERMINE.txt absent)
    est sauté.
  - Patiente sans aucune coupe valide (cas de nnU-Net : 801_SD, 848_DG) :
      * Pipeline A : probabilité imputée par la médiane des probabilités
        des patientes d'entraînement du pli ;
      * Pipeline C fine-tuné : features imputées par la médiane des
        patientes d'entraînement du pli (comme dans les pipelines B et C).

RÉVISION JRRAS - ÉTAPE 5 : validation croisée (5 plis) du Pipeline A et
du Pipeline C "fine-tuné" (commentaire 5 du reviewer), sur les 72
patientes et les 12 jeux de masques.

À chaque pli et pour chaque jeu de masques :
  1. Un ResNet-18 (initialisé avec les poids ImageNet) est AFFINÉ
     (fine-tuning) pour prédire la pCR coupe par coupe, sur les patientes
     d'entraînement du pli uniquement.
     L'arrêt anticipé (early stopping) se fait sur une partie des
     patientes d'ENTRAÎNEMENT (validation interne, 20 %), jamais sur le
     pli de test.
  2. Pipeline A : probabilité moyenne des coupes -> probabilité patiente.
  3. Pipeline C fine-tuné : les caractéristiques (512) du réseau affiné
     sont moyennées par patiente puis données à un Random Forest
     (200 arbres) entraîné sur les patientes d'entraînement du pli.

Plis : les 5 plis de la 1re répétition de la validation croisée des
étapes 2 et 4 (mêmes patientes dans chaque pli).
Coupes : même règle que partout (masque absent, vide ou < 2 pixels ->
coupe exclue).

Le script est LONG (~25-30 h). Il enregistre chaque pli terminé et peut
être arrêté puis relancé : les plis déjà faits sont sautés.

Sorties : results/CV_revision/cv_A_v2/
  oof_<MODELE>_pli<k>.csv        : prédictions du pli k (A et C fine-tuné)
  journal_entrainement.csv       : époques, pertes, temps par pli
  resultats_cv_A_Cft.csv         : résumé final (calculé quand tout est fini)
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
from scipy.stats import mannwhitneyu, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import shutil
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
OUT_DIR = PROJECT_ROOT / "results" / "CV_revision" / "cv_A_v2"
OLD_GT_DIR = PROJECT_ROOT / "results" / "CV_revision" / "cv_A"   # résultats GT déjà valables

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

SEED = 42
N_SPLITS = 5
N_REPEATS_REF = 10        # pour retrouver exactement les plis des étapes 2 et 4
IMG_SIZE = (256, 256)
MIN_ROI = 2
BATCH = 16
LR = 1e-4
MAX_EPOCHS = 30
PATIENCE = 5
MIN_DELTA = 1e-4
VAL_FRACTION = 0.2        # part des patientes d'entraînement pour l'arrêt anticipé
N_TREES = 200
N_BOOT = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ---------------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(OUT_DIR / "cv_A.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("etape5_v2")


# ---------------------------------------------------------------------------
# DONNÉES
# ---------------------------------------------------------------------------
def patient_id(nom):
    base = nom.replace(".nii.gz", "").replace(".nii", "")
    return "_".join(base.split("_")[:2])


def lire_2d(chemin):
    arr = np.asarray(nib.load(str(chemin)).dataobj, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, arr.shape[2] // 2]
    return arr


def trouver_masque(nom_img, index_masques):
    for cand in (nom_img,
                 nom_img.replace(".nii.gz", "_mask.nii.gz"),
                 nom_img.replace(".nii", "_mask.nii")):
        if cand in index_masques:
            return index_masques[cand]
    return None


def charger_coupes(dossier_masques, p72):
    """Renvoie (X uint8 [N,256,256], liste des patient_id) pour un jeu de masques."""
    index = {f.name: f for f in dossier_masques.glob("*.nii*")}
    X, pids = [], []
    for f_img in sorted(IMAGES_DIR.glob("*.nii*")):
        pid = patient_id(f_img.name)
        if pid not in p72:
            continue
        f_m = trouver_masque(f_img.name, index)
        if f_m is None:
            continue
        m = lire_2d(f_m) > 0
        if m.sum() < MIN_ROI:
            continue
        img = lire_2d(f_img)
        img = (img - img.min()) / (img.max() - img.min()) if img.max() > img.min() else img * 0
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize(IMG_SIZE, Image.BILINEAR))
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(IMG_SIZE, Image.NEAREST)) > 0
        X.append((img * m).astype(np.uint8))
        pids.append(pid)
    return np.stack(X), np.array(pids)


def vers_tenseur(x_uint8):
    x = torch.from_numpy(x_uint8.astype(np.float32) / 255.0)[:, None]
    return (x.repeat(1, 3, 1, 1) - IMAGENET_MEAN) / IMAGENET_STD


# ---------------------------------------------------------------------------
# RÉSEAU
# ---------------------------------------------------------------------------
class ResNetPCR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        self.tete = nn.Linear(512, 1)

    def forward(self, x, renvoyer_features=False):
        f = self.backbone(x)
        out = self.tete(f).squeeze(1)
        return (out, f) if renvoyer_features else out


@torch.no_grad()
def predire(net, X):
    net.eval()
    probs, feats = [], []
    for i in range(0, len(X), 64):
        logit, f = net(vers_tenseur(X[i:i + 64]).to(DEVICE), renvoyer_features=True)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        feats.append(f.cpu().numpy())
    return np.concatenate(probs), np.concatenate(feats)


def entrainer(X_tr, y_tr, X_va, y_va, graine):
    torch.manual_seed(graine)
    np.random.seed(graine)
    net = ResNetPCR().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(graine)
    meilleur, etat, sans_progres, historique = np.inf, None, 0, []
    for ep in range(1, MAX_EPOCHS + 1):
        net.train()
        ordre = rng.permutation(len(X_tr))
        perte_tr = 0.0
        for i in range(0, len(ordre), BATCH):
            idx = ordre[i:i + BATCH]
            xb = vers_tenseur(X_tr[idx]).to(DEVICE)
            yb = torch.from_numpy(y_tr[idx].astype(np.float32)).to(DEVICE)
            opt.zero_grad()
            perte = crit(net(xb), yb)
            perte.backward()
            opt.step()
            perte_tr += perte.item() * len(idx)
        perte_tr /= len(X_tr)
        # perte de validation interne (coupes des patientes de validation)
        net.eval()
        with torch.no_grad():
            pv = []
            for i in range(0, len(X_va), 64):
                pv.append(net(vers_tenseur(X_va[i:i + 64]).to(DEVICE)).cpu())
            perte_va = crit(torch.cat(pv), torch.from_numpy(y_va.astype(np.float32))).item()
        historique.append((ep, perte_tr, perte_va))
        log.info(f"      époque {ep:2d} : perte entraînement {perte_tr:.4f}, "
                 f"validation interne {perte_va:.4f}")
        if perte_va < meilleur - MIN_DELTA:
            meilleur, sans_progres = perte_va, 0
            etat = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            sans_progres += 1
            if sans_progres >= PATIENCE:
                break
    net.load_state_dict(etat)
    return net, historique


def moyenne_par_patiente(valeurs, pids, ordre_patientes):
    df = pd.DataFrame(valeurs if valeurs.ndim == 2 else valeurs[:, None])
    df["pid"] = pids
    m = df.groupby("pid").mean()
    return m.reindex(ordre_patientes).to_numpy()


def seuil_youden(y, p):
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    return float(min(thr[np.argmax(tpr - fpr)], 1.0))


# ---------------------------------------------------------------------------
# UN PLI
# ---------------------------------------------------------------------------
def traiter_pli(nom, k, X, pids, labels_df, tr_pat, te_pat):
    sortie = OUT_DIR / f"oof_{nom}_pli{k}.csv"
    if sortie.exists():
        log.info(f"[{nom}] pli {k} déjà fait -> on passe")
        return
    t0 = time.time()
    lab = labels_df.set_index("patient_id")["response"]
    y_slice = lab.reindex(pids).to_numpy().astype(int)

    # validation interne pour l'arrêt anticipé (patientes d'entraînement)
    y_tr_pat = lab.reindex(tr_pat).to_numpy()
    fit_pat, va_pat = train_test_split(tr_pat, test_size=VAL_FRACTION,
                                       stratify=y_tr_pat, random_state=SEED + k)
    m_fit, m_va = np.isin(pids, fit_pat), np.isin(pids, va_pat)
    m_tr, m_te = np.isin(pids, tr_pat), np.isin(pids, te_pat)
    log.info(f"[{nom}] pli {k} : {m_fit.sum()} coupes d'entraînement, "
             f"{m_va.sum()} de validation interne, {m_te.sum()} de test")

    net, hist = entrainer(X[m_fit], y_slice[m_fit], X[m_va], y_slice[m_va], SEED + k)

    p_tr, f_tr = predire(net, X[m_tr])
    p_va, _ = predire(net, X[m_va])
    p_te, f_te = predire(net, X[m_te])

    # Pipeline A : moyenne des probabilités par patiente
    te_list = [p for p in te_pat]
    tr_list = list(tr_pat)
    pa_tr = moyenne_par_patiente(p_tr, pids[m_tr], tr_list).ravel()
    med_A = float(np.nanmedian(pa_tr))          # pour les patientes sans coupe
    pa_te = moyenne_par_patiente(p_te, pids[m_te], te_list).ravel()
    imputees = [p for p, v in zip(te_list, pa_te) if np.isnan(v)]
    if imputees:
        log.warning(f"[{nom}] pli {k} : patientes de test sans coupe (imputées) : {imputees}")
    pa_te = np.where(np.isnan(pa_te), med_A, pa_te)
    va_list = list(va_pat)
    pa_va = moyenne_par_patiente(p_va, pids[m_va], va_list).ravel()
    pa_va = np.where(np.isnan(pa_va), med_A, pa_va)
    seuil_A = seuil_youden(lab.reindex(va_list).to_numpy(), pa_va)

    # Pipeline C fine-tuné : features moyennes -> imputation -> Random Forest
    F_tr = moyenne_par_patiente(f_tr, pids[m_tr], tr_list)
    F_te = moyenne_par_patiente(f_te, pids[m_te], te_list)
    y_tr = lab.reindex(tr_list).to_numpy().astype(int)
    imp = SimpleImputer(strategy="median").fit(F_tr)
    F_tr, F_te = imp.transform(F_tr), imp.transform(F_te)
    sc = StandardScaler().fit(F_tr)
    rf = RandomForestClassifier(n_estimators=N_TREES, random_state=SEED, n_jobs=-1)
    rf.fit(sc.transform(F_tr), y_tr)
    pc_te = rf.predict_proba(sc.transform(F_te))[:, 1]
    # seuil de Youden du RF par validation croisée interne (entraînement seulement)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    p_in = np.zeros(len(y_tr))
    for a, b in inner.split(F_tr, y_tr):
        s2 = StandardScaler().fit(F_tr[a])
        r2 = RandomForestClassifier(n_estimators=N_TREES, random_state=SEED, n_jobs=-1)
        r2.fit(s2.transform(F_tr[a]), y_tr[a])
        p_in[b] = r2.predict_proba(s2.transform(F_tr[b]))[:, 1]
    seuil_C = seuil_youden(y_tr, p_in)

    pd.DataFrame({
        "patient_id": te_list, "pli": k,
        "pCR": lab.reindex(te_list).to_numpy().astype(int),
        "proba_A": pa_te, "seuil_youden_A": seuil_A,
        "proba_Cft": pc_te, "seuil_youden_Cft": seuil_C,
        "sans_coupe_imputee": [int(p in imputees) for p in te_list],
    }).to_csv(sortie, index=False)

    journal = OUT_DIR / "journal_entrainement.csv"
    lignes = pd.DataFrame([{"modele": nom, "pli": k, "epoque": e, "perte_entrainement": a,
                            "perte_validation_interne": b} for e, a, b in hist])
    lignes["duree_pli_s"] = round(time.time() - t0)
    lignes.to_csv(journal, mode="a", header=not journal.exists(), index=False)
    log.info(f"[{nom}] pli {k} terminé en {(time.time() - t0) / 60:.1f} min "
             f"({len(hist)} époques)")


# ---------------------------------------------------------------------------
# RÉSUMÉ
# ---------------------------------------------------------------------------
def wilson(k, n, z=norm.ppf(0.975)):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    d = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0, c - d), min(1, c + d))


def ic_auc(y, p):
    rng = np.random.default_rng(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = []
    for _ in range(N_BOOT):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(aucs, [2.5, 97.5])


def bloc_metriques(y, pred, prefixe):
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    se_lo, se_hi = wilson(tp, tp + fn); sp_lo, sp_hi = wilson(tn, tn + fp)
    return {f"{prefixe}_tp": tp, f"{prefixe}_tn": tn, f"{prefixe}_fp": fp, f"{prefixe}_fn": fn,
            f"{prefixe}_sensibilite": tp / (tp + fn), f"{prefixe}_sensibilite_IC95": f"[{se_lo:.3f}-{se_hi:.3f}]",
            f"{prefixe}_specificite": tn / (tn + fp), f"{prefixe}_specificite_IC95": f"[{sp_lo:.3f}-{sp_hi:.3f}]",
            f"{prefixe}_exactitude_equilibree": (tp / (tp + fn) + tn / (tn + fp)) / 2}


def resumer():
    lignes = []
    for nom in MODELES:
        fichiers = [OUT_DIR / f"oof_{nom}_pli{k}.csv" for k in range(1, N_SPLITS + 1)]
        if not all(f.exists() for f in fichiers):
            continue
        df = pd.concat(pd.read_csv(f) for f in fichiers)
        y = df["pCR"].to_numpy()
        for pipe, col, cs in [("A", "proba_A", "seuil_youden_A"),
                              ("C_finetune", "proba_Cft", "seuil_youden_Cft")]:
            p = df[col].to_numpy()
            auc = roc_auc_score(y, p)
            lo, hi = ic_auc(y, p)
            # test de Mann-Whitney (AUC = 0,5) sur les prédictions out-of-fold
            pval = mannwhitneyu(p[y == 1], p[y == 0], alternative="two-sided").pvalue
            r = {"pipeline": pipe, "modele": nom, "n_patientes": len(y), "n_pCR": int(y.sum()),
                 "AUC_oof": auc, "AUC_IC95_bootstrap": f"[{lo:.3f}-{hi:.3f}]",
                 "p_MannWhitney_AUC_0.5": pval,
                 "seuil_youden_mediane": df[cs].median()}
            r.update(bloc_metriques(y, (p >= 0.5).astype(int), "s05"))
            r.update(bloc_metriques(y, (p >= df[cs].to_numpy()).astype(int), "youden"))
            lignes.append(r)
    if lignes:
        res = pd.DataFrame(lignes)
        res.to_csv(OUT_DIR / "resultats_cv_A_Cft.csv", index=False)
        pd.set_option("display.width", 200)
        print("\n" + res[["pipeline", "modele", "AUC_oof", "AUC_IC95_bootstrap",
                          "p_MannWhitney_AUC_0.5", "s05_sensibilite", "s05_specificite",
                          "youden_sensibilite", "youden_specificite"]].round(3).to_string(index=False))


# ---------------------------------------------------------------------------
def main():
    log.info(f"=== ÉTAPE 5 v2 : Pipeline A + Pipeline C fine-tuné (appareil : {DEVICE}) ===")
    # recopier les résultats GT déjà valables (masques manuels, inchangés)
    for k in range(1, N_SPLITS + 1):
        src, dst = OLD_GT_DIR / f"oof_GT_pli{k}.csv", OUT_DIR / f"oof_GT_pli{k}.csv"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            log.info(f"[GT] pli {k} recopié depuis cv_A")
    labels = pd.read_csv(LABELS_CSV)
    labels["patient_id"] = labels["patient_id"].astype(str)
    labels = labels.sort_values("patient_id").reset_index(drop=True)   # même ordre que l'étape 2
    p72 = set(labels["patient_id"])
    y = labels["response"].astype(int).to_numpy()
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS_REF, random_state=SEED)
    plis = list(rskf.split(np.zeros(len(y)), y))[:N_SPLITS]           # 1re répétition
    pd.DataFrame([{"pli": k + 1, "patient_id": labels["patient_id"][i]}
                  for k, (_, te) in enumerate(plis) for i in te]).to_csv(OUT_DIR / "plis.csv", index=False)

    for nom, dossier in MODELES.items():
        if all((OUT_DIR / f"oof_{nom}_pli{k}.csv").exists() for k in range(1, N_SPLITS + 1)):
            log.info(f"[{nom}] déjà terminé -> on passe")
            continue
        if nom != "GT" and not (dossier / "_TERMINE.txt").exists():
            log.warning(f"[{nom}] masques v2 pas terminés (_TERMINE.txt absent) -> on passe")
            continue
        t0 = time.time()
        X, pids = charger_coupes(dossier, p72)
        manquantes = sorted(p72 - set(pids))
        if manquantes:
            log.warning(f"[{nom}] patientes sans coupe valide : {manquantes}")
        log.info(f"[{nom}] {len(X)} coupes chargées ({time.time() - t0:.0f} s)")
        for k, (tr, te) in enumerate(plis, 1):
            traiter_pli(nom, k, X, pids, labels,
                        labels["patient_id"].to_numpy()[tr], labels["patient_id"].to_numpy()[te])
        resumer()

    resumer()
    log.info("=== Terminé ===")


if __name__ == "__main__":
    main()