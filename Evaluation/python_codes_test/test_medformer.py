import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# =====================
# Métriques
# =====================
def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f, y_pred_f = y_true.view(-1), y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def iou_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f, y_pred_f = y_true.view(-1), y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def precision_score(y_true, y_pred):
    y_true_f, y_pred_f = y_true.view(-1), y_pred.view(-1)
    tp = (y_true_f * y_pred_f).sum()
    fp = ((1 - y_true_f) * y_pred_f).sum()
    return tp / (tp + fp + 1e-6)


def recall_score(y_true, y_pred):
    y_true_f, y_pred_f = y_true.view(-1), y_pred.view(-1)
    tp = (y_true_f * y_pred_f).sum()
    fn = (y_true_f * (1 - y_pred_f)).sum()
    return tp / (tp + fn + 1e-6)


# =====================
# Chargement sécurisé du checkpoint
# =====================
def load_checkpoint_safe(model, ckpt_path, device):
    print(f"📥 Chargement du checkpoint : {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_state[nk] = v

    res = model.load_state_dict(new_state, strict=False)
    print("🔧 Résultat load_state_dict (missing/unexpected keys) :")
    print(res)
    return model


# =====================
# Overlay style TransUNet
# =====================
def save_prediction_overlay(image, mask, pred, save_path):
    img = image.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img, cmap="gray"); axs[0].set_title("Image"); axs[0].axis("off")
    axs[1].imshow(mask, cmap="gray"); axs[1].set_title("Mask réel"); axs[1].axis("off")
    axs[2].imshow(img, cmap="gray"); axs[2].imshow(pred, cmap="Reds", alpha=0.5)
    axs[2].set_title("Prédiction (rouge)"); axs[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# =====================
# Fonction de test MedFormer
# =====================
def test_medformer(model, test_loader, device, output_dir):
    model.eval()

    # === Dossier des prédictions ===
    pred_dir = os.path.join(output_dir, "prediction", "medformer_model")
    os.makedirs(pred_dir, exist_ok=True)

    patients, slices = [], []
    dice_scores, iou_scores, precision_scores, recall_scores = [], [], [], []
    thresholds = [0.3, 0.5, 0.7]
    thr_records = {thr: {"Patient": [], "Slice": [], "Dice": []} for thr in thresholds}

    print(f"🧩 Nombre total d’images de test : {len(test_loader.dataset)}")

    with torch.no_grad():
        for batch_idx, (images, masks, patient_ids, slice_idxs) in enumerate(test_loader):
            print(f"\n[MedFormer] Batch {batch_idx} — shape: {images.shape}")
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            print(f"  logits: min {outputs.min().item():.4f}, max {outputs.max().item():.4f}, mean {outputs.mean().item():.4f}")
            print(f"  probs : min {probs.min().item():.4f}, max {probs.max().item():.4f}, mean {probs.mean().item():.4f}")

            for b in range(probs.size(0)):
                patient_id = patient_ids[b]
                slice_idx = slice_idxs[b].item()
                mask_b = masks[b]
                pred_bin = preds[b]

                # Sauvegarde overlay
                save_path = os.path.join(pred_dir, f"{patient_id}_slice{slice_idx}_overlay.png")
                save_prediction_overlay(images[b], mask_b, pred_bin, save_path)

                # Calcul métriques
                dice_val = dice_coef(mask_b, pred_bin).item()
                iou_val = iou_coef(mask_b, pred_bin).item()
                prec_val = precision_score(mask_b, pred_bin).item()
                rec_val = recall_score(mask_b, pred_bin).item()

                patients.append(patient_id)
                slices.append(slice_idx)
                dice_scores.append(dice_val)
                iou_scores.append(iou_val)
                precision_scores.append(prec_val)
                recall_scores.append(rec_val)

                for thr in thresholds:
                    thr_bin = (probs[b] > thr).float()
                    thr_dice = dice_coef(mask_b, thr_bin).item()
                    thr_records[thr]["Patient"].append(patient_id)
                    thr_records[thr]["Slice"].append(slice_idx)
                    thr_records[thr]["Dice"].append(thr_dice)

    # === Sauvegarde CSV directement dans "results" ===
    metrics_path = os.path.join(output_dir, "metrics_medformer.csv")
    summary_path = os.path.join(output_dir, "summary_medformer.csv")

    metrics_df = pd.DataFrame({
        "Patient": patients,
        "Slice": slices,
        "Dice": dice_scores,
        "IoU": iou_scores,
        "Precision": precision_scores,
        "Recall": recall_scores
    })
    metrics_df.to_csv(metrics_path, index=False)

    # Résumé global
    summary = metrics_df[["Dice", "IoU", "Precision", "Recall"]].mean().to_dict()
    summary.update({k + "_std": metrics_df[k].std() for k in ["Dice", "IoU", "Precision", "Recall"]})
    summary.update({k + "_median": metrics_df[k].median() for k in ["Dice", "IoU", "Precision", "Recall"]})
    pd.DataFrame([summary], index=["medformer_model"]).to_csv(summary_path)

    # Sauvegarde des thresholds
    thr_dir = os.path.join(output_dir, "thresholds_medformer")
    os.makedirs(thr_dir, exist_ok=True)
    for thr, rec in thr_records.items():
        df_thr = pd.DataFrame(rec)
        df_thr.to_csv(os.path.join(thr_dir, f"threshold_{int(thr*100)}.csv"), index=False)
        print(f"👉 Threshold {thr}: mean dice = {df_thr['Dice'].mean():.4f}")

    print("\n✅ MedFormer terminé. Résumé global :")
    print(pd.DataFrame([summary], index=["medformer_model"]).round(4))
    print(f"📊 Fichier des métriques : {metrics_path}")
    print(f"📈 Fichier du résumé     : {summary_path}")
    print(f"🖼️  Prédictions sauvegardées dans : {pred_dir}")

    return summary


# =====================
# Main
# =====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = r"C:\Users\Admin\PycharmProjects\pythonProjectTest"

    image_dir = os.path.join(base_dir, "dataset", "images")
    mask_dir = os.path.join(base_dir, "dataset", "masks")

    from dataset import MedicalDataset2D
    test_dataset = MedicalDataset2D(image_dir, mask_dir, empty_ratio=0.2, target_size=(224, 224))
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    from medformer_model import MedFormer2D
    ckpt_path = os.path.join(base_dir, "checkpoints", "best_model_MedFormer.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"⚠️ Checkpoint manquant : {ckpt_path}")

    model = MedFormer2D().to(device)
    model = load_checkpoint_safe(model, ckpt_path, device)

    output_dir = os.path.join(base_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    test_medformer(model, test_loader, device, output_dir)
