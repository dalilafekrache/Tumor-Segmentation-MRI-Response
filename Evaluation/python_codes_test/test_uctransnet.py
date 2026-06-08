import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# =====================
# Dataset
# =====================
from dataset import MedicalDataset2D   # ton dataset.py avec target_size


# =====================
# Métriques
# =====================
def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f, y_pred_f = y_true.view(-1), y_pred.view(-1)
    inter = (y_true_f * y_pred_f).sum()
    return (2. * inter + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def iou_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f, y_pred_f = y_true.view(-1), y_pred.view(-1)
    inter = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - inter
    return (inter + smooth) / (union + smooth)


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
# Sauvegarde overlay
# =====================
def save_prediction_overlay(image, mask, pred, save_path):
    img, mask, pred = image.squeeze().cpu().numpy(), mask.squeeze().cpu().numpy(), pred.squeeze().cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img, cmap="gray"); axs[0].set_title("Image"); axs[0].axis("off")
    axs[1].imshow(mask, cmap="gray"); axs[1].set_title("Mask réel"); axs[1].axis("off")
    axs[2].imshow(img, cmap="gray"); axs[2].imshow(pred, cmap="Reds", alpha=0.5)
    axs[2].set_title("Prédiction (rouge)"); axs[2].axis("off")
    plt.tight_layout(); plt.savefig(save_path, bbox_inches="tight"); plt.close()


# =====================
# Test UCTransNet
# =====================
def test_uctransnet(model, test_loader, device, output_dir):
    model.eval()
    patients, slices = [], []
    dice_scores, iou_scores, precision_scores, recall_scores = [], [], [], []

    seg_dir = os.path.join(output_dir, "prediction", "uctransnet_model")
    os.makedirs(seg_dir, exist_ok=True)

    with torch.no_grad():
        for images, masks, patient_ids, slice_idxs in test_loader:
            print(f"[UCTransNet] Batch shape : {images.shape}")
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            for b in range(preds.size(0)):
                patient_id, slice_idx = patient_ids[b], slice_idxs[b].item()
                save_path = os.path.join(seg_dir, f"{patient_id}_slice{slice_idx}_overlay.png")
                save_prediction_overlay(images[b], masks[b], preds[b], save_path)

                patients.append(patient_id); slices.append(slice_idx)
                dice_scores.append(dice_coef(masks[b], preds[b]).item())
                iou_scores.append(iou_coef(masks[b], preds[b]).item())
                precision_scores.append(precision_score(masks[b], preds[b]).item())
                recall_scores.append(recall_score(masks[b], preds[b]).item())

    # CSV slice-par-slice
    metrics_df = pd.DataFrame({
        "Patient": patients, "Slice": slices,
        "Dice": dice_scores, "IoU": iou_scores,
        "Precision": precision_scores, "Recall": recall_scores
    })
    metrics_df.to_csv(os.path.join(output_dir, "metrics_uctransnet.csv"), index=False)

    # Résumé global
    summary = metrics_df[["Dice", "IoU", "Precision", "Recall"]].mean().to_dict()
    summary.update({k+"_std": metrics_df[k].std() for k in ["Dice","IoU","Precision","Recall"]})
    summary.update({k+"_median": metrics_df[k].median() for k in ["Dice","IoU","Precision","Recall"]})
    pd.DataFrame([summary], index=["uctransnet"]).to_csv(os.path.join(output_dir, "summary_uctransnet.csv"))

    print("\n✅ UCTransNet terminé. Résumé :")
    print(pd.DataFrame([summary], index=["uctransnet"]).round(4))
    return summary


# =====================
# Main
# =====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = r"C:\Users\Admin\PycharmProjects\pythonProjectTest"

    image_dir, mask_dir = os.path.join(base_dir, "dataset", "images"), os.path.join(base_dir, "dataset", "masks")
    test_dataset = MedicalDataset2D(image_dir, mask_dir, empty_ratio=0.2, target_size=(224, 224))
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    from uctransnet_model import UCTransNet2D   # ton modèle UCTransNet

    ckpt_path = os.path.join(base_dir, "checkpoints", "best_model_Uctransnet.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"⚠️ Checkpoint manquant : {ckpt_path}")

    model = UCTransNet2D().to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # strict=False si mismatch léger

    output_dir = os.path.join(base_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    test_uctransnet(model, test_loader, device, output_dir)
