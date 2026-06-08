import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =====================
# Métriques
# =====================
def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def iou_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def precision_score(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    tp = (y_true_f * y_pred_f).sum()
    fp = ((1 - y_true_f) * y_pred_f).sum()
    return tp / (tp + fp + 1e-6)

def recall_score(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    tp = (y_true_f * y_pred_f).sum()
    fn = (y_true_f * (1 - y_pred_f)).sum()
    return tp / (tp + fn + 1e-6)


# =====================
# Sauvegarde overlay prédictions
# =====================
def save_prediction_overlay(image, mask, pred, save_path):
    img = image.squeeze().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Image")
    axs[0].axis("off")

    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Mask réel")
    axs[1].axis("off")

    axs[2].imshow(img, cmap="gray")
    axs[2].imshow(pred, cmap="Reds", alpha=0.5)
    axs[2].set_title("Prédiction (rouge)")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# =====================
# Fonction de test d’un modèle
# =====================
def test_model(model, test_loader, device, model_name, output_dir):
    model.eval()
    patients, slices = [], []
    dice_scores, iou_scores, precision_scores, recall_scores = [], [], [], []

    seg_dir = os.path.join(output_dir, "prediction", model_name)
    os.makedirs(seg_dir, exist_ok=True)

    with torch.no_grad():
        for batch in test_loader:
            images, masks, patient_ids, slice_idxs = batch
            print(f"[{model_name}] Batch shape : {images.shape}")

            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            for b in range(preds.size(0)):
                patient_id = patient_ids[b]
                slice_idx = slice_idxs[b].item()

                save_name = f"{patient_id}_slice{slice_idx}_overlay.png"
                save_path = os.path.join(seg_dir, save_name)
                save_prediction_overlay(images[b], masks[b], preds[b], save_path)

                patients.append(patient_id)
                slices.append(slice_idx)
                dice_scores.append(dice_coef(masks[b], preds[b]).item())
                iou_scores.append(iou_coef(masks[b], preds[b]).item())
                precision_scores.append(precision_score(masks[b], preds[b]).item())
                recall_scores.append(recall_score(masks[b], preds[b]).item())

    metrics_df = pd.DataFrame({
        "Patient": patients,
        "Slice": slices,
        "Dice": dice_scores,
        "IoU": iou_scores,
        "Precision": precision_scores,
        "Recall": recall_scores
    })
    metrics_path = os.path.join(output_dir, model_name, f"metrics_{model_name}.csv")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)

    summary = metrics_df[["Dice", "IoU", "Precision", "Recall"]].mean().to_dict()
    summary.update({k + "_std": metrics_df[k].std() for k in ["Dice", "IoU", "Precision", "Recall"]})
    summary.update({k + "_median": metrics_df[k].median() for k in ["Dice", "IoU", "Precision", "Recall"]})

    return summary, metrics_df


# =====================
# Exécution d’un seul modèle
# =====================
def run_single_model(model_name, model_class, ckpt_file, test_loader, device, base_dir):
    ckpt_dir = os.path.join(base_dir, "checkpoints")
    output_dir = os.path.join(base_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, ckpt_file)
    if not os.path.exists(ckpt_path):
        print(f"⚠️ Checkpoint manquant : {ckpt_path}")
        return None

    try:
        model = model_class().to(device)
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {model_name} : {e}")
        return None

    summary, metrics_df = test_model(model, test_loader, device, model_name, output_dir)

    summary_df = pd.DataFrame([summary], index=[model_name])
    summary_df.to_csv(os.path.join(output_dir, f"summary_{model_name}.csv"))

    print(f"\n✅ {model_name} terminé. Résumé sauvegardé.")
    print(summary_df.round(4))
    return summary


# =====================
# Exemple d'utilisation
# =====================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = r"C:\Users\Admin\PycharmProjects\pythonProjectTest"

    image_dir = os.path.join(base_dir, "dataset", "images")
    mask_dir = os.path.join(base_dir, "dataset", "masks")

    from dataset import MedicalDataset2D
    test_dataset = MedicalDataset2D(image_dir, mask_dir, empty_ratio=0.2, target_size=(224, 224))
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Imports modèles
    from unet_model import UNet
    from resunet_model import ResUNet
    from resunetpp_model import ResUNetPP
    from transunet_model import TransUNet2D
    from attention_unet_model import AttentionUNet2D
    from unetpp_model import UNetPP
    from VNet2d_model import VNet2D
    from uctransnet_model import UCTransNet2D
    from swin_unet_model import SwinUnet
    from medformer_model import MedFormer2D

    # Mapping modèle -> checkpoint + classe
    model_ckpt_names = {
        "unet_model": ("best_model_Unet.pth", UNet),
        "resunet_model": ("best_model_Resunet.pth", ResUNet),
        "resunetpp_model": ("best_model_Resunetpp.pth", ResUNetPP),
        "transunet_model": ("best_model_Transunet.pth", TransUNet2D),
        "attention_unet_model": ("best_model_AttentionUnet.pth", AttentionUNet2D),
        "unetpp_model": ("best_model_Unetpp.pth", UNetPP),
        "VNet2d_model": ("best_model_Vnet.pth", VNet2D),
        "uctransnet_model": ("best_model_Uctransnet.pth", UCTransNet2D),
        "swinunet_model": ("best_model_swinunet.pth", SwinUnet),
        "medformer_model": ("best_model_medformer.pth", MedFormer2D),
    }

    # 👉 Choisis ici le modèle que tu veux exécuter
    run_single_model(
        "unet_model",
        model_ckpt_names["unet_model"][1],
        model_ckpt_names["unet_model"][0],
        test_loader,
        device,
        base_dir
    )

    # Exemple pour lancer SwinUnet à la place :
    # run_single_model(
    #     "swinunet_model",
    #     model_ckpt_names["swinunet_model"][1],
    #     model_ckpt_names["swinunet_model"][0],
    #     test_loader,
    #     device,
    #     base_dir
    # )
