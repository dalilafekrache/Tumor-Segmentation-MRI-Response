import os
import argparse
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import MedicalDataset2D

# Import des modèles
from unet_model import UNet
from unetpp_model import UNetPP
from resunet_model import ResUNet
from resunetpp_model import ResUNetPP
from attention_unet_model import AttentionUNet2D
from swin_unet_model import SwinUnet
from transunet_model import TransUNet2D
from uctransnet_model import UCTransNet2D
from VNet2d_model import VNet2D
from medformer_model import MedFormer2D


def charger_modele(nom_modele, chemin_checkpoint, device, no_checkpoint=False):
    nom = nom_modele.lower()
    img_size = 224  # valeur par défaut

    if nom == "unet":
        modele = UNet(in_channels=1, out_channels=1)
    elif nom == "unetpp":
        modele = UNetPP(in_ch=1, out_ch=1)
    elif nom == "resunet":
        modele = ResUNet(in_channels=1, out_channels=1)
    elif nom == "resunetpp":
        modele = ResUNetPP(in_channels=1, out_channels=1)
    elif nom == "attentionunet":
        modele = AttentionUNet2D(in_ch=1, out_ch=1)
    elif nom == "swinunet":
        modele = SwinUnet(num_classes=1, pretrained=True)
    elif nom == "transunet":
        img_size = 256
        modele = TransUNet2D(img_size=img_size, in_channels=1, out_channels=1)
    elif nom == "uctransnet":
        modele = UCTransNet2D(in_channels=1, out_channels=1)
    elif nom == "vnet":
        modele = VNet2D(in_channels=1, out_channels=1)
    elif nom == "medformer":
        img_size = 224  # ✅ correction ici
        modele = MedFormer2D(in_channels=1, out_channels=1)
    else:
        raise ValueError(f"❌ Modèle inconnu : {nom_modele}")

    # Charger checkpoint si disponible
    if not no_checkpoint and chemin_checkpoint and os.path.exists(chemin_checkpoint):
        checkpoint = torch.load(chemin_checkpoint, map_location=device)

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model_dict = modele.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items()
                         if k in model_dict and v.shape == model_dict[k].shape}

        print(f"✅ Chargement filtré : {len(filtered_dict)} / {len(model_dict)} couches chargées.")
        model_dict.update(filtered_dict)
        modele.load_state_dict(model_dict, strict=False)
    else:
        print("⚠️ Aucun checkpoint chargé → modèle utilisé avec poids aléatoires.")

    modele.to(device)
    modele.eval()
    return modele, img_size


def diagnostic_prediction(modele, dataloader, device):
    """Affiche un diagnostic rapide : image, GT, préd brute et seuillée"""
    modele.eval()
    images, masks, patient_ids, slice_indices = next(iter(dataloader))
    images, masks = images.to(device), masks.to(device)

    with torch.no_grad():
        raw_preds = torch.sigmoid(modele(images))

    # Stats globales
    print(f"🔍 Stats prédictions brutes : min={raw_preds.min().item():.4f}, "
          f"max={raw_preds.max().item():.4f}, mean={raw_preds.mean().item():.4f}")

    preds_seuil = (raw_preds > 0.5).float()

    # Premier exemple
    img = images[0, 0].cpu().numpy()
    mask = masks[0, 0].cpu().numpy()
    pred_raw = raw_preds[0, 0].cpu().numpy()
    pred_bin = preds_seuil[0, 0].cpu().numpy()

    # Figures
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Image")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Masque GT")
    axes[2].imshow(pred_raw, cmap="viridis"); axes[2].set_title("Préd brute (sigmoid)")
    axes[3].imshow(pred_bin, cmap="gray"); axes[3].set_title("Préd seuillée (0.5)")
    plt.show()

    # ✅ Histogramme des prédictions
    plt.figure(figsize=(6, 4))
    plt.hist(pred_raw.ravel(), bins=50, color="blue", alpha=0.7)
    plt.title("Histogramme des valeurs de prédiction (sigmoid)")
    plt.xlabel("Valeur prédite")
    plt.ylabel("Fréquence")
    plt.show()


def predire_et_sauvegarder(modele, dataloader, device, dossier_sortie, nom_modele):
    os.makedirs(dossier_sortie, exist_ok=True)

    with torch.no_grad():
        for images, masques, patient_ids, slice_indices in dataloader:
            images = images.to(device)
            raw_preds = torch.sigmoid(modele(images))

            print(f"[DEBUG] {nom_modele} → min={raw_preds.min().item():.4f}, "
                  f"max={raw_preds.max().item():.4f}, mean={raw_preds.mean().item():.4f}")

            preds = (raw_preds > 0.5).float().cpu().numpy()

            images = images.cpu().numpy()
            masques = masques.cpu().numpy()

            for i in range(len(images)):
                patient_file = patient_ids[i]
                base_name = os.path.splitext(os.path.splitext(patient_file)[0])[0]

                pred_nifti = nib.Nifti1Image(preds[i, 0], np.eye(4))
                nib.save(pred_nifti, os.path.join(dossier_sortie, f"{base_name}_nifti.nii.gz"))

                # Visualisation
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(images[i, 0], cmap="gray"); axes[0].set_title("Image")
                axes[1].imshow(masques[i, 0], cmap="gray"); axes[1].set_title("GT")
                axes[2].imshow(preds[i, 0], cmap="gray"); axes[2].set_title("Prediction")
                plt.tight_layout()
                plt.savefig(os.path.join(dossier_sortie, f"{base_name}.png"))
                plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modele, img_size = charger_modele(args.model, args.checkpoint, device, args.no_checkpoint)

    dataset = MedicalDataset2D(args.images_dir, args.masks_dir, img_size=img_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # ✅ Diagnostic automatique si MedFormer
    if args.model.lower() == "medformer":
        print("=== Diagnostic MedFormer ===")
        diagnostic_prediction(modele, dataloader, device)

    # Lancer les prédictions
    predire_et_sauvegarder(modele, dataloader, device, args.output_dir, args.model)
