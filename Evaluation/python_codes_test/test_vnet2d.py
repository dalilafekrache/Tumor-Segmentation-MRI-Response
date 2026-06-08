import os
import sys
import torch
import pandas as pd
import numpy as np
import SimpleITK as sitk
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = r"F:\SEGMENTATION IRM\pythonProjectTest"
sys.path.append(PROJECT_ROOT)

from dataset import NiftiSegmentationDataset2D
from modèles.VNet2d_model import VNet2D


def test_vnet2d(model, loader, device, output_dir, image_dir):

    model.eval()

    seg_root = os.path.join(output_dir, "prediction", "vnet2d_model")
    os.makedirs(seg_root, exist_ok=True)

    with torch.no_grad():

        for images, masks, filenames, slice_idxs in tqdm(loader):

            images = images.to(device)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            for b in range(preds.size(0)):

                fname = filenames[b]
                slice_idx = slice_idxs[b].item()

                base_name = fname.replace(".nii.gz","").replace(".nii","")

                original_path = os.path.join(image_dir, fname)
                original_img = sitk.ReadImage(original_path)

                # masque 2D prédiction
                mask_np = preds[b].squeeze().cpu().numpy().astype(np.uint8)

                mask_np = np.expand_dims(mask_np, axis=0)

                mask_itk = sitk.GetImageFromArray(mask_np)

                # resampling vers image originale
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(original_img)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)

                mask_itk = resampler.Execute(mask_itk)

                out_name = f"{base_name}_slice{slice_idx:03d}.nii.gz"

                sitk.WriteImage(
                    mask_itk,
                    os.path.join(seg_root, out_name)
                )


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    image_dir = os.path.join(PROJECT_ROOT,"test","dataset","images")
    mask_dir  = os.path.join(PROJECT_ROOT,"test","dataset","masks")

    dataset = NiftiSegmentationDataset2D(image_dir, mask_dir)

    loader = DataLoader(dataset,batch_size=2,shuffle=False)

    model = VNet2D().to(device)

    ckpt_path = os.path.join(PROJECT_ROOT,"checkpoints","best_model_Vnet.pth")

    model.load_state_dict(
        torch.load(ckpt_path,map_location=device),
        strict=False
    )

    output_dir = os.path.join(PROJECT_ROOT,"results")

    test_vnet2d(model,loader,device,output_dir,image_dir)