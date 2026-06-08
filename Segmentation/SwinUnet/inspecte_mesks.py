import matplotlib.pyplot as plt
from dataset_swinunet import NiftiSegmentationDataset

dataset = NiftiSegmentationDataset("dataset/images", "dataset/masks", transform_size=(256, 256))

for i in range(10):
    img, mask = dataset[i]
    print(f"[Sample {i}] mask.unique() = {mask.unique()} | mask.sum() = {mask.sum()}")

    plt.subplot(1, 2, 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title("Image")

    plt.subplot(1, 2, 2)
    plt.imshow(mask.squeeze(), cmap='gray')
    plt.title("Mask")

    plt.tight_layout()
    plt.show()
