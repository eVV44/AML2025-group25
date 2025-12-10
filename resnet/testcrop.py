from pathlib import Path
import random

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main(img_size=224, num_samples=10, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    root = Path("data/train_images/train_images")
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in root.iterdir() if p.suffix.lower() in exts]
    if not images:
        raise FileNotFoundError(f"No images found in {root}")

    samples = random.sample(images, k=min(num_samples, len(images)))

    rrc = A.RandomResizedCrop(
        size=(img_size, img_size),
        scale=(0.6, 1.0),
        ratio=(0.75, 1.33),
        p=1.0,
    )

    aspect_preserve = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=0,            # constant
            fill=(255, 255, 255),     # pad with white
            fill_mask=0
        ),
        A.RandomCrop(height=img_size, width=img_size)
    ])

    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 4 * len(samples)))
    if len(samples) == 1:
        axes = np.array([axes])

    for i, path in enumerate(samples):
        img = np.array(Image.open(path).convert("RGB"))
        cropped = rrc(image=img)["image"]
        padded = aspect_preserve(image=img)["image"]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original ({path.name})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(cropped)
        axes[i, 1].set_title("RandomResizedCrop")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(padded)
        axes[i, 2].set_title("AspectPreserve+Pad")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("testcrop_debug.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
