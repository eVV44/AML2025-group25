import io
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from rembg import remove, new_session

def main():
    train_dir = Path("data/train_images/train_images")
    image_files = [p for p in train_dir.glob("*") if p.suffix.lower() in {".jpg"}]
    if not image_files:
        raise FileNotFoundError(f"No images found in {train_dir}")

    session = new_session()  # keep the model in memory
    samples = random.sample(image_files, k=min(5, len(image_files)))

    fig, axes = plt.subplots(len(samples), 2, figsize=(10, 3.5 * len(samples)))
    axes = axes if len(samples) > 1 else [axes]  # normalize indexing
    fig.suptitle("Random train samples with background removal", fontsize=14)

    for row_idx, file in enumerate(samples):
        with open(file, "rb") as f_in:
            data = f_in.read()

        removed = remove(data, session=session)
        original = Image.open(io.BytesIO(data)).convert("RGB")
        no_bg = Image.open(io.BytesIO(removed)).convert("RGBA")

        axes[row_idx][0].imshow(original)
        axes[row_idx][0].set_title(f"Original: {file.name}")
        axes[row_idx][0].axis("off")

        axes[row_idx][1].imshow(no_bg)
        axes[row_idx][1].set_title("Background Removed")
        axes[row_idx][1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
