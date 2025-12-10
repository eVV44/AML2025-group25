import io
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rembg import remove, new_session


def load_rgba_with_mask(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Run rembg (U^2-Net) to get saliency mask and RGBA cutout."""
    cutout = remove(data, session=session)
    rgba = np.array(Image.open(io.BytesIO(cutout)).convert("RGBA"))
    mask = rgba[:, :, 3]  # alpha channel as mask
    return rgba, mask


def masked_edges(image_rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask_bin = (mask > 0).astype("uint8")
    masked = image_rgb.copy()
    masked[mask_bin == 0] = 0

    gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
    if np.any(mask_bin):
        v = np.median(gray[mask_bin > 0])
    else:
        v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lower, upper)
    edges[mask_bin == 0] = 0
    return masked, edges


def overlay_edges(image_rgb: np.ndarray, edges: np.ndarray, color=(255, 0, 0), alpha=0.8) -> np.ndarray:
    overlay = image_rgb.copy()
    overlay[edges > 0] = color
    return cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)


if __name__ == "__main__":
    data_dir = Path("data/train_images/train_images")
    image_files = [p for p in data_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not image_files:
        raise FileNotFoundError(f"No images found in {data_dir}")

    session = new_session()  # U^2-Net saliency model
    samples = random.sample(image_files, k=min(5, len(image_files)))

    fig, axes = plt.subplots(len(samples), 4, figsize=(14, 3.5 * len(samples)))
    axes = axes if len(samples) > 1 else [axes]
    fig.suptitle("Saliency-guided edge detection (rembg/U2Net mask)", fontsize=14)

    for row_idx, path in enumerate(samples):
        with open(path, "rb") as f_in:
            data = f_in.read()

        original = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        _, mask = load_rgba_with_mask(data)
        masked_img, edges = masked_edges(original, mask)
        overlay = overlay_edges(original, edges)

        axes[row_idx][0].imshow(original)
        axes[row_idx][0].set_title(f"Original: {path.name}")
        axes[row_idx][0].axis("off")

        axes[row_idx][1].imshow(mask, cmap="gray")
        axes[row_idx][1].set_title("Saliency mask (U2Net)")
        axes[row_idx][1].axis("off")

        axes[row_idx][2].imshow(edges, cmap="gray")
        axes[row_idx][2].set_title("Edges (masked)")
        axes[row_idx][2].axis("off")

        axes[row_idx][3].imshow(overlay)
        axes[row_idx][3].set_title("Overlay")
        axes[row_idx][3].axis("off")

    plt.tight_layout()
    plt.show()
