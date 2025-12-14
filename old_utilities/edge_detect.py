import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_image_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def detect_edges(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    v = np.median(gray)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, threshold1=lower, threshold2=upper)
    return edges


def overlay_edges(image_rgb: np.ndarray, edges: np.ndarray, color=(255, 0, 0), alpha=0.8) -> np.ndarray:
    overlay = image_rgb.copy()
    overlay[edges > 0] = color
    return cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)


def main():
    data_dir = Path("data/train_images/train_images")
    image_files = [p for p in data_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not image_files:
        raise FileNotFoundError(f"No images found in {data_dir}")

    samples = random.sample(image_files, k=min(5, len(image_files)))

    fig, axes = plt.subplots(len(samples), 3, figsize=(12, 3.5 * len(samples)))
    axes = axes if len(samples) > 1 else [axes]  # normalize indexing
    fig.suptitle("Edge detection (Canny) on random train samples", fontsize=14)

    for row_idx, path in enumerate(samples):
        img = load_image_rgb(path)
        edges = detect_edges(img)
        overlay = overlay_edges(img, edges)

        axes[row_idx][0].imshow(img)
        axes[row_idx][0].set_title(f"Original: {path.name}")
        axes[row_idx][0].axis("off")

        axes[row_idx][1].imshow(edges, cmap="gray")
        axes[row_idx][1].set_title("Edges")
        axes[row_idx][1].axis("off")

        axes[row_idx][2].imshow(overlay)
        axes[row_idx][2].set_title("Overlay")
        axes[row_idx][2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
