from pathlib import Path
import random
import numpy as np
from PIL import Image
import albumentations as A
import matplotlib.pyplot as plt
from augmentation import RandomBackgroundSwap, RandomEdgeOverlay

def get_crop_box(bbox, img_w: int, img_h: int, margin: float):
    left, top, right, bottom = bbox
    w = max(1, right - left)
    h = max(1, bottom - top)
    side = int(max(w, h) * (1.0 + 2.0 * float(margin)))
    side = max(1, min(side, img_w, img_h))

    cx = (left + right) // 2
    cy = (top + bottom) // 2
    left = max(0, min(cx - side // 2, img_w - side))
    top = max(0, min(cy - side // 2, img_h - side))
    return left, top, left + side, top + side

class CropToForegroundBox(A.ImageOnlyTransform):
    def __init__(self, fg_root, margin=0.1, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.fg_root = Path(fg_root) if fg_root else None
        self.margin = float(margin)

    def apply(self, img, **params):
        name = params.get("image_name")
        if not (self.fg_root and name):
            print("No fg_root or image_name provided.")
            return img

        fg_path = self.fg_root / f"{Path(name).stem}.png"
        if not fg_path.exists():
            print(f"Foreground path does not exist: {fg_path}")
            return img

        try:
            alpha = Image.open(fg_path).convert("RGBA").split()[-1]
            bbox = alpha.point(lambda v: 255 if v > 25 else 0).getbbox()
            print(f"Foreground bbox for {name}: {bbox}")
        except Exception:
            return img
        if bbox is None:
            print("No foreground detected; skipping crop.")
            return img

        img_h, img_w = img.shape[:2]
        left, top, right, bottom = get_crop_box(bbox, img_w, img_h, self.margin)
        plt.imshow(img)
        plt.imshow(img[top:bottom, left:right])
        return img[top:bottom, left:right]

# Pick some backgrounds and set the foreground root
bg_root = Path("data/train_images/background")
fg_root = Path("data/train_images/foreground")
edge_root = Path("data/train_images/edges")

tform = RandomBackgroundSwap(bg_root=bg_root, fg_root=fg_root, p_remove=0.3, p_swap=0.5)
tform_edge = RandomEdgeOverlay(edge_root=edge_root, p=1.0)
tform_crop = CropToForegroundBox(fg_root=fg_root, margin=0.1, p=1)

# Load a sample RGB image
image = random.randint(0, 3000)
img_path = Path(f"data/train_images/train_images/{image}.jpg")
img = np.array(Image.open(img_path).convert("RGB"))

# Load edges for visualization (precomputed)
edges = tform_edge.load_edges(img_path.name, img.shape)
if edges is None:
    edges = np.zeros(img.shape[:2], dtype=np.uint8)

# Apply; Albumentations passes image_name via params
# out = tform.apply(img, image_name=img_path.name)
out = tform_crop.apply(img, image_name=img_path.name)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(edges, cmap="gray")
axes[1].set_title("Edges")
axes[1].axis("off")

axes[2].imshow(out)
axes[2].set_title("Transformed")
axes[2].axis("off")

plt.tight_layout()
plt.show()
