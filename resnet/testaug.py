from pathlib import Path
import random
import numpy as np
from PIL import Image
import albumentations as A
import matplotlib.pyplot as plt
from augmentation import RandomBackgroundSwap, RandomEdgeOverlay

# Pick some backgrounds and set the foreground root
bg_root = Path("data/train_images/background")
fg_root = Path("data/train_images/foreground")
edge_root = Path("data/train_images/edges")

tform = RandomBackgroundSwap(bg_root=bg_root, fg_root=fg_root, p_remove=0.3, p_swap=0.5)
tform_edge = RandomEdgeOverlay(edge_root=edge_root, p=1.0)

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
out = tform_edge.apply(img, image_name=img_path.name)

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
