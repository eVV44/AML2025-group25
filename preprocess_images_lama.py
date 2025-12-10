import io
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove
from tqdm import tqdm

try:
    from lama_cleaner.model_manager import ModelManager 
    from lama_cleaner.schema import Config 
except ImportError as e:
    raise ImportError("LaMa runtime not installed. Please install lama-cleaner.") from e


def load_lama_inpainter(device: str = "cuda"):
    config = Config(
        ldm_steps=20,
        hd_strategy="Crop",
        prompt="",
        negative_prompt="",
        num_samples=1,
        image_resolution=512,
        hd_strategy_crop_margin=128,
        hd_strategy_crop_trigger_size=512,
        hd_strategy_resize_limit=1024,
    )

    manager = ModelManager(name="lama", device=device)

    def inpaint_fn(image, mask):
        return manager(image, mask, config)

    return inpaint_fn

def inpaint_background_lama(image_rgb: np.ndarray, mask: np.ndarray, lama_model) -> np.ndarray:
    mask_255 = ((mask > 0).astype(np.uint8) * 255).copy()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result = lama_model(image_bgr, mask_255) 

    if result is not None and result.ndim == 3 and result.shape[2] == 3:
        try: result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        except Exception: pass
    return result

def get_inpainted_background(data: bytes, mask: np.ndarray, lama_model) -> Image.Image:
    # Background: inpaint the bird region to fill with plausible content
    if lama_model is None:
        raise RuntimeError("LaMa model is not loaded; please provide lama_model.")
    
    rgb_image_data = Image.open(io.BytesIO(data)).convert("RGB")
    full_image = np.array(rgb_image_data)

    bg_filled = inpaint_background_lama(full_image, mask, lama_model)
    if isinstance(bg_filled, np.ndarray) and bg_filled.dtype != np.uint8:
        bg_filled = np.clip(bg_filled, 0, 255).astype(np.uint8)
    bg_np = np.dstack([bg_filled, np.full_like(mask, 255)]) 
    return Image.fromarray(bg_np, mode="RGBA")

def get_edge_map(mask: np.ndarray) -> Image.Image:
    mask_blur = cv2.GaussianBlur(mask, (3, 3), 0)
    v = np.median(mask_blur)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(mask_blur, lower, upper)
    edges_img = Image.fromarray(edges)
    return edges_img

def process_image_bytes(data: bytes, session=None, lama_model=None) -> Dict[str, Image.Image]:
    # Run rembg to get foreground cutout
    cutout = remove(data, session=session)
    fg = Image.open(io.BytesIO(cutout)).convert("RGBA")
    mask = np.array(fg)[:, :, 3]

    background = get_inpainted_background(data, mask, lama_model)
    edges_img = get_edge_map(mask)

    return {"foreground": fg, "background": background, "edges": edges_img}

def save_outputs(outputs: Dict[str, Image.Image], stem: str, out_root: Path):
    for key, img in outputs.items():
        subdir = out_root / key
        subdir.mkdir(parents=True, exist_ok=True)
        ext = ".png" if img.mode in ("RGBA", "LA") else ".jpg"
        img.save(subdir / f"{stem}{ext}")

def main():
    data_dir = Path("data/train_images/train_images")
    out_root = Path("data/train_images")
    image_files = [p for p in data_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not image_files:
        raise FileNotFoundError(f"No images found in {data_dir}")

    session = new_session()  # reuse rembg model
    lama_model = load_lama_inpainter(device="cuda")

    for path in tqdm(image_files, desc="Processing images (LaMa)"):
        with open(path, "rb") as f_in:
            data = f_in.read()
        outputs = process_image_bytes(data, session=session, lama_model=lama_model)
        save_outputs(outputs, path.stem, out_root)


if __name__ == "__main__":
    main()
