from pathlib import Path
import torchvision.transforms as T
import albumentations as A
import numpy as np
import random
import cv2

from albumentations.pytorch import ToTensorV2
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_DEFAULT_CROP_PCT = 0.875

def get_train_transforms_torch(img_size=224):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(3/4, 4/3)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)) 
    ])

def get_test_transforms_torch(img_size=224):
    resize_size = int(round(img_size / IMAGENET_DEFAULT_CROP_PCT))
    return T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_test_transforms_torch_tta(img_size=224):
    resize_size = int(round(img_size / IMAGENET_DEFAULT_CROP_PCT))
    return T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(resize_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

class RandomBackgroundSwap(A.ImageOnlyTransform):
    def __init__(self, bg_root, fg_root, p_remove=0.3, p_swap=0.4, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.bg_root = Path(bg_root) if bg_root else None
        self.bg_paths = self.load_bgs(self.bg_root)
        self.fg_root = fg_root
        self.p_remove = p_remove
        self.p_swap = p_swap  

    def load_bgs(self, root):
        if not root: return []
        exts = {".jpg", ".jpeg", ".png"}
        return [p for p in root.glob("*") if p.suffix.lower() in exts]

    def remove_background(self, name):
        fg_path = self.fg_root / f"{Path(name).stem}.png"
        fg = Image.open(fg_path).convert("RGBA")
        bg = Image.new("RGB", fg.size, (255, 255, 255))
        bg.paste(fg, mask=fg.split()[-1])
        return np.array(bg)
        
    def swap_background(self, img, name):
        bg = Image.open(random.choice(self.bg_paths)).convert("RGB")
        fg_path = self.fg_root / f"{Path(name).stem}.png" 
        fg = Image.open(fg_path).convert("RGBA")
        bg = bg.resize((img.shape[1], img.shape[0]))
        bg.paste(fg, mask=fg.split()[-1])
        return np.array(bg)

    def apply(self, img, **params):
        name = params.get("image_name")
        r = random.random()
        if r < self.p_remove:
            return self.remove_background(name)
        elif r < self.p_remove + self.p_swap and self.bg_paths and self.fg_root and name:
            return self.swap_background(img, name)
        return img

class RandomEdgeOverlay(A.ImageOnlyTransform):
    def __init__(self, edge_root=None, color=(0, 255, 255), alpha=0.8, always_apply=False, p=0.2):
        super().__init__(always_apply, p)
        self.edge_root = Path(edge_root) if edge_root else None
        self.color = color
        self.alpha = alpha
    
    def load_edges(self, name, target_shape):
        if not (self.edge_root and name): return None

        edge_path = self.edge_root / f"{Path(name).stem}.jpg"
        if edge_path.exists():
            edges = np.array(Image.open(edge_path).convert("L"))
            if edges.shape != target_shape[:2]:
                edges = cv2.resize(edges, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
            return edges
        return None
        
    # Some image magic with help from repos and chatgpt, basically overlays edges onto image
    def apply(self, img, **params):
        name = params.get("image_name")
        edges = self.load_edges(name, img.shape)
        if edges is None: return img

        # Threshold to drop JPEG noise but keep edge visibility, then thicken/smooth
        mask = (edges > 40).astype(np.uint8)
        if mask.max() > 0:
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
            mask = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)
        mask = mask[..., None]  

        color_arr = np.array(self.color, dtype=np.float32)
        img_f = img.astype(np.float32)
        overlay = img_f * (1 - mask) + color_arr * mask
        blended = img_f * (1 - self.alpha * mask) + overlay * (self.alpha * mask)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return blended

def get_train_transforms_album(img_size=224):
    train_root = Path("data/train_images/")
    bg_root = train_root / "background"
    fg_root = train_root / "foreground"
    edge_root = train_root / "edges"
    return A.Compose([
        # RandomBackgroundSwap(bg_root=bg_root, fg_root=fg_root, p_remove=0.3, p_swap=0.2),
        # RandomEdgeOverlay(edge_root=edge_root, color=(255, 0, 0), alpha=0.8, p=0.1),

        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.75, 1.33), p=1.0),

        # A.LongestMaxSize(max_size=img_size),
        # A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, fill=(255, 255, 255), fill_mask=0),
        # A.RandomCrop(height=img_size, width=img_size),

        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.05, 0.05),
            rotate=(-15, 15),
            border_mode=0,
            fit_output=False,
            keep_ratio=True,
            p=0.5
        ),

        A.OneOf([
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
            A.ToGray()
        ], p=0.6),

        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3))
        ], p=0.15),

        A.GaussNoise(std_range=(0.01, 0.05), mean_range=(0.0, 0.0), p=0.1),

        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(int(0.05 * img_size), int(0.2 * img_size)),
            hole_width_range=(int(0.05 * img_size), int(0.2 * img_size)),
            fill=0,
            p=0.2
        ),

        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

# shits broken yo, dont use
def get_test_transforms_album(img_size=224):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, fill=(255, 255, 255), fill_mask=0),
        A.RandomCrop(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

get_train_transforms = get_train_transforms_torch
get_test_transforms = get_test_transforms_torch
get_test_transforms_tta = get_test_transforms_torch_tta

get_train_transforms = get_train_transforms_album
# get_test_transforms = get_test_transforms_album
