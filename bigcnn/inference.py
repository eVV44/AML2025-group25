from pathlib import Path

import torch
from torchvision.transforms import functional as TVF
from augmentation import get_crop_box
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_DEFAULT_CROP_PCT = 0.875
FG_CROP_MARGIN = 0.1
ALPHA_BBOX_THRESH = 25

def predict_tta(
    model,
    images: torch.Tensor,
    attr_vectors: torch.Tensor | None,
    *,
    attr_temp: float,
    attr_mix: float,
    tta_crop_size: int,
    tta_mode: str,
) -> torch.Tensor:
    mode = str(tta_mode).lower().strip()

    if mode in {"none", "center"}:
        views = [TVF.center_crop(images, [tta_crop_size, tta_crop_size])]
    elif mode == "flip":
        center = TVF.center_crop(images, [tta_crop_size, tta_crop_size])
        views = [center, torch.flip(center, dims=[3])]
    elif mode == "five_crop":
        views = list(TVF.five_crop(images, [tta_crop_size, tta_crop_size]))
    elif mode == "ten_crop":
        views = list(TVF.ten_crop(images, [tta_crop_size, tta_crop_size]))

    logits_sum = None
    for view in views:
        logits = model.predict(view, attr_vectors, attr_temp=attr_temp, attr_mix=attr_mix)
        logits_sum = logits if logits_sum is None else (logits_sum + logits)

    return logits_sum / float(len(views))


def load_fg_img(
    image_names: list[str] | tuple[str, ...],
    *,
    device: torch.device,
    target_size: int,
    foreground_dir: str | Path,
) -> torch.Tensor:
    foreground_root = Path(foreground_dir)
    resize_size = int(round(target_size / IMAGENET_DEFAULT_CROP_PCT))

    batch = []
    for name in image_names:
        stem = Path(name).stem
        fg_path = foreground_root / f"{stem}.png"

        fg = Image.open(fg_path).convert("RGBA")
        alpha = fg.split()[-1]
        bbox = alpha.point(lambda v: 255 if v > ALPHA_BBOX_THRESH else 0).getbbox()
        if bbox is not None:
            img_w, img_h = fg.size
            fg = fg.crop(get_crop_box(bbox, img_w, img_h, FG_CROP_MARGIN))

        bg = Image.new("RGB", fg.size, (255, 255, 255))
        bg.paste(fg, mask=fg.split()[-1])

        resized = TVF.resize(bg, resize_size, antialias=True)
        cropped = TVF.center_crop(resized, [target_size, target_size])

        t = TVF.to_tensor(cropped)
        t = TVF.normalize(t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        batch.append(t)

    return torch.stack(batch, dim=0).to(device)


def predict_logits(
    model,
    images: torch.Tensor,
    attr_vectors: torch.Tensor | None = None,
    *,
    attr_temp: float = 10.0,
    attr_mix: float = 0.3,
    use_tta: bool = False,
    tta_crop_size: int | None = None,
    tta_mode: str = "ten_crop",
    use_foreground: bool = False,
    image_names: list[str] | tuple[str, ...] | None = None,
    foreground_dir: str | Path = "data/test_images/foreground",
) -> torch.Tensor:
    image_set = [images]

    if use_foreground:
        target_size = int(images.shape[-1])
        fg_images = load_fg_img(image_names, device=images.device, target_size=target_size, foreground_dir=foreground_dir)
        image_set.append(fg_images)
    
    all_logits = []
    for img in image_set:
        if not use_tta:
            main_logits = model.predict(img, attr_vectors, attr_temp=attr_temp, attr_mix=attr_mix)
        else:
            main_logits = predict_tta(
                model,
                img,
                attr_vectors,
                attr_temp=attr_temp,
                attr_mix=attr_mix,
                tta_crop_size=tta_crop_size,
                tta_mode=tta_mode,
            )
        all_logits.append(main_logits)

    return torch.stack(all_logits, dim=0).mean(dim=0)
