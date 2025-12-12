from pathlib import Path

import torch
from torchvision.transforms import functional as TVF
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
    else:
        raise ValueError(f"Unknown tta_mode: {tta_mode}")

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

    batch = []
    for name in image_names:
        stem = Path(name).stem
        fg_path = foreground_root / f"{stem}.png"

        fg = Image.open(fg_path).convert("RGBA")
        bg = Image.new("RGB", fg.size, (255, 255, 255))
        bg.paste(fg, mask=fg.split()[-1])

        resized = TVF.resize(bg, target_size, antialias=True)
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
    if not use_tta:
        main_logits = model.predict(images, attr_vectors, attr_temp=attr_temp, attr_mix=attr_mix)
    else:
        if tta_crop_size is None:
            raise ValueError("tta_crop_size must be provided when use_tta=True")
        main_logits = predict_tta(
            model,
            images,
            attr_vectors,
            attr_temp=attr_temp,
            attr_mix=attr_mix,
            tta_crop_size=tta_crop_size,
            tta_mode=tta_mode,
        )

    if not use_foreground:
        return main_logits

    target_size = int(images.shape[-1])
    if images.shape[-2] != images.shape[-1]:
        raise ValueError(f"Expected square input tensor, got {tuple(images.shape[-2:])}")

    fg_images = load_fg_img(
        image_names,
        device=images.device,
        target_size=target_size,
        foreground_dir=foreground_dir,
    )

    if not use_tta:
        fg_logits = model.predict(fg_images, attr_vectors, attr_temp=attr_temp, attr_mix=attr_mix)
    else:
        fg_logits = predict_tta(
            model,
            fg_images,
            attr_vectors,
            attr_temp=attr_temp,
            attr_mix=attr_mix,
            tta_crop_size=tta_crop_size,
            tta_mode=tta_mode,
        )

    return 0.5 * (main_logits + fg_logits)
