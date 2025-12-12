import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TVF

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
    tta_scales: list[float] | tuple[float, ...] | None = None,
) -> torch.Tensor:
    if not use_tta:
        return model.predict(images, attr_vectors, attr_temp=attr_temp, attr_mix=attr_mix)

    if tta_crop_size is None:
        raise ValueError("tta_crop_size must be provided when use_tta=True")

    mode = str(tta_mode).lower().strip()
    if tta_scales is None:
        tta_scales = [1.0]

    logits_sum = None
    num_views_total = 0

    for scale in tta_scales:
        scale = float(scale)
        if scale <= 0:
            continue

        if abs(scale - 1.0) < 1e-6:
            scaled = images
        else:
            _, _, height, width = images.shape
            scaled_size = (int(round(height * scale)), int(round(width * scale)))
            if scaled_size[0] < tta_crop_size or scaled_size[1] < tta_crop_size:
                continue
            scaled = F.interpolate(images, size=scaled_size, mode="bilinear", align_corners=False)

        if mode in {"none", "center"}:
            views = [TVF.center_crop(scaled, [tta_crop_size, tta_crop_size])]
        elif mode == "flip":
            center = TVF.center_crop(scaled, [tta_crop_size, tta_crop_size])
            views = [center, torch.flip(center, dims=[3])]
        elif mode == "five_crop":
            views = list(TVF.five_crop(scaled, [tta_crop_size, tta_crop_size]))
        elif mode == "ten_crop":
            views = list(TVF.ten_crop(scaled, [tta_crop_size, tta_crop_size]))
        else:
            raise ValueError(f"Unknown tta_mode: {tta_mode}")

        for view in views:
            view_logits = model.predict(view, attr_vectors, attr_temp=attr_temp, attr_mix=attr_mix)
            logits_sum = view_logits if logits_sum is None else (logits_sum + view_logits)
            num_views_total += 1

    return logits_sum / float(num_views_total)
