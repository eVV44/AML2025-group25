import torch

# This code was taken from some implementations of Mixup and CutMix and adapted to fit this codebase using genAI

def sample_mix_lam(alpha: float, device: torch.device) -> float:
    if alpha <= 0:
        return 1.0
    alpha_t = torch.tensor(alpha, device=device)
    dist = torch.distributions.Beta(alpha_t, alpha_t)
    return float(dist.sample(()).item())


def rand_bbox(images: torch.Tensor, lam: float) -> tuple[int, int, int, int]:
    _, _, height, width = images.shape
    device = images.device

    cut_ratio = (1.0 - lam) ** 0.5
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = int(torch.randint(0, width, (1,), device=device).item())
    cy = int(torch.randint(0, height, (1,), device=device).item())

    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, width)
    bby2 = min(cy + cut_h // 2, height)
    return bbx1, bby1, bbx2, bby2


def apply_mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    lam = sample_mix_lam(alpha, images.device)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1.0 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]
    return mixed_images, labels_a, labels_b, lam


def apply_cutmix(images: torch.Tensor, labels: torch.Tensor, alpha: float):
    lam = sample_mix_lam(alpha, images.device)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = images.clone()
    bbx1, bby1, bbx2, bby2 = rand_bbox(images, lam)
    mixed_images[:, :, bby1:bby2, bbx1:bbx2] = mixed_images[index, :, bby1:bby2, bbx1:bbx2]

    box_area = (bbx2 - bbx1) * (bby2 - bby1)
    total_area = images.size(-1) * images.size(-2)
    lam = 1.0 - (box_area / float(total_area))

    labels_a = labels
    labels_b = labels[index]
    return mixed_images, labels_a, labels_b, lam


def apply_mixup_cutmix(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    mixup_alpha: float,
    mixup_prob: float,
    cutmix_alpha: float,
    cutmix_prob: float,
):
    if images.size(0) < 2:
        return images, labels, None, 1.0

    r = float(torch.rand((), device=images.device).item())
    if cutmix_alpha > 0 and cutmix_prob > 0 and r < cutmix_prob:
        return apply_cutmix(images, labels, cutmix_alpha)
    if mixup_alpha > 0 and mixup_prob > 0 and r < (cutmix_prob + mixup_prob):
        return apply_mixup(images, labels, mixup_alpha)

    return images, labels, None, 1.0

