from pathlib import Path
import random
from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

import open_clip
from sklearn.model_selection import train_test_split

# Lots of boilerplate code was taken from public repos using CLIP, and examples of fine tuning CLIP

PRETRAINED_TAGS = {
    "ViT-B/16": "laion2b_s34b_b88k",
    "ViT-B/32": "laion2b_s34b_b79k",
    "RN50": "openai",
}

CONFIG = {
    "data_dir": Path("data"),
    "model_name": "ViT-B/16",
    "pretrained_tag": "auto",
    "batch_size": 32,
    "epochs": 60,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "val_fraction": 0.1,
    "num_workers": 4,
    "seed": None,
    "submission_path": Path("submissions/clip.csv"),
}


def set_seed(seed: int | None) -> int:
    seed = int(seed or int(time()))
    print(f"Using seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    return (base_dir / raw_path.lstrip("/")).resolve()


def add_resolved_path(
    df: pd.DataFrame, base_dir: Path, image_path_col: str = "image_path"
) -> pd.DataFrame:
    df = df.copy()
    df["resolved_path"] = df[image_path_col].apply(lambda p: resolve_path(base_dir, p))
    return df


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, target_col: str):
        self.paths = df["resolved_path"].tolist()
        self.targets = df[target_col].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


def load_class_names(path: Path):
    mapping = np.load(path, allow_pickle=True).item()  # {name: id}
    id_to_name = {v: k for k, v in mapping.items()}
    return [id_to_name[i] for i in sorted(id_to_name)]


@torch.inference_mode()
def encode_images(model, imgs: torch.Tensor) -> torch.Tensor:
    feats = model.encode_image(imgs)
    return F.normalize(feats, dim=-1)


def train_linear_head(
    model,
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: dict,
):
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    optim = torch.optim.AdamW(
        head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    def run_epoch(loader: DataLoader, training: bool):
        head.train(training)
        total_loss, correct, total = 0.0, 0, 0
        for imgs, targets in loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            feats = encode_images(model, imgs)
            logits = head(feats)
            loss = criterion(logits, targets)

            if training:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)
        return total_loss / total, correct / total

    best_val = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = run_epoch(train_loader, training=True)
        val_loss, val_acc = run_epoch(val_loader, training=False)
        best_val = max(best_val, val_acc)
        print(
            f"Epoch {epoch:02d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

    print(f"Best val acc: {best_val:.4f}")
    return best_val


@torch.inference_mode()
def predict_test(model, head: nn.Module, test_loader: DataLoader, device: torch.device):
    head.eval()

    preds, ids = [], []
    for imgs, batch_ids in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = encode_images(model, imgs)
        logits = head(feats)
        preds.extend(logits.argmax(dim=1).tolist())
        ids.extend(batch_ids.tolist())
    return preds, ids


def write_submission(ids, preds, path: Path):
    submission = pd.DataFrame({"id": ids, "label": [p + 1 for p in preds]})
    submission.sort_values("id", inplace=True)
    submission.reset_index(drop=True, inplace=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    print(f"Wrote submission to {path}")


def load_train_df(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "train_images.csv")
    df = add_resolved_path(df, data_dir / "train_images")
    df["label_idx"] = df["label"].astype(int) - 1
    return df


def load_test_df(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "test_images_path.csv")
    return add_resolved_path(df, data_dir / "test_images")


def main():
    cfg = CONFIG
    seed = set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    data_dir: Path = cfg["data_dir"]
    num_classes = len(load_class_names(data_dir / "class_names.npy"))

    train_df = load_train_df(data_dir)
    train_df, val_df = train_test_split(
        train_df,
        test_size=cfg["val_fraction"],
        random_state=seed,
        shuffle=True,
        stratify=train_df["label_idx"],
    )

    tag = (
        PRETRAINED_TAGS[cfg["model_name"]]
        if cfg["pretrained_tag"] == "auto"
        else cfg["pretrained_tag"]
    )
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        cfg["model_name"], pretrained=tag
    )
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    head = nn.Linear(model.visual.output_dim, num_classes).to(device)

    train_loader = DataLoader(
        ImageDataset(train_df, preprocess_train, target_col="label_idx"),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        ImageDataset(val_df, preprocess_val, target_col="label_idx"),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
    )

    train_linear_head(model, head, train_loader, val_loader, device, cfg)

    test_df = load_test_df(data_dir)
    test_loader = DataLoader(
        ImageDataset(test_df, preprocess_val, target_col="id"),
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
    )

    preds, ids = predict_test(model, head, test_loader, device)
    write_submission(ids, preds, cfg["submission_path"])


if __name__ == "__main__":
    main()
