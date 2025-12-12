import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A


class BirdDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.root_dir = os.path.dirname(os.path.abspath(csv_path))
        self.transform = transform
        self.has_id = 'id' in self.df.columns

    def get_full_path(self, image_path: str) -> str:
        csv_path = image_path.lstrip('/')
        parts = csv_path.split('/', 1)
        if len(parts) == 2:
            return os.path.join(self.root_dir, parts[0], parts[0], parts[1])
        return os.path.join(self.root_dir, csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.get_full_path(row['image_path'])
        image = Image.open(img_path).convert('RGB')
        basename = os.path.basename(img_path)

        if self.transform:
            if isinstance(self.transform, (A.BasicTransform, A.BaseCompose)):
                augmented = self.transform(image=np.array(image), image_name=basename)
                image = augmented['image']
            else: # Backward compatibility with torchvision transforms
                image = self.transform(image)

        if self.has_id: target = int(row['id'])
        else: target = int(row['label']) - 1

        return image, target, basename
