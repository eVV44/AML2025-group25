# -- IMPORTS --
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from PIL import Image

# dataset augmentation, loading, splitting, transform to uniform size
# maybe use torch dataset and dataloader?

class BirdDataset(Dataset):
    """
    Dataset tranformation.
    """
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
    
        img_path = row["image_path"]
        label = row["label_idx"]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label