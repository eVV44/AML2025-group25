# -- IMPORTS --
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from typing import Tuple
from datetime import datetime
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# dataset augmentation, loading, splitting, transform to uniform size
# maybe use torch dataset and dataloader?

def transform_augment(img_size=(224,224), use_imagenet_norm=True, augment=True):
    """
    Transformation and augmentation.
    """
    # normalization
    if use_imagenet_norm:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std  = [0.5, 0.5, 0.5]

    # train transform
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size[0] + 32, img_size[1] + 32)),
            transforms.RandomResizedCrop(img_size[0], scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02,0.1), ratio=(0.3,3.3), inplace=True)])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    
    test_transform = val_transform

    return train_transform, val_transform, test_transform

def train_val_loaders(train_df, img_size=(224,224), batch_size=64, val_size=0.2,
                      random_state=42, num_workers=4, use_imagenet_norm=True, augment=True):
    """
    Create dataloaders with transforms + augmentation.
    """
    # train val split
    train_df_split, val_df_split = train_test_split(
        train_df, 
        test_size=val_size, 
        stratify=train_df["label_idx"], 
        random_state=random_state)

    train_transform, val_transform, _ = transform_augment(
        img_size=img_size, 
        use_imagenet_norm=use_imagenet_norm, 
        augment=augment)

    train_dataset = BirdDataset(train_df_split, transform=train_transform)
    val_dataset = BirdDataset(val_df_split, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_df_split, val_df_split


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
    

class TestSet():
    def __init__(self, img_transform_size: Tuple[int] = (224,224)):
        self.test_df = pd.read_csv('data/test_images_path.csv')
        self.test_df['image_path'] = 'data/test_images' + self.test_df['image_path']
        self.transform = transforms.Compose(
            [
                transforms.Resize(img_transform_size),
                transforms.ToTensor()
            ]
        )

    def predict(self, img_path, model):
        model.eval()

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            logits = model(img)
            pred = logits.argmax(dim=1).item()
        
        return pred

    def predict_all(self, prediction_model):
        self.test_df['label'] = self.test_df['image_path'].apply(self.predict, model=prediction_model)

    def get_df(self):
        return self.test_df
    
    def convert_to_submission(self):
        submission = self.test_df.copy(deep=True)
        submission.drop(columns=['image_path'], inplace=True)
        now_str = datetime.now().strftime("%d-%m-%Y_at_%H_%M") 
        submission.to_csv(f'predictions/predictions_{now_str}.csv', index=False)
