# -- IMPORTS --
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from typing import Tuple
from datetime import datetime

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
