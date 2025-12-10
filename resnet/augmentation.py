import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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

def get_train_transforms_album(img_size=224):
    return A.Compose([
        A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.7),

        A.OneOf([
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
            A.ToGray()
        ], p=0.8),

        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3))
        ], p=0.3),

        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

        A.CoarseDropout(
            max_holes=1,
            max_height=int(0.2 * img_size),
            max_width=int(0.2 * img_size),
            min_height=int(0.05 * img_size),
            min_width=int(0.05 * img_size),
            fill_value=0,
            p=0.5
        ),

        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def get_test_transforms_torch(img_size=224):
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_test_transforms_album(img_size=224):
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

get_train_transforms = get_train_transforms_torch
get_test_transforms = get_test_transforms_torch

# get_train_transforms = get_train_transforms_album
# get_test_transforms = get_test_transforms_album