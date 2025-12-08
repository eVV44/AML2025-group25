# -- IMPROTS --
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBackbone(nn.Module):
    """
    CNN backbone.
    """
    def __init__(self, feature_dim: int = 256):
        super().__init__()

        self.conv = nn.Sequential(
            # block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3))

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.fc(x)
        return x
    

class BirdMultiTaskModel(nn.Module):
    """
    Multi-task model: 
    - main task = bird class (200 classes)
    - auxiliary task = attribute prediction (312 attributes)
    """
    def __init__(self, n_classes: int = 200, n_attributes: int = 312, feature_dim: int = 256):
        super().__init__()
        self.backbone = CNNBackbone(feature_dim=feature_dim)

        # classification head
        self.class_head = nn.Linear(feature_dim, n_classes)

        # attribute head
        self.attr_head = nn.Linear(feature_dim, n_attributes)

    def forward(self, images: torch.Tensor):
        features = self.backbone(images)
        class_logits = self.class_head(features)
        attr_logits  = self.attr_head(features)
        return class_logits, attr_logits