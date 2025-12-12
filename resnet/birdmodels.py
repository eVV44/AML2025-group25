import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50

def calc_cosine_similarity(a, b):
    # Compute cosine similarity through normalization together with dot product 
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return F.linear(a_norm, b_norm)

# Roughly 6M parameters
class BiggerCNNEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()

        def conv_block(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2),

            conv_block(128, 192),
            nn.MaxPool2d(kernel_size=2),

            conv_block(192, 256),
            nn.MaxPool2d(kernel_size=2),

            conv_block(256, 320),
            nn.MaxPool2d(kernel_size=2),

            conv_block(320, 384),
            # nn.Dropout(p=0.1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # self.dropout = nn.Dropout(p=0.2)
        self.projection = nn.Identity()
        out_dim = 384
        if embedding_dim != out_dim:
            self.projection = nn.Linear(out_dim, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        x = self.projection(x)
        return x

class ResNetEncoder(nn.Module):
    resnet_dim = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048
    }

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.fc = nn.Identity() 

        RESNET_OUT_DIM = self.resnet_dim["resnet18"]
        self.projection = nn.Identity()
        if embedding_dim != RESNET_OUT_DIM:
            self.projection = nn.Linear(RESNET_OUT_DIM, embedding_dim)
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.projection(x)
        return x

class CosineClassifier(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))
        self.log_temp = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, embeddings):
        # Compute cosine similarity with temperature scaling
        logits = calc_cosine_similarity(embeddings, self.prototypes)
        temp = torch.exp(self.log_temp).clamp(0.1, 10.0)
        return logits * temp

class AttributePredictor(nn.Module):
    def __init__(self, embedding_dim: int, attribute_dim: int = 312):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, attribute_dim)
        )
    
    def forward(self, embeddings):
        return self.predictor(embeddings)

class AttributeEmbedder(nn.Module):
    def __init__(self, attribute_dim: int = 312, embedding_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(attribute_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
    
    def forward(self, attributes):
        return self.mlp(attributes)

# ============================================================================
# COMPLETE MODEL
# ============================================================================

class BirdClassifier(nn.Module):
    def __init__(self, num_classes: int = 200, attribute_dim: int = 312, embedding_dim: int = 512, 
                 use_attr_pred: bool = True, use_attr_emb : bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.use_attr_pred = use_attr_pred
        self.use_attr_emb = use_attr_emb
            
        # self.encoder = ResNetEncoder(embedding_dim)
        # self.encoder = SimpleCNNEncoder(embedding_dim)
        self.encoder = BiggerCNNEncoder(embedding_dim)
        self.classifier = CosineClassifier(embedding_dim, num_classes)
        
        # Optional auxiliary attribute heads
        if self.use_attr_pred:
            self.attr_pred = AttributePredictor(embedding_dim, attribute_dim)
        
        if self.use_attr_emb:
            self.attr_emb = AttributeEmbedder(attribute_dim, embedding_dim)
    
    def forward(self, images):
        """
        Args:
            images: (batch_size, 3, H, W)
        Returns:
            logits: (batch_size, num_classes)
            embeddings: (batch_size, embedding_dim) 
            attr_pred: (batch_size, 312) or None
        """
        # Extract embeddings
        embeddings = self.encoder(images)
        
        # Classify
        logits = self.classifier(embeddings)
        
        attr_pred = None
        if self.use_attr_pred:
            attr_pred = self.attr_pred(embeddings)
        
        return logits, embeddings, attr_pred

    def predict(self, images, all_attributes = None, attr_temp: float = 10.0, attr_mix: float = 0.3):
        """
        Inference using learned classifier and optional attribute-based predictions.
        
        Args:
            images: (batch_size, 3, H, W)
            all_attributes: (num_classes, 312), optional - class attribute embeddings
            attr_temp: temperature scaling for attribute logits
            attr_mix: mixing weight for attribute logits, in range [0, 1]
        
        Returns:
            logits: (batch_size, num_classes) - class logits, or weighted combination with 
                    attribute logits if self.use_attr_emb is True and attr_mix > 0
        """
        
        embeddings = self.encoder(images)
        class_logits = self.classifier(embeddings)

        if not self.use_attr_emb or attr_mix <= 0:
            return class_logits

        attr_embeddings = self.attr_emb(all_attributes)
        attr_logits = calc_cosine_similarity(embeddings, attr_embeddings) * attr_temp
        combined_logits = (1 - attr_mix) * class_logits + attr_mix * attr_logits
        return combined_logits
