import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingCNN(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        
        # Convolutional layers, structure per block: convolution, activation, pooling
        self.conv = nn.Sequential(
            nn.Conv2d(3, out_channels=32, kernel_size=3, padding=1, padding_mode='zeros'), # convolutional layer 32 filters (channels), preserve image height and width with 1 pixel zeropadding
            nn.ReLU(), # activation layer
            nn.MaxPool2d(2), # pooling layer: kernel size 2 and stride 2, output height and width are half of input

            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)) # transform output to height x width: 1x1
        )

        # Fully connected (linear) layer(s)
        self.fc = nn.Sequential(
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1) # flatten to prepare for linear layer(s)
        x = self.fc(x)
        x = F.normalize(x, dim=1) # normalize (L2) for cosine similarity
        return x


class BirdEmbeddingModel(nn.Module):
    def __init__(self, n_classes: int = 200, embedding_dim: int = 128):
        super().__init__()
        self.cnn = EmbeddingCNN(embedding_dim)
        self.class_embedding_anchors = nn.Parameter(torch.randn(n_classes, embedding_dim))
        nn.init.xavier_uniform_(self.class_embedding_anchors) # Apply xavier uniform normalization to the initial state for better starting optimization
    
    def get_class_anchors(self) -> torch.Tensor:
        return F.normalize(self.class_embedding_anchors, dim=1)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # image embeddings
        image_embedding = self.cnn(images)
        # class anchors
        class_anchors = self.get_class_anchors()
        
        # cosine similarity
        logits = torch.matmul(image_embedding, class_anchors.T)

        return logits
