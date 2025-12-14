# -- IMPORTS --
import torch.nn as nn
from embedder_models import BirdEmbeddingModel
from multi_task_models import BirdMultiTaskModel

def create_model(model_name: str, n_classes=200, embedding_dim=128,
                n_attributes: int = 312, feature_dim: int = 256):
    """
    Call models by name.
    """
    model_name = model_name.lower()

    if model_name == "bird_embedding":
        return BirdEmbeddingModel(
            n_classes=n_classes,
            embedding_dim=embedding_dim)

    elif model_name == "multi_task_cnn":
        return BirdMultiTaskModel(
            n_classes=n_classes,
            n_attributes=n_attributes,
            feature_dim=feature_dim)

    else:
        raise ValueError(f"Unknown model name: {model_name}")