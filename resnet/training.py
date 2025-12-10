import torch
import torch.nn as nn
import torch.nn.functional as F

from birdmodels import calc_cosine_similarity

def train_step(model, images, labels, attributes, optimizer, criterion, device,
                       use_attr_pred=True, lambda_attr_pred=0.5,
                       use_attr_emb=True, lambda_attr_emb=0.3,
                       attr_temp=10.0):
    """
    Single training step with optional attribute prediction and attribute-embedding logits.
    
    Args:
        model: BirdClassifier
        images: (batch_size, 3, H, W)
        labels: (batch_size,)
        attributes: (num_classes, 312) - all class attributes
        optimizer: torch optimizer
        criterion: loss function applied to class logits (e.g., CrossEntropyLoss)
        device: cuda/cpu (unused, kept for API compatibility)
        use_attr_pred: enable auxiliary attribute prediction (BCE) if head exists
        lambda_attr_pred: weight for attribute prediction loss
        use_attr_emb: enable attribute-similarity logits supervision if head exists
        lambda_attr_emb: weight for attribute-logits classification loss
        attr_temp: temperature scaling for attribute logits
    
    Returns:
        loss: total loss value (float)
        accuracy: batch accuracy (float)
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    logits, embeddings, attr_pred = model(images)
    class_loss = criterion(logits, labels)

    use_attr_pred = use_attr_pred and model.use_attr_pred

    # Attribute prediction loss (BCE against ground truth attributes)
    attr_loss = 0.0
    if use_attr_pred:
        batch_attributes = attributes[labels].float()  # (batch_size, 312)
        attr_loss = F.binary_cross_entropy_with_logits(attr_pred, batch_attributes)

    # Attribute-similarity logits loss (align image embeddings to attribute embeddings)
    attr_class_loss = 0.0
    if use_attr_emb:
        attr_embeddings = model.attr_emb(attributes)
        attr_logits = calc_cosine_similarity(embeddings, attr_embeddings) * attr_temp
        attr_class_loss = criterion(attr_logits, labels)

    attr_pred_loss = lambda_attr_pred * attr_loss if use_attr_pred else torch.tensor(0.0)
    attr_emb_loss = lambda_attr_emb * attr_class_loss if use_attr_emb else torch.tensor(0.0)

    total_loss = class_loss + attr_pred_loss + attr_emb_loss
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Compute accuracy
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == labels).float().mean()
    
    loss_list = [total_loss, class_loss, attr_pred_loss, attr_emb_loss]
    loss_list = [l.item() for l in loss_list]
    return loss_list, accuracy.item()


def evaluate_improved(model, dataloader, device, attributes=None, use_tta=False, attr_mix=0.3, attr_temp=10.0):
    """
    Evaluation with optional attribute-augmented logits and test-time augmentation.
    
    Args:
        model: BirdClassifier
        dataloader: validation/test dataloader
        device: cuda/cpu
        attributes: (num_classes, 312) attribute matrix (on device) for attribute logits
        use_tta: whether to use test-time augmentation
        attr_mix: weight for attribute logits in the blend (0-1) inside model.predict
        attr_temp: temperature scaling for attribute logits inside model.predict
    
    Returns:
        accuracy: validation accuracy (float)
        predictions: concatenated predictions on CPU
        labels: concatenated labels on CPU
    """
    model.eval()
    
    attr_vectors = None
    if attributes is not None:
        attr_vectors = attributes.to(device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            def forward(batch):
                return model.predict(batch, attr_vectors, attr_temp=attr_temp, attr_mix=attr_mix)
            
            if use_tta:
                # Test-time augmentation | Add multiple augmentations later
                logits1 = forward(images)
                logits2 = forward(torch.flip(images, dims=[3])) 
                logits = (logits1 + logits2) / 2
            else:
                logits = forward(images)
            
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    accuracy = (all_predictions == all_labels).float().mean().item()
    
    return accuracy, all_predictions, all_labels
