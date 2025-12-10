import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import BirdDataset
from augmentation import get_train_transforms, get_test_transforms
from birdmodels import BirdClassifier
from training import train_step, evaluate_improved

# Configuration dictionary
config = {
    'data_root': 'data',
    'num_classes': 200,
    'attribute_dim': 312,
    'embedding_dim': 512,
    'img_size': 244,
    
    # Training
    'batch_size': 48,
    'eval_batch_size': 64,
    'val_split': 0.15,
    'num_epochs': 500,
    'learning_rate': 1.5e-3,
    'weight_decay': 1e-4,
    'label_smoothing': 0.1,
    
    # Attribute heads
    'use_attr_pred_head': True,
    'use_attr_emb_head': False,
    'lambda_attr_pred': 1.0,
    'lambda_attr_emb': 0.3,
    'attr_temp': 10.0,
    'attr_mix': 0.3,
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'use_tta': False,
    'seed': None,
}

def print_config(config):
    print(f"  Device: {config['device']}")
    print(f"  Embedding dim: {config['embedding_dim']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Weight decay: {config['weight_decay']}")
    print(f"  Label smoothing: {config['label_smoothing']}")
    print(f"  Random seed: {config['seed']}\n")
    print(f"  Use Attribute Prediction Head: {config['use_attr_pred_head']}{' with λ=' + str(config['lambda_attr_pred']) if config['use_attr_pred_head'] else ''}")
    print(f"  Use Attribute Embedding Head: {config['use_attr_emb_head']}{' with λ=' + str(config['lambda_attr_emb']) if config['use_attr_emb_head'] else ''}")

def setup_environment(config):
    # Sets seeds and prints configuration for reproducibility
    config['seed'] = config['seed'] or int(time.time())
    
    print("="*70)
    print("Birdies Classification Training")
    print("="*70)
    print("\nConfig:")
    print_config(config)
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    return torch.device(config['device'])

def prepare_data(config, device):
    # Loads attributes, transforms, splits data, and returns loaders
    print("\nPreparing datasets...")
    
    # Load attributes
    attributes = np.load(os.path.join(config['data_root'], 'attributes.npy'))
    attr_vectors = torch.FloatTensor(attributes).to(device)
    
    # Transforms
    train_transform = get_train_transforms(img_size=config['img_size'])
    val_transform = get_test_transforms(img_size=config['img_size'])
    
    # Datasets
    train_csv = os.path.join(config['data_root'], 'train_images.csv')
    full_train_dataset = BirdDataset(train_csv, transform=train_transform)
    full_val_dataset = BirdDataset(train_csv, transform=val_transform)

    # Split, TODO: Ensure stratified split
    indices = range(len(full_train_dataset))
    labels = full_train_dataset.df['label']
    train_idx, val_idx = train_test_split(
        indices, test_size=config['val_split'], shuffle=True, random_state=config['seed'], stratify=labels
    )

    train_ds = Subset(full_train_dataset, train_idx)
    val_ds = Subset(full_val_dataset, val_idx)
    
    # Loaders
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['eval_batch_size'], 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")
    return train_loader, val_loader, attr_vectors

def setup_model_components(config, device):
    # Initializes model, optimizer, scheduler, and loss function
    print("\nCreating model components...")
    
    model = BirdClassifier(
        num_classes=config['num_classes'],
        attribute_dim=config['attribute_dim'],
        embedding_dim=config['embedding_dim'],
        use_attr_pred=config['use_attr_pred_head'],
        use_attr_emb=config['use_attr_emb_head']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                                  weight_decay=config['weight_decay'])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=config['num_epochs'], 
                                                           eta_min=1e-6)
    
    return model, optimizer, scheduler, criterion

def training_loop(config, model, train_loader, val_loader, optimizer, scheduler, criterion, attr_vectors, device):
    print("="*70)
    print(f"Starting Training for {config['num_epochs']} epochs")
    print("="*70)

    best_val_acc = 0.0
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss, class_loss, attr_pred_loss, attr_emb_loss = 0.0, 0.0, 0.0, 0.0
        train_acc = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            loss_list, acc = train_step(
                model, images, labels, attr_vectors, optimizer, criterion, device,
                use_attr_pred=config['use_attr_pred_head'],
                use_attr_emb=config['use_attr_emb_head'],
                lambda_attr_pred=config['lambda_attr_pred'],
                lambda_attr_emb=config['lambda_attr_emb'],
                attr_temp=config['attr_temp']
            )

            t_loss, c_loss, ap_loss, ae_loss = loss_list

            train_loss += t_loss
            class_loss += c_loss
            attr_pred_loss += ap_loss
            attr_emb_loss += ae_loss

            train_acc += acc

            pbar.set_postfix({'loss': f'{t_loss:.4f}', 'acc': f'{acc:.4f}'})
        
        # Averages
        train_loss /= len(train_loader)
        class_loss /= len(train_loader)
        attr_pred_loss /= len(train_loader)
        attr_emb_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation
        val_acc, _, _ = evaluate_improved(
            model, val_loader, device, attributes=attr_vectors,
            use_tta=config['use_tta'], attr_mix=config['attr_mix'], attr_temp=config['attr_temp']
        )
        
        scheduler.step()
        loss_print_list = [f"Train Loss: {train_loss:.4f}", f"Class Loss: {class_loss:.4f}"]
        if config['use_attr_pred_head']: loss_print_list.append(f"Attr Pred Loss: {attr_pred_loss:.4f}")
        if config['use_attr_emb_head']: loss_print_list.append(f"Attr Emb Loss: {attr_emb_loss:.4f}")
        print(f"\nEpoch {epoch+1}: {' | '.join(loss_print_list)}")
        
        whitespace_pad = " " * len(f"Epoch {epoch+1}: ")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{whitespace_pad}LR: {current_lr:.6f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save Best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, 'best_improved_model.pth')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.4f})")
        print()
        
    print(f"Training Complete. Best Validation Accuracy: {best_val_acc:.4f}")

def main():
    # Orchestrate the training pipeline
    device = setup_environment(config)
    train_loader, val_loader, attr_vectors = prepare_data(config, device)
    model, optimizer, scheduler, criterion = setup_model_components(config, device)
    
    training_loop(
        config, model, train_loader, val_loader, 
        optimizer, scheduler, criterion, attr_vectors, device
    )

if __name__ == '__main__':
    main()