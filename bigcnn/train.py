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
    'img_size': 384,
    
    # Training
    'batch_size': 32,
    'eval_batch_size': 32,
    'val_split': 0.15,
    'num_epochs': 400,
    'optimizer': 'sgd',  # 'sgd' or 'adamw'
    'learning_rate': 2e-2,
    'weight_decay': 5e-4,
    'label_smoothing': 0.0,
    'warmup_epochs': 0,

    # Mixup / CutMix
    'mixup_alpha': 0.2,
    'mixup_prob': 0.25,
    'cutmix_alpha': 1.0,
    'cutmix_prob': 0.25,

    # Cosine-margin (ArcFace-style)
    'classifier_scale': 30.0,
    'classifier_margin': 0.2,
    
    # Attribute heads
    'use_attr_pred_head': False,
    'use_attr_emb_head': False,
    'lambda_attr_pred': 0.5,
    'lambda_attr_emb': 0.4,
    'attr_temp': 10.0,
    'attr_mix': 0.3,

    # Eval
    # If True: validation averages logits from original + foreground images.
    'use_foreground_eval': True,
    
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': None,

    # I/O
    'save_path': 'best_bigcnn_model.pth',

    # Resume
    # Set to a checkpoint path (e.g., 'best_bigcnn_model.pth') to continue training.
    # 'resume_from': 'best_bigcnn_model.pth',
    'resume_from': None,
}

def print_config(config):
    print(f"  Device: {config['device']}")
    print(f"  Embedding dim: {config['embedding_dim']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Optimizer: {config['optimizer']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Weight decay: {config['weight_decay']}")
    print(f"  Label smoothing: {config['label_smoothing']}")
    print(f"  Warmup epochs: {config['warmup_epochs']}")
    print(f"  Mixup: a={config['mixup_alpha']} p={config['mixup_prob']}")
    print(f"  CutMix: a={config['cutmix_alpha']} p={config['cutmix_prob']}")
    print(f"  ArcFace: m={config['classifier_margin']} s={config['classifier_scale']}")
    print(f"  Random seed: {config['seed']}\n")
    print(f"  Resume from: {config.get('resume_from')}\n")
    print(f"  Use Attribute Prediction Head: {config['use_attr_pred_head']}{' with λ=' + str(config['lambda_attr_pred']) if config['use_attr_pred_head'] else ''}")
    print(f"  Use Attribute Embedding Head: {config['use_attr_emb_head']}{' with λ=' + str(config['lambda_attr_emb']) if config['use_attr_emb_head'] else ''}")

def setup_environment(config):
    # Sets seeds and prints configuration for reproducibility
    resume_path = config.get('resume_from')
    if resume_path:
        try:
            checkpoint = torch.load(resume_path, map_location="cpu")
            ckpt_config = checkpoint.get("config", {})
            if config.get("seed") is None and ckpt_config.get("seed") is not None:
                config["seed"] = ckpt_config["seed"]
        except Exception as e:
            print(f"Warning: failed to read resume checkpoint '{resume_path}': {e}")

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

def resume_checkpoint(config, model, optimizer, scheduler, device):
    resume_path = config.get('resume_from')
    if not resume_path:
        return 0, 0.0

    print(f"\nResuming from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    if checkpoint.get('optimizer_state_dict', None) is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = int(checkpoint.get('epoch', -1)) + 1
    best_val_acc = float(checkpoint.get('best_val_acc', checkpoint.get('val_acc', 0.0)))

    if checkpoint.get('scheduler_state_dict', None) is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        for _ in range(max(0, start_epoch)):
            scheduler.step()

    print(f"  Start epoch: {start_epoch} | Best val acc so far: {best_val_acc:.4f}")
    return start_epoch, best_val_acc

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
        use_attr_emb=config['use_attr_emb_head'],
        classifier_scale=config['classifier_scale'],
        classifier_margin=config['classifier_margin'],
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])

    optimizer_name = str(config['optimizer']).lower()
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=0.9,
            weight_decay=config['weight_decay'],
            nesterov=True,
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
        )
    
    warmup_epochs = int(config.get('warmup_epochs', 0) or 0)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config['num_epochs'] - warmup_epochs),
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=1e-6,
        )
    
    return model, optimizer, scheduler, criterion

def training_loop(config, model, train_loader, val_loader, optimizer, scheduler, criterion, attr_vectors, device, *, start_epoch=0, best_val_acc=0.0):
    print("="*70)
    print(f"Starting Training for {config['num_epochs']} epochs")
    print("="*70)

    for epoch in range(start_epoch, config['num_epochs']):
        model.train()
        train_loss, class_loss, attr_pred_loss, attr_emb_loss = 0.0, 0.0, 0.0, 0.0
        train_acc = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for images, labels, _ in pbar:
            images, labels = images.to(device), labels.to(device)
            
            loss_list, acc = train_step(
                model, images, labels, attr_vectors, optimizer, criterion, device,
                use_attr_pred=config['use_attr_pred_head'],
                use_attr_emb=config['use_attr_emb_head'],
                lambda_attr_pred=config['lambda_attr_pred'],
                lambda_attr_emb=config['lambda_attr_emb'],
                attr_temp=config['attr_temp'],
                mixup_alpha=config['mixup_alpha'],
                mixup_prob=config['mixup_prob'],
                cutmix_alpha=config['cutmix_alpha'],
                cutmix_prob=config['cutmix_prob'],
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
            attr_mix=config['attr_mix'],
            attr_temp=config['attr_temp'],
            use_foreground=bool(config.get('use_foreground_eval', False)),
            foreground_dir=os.path.join(config['data_root'], 'train_images', 'foreground'),
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
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'config': config
            }, config['save_path'])
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.4f})")
        print()
        
    print(f"Training Complete. Best Validation Accuracy: {best_val_acc:.4f}")

def main():
    # Orchestrate the training pipeline
    device = setup_environment(config)
    train_loader, val_loader, attr_vectors = prepare_data(config, device)
    model, optimizer, scheduler, criterion = setup_model_components(config, device)

    start_epoch, best_val_acc = resume_checkpoint(config, model, optimizer, scheduler, device)
    
    training_loop(
        config, model, train_loader, val_loader, 
        optimizer, scheduler, criterion, attr_vectors, device,
        start_epoch=start_epoch,
        best_val_acc=best_val_acc,
    )

if __name__ == '__main__':
    main()
