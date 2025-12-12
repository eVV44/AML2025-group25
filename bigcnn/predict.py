"""
Generate predictions for test set using trained model.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from dataset import BirdDataset
from augmentation import get_test_transforms, get_test_transforms_tta
from birdmodels import BirdClassifier
from inference import predict_logits

def generate_predictions(model_path='best_bigcnn_model.pth', output_path=None, use_tta=True):
    use_tta_override = bool(use_tta)
    if output_path is None:
        timestamp = datetime.now().strftime('%m-%d %H_%M_%S')
        output_path = f'submissions/prediction_{timestamp}.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("Generating Test Predictions")
    print("="*70)
    print(f"\nModel: {model_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    
    # Create model
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    ckpt_config = checkpoint.get('config', {})

    # Configuration (prioritize checkpoint config, fallback to defaults)
    default_config = {
        'data_root': 'data',
        'num_classes': 200,
        'attribute_dim': 312,
        'embedding_dim': 512,
        'img_size': 384,
        'attr_temp': 10.0,
        'attr_mix': 0.3,
        'eval_batch_size': 32,
        'use_attr_pred_head': True,
        'use_attr_emb_head': True,
        'use_tta': use_tta_override,
        'classifier_scale': 30.0,
        'classifier_margin': 0.2,
        'tta_mode': 'ten_crop',
        'tta_scales': [1.0],
    }
    config = {k: ckpt_config.get(k, v) for k, v in default_config.items()}
    config['use_tta'] = use_tta_override
    use_tta = config['use_tta']

    print("Inference configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nCreating model...")
    model = BirdClassifier(
        num_classes=config['num_classes'],
        attribute_dim=config['attribute_dim'],
        embedding_dim=config['embedding_dim'],
        use_attr_pred=config['use_attr_pred_head'],
        use_attr_emb=config['use_attr_emb_head'],
        classifier_scale=config['classifier_scale'],
        classifier_margin=config['classifier_margin'],
    ).to(device)
    
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']} with val_acc={checkpoint['val_acc']:.4f}")
    
    # Load attributes
    print("Loading attributes...")
    attributes = np.load(os.path.join(config['data_root'], 'attributes.npy'))
    attribute_vectors = torch.FloatTensor(attributes).to(device)
    
    # Load test data
    print("\nPreparing test dataset...")
    test_transform = (
        get_test_transforms_tta(img_size=config['img_size'])
        if use_tta
        else get_test_transforms(img_size=config['img_size'])
    )

    test_dataset = BirdDataset(
        os.path.join(config['data_root'], 'test_images_path.csv'),
        transform=test_transform,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['eval_batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    all_predictions = []
    all_ids = []
    
    with torch.no_grad():
        for images, test_ids in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)

            logits = predict_logits(
                model,
                images,
                attribute_vectors,
                attr_temp=config['attr_temp'],
                attr_mix=config['attr_mix'],
                use_tta=use_tta,
                tta_crop_size=config['img_size'],
                tta_mode=config.get('tta_mode', 'ten_crop'),
                tta_scales=config.get('tta_scales', [1.0]),
            )
            
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_ids.extend(test_ids.cpu().numpy() if torch.is_tensor(test_ids) else test_ids)
    
    # Convert back to 1-indexed labels (as required by submission format)
    all_predictions = np.array(all_predictions) + 1
    all_ids = np.array(all_ids)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': all_ids,
        'label': all_predictions
    })
    
    # Save submission
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"Submission saved to: {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nSample predictions:")
    print(submission_df.head(10))
    print(f"\nLabel distribution:")
    print(f"  Min: {all_predictions.min()}")
    print(f"  Max: {all_predictions.max()}")
    print(f"  Unique classes: {len(np.unique(all_predictions))}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    # Generate predictions with best model
    generate_predictions(
        model_path='best_bigcnn_model.pth',
        use_tta=True  # Set to True for better accuracy (slower)
    )
