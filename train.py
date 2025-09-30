#!/usr/bin/env python3
"""
Simple Training Script for Surgical Scene Understanding
Uses only: ssg_dataset.py, model.py, training_utils.py

Usage:
    python train.py
"""

import torch
from torch.utils.data import DataLoader
import argparse
import logging
from pathlib import Path

from ssg_dataset import SSGVQA, multi_triplet_collate_fn
from model import SurgicalSceneGraphVQA
from training_utils import GNNTrainer
import config as cfg

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_dataloaders(batch_size=8, num_workers=0, use_debug=True):
    """Create train, val, test dataloaders"""
    logger = logging.getLogger(__name__)
    
    # Create datasets
    logger.info("Creating datasets...")
    if use_debug:
        logger.info("🐛 Using DEBUG datasets (smaller size for fast testing)")
        train_dataset = SSGVQA(mode="debug")
        val_dataset = SSGVQA(mode="debug") 
        test_dataset = SSGVQA(mode="debug")
    else:
        train_dataset = SSGVQA(mode="train")
        val_dataset = SSGVQA(mode="val") 
        test_dataset = SSGVQA(mode="test")
    
    logger.info(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=multi_triplet_collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multi_triplet_collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multi_triplet_collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    logger.info("Dataloaders created successfully")
    return train_loader, val_loader, test_loader

def train_model(gnn_type='gcn', num_epochs=20, batch_size=8, learning_rate=1e-4, 
                vqa_classes=51, save_dir='checkpoints', use_uncertainty_weights=True, 
                use_debug=True, resume_from=None):
    """Train the model"""
    logger = logging.getLogger(__name__)
    
    # Setup device
    device = cfg.DEVICE
    logger.info(f"Using device: {device}")
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(batch_size, use_debug=use_debug)
    
    # Initialize trainer
    logger.info(f"Initializing trainer with GNN type: {gnn_type}")
    logger.info(f"VQA classes: {vqa_classes}, Triplet classes: 101")
    logger.info(f"Using uncertainty weights: {use_uncertainty_weights}")
    
    # Debug mode always uses fresh start
    if use_debug and resume_from:
        logger.info("🐛 Debug mode: Ignoring resume_from, using fresh start")
        resume_from = None
    
    trainer = GNNTrainer(
        gnn_type=gnn_type,
        num_classes=vqa_classes,  # VQA classes (51)
        learning_rate=learning_rate,
        device=device,
        use_uncertainty_weights=use_uncertainty_weights
    )
    
    # Load checkpoint if specified
    start_epoch = 0
    if resume_from and not use_debug:
        start_epoch = trainer.load_checkpoint(resume_from)
        logger.info(f"📂 Resumed from checkpoint: {resume_from} (epoch {start_epoch})")
    
    # Train model
    logger.info("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        save_dir=save_path,
        start_epoch=start_epoch
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    best_model_path = save_path / f'best_{gnn_type}_model.pth'
    if best_model_path.exists():
        test_results = trainer.evaluate(
            test_loader=test_loader,
            model_path=str(best_model_path)
        )
        logger.info(f"Test Results: {test_results}")
    else:
        logger.warning("Best model not found, evaluating current model...")
        test_results = trainer.evaluate(test_loader=test_loader)
        logger.info(f"Test Results: {test_results}")
    
    return trainer, test_results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Surgical Scene Understanding Model')
    parser.add_argument('--gnn_type', type=str, default='gcn', 
                       choices=['none', 'gcn', 'gat', 'gin'],
                       help='GNN architecture type')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--vqa_classes', type=int, default=51,
                       help='Number of VQA answer classes')
    parser.add_argument('--use_uncertainty', action='store_true',
                       help='Use uncertainty-based loss weighting')
    parser.add_argument('--debug', action='store_true',
                       help='Use debug dataset (smaller size for fast testing)')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Path to checkpoint to resume from (ignored in debug mode)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("SURGICAL SCENE UNDERSTANDING TRAINING")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  GNN Type: {args.gnn_type}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Learning Rate: {args.lr}")
    logger.info(f"  VQA Classes: {args.vqa_classes}")
    logger.info(f"  Triplet Classes: 101")
    logger.info(f"  Use Uncertainty: {args.use_uncertainty}")
    logger.info(f"  Debug Mode: {args.debug}")
    logger.info(f"  Resume From: {args.resume_from if args.resume_from else 'None (fresh start)'}")
    logger.info(f"  Save Directory: {args.save_dir}")
    logger.info("=" * 60)
    
    try:
        # Train model
        trainer, test_results = train_model(
            gnn_type=args.gnn_type,
            num_epochs=args.epochs,
            batch_size=args.batch_size,  
            learning_rate=args.lr,
            vqa_classes=args.vqa_classes,
            use_uncertainty_weights=args.use_uncertainty,
            use_debug=args.debug,
            resume_from=args.resume_from,
            save_dir=args.save_dir
        )
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Final Test Results: {test_results}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()