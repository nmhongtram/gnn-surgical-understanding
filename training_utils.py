#!/usr/bin/env python3
"""
Training Utilities for GNN Architecture Comparison
Support training and evaluation for: none, gcn, gat, gin architectures

Usage:
    from training_utils import GNNTrainer, setup_training, evaluate_model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import json
import logging
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, precision_score, recall_score
from tqdm import tqdm

from model import SurgicalSceneGraphVQA
import config as cfg


def compute_detailed_metrics(vqa_targets, vqa_predictions, triplet_targets, triplet_predictions, triplet_to_id_mapping):
    """
    Compute detailed metrics for VQA and triplet prediction tasks
    
    Args:
        vqa_targets: VQA ground truth labels (list)
        vqa_predictions: VQA predicted labels (list) 
        triplet_targets: Triplet ground truth (list of lists of triplet IDs)
        triplet_predictions: Triplet predictions (tensor: [batch, 101])
        triplet_to_id_mapping: Mapping from triplet components to IDs
        
    Returns:
        dict: Detailed metrics for both tasks
    """
    metrics = {}
    
    # =============================================================================
    # VQA METRICS
    # =============================================================================
    if vqa_targets and vqa_predictions:
        try:
            metrics['vqa_accuracy'] = accuracy_score(vqa_targets, vqa_predictions)
            metrics['vqa_precision'] = precision_score(vqa_targets, vqa_predictions, average='macro', zero_division=0)
            metrics['vqa_recall'] = recall_score(vqa_targets, vqa_predictions, average='macro', zero_division=0)
            metrics['vqa_f1_macro'] = f1_score(vqa_targets, vqa_predictions, average='macro', zero_division=0)
            metrics['vqa_f1_weighted'] = f1_score(vqa_targets, vqa_predictions, average='weighted', zero_division=0)
        except Exception as e:
            print(f"Warning: VQA metrics computation failed: {e}")
            metrics.update({
                'vqa_accuracy': 0.0, 'vqa_precision': 0.0, 'vqa_recall': 0.0, 
                'vqa_f1_macro': 0.0, 'vqa_f1_weighted': 0.0
            })
    
    # =============================================================================
    # TRIPLET PREDICTION METRICS
    # =============================================================================
    if triplet_targets and triplet_predictions is not None:
        try:
            # Convert triplet predictions to probabilities
            triplet_probs = torch.sigmoid(triplet_predictions).cpu().numpy()  # [batch, 101]
            batch_size = triplet_probs.shape[0]
            
            # Create binary ground truth matrix
            triplet_gt = np.zeros((batch_size, 101))
            for i, sample_triplets in enumerate(triplet_targets):
                for triplet_id in sample_triplets:
                    if isinstance(triplet_id, torch.Tensor):
                        triplet_id = triplet_id.item()
                    if 0 <= triplet_id < 101:  # Skip padding -1
                        triplet_gt[i, triplet_id] = 1
            
            # Load triplet component mapping from maps.txt
            triplet_components = {}
            try:
                with open(cfg.META_INFO_DIR / 'maps.txt', 'r') as f:
                    lines = f.readlines()[1:]  # Skip header
                    for line in lines:
                        if line.strip() and not line.startswith('#'):
                            parts = line.strip().split(',')
                            if len(parts) >= 4:
                                triplet_id = int(parts[0])
                                instrument_id = int(parts[1])
                                verb_id = int(parts[2])
                                target_id = int(parts[3])
                                triplet_components[triplet_id] = {
                                    'instrument': instrument_id,
                                    'verb': verb_id,
                                    'target': target_id
                                }
            except Exception as e:
                print(f"Warning: Could not load maps.txt: {e}")
                triplet_components = {}
            
            # Overall triplet mAP (IVT - complete triplets)
            valid_samples = triplet_gt.sum(axis=1) > 0  # Samples with at least one triplet
            if valid_samples.sum() > 0:
                metrics['triplet_mAP_IVT'] = average_precision_score(
                    triplet_gt[valid_samples], triplet_probs[valid_samples], average='macro'
                )
            else:
                metrics['triplet_mAP_IVT'] = 0.0
            
            # Component-wise mAP computation
            if triplet_components:
                # Initialize component ground truth matrices
                iv_gt = np.zeros((batch_size, 70))  # Max IV combinations
                it_gt = np.zeros((batch_size, 105)) # Max IT combinations  
                i_gt = np.zeros((batch_size, 7))    # Instrument classes
                v_gt = np.zeros((batch_size, 10))   # Verb classes
                t_gt = np.zeros((batch_size, 15))   # Target classes
                
                # Initialize component prediction matrices
                iv_probs = np.zeros((batch_size, 70))
                it_probs = np.zeros((batch_size, 105))
                i_probs = np.zeros((batch_size, 7))
                v_probs = np.zeros((batch_size, 10))
                t_probs = np.zeros((batch_size, 15))
                
                # Decompose triplets into components
                for i in range(batch_size):
                    for triplet_id in range(101):
                        if triplet_gt[i, triplet_id] == 1:
                            # Get triplet components using integer key
                            if triplet_id in triplet_components:
                                components = triplet_components[triplet_id]
                                inst_id, verb_id, target_id = components['instrument'], components['verb'], components['target']
                                
                                # Set ground truth
                                if inst_id < 7: i_gt[i, inst_id] = 1
                                if verb_id < 10: v_gt[i, verb_id] = 1  
                                if target_id < 15: t_gt[i, target_id] = 1
                                
                                # IV combination (simplified mapping)
                                iv_idx = min(inst_id * 10 + verb_id, 69)
                                iv_gt[i, iv_idx] = 1
                                
                                # IT combination (simplified mapping)
                                it_idx = min(inst_id * 15 + target_id, 104)
                                it_gt[i, it_idx] = 1
                        
                        # Aggregate predictions for components
                        prob = triplet_probs[i, triplet_id]
                        if prob > 0.01 and triplet_id in triplet_components:  # Only consider significant predictions
                            components = triplet_components[triplet_id]
                            inst_id, verb_id, target_id = components['instrument'], components['verb'], components['target']
                            
                            # Accumulate probabilities (use max pooling)
                            if inst_id < 7: i_probs[i, inst_id] = max(i_probs[i, inst_id], prob)
                            if verb_id < 10: v_probs[i, verb_id] = max(v_probs[i, verb_id], prob)
                            if target_id < 15: t_probs[i, target_id] = max(t_probs[i, target_id], prob)
                            
                            iv_idx = min(inst_id * 10 + verb_id, 69)
                            iv_probs[i, iv_idx] = max(iv_probs[i, iv_idx], prob)
                            
                            it_idx = min(inst_id * 15 + target_id, 104)
                            it_probs[i, it_idx] = max(it_probs[i, it_idx], prob)
                
                # Compute component mAPs
                def safe_map(gt, pred):
                    valid = gt.sum(axis=1) > 0
                    if valid.sum() > 0:
                        return average_precision_score(gt[valid], pred[valid], average='macro')
                    return 0.0
                
                metrics['triplet_mAP_IV'] = safe_map(iv_gt, iv_probs)
                metrics['triplet_mAP_IT'] = safe_map(it_gt, it_probs) 
                metrics['triplet_mAP_I'] = safe_map(i_gt, i_probs)
                metrics['triplet_mAP_V'] = safe_map(v_gt, v_probs)
                metrics['triplet_mAP_T'] = safe_map(t_gt, t_probs)
            else:
                # Fallback values if triplet_components.json not available
                metrics.update({
                    'triplet_mAP_IV': 0.0, 'triplet_mAP_IT': 0.0,
                    'triplet_mAP_I': 0.0, 'triplet_mAP_V': 0.0, 'triplet_mAP_T': 0.0
                })
                
        except Exception as e:
            print(f"Warning: Triplet metrics computation failed: {e}")
            metrics.update({
                'triplet_mAP_IVT': 0.0, 'triplet_mAP_IV': 0.0, 'triplet_mAP_IT': 0.0,
                'triplet_mAP_I': 0.0, 'triplet_mAP_V': 0.0, 'triplet_mAP_T': 0.0
            })
    
    return metrics


class GNNTrainer:
    """
    Unified trainer for all GNN architectures comparison
    Supports: none (baseline), gcn, gat, gin
    """
    
    def __init__(self, gnn_type, num_classes=100, learning_rate=1e-4, 
                 device=None,
                 use_uncertainty_weights=False):
        """
        Initialize trainer for specific GNN architecture
        
        Args:
            gnn_type: 'none', 'gcn', 'gat', 'gin'
            num_classes: Number of VQA answer classes
            learning_rate: Learning rate for optimizer
            device: Training device (if None, uses config.DEVICE)
            use_uncertainty_weights: Whether to use uncertainty-based loss weighting
        """
        self.gnn_type = gnn_type
        self.device = device if device is not None else cfg.DEVICE
        self.num_classes = num_classes
        self.use_uncertainty_weights = use_uncertainty_weights
        
        # Initialize model (disable triplet detection for debug)
        self.model = SurgicalSceneGraphVQA(
            num_classes=num_classes,
            gnn_type=gnn_type,
            hidden_dim=768,  # Use enhanced hidden_dim
            use_uncertainty_weights=use_uncertainty_weights,
            enable_triplet_detection=True   # Multi-task loss fixed!
        ).to(device)
        
        # Setup optimizer with stronger weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-3  # Increased from 1e-5 to 1e-3 to reduce overfitting
        )
        
        # Setup scheduler with more aggressive learning rate reduction
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.3, patience=5  # More aggressive: factor=0.3, patience=5
        )
        
        # Metrics tracking
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_metrics = {}
        
        # Checkpoint tracking
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_acc = 0.0
        
        print(f"✅ {gnn_type.upper()} Trainer initialized")
        print(f"   Device: {device}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Trainable: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            int: Start epoch number
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                print(f"❌ Checkpoint not found: {checkpoint_path}")
                return 0
            
            print(f"📂 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load model state - try full model first, fallback to state_dict
            if 'model' in checkpoint:
                print("🔧 Loading full model object")
                self.model = checkpoint['model'].to(self.device)
            else:
                print("🔧 Loading model from state_dict")
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training state
            start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
            self.best_val_loss = checkpoint.get('val_loss', float('inf'))
            self.best_metrics = checkpoint.get('metrics', {})
            
            # Load history if available
            if 'train_history' in checkpoint:
                self.train_history = checkpoint['train_history']
            if 'val_history' in checkpoint:
                self.val_history = checkpoint['val_history']
            
            print(f"✅ Checkpoint loaded successfully")
            print(f"   Starting from epoch: {start_epoch}")
            print(f"   Best validation loss: {self.best_val_loss:.4f}")
            print(f"   Best metrics: {self.best_metrics}")
            
            return start_epoch
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print(f"   Starting fresh training...")
            return 0
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = defaultdict(list)
        
        # Create progress bar
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", 
                   leave=False, ncols=100, colour='green')
        
        for batch_idx, batch_data in enumerate(pbar):
            # Move data to device
            graph_data, questions, vqa_labels, triplet_labels = batch_data
            graph_data = graph_data.to(self.device)
            vqa_labels = vqa_labels.to(self.device)
            
            # Move triplet labels to device - triplet_labels is now a batched tensor [batch_size, max_triplets]
            if isinstance(triplet_labels, torch.Tensor):
                triplet_labels = triplet_labels.to(self.device)
                # Convert batched tensor back to list of lists for model compatibility
                triplet_labels_device = []
                for i in range(triplet_labels.shape[0]):
                    sample_triplets = triplet_labels[i]
                    # Convert to list, filtering out padding values (-1)
                    valid_triplets = sample_triplets[sample_triplets != -1].tolist()
                    triplet_labels_device.append(valid_triplets)
            else:
                # Fallback for old format
                triplet_labels_device = []
                for sample_triplets in triplet_labels:
                    if isinstance(sample_triplets, torch.Tensor):
                        triplet_labels_device.append(sample_triplets.to(self.device))
                    else:
                        triplet_labels_device.append(sample_triplets)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(graph_data, questions)
            
            # Multi-task loss with enhanced class imbalance handling
            if isinstance(outputs, dict):
                # Choose loss computation method based on configuration
                if self.use_uncertainty_weights and hasattr(self.model, 'compute_uncertainty_weighted_loss'):
                    # Use uncertainty-based loss weighting
                    losses = self.model.compute_uncertainty_weighted_loss(
                        outputs, vqa_labels, triplet_labels_device
                    )
                else:
                    # Use direct triplet loss
                    losses = self.model.compute_multi_task_loss(
                        outputs, vqa_labels, triplet_labels_device,
                        vqa_weight=0.7, direct_triplet_weight=0.3,
                        use_multi_label=True
                    )
                total_loss = losses['total']
            else:
                # Fallback to VQA-only for backward compatibility
                vqa_loss = nn.CrossEntropyLoss()(outputs, vqa_labels)
                total_loss = vqa_loss
                losses = {'vqa': vqa_loss.item(), 'total': vqa_loss.item()}
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track losses (skip uncertainty_params as it's a dict)
            for key, loss in losses.items():
                if key == 'uncertainty_params':
                    continue  # Skip dict parameters, they're not losses
                if isinstance(loss, torch.Tensor):
                    epoch_losses[key].append(loss.item())
                else:
                    epoch_losses[key].append(loss)  # Already a float
            
            # Update progress bar with current loss and weights
            if self.use_uncertainty_weights and 'uncertainty_params' in losses:
                uncertainty_params = losses['uncertainty_params']
                vqa_precision = 1.0 / uncertainty_params['vqa_sigma2']
                triplet_precision = 1.0 / uncertainty_params['triplet_sigma2']
                total_precision = vqa_precision + triplet_precision
                vqa_weight = vqa_precision / total_precision
                triplet_weight = triplet_precision / total_precision
                
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'VQA_w': f'{vqa_weight:.2f}',
                    'Triplet_w': f'{triplet_weight:.2f}'
                })
            else:
                pbar.set_postfix({'Loss': f'{total_loss.item():.4f}'})
        
        pbar.close()
        
        # Average losses for epoch
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        # Update history
        for key, value in avg_losses.items():
            self.train_history[f'train_{key}'].append(value)
        
        return avg_losses
    
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch with detailed metrics"""
        self.model.eval()
        epoch_losses = defaultdict(list)
        
        # Reset metrics tracking
        vqa_predictions = []
        vqa_targets = []
        triplet_predictions = []
        triplet_targets = []
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Move data to device
                graph_data, questions, vqa_labels, triplet_labels = batch_data
                graph_data = graph_data.to(self.device)
                vqa_labels = vqa_labels.to(self.device)
                
                # Move triplet labels to device - triplet_labels is now a batched tensor [batch_size, max_triplets]
                if isinstance(triplet_labels, torch.Tensor):
                    triplet_labels = triplet_labels.to(self.device)
                    # Convert batched tensor back to list of lists for model compatibility
                    triplet_labels_device = []
                    for i in range(triplet_labels.shape[0]):
                        sample_triplets = triplet_labels[i]
                        # Convert to list, filtering out padding values (-1)
                        valid_triplets = sample_triplets[sample_triplets != -1].tolist()
                        triplet_labels_device.append(valid_triplets)
                else:
                    # Fallback for old format
                    triplet_labels_device = []
                    for sample_triplets in triplet_labels:
                        if isinstance(sample_triplets, torch.Tensor):
                            triplet_labels_device.append(sample_triplets.to(self.device))
                        else:
                            triplet_labels_device.append(sample_triplets)
                
                # Forward pass
                outputs = self.model(graph_data, questions)
                
                # Multi-task loss (VQA + Triplet + Constraints)
                if isinstance(outputs, dict):
                    # Use direct triplet loss computation
                    if self.use_uncertainty_weights and hasattr(self.model, 'compute_uncertainty_weighted_loss'):
                        losses = self.model.compute_uncertainty_weighted_loss(outputs, vqa_labels, triplet_labels_device)
                    else:
                        losses = self.model.compute_multi_task_loss(outputs, vqa_labels, triplet_labels_device)
                else:
                    # Fallback to VQA-only for backward compatibility
                    vqa_loss = nn.CrossEntropyLoss()(outputs, vqa_labels)
                    losses = {'vqa': vqa_loss.item(), 'total': vqa_loss.item()}
                
                # Track losses (skip uncertainty_params as it's a dict)
                for key, loss in losses.items():
                    if key == 'uncertainty_params':
                        continue  # Skip dict parameters, they're not losses
                    if isinstance(loss, torch.Tensor):
                        epoch_losses[key].append(loss.item())
                    else:
                        epoch_losses[key].append(loss)  # Already a float
                
                # Track predictions for detailed metrics
                if isinstance(outputs, dict) and 'vqa' in outputs:
                    vqa_preds = torch.argmax(outputs['vqa'], dim=1).cpu().numpy()
                    vqa_targets.extend(vqa_labels.cpu().numpy())
                    vqa_predictions.extend(vqa_preds)
                    
                    # Track triplet predictions
                    if 'direct_triplets' in outputs:
                        triplet_preds = outputs['direct_triplets'].cpu()  # [batch, 101]
                        triplet_predictions.append(triplet_preds)
                        
                        # Convert triplet targets to list format
                        batch_triplet_targets = []
                        for sample_triplets in triplet_labels:
                            if isinstance(sample_triplets, torch.Tensor):
                                batch_triplet_targets.append(sample_triplets.cpu().tolist())
                            else:
                                batch_triplet_targets.append(sample_triplets)
                        triplet_targets.extend(batch_triplet_targets)
                        
                elif not isinstance(outputs, dict):
                    vqa_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    vqa_targets.extend(vqa_labels.cpu().numpy())
                    vqa_predictions.extend(vqa_preds)
        
        # Average losses for epoch
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        # Compute detailed metrics
        try:
            # Concatenate triplet predictions
            if triplet_predictions:
                triplet_predictions_tensor = torch.cat(triplet_predictions, dim=0)
            else:
                triplet_predictions_tensor = None
                
            # Compute detailed metrics using the new function
            all_metrics = compute_detailed_metrics(
                vqa_targets=vqa_targets,
                vqa_predictions=vqa_predictions,
                triplet_targets=triplet_targets,
                triplet_predictions=triplet_predictions_tensor,
                triplet_to_id_mapping=getattr(self.model, 'triplet_to_id', {})
            )
        except Exception as e:
            print(f"Warning: Detailed metrics computation failed: {e}")
            # Fallback to basic metrics
            all_metrics = {}
            if vqa_predictions:
                all_metrics['vqa_accuracy'] = accuracy_score(vqa_targets, vqa_predictions)
                all_metrics['vqa_f1_weighted'] = f1_score(vqa_targets, vqa_predictions, average='weighted')
        
        # Update history
        for key, value in avg_losses.items():
            self.val_history[f'val_{key}'].append(value)
        
        for key, value in all_metrics.items():
            self.val_history[f'val_{key}'].append(value)
        
        return avg_losses, all_metrics
    
    def train(self, train_loader, val_loader, num_epochs=20, save_dir=None, start_epoch=0):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader  
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            start_epoch: Epoch to start from (for resuming)
        """
        print(f"\n🚀 Starting {self.gnn_type.upper()} Training")
        print("=" * 50)
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            
            print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            print("-" * 30)
            
            # Training
            print("🏋️  Training...")
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Validation
            print("🔍 Validating...")
            val_losses, val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_losses['total'])
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\n📊 Epoch {epoch+1} Summary ({epoch_time:.1f}s):")
            print(f"   Train Loss: {train_losses['total']:.4f}")
            if 'vqa' in train_losses:
                print(f"     VQA: {train_losses['vqa']:.4f} | Triplet: {train_losses.get('triplet', 0):.4f}")
            print(f"   Val Loss:   {val_losses['total']:.4f}")
            if 'vqa' in val_losses:
                print(f"     VQA: {val_losses['vqa']:.4f} | Triplet: {val_losses.get('triplet', 0):.4f}")
            
            # Log uncertainty parameters if using uncertainty weighting
            if self.use_uncertainty_weights and 'uncertainty_params' in train_losses:
                uncertainty_params = train_losses['uncertainty_params']
                
                # Calculate precision weights (1/σ²)
                vqa_precision = 1.0 / uncertainty_params['vqa_sigma2']
                triplet_precision = 1.0 / uncertainty_params['triplet_sigma2']
                
                # Calculate relative weights (normalized)
                total_precision = vqa_precision + triplet_precision
                vqa_relative_weight = vqa_precision / total_precision
                triplet_relative_weight = triplet_precision / total_precision
                
                print(f"   🎯 Uncertainty Weighting:")
                print(f"     VQA σ²:           {uncertainty_params['vqa_sigma2']:.4f}")
                print(f"     Triplet σ²:       {uncertainty_params['triplet_sigma2']:.4f}")
                print(f"     VQA Weight:       {vqa_relative_weight:.3f} (precision: {vqa_precision:.2f})")
                print(f"     Triplet Weight:   {triplet_relative_weight:.3f} (precision: {triplet_precision:.2f})")
                
                # Show raw vs weighted losses
                if 'vqa' in train_losses and 'vqa_weighted' in train_losses:
                    print(f"   📊 Loss Analysis:")
                    print(f"     VQA Raw:          {train_losses['vqa']:.4f}")
                    print(f"     VQA Weighted:     {train_losses['vqa_weighted']:.4f}")
                    print(f"     Triplet Raw:      {train_losses.get('triplet', 0):.4f}")
                    print(f"     Triplet Weighted: {train_losses.get('triplet_weighted', 0):.4f}")
            
            # VQA Metrics
            print(f"   📊 VQA Metrics:")
            print(f"     Accuracy:     {val_metrics.get('vqa_accuracy', 0):.3f}")
            print(f"     Precision:    {val_metrics.get('vqa_precision', 0):.3f}")
            print(f"     Recall:       {val_metrics.get('vqa_recall', 0):.3f}")
            print(f"     F1-Macro:     {val_metrics.get('vqa_f1_macro', 0):.3f}")
            print(f"     F1-Weighted:  {val_metrics.get('vqa_f1_weighted', 0):.3f}")
            
            # Triplet Metrics
            print(f"   🔺 Triplet Metrics:")
            print(f"     mAP-IVT:      {val_metrics.get('triplet_mAP_IVT', 0):.3f}")
            print(f"     mAP-IV:       {val_metrics.get('triplet_mAP_IV', 0):.3f}")
            print(f"     mAP-IT:       {val_metrics.get('triplet_mAP_IT', 0):.3f}")
            print(f"     mAP-I:        {val_metrics.get('triplet_mAP_I', 0):.3f}")
            print(f"     mAP-V:        {val_metrics.get('triplet_mAP_V', 0):.3f}")
            print(f"     mAP-T:        {val_metrics.get('triplet_mAP_T', 0):.3f}")
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.best_metrics = val_metrics.copy()
                
                if save_dir:
                    best_path = save_dir / f"best_{self.gnn_type}_model.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_losses['total'],
                        'metrics': val_metrics,
                        'train_history': dict(self.train_history),
                        'val_history': dict(self.val_history),
                        'gnn_type': self.gnn_type,
                        'model_config': {
                            'num_classes': self.num_classes,
                            'use_uncertainty_weights': self.use_uncertainty_weights
                        }
                    }, best_path)
                    print(f"💾 Best model saved: {best_path}")
            
            # Save checkpoint every 5 epochs
            if save_dir and (epoch + 1) % 5 == 0:
                checkpoint_path = save_dir / f"checkpoint_{self.gnn_type}_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_history': dict(self.train_history),
                    'val_history': dict(self.val_history),
                    'gnn_type': self.gnn_type,
                    'model_config': {
                        'num_classes': self.num_classes,
                        'use_uncertainty_weights': self.use_uncertainty_weights
                    }
                }, checkpoint_path)
        
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time/60:.1f} minutes")
        print(f"🏆 Best validation loss: {self.best_val_loss:.4f}")
        print(f"🎯 Best metrics: VQA={self.best_metrics.get('vqa_accuracy', 0):.3f}, "
              f"Triplet={self.best_metrics.get('complete_triplet_accuracy', 0):.3f}")
        
        return self.best_metrics
    
    def evaluate(self, test_loader, model_path=None):
        """
        Evaluate model on test set
        
        Args:
            test_loader: Test data loader
            model_path: Path to saved model (optional)
        """
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            # Try to load full model first, fallback to state_dict
            if 'model' in checkpoint:
                print("🔧 Loading full model object for evaluation")
                self.model = checkpoint['model'].to(self.device)
            else:
                print("🔧 Loading model from state_dict for evaluation")
                self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📂 Loaded model from {model_path}")
        
        self.model.eval()
        test_losses = defaultdict(list)
        vqa_predictions = []
        vqa_targets = []
        
        print(f"\n🔬 Evaluating {self.gnn_type.upper()} on test set...")
        
        with torch.no_grad():
            for batch_data in test_loader:
                # Move data to device
                graph_data, questions, vqa_labels, triplet_labels = batch_data
                graph_data = graph_data.to(self.device)
                vqa_labels = vqa_labels.to(self.device)
                
                # Move triplet labels to device - triplet_labels is now a batched tensor [batch_size, max_triplets]
                if isinstance(triplet_labels, torch.Tensor):
                    triplet_labels = triplet_labels.to(self.device)
                    # Convert batched tensor back to list of lists for model compatibility
                    triplet_labels_device = []
                    for i in range(triplet_labels.shape[0]):
                        sample_triplets = triplet_labels[i]
                        # Convert to list, filtering out padding values (-1)
                        valid_triplets = sample_triplets[sample_triplets != -1].tolist()
                        triplet_labels_device.append(valid_triplets)
                else:
                    # Fallback for old format
                    triplet_labels_device = []
                    for sample_triplets in triplet_labels:
                        if isinstance(sample_triplets, torch.Tensor):
                            triplet_labels_device.append(sample_triplets.to(self.device))
                        else:
                            triplet_labels_device.append(sample_triplets)
                
                # Forward pass
                outputs = self.model(graph_data, questions)
                
                # Compute losses
                if self.use_uncertainty_weights and hasattr(self.model, 'compute_uncertainty_weighted_loss'):
                    losses = self.model.compute_uncertainty_weighted_loss(outputs, vqa_labels, triplet_labels_device)
                else:
                    losses = self.model.compute_multi_task_loss(outputs, vqa_labels, triplet_labels_device)
                
                # Track losses
                for key, loss in losses.items():
                    if key == 'uncertainty_params':
                        continue  # Skip dict parameters, they're not losses
                    if isinstance(loss, torch.Tensor):
                        test_losses[key].append(loss.item())
                    else:
                        test_losses[key].append(loss)  # Already a float
                
                # Track VQA predictions for accuracy calculation
                if isinstance(outputs, dict) and 'vqa' in outputs:
                    vqa_preds = torch.argmax(outputs['vqa'], dim=1).cpu().numpy()
                    vqa_targets.extend(vqa_labels.cpu().numpy())
                    vqa_predictions.extend(vqa_preds)
                elif not isinstance(outputs, dict):
                    vqa_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    vqa_targets.extend(vqa_labels.cpu().numpy())
                    vqa_predictions.extend(vqa_preds)
        
        # Compute final metrics
        avg_losses = {key: np.mean(values) for key, values in test_losses.items()}
        test_metrics = {}
        if vqa_predictions:
            test_metrics['vqa_accuracy'] = accuracy_score(vqa_targets, vqa_predictions)
            test_metrics['vqa_f1_weighted'] = f1_score(vqa_targets, vqa_predictions, average='weighted')
        
        print(f"📊 Test Results for {self.gnn_type.upper()}:")
        print(f"   Test Loss: {avg_losses['total']:.4f}")
        print(f"   VQA Accuracy: {test_metrics.get('vqa_accuracy', 0):.3f}")
        print(f"   VQA F1 (Weighted): {test_metrics.get('vqa_f1_weighted', 0):.3f}")

        
        return avg_losses, test_metrics


def setup_training(architectures=['none', 'gcn', 'gat', 'gin'], 
                  num_classes=100, learning_rate=1e-4, device=None):
    """
    Setup trainers for multiple architectures
    
    Args:
        architectures: List of GNN types to train
        num_classes: Number of VQA classes
        learning_rate: Learning rate
        device: Training device
    
    Returns:
        Dictionary of trainers
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainers = {}
    
    print(f"🛠️  Setting up trainers for {len(architectures)} architectures")
    print(f"   Device: {device}")
    print(f"   Architectures: {architectures}")
    
    for arch in architectures:
        try:
            trainer = GNNTrainer(
                gnn_type=arch,
                num_classes=num_classes,
                learning_rate=learning_rate,
                device=device
            )
            trainers[arch] = trainer
        except Exception as e:
            print(f"❌ Failed to setup {arch} trainer: {e}")
    
    print(f"✅ Successfully setup {len(trainers)} trainers")
    return trainers


def compare_architectures(trainers, train_loader, val_loader, test_loader, 
                         num_epochs=20, save_dir="checkpoints/comparison"):
    """
    Train and compare multiple architectures
    
    Args:
        trainers: Dictionary of trainers
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        num_epochs: Number of epochs
        save_dir: Save directory
    
    Returns:
        Comparison results
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print(f"\n🏁 Starting Architecture Comparison")
    print(f"   Architectures: {list(trainers.keys())}")
    print(f"   Epochs: {num_epochs}")
    print("=" * 60)
    
    # Train each architecture
    for arch_name, trainer in trainers.items():
        print(f"\n🚀 Training {arch_name.upper()}")
        
        arch_save_dir = save_dir / arch_name
        
        try:
            # Train
            best_metrics = trainer.train(
                train_loader, val_loader, 
                num_epochs=num_epochs, 
                save_dir=arch_save_dir
            )
            
            # Test
            test_losses, test_metrics = trainer.evaluate(
                test_loader, 
                model_path=arch_save_dir / f"best_{arch_name}_model.pt"
            )
            
            # Store results
            results[arch_name] = {
                'best_val_metrics': best_metrics,
                'test_losses': test_losses,
                'test_metrics': test_metrics,
                'train_history': dict(trainer.train_history),
                'val_history': dict(trainer.val_history)
            }
            
        except Exception as e:
            print(f"❌ {arch_name.upper()} training failed: {e}")
            results[arch_name] = {'error': str(e)}
    
    # Save comparison results
    results_path = save_dir / "comparison_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to python types for JSON serialization
        json_results = {}
        for arch, data in results.items():
            if 'error' not in data:
                json_results[arch] = {
                    'test_metrics': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                   for k, v in data['test_metrics'].items()},
                    'test_losses': {k: float(v) for k, v in data['test_losses'].items()}
                }
            else:
                json_results[arch] = data
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_path}")
    
    # Print comparison summary
    print_comparison_summary(results)
    
    return results


def print_comparison_summary(results):
    """Print formatted comparison summary"""
    print(f"\n📊 ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 70)
    
    # Filter successful results
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not successful_results:
        print("❌ No successful training runs to compare")
        return
    
    # Print header
    print(f"{'Architecture':<12} {'VQA Acc':<10} {'Triplet Acc':<12} {'Test Loss':<10} {'Status':<10}")
    print("-" * 70)
    
    # Sort by VQA accuracy (descending)
    sorted_results = sorted(
        successful_results.items(),
        key=lambda x: x[1]['test_metrics'].get('vqa_accuracy', 0),
        reverse=True
    )
    
    for arch_name, data in sorted_results:
        vqa_acc = data['test_metrics'].get('vqa_accuracy', 0)
        triplet_acc = data['test_metrics'].get('complete_triplet_accuracy', 0)
        test_loss = data['test_losses'].get('total', 0)
        
        print(f"{arch_name.upper():<12} {vqa_acc:<10.3f} {triplet_acc:<12.3f} {test_loss:<10.3f} {'✅':<10}")
    
    # Print failed runs
    failed_results = {k: v for k, v in results.items() if 'error' in v}
    for arch_name, data in failed_results.items():
        print(f"{arch_name.upper():<12} {'N/A':<10} {'N/A':<12} {'N/A':<10} {'❌':<10}")
    
    # Best architecture
    if sorted_results:
        best_arch, best_data = sorted_results[0]
        print(f"\n🏆 Best Architecture: {best_arch.upper()}")
        print(f"   VQA Accuracy: {best_data['test_metrics'].get('vqa_accuracy', 0):.3f}")
        print(f"   Triplet Accuracy: {best_data['test_metrics'].get('complete_triplet_accuracy', 0):.3f}")
        print(f"   Test Loss: {best_data['test_losses'].get('total', 0):.3f}")


def evaluate_model(model_path, test_loader, device=None):
    """
    Evaluate a single saved model
    
    Args:
        model_path: Path to saved model
        test_loader: Test data loader
        device: Evaluation device
    
    Returns:
        Test metrics
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    gnn_type = checkpoint.get('gnn_type', 'gcn')
    
    # Create trainer and evaluate
    trainer = GNNTrainer(gnn_type=gnn_type, device=device)
    test_losses, test_metrics = trainer.evaluate(test_loader, model_path)
    
    return test_losses, test_metrics


class CheckpointManager:
    """Manages model checkpoints and dataset caching"""
    
    @staticmethod
    def save_checkpoint(trainer, checkpoint_dir, epoch, step, save_name=None):
        """Save model checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if save_name is None:
            save_name = f"{trainer.gnn_type}_epoch_{epoch}_step_{step}.pth"
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'best_val_loss': trainer.best_val_loss,
            'best_val_acc': trainer.best_val_acc,
            'train_history': dict(trainer.train_history),
            'val_history': dict(trainer.val_history),
            'gnn_type': trainer.gnn_type,
            'model_config': {
                'num_classes': trainer.num_classes,
                'use_uncertainty_weights': trainer.use_uncertainty_weights
            }
        }
        
        checkpoint_path = checkpoint_dir / save_name
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    @staticmethod
    def load_checkpoint(trainer, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        
        # Load model from state_dict (reliable method)
        print("🔧 Loading model from state_dict")
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        trainer.current_epoch = checkpoint['epoch']
        trainer.current_step = checkpoint['step']
        trainer.best_val_loss = checkpoint['best_val_loss']
        trainer.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        trainer.train_history = defaultdict(list, checkpoint.get('train_history', {}))
        trainer.val_history = defaultdict(list, checkpoint.get('val_history', {}))
        
        print(f"📂 Checkpoint loaded: {checkpoint_path}")
        print(f"   Resuming from epoch {trainer.current_epoch}, step {trainer.current_step}")
        return checkpoint
    
    @staticmethod
    def save_dataset_cache(datasets, cache_dir):
        """Save preprocessed datasets as .pt files"""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, dataset in datasets.items():
            cache_path = cache_dir / f"{split_name}_dataset.pth"
            torch.save(dataset, cache_path)
            print(f"💾 Dataset cache saved: {cache_path}")
        
        return cache_dir
    
    @staticmethod
    def load_dataset_cache(cache_dir):
        """Load preprocessed datasets from .pth files"""
        cache_dir = Path(cache_dir)
        datasets = {}
        
        for split in ['train', 'val', 'test']:
            cache_path = cache_dir / f"{split}_dataset.pth"
            if cache_path.exists():
                datasets[split] = torch.load(cache_path)
                print(f"📂 Dataset cache loaded: {cache_path}")
        
        return datasets


if __name__ == "__main__":
    # Example usage
    print("🧪 Training Utilities for GNN Architecture Comparison")
    print("Usage:")
    print("  from training_utils import setup_training, compare_architectures")
    print("  trainers = setup_training(['none', 'gcn', 'gat', 'gin'])")
    print("  results = compare_architectures(trainers, train_loader, val_loader, test_loader)")