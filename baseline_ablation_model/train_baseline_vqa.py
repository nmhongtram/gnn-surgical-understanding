#!/usr/bin/env python3
"""
Baseline VQA Training Script with Mixed Precision Support

Training script for Full Frame Baseline Model (ablation study)
Uses only global frame features without graph structure or GNN processing
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler  # ✨ Mixed Precision imports
from tqdm import tqdm
import time

import config as cfg

# Configure tqdm for Kaggle environment
tqdm.pandas()
os.environ['TQDM_DISABLE'] = '0'

# Progress bar settings
USE_TQDM = True
MANUAL_PROGRESS_INTERVAL = 10

# Local imports
from baseline_dataset import FullFrameBaselineDataset, baseline_collate_fn
from full_frame_baseline_model import create_full_frame_baseline_model
from collections import defaultdict
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report
from scipy.stats import chisquare
import re


def create_progress_bar(iterable, desc, leave=False):
    """Create progress bar with fallback to manual progress"""
    if USE_TQDM:
        return tqdm(iterable, desc=desc, leave=leave, dynamic_ncols=True, 
                   ascii=True, file=sys.stdout, miniters=1, mininterval=0.1)
    else:
        return iterable


def train_baseline_vqa(resume_from=None, use_mixed_precision=True):
    """Train Baseline VQA model with Mixed Precision support
    
    Args:
        resume_from (str): Path to checkpoint file to resume from
        use_mixed_precision (bool): Enable Automatic Mixed Precision training
    """
    
    print("🚀 Baseline VQA Training (Ablation Study)")
    print("📊 No Graph Structure | No GNN Processing | Global Frame Features Only")
    print("=" * 70)
    
    # Setup
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 🔥 Mixed Precision Setup
    if use_mixed_precision and device.type == 'cuda':
        # Check if GPU supports mixed precision
        gpu_name = torch.cuda.get_device_name(0)
        has_tensor_cores = 'T4' in gpu_name or 'V100' in gpu_name or 'A100' in gpu_name or 'RTX' in gpu_name
        
        print(f"GPU: {gpu_name}")
        print(f"Tensor Cores: {'✅ Available' if has_tensor_cores else '❌ Not available (P100 etc.)'}")
        
        if has_tensor_cores:
            print("🚀 Mixed Precision Training: ENABLED (with speed boost)")
        else:
            print("💾 Mixed Precision Training: ENABLED (memory savings only)")
            
        scaler = GradScaler()
    else:
        use_mixed_precision = False
        scaler = None
        print("❌ Mixed Precision Training: DISABLED")
    
    # Training config
    # 🎯 Increase batch size if using mixed precision (more memory available)
    BATCH_SIZE = 128 if use_mixed_precision else 64  # ✨ Larger batch with mixed precision
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 10
    GRAD_CLIP_NORM = 1.0
    
    print(f"Batch size: {BATCH_SIZE} {'(increased due to mixed precision)' if use_mixed_precision and BATCH_SIZE > 64 else ''}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Gradient clipping: {GRAD_CLIP_NORM}")
    
    if resume_from:
        print(f"Resume from: {resume_from}")
    
    # Load datasets
    print("\n📁 Loading baseline datasets...")
    train_dataset = FullFrameBaselineDataset(mode="train")
    val_dataset = FullFrameBaselineDataset(mode="val")
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Memory usage: ~{train_dataset.get_memory_usage():.1f}MB (frame features)")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=baseline_collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=baseline_collate_fn,
        num_workers=2
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create baseline model
    print("\n🤖 Creating full frame baseline model...")
    model_config = {
        'frame_feature_dim': 512,  # Global frame feature dimension
        'hidden_dim': 768,
        'question_dim': 768,
        'num_classes': 50,
        'bert_model_name': "emilyalsentzer/Bio_ClinicalBERT",
        'num_transformer_layers': 3,
        'dropout_prob': 0.1
    }
    model = create_full_frame_baseline_model(**model_config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    print(f"Frozen parameters: {frozen_params:,} (BERT)")
    print(f"🔬 Model type: Baseline (no GNN, global frame features only)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=2, verbose=True, min_lr=1e-8
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_acc = 0.0
    best_val_loss = 100.0
    
    if resume_from and os.path.exists(resume_from):
        print(f"\n📂 Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('val_acc', 0.0)
        best_val_loss = checkpoint.get('val_loss', 100.0)
        
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 🔥 Resume scaler state for mixed precision
        if use_mixed_precision and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("✅ Mixed precision scaler state resumed")
        
        print(f"✅ Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
        
        if hasattr(scheduler, 'num_bad_epochs'):
            print(f"📊 Scheduler state: patience={scheduler.num_bad_epochs}/{scheduler.patience}, best={scheduler.best:.4f}")
    elif resume_from:
        print(f"⚠️ Checkpoint file not found: {resume_from}")
        print("Starting training from scratch...")
    
    # Training loop
    print(f"\n🏋️ Starting baseline training from epoch {start_epoch + 1}...")
    
    # 📊 Track mixed precision statistics
    if use_mixed_precision:
        mp_stats = {'scale_updates': 0, 'skipped_steps': 0, 'total_steps': 0}
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 20)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = create_progress_bar(enumerate(train_loader), desc="Training", leave=False)
        
        # 📊 Track training time for mixed precision comparison
        epoch_start_time = time.time()
        
        for batch_idx, batch in train_pbar:
            
            # Extract data from baseline batch format
            frame_features = batch['frame_features'].to(device)  # [B, 512]
            questions_batch = {
                'input_ids': batch['questions_batch']['input_ids'].to(device),
                'attention_mask': batch['questions_batch']['attention_mask'].to(device),
                'token_type_ids': batch['questions_batch']['token_type_ids'].to(device)
            }
            vqa_labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # 🚀 Mixed Precision Forward Pass
            if use_mixed_precision:
                with autocast():
                    outputs = model(frame_features, questions_batch)
                    loss = model.compute_loss(outputs, vqa_labels)
            else:
                outputs = model(frame_features, questions_batch)
                loss = model.compute_loss(outputs, vqa_labels)
            
            # 🚀 Mixed Precision Backward Pass
            if use_mixed_precision:
                # Scale loss to prevent gradient underflow
                scaler.scale(loss).backward()
                
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                
                # Gradient clipping on unscaled gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                
                # Update with scaler
                scaler.step(optimizer)
                scale_before = scaler.get_scale()
                scaler.update()
                scale_after = scaler.get_scale()
                
                # Track mixed precision statistics
                mp_stats['total_steps'] += 1
                if scale_after != scale_before:
                    mp_stats['scale_updates'] += 1
                if scale_after < scale_before:  # Scale decreased = step was skipped
                    mp_stats['skipped_steps'] += 1
                    
            else:
                # Regular backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += vqa_labels.size(0)
            train_correct += (predicted == vqa_labels).sum().item()
            
            # Update progress bar with mixed precision info
            if batch_idx % 10 == 0:
                progress_dict = {
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * train_correct / train_total:.2f}%'
                }
                
                # Add mixed precision info to progress bar
                if use_mixed_precision:
                    progress_dict['Scale'] = f'{scaler.get_scale():.0f}'
                
                train_pbar.set_postfix(progress_dict)
                train_pbar.refresh()
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = create_progress_bar(enumerate(val_loader), desc="Validation", leave=False)
            
            for batch_idx, batch in val_pbar:
                
                # Extract data from baseline batch format
                frame_features = batch['frame_features'].to(device)
                questions_batch = {
                    'input_ids': batch['questions_batch']['input_ids'].to(device),
                    'attention_mask': batch['questions_batch']['attention_mask'].to(device),
                    'token_type_ids': batch['questions_batch']['token_type_ids'].to(device)
                }
                vqa_labels = batch['labels'].to(device)
                
                # 🚀 Mixed Precision Validation (optional, saves memory)
                if use_mixed_precision:
                    with autocast():
                        outputs = model(frame_features, questions_batch)
                        loss = model.compute_loss(outputs, vqa_labels)
                else:
                    outputs = model(frame_features, questions_batch)
                    loss = model.compute_loss(outputs, vqa_labels)
                
                # Statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += vqa_labels.size(0)
                val_correct += (predicted == vqa_labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })
                val_pbar.refresh()
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Step scheduler with validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_cached = torch.cuda.memory_reserved(0) / 1024**3
        
        # Print epoch results with mixed precision stats
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Epoch Time: {epoch_time:.1f}s")
        
        if torch.cuda.is_available():
            print(f"GPU Memory: {gpu_allocated:.1f}GB/{gpu_memory:.1f}GB (cached: {gpu_cached:.1f}GB)")
        
        # 📊 Mixed precision statistics
        if use_mixed_precision:
            skip_rate = 100 * mp_stats['skipped_steps'] / max(mp_stats['total_steps'], 1)
            print(f"Mixed Precision: Scale={scaler.get_scale():.0f}, Skipped Steps={skip_rate:.1f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            checkpoint_path = '/kaggle/working/checkpoints/best_baseline_vqa_model.pth'
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            # 🔥 Save checkpoint with mixed precision state
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'model_config': model_config,
                'mixed_precision_enabled': use_mixed_precision,
                'model_type': 'baseline'  # Mark as baseline model
            }
            
            # Add scaler state if using mixed precision
            if use_mixed_precision:
                checkpoint_dict['scaler_state_dict'] = scaler.state_dict()
                checkpoint_dict['mp_stats'] = mp_stats.copy()
            
            torch.save(checkpoint_dict, checkpoint_path)
            print(f"✅ New best baseline model saved! Val Loss: {val_loss:.4f}")
        
        # Save regular checkpoint
        checkpoint_path = f'/kaggle/working/checkpoints/baseline_checkpoint_epoch_{epoch + 1}.pth'
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'model_config': model_config,
            'mixed_precision_enabled': use_mixed_precision,
            'model_type': 'baseline'
        }
        
        if use_mixed_precision:
            checkpoint_dict['scaler_state_dict'] = scaler.state_dict()
            checkpoint_dict['mp_stats'] = mp_stats.copy()
        
        torch.save(checkpoint_dict, checkpoint_path)
        print(f"📁 Baseline checkpoint saved at epoch {epoch + 1}")
    
    print(f"\n🎉 Baseline training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # 📊 Final mixed precision statistics
    if use_mixed_precision:
        print(f"\n🚀 Mixed Precision Statistics:")
        print(f"Total training steps: {mp_stats['total_steps']}")
        print(f"Scale updates: {mp_stats['scale_updates']}")
        print(f"Skipped steps: {mp_stats['skipped_steps']} ({100*mp_stats['skipped_steps']/max(mp_stats['total_steps'],1):.1f}%)")
        print(f"Final scale factor: {scaler.get_scale():.0f}")


def get_valid_answers_for_question(metadata, answer_vocab):
    """
    Determine valid answers for a question based on query_type metadata
    
    Args:
        metadata (dict): Question metadata containing query_type, ana_type, etc.
        answer_vocab (list): List of all possible answers
        
    Returns:
        set: Set of valid answer indices for this question type
    """
    # Get query_type and ana_type from metadata
    query_type = metadata.get('query_type', 'general') if isinstance(metadata, dict) else 'general'
    ana_type = metadata.get('ana_type', 'general') if isinstance(metadata, dict) else 'general'
    
    # Count questions (ana_type indicates counting)
    if query_type == 'count' or (ana_type and 'count' in str(ana_type).lower()):
        numeric_answers = {str(i) for i in range(20)}  # 0-19
        return {i for i, ans in enumerate(answer_vocab) if ans in numeric_answers}
    
    # Exist questions
    elif query_type == 'exist':
        binary_answers = {'True', 'False'}
        return {i for i, ans in enumerate(answer_vocab) if ans in binary_answers}
    
    # Color questions
    elif query_type == 'query_color':
        color_answers = {'white', 'yellow', 'silver', 'red', 'blue', 'brown'}
        return {i for i, ans in enumerate(answer_vocab) if ans in color_answers}
    
    # Type questions (instrument vs anatomy)
    elif query_type == 'query_type':
        type_answers = {'instrument', 'anatomy'}
        return {i for i, ans in enumerate(answer_vocab) if ans in type_answers}
    
    # Component questions (instrument + anatomy objects)
    elif query_type == 'query_component':
        component_answers = {
            "abdominal_wall_cavity", "adhesion", "aspirate", "bipolar", "blood_vessel", 
            "clipper", "cystic_artery", "cystic_duct", "cystic_pedicle", "cystic_plate",
            "fluid", "gallbladder", "grasper", "gut", "hook", "irrigator", "liver",
            "omentum", "peritoneum", "scissors", "silver", "specimen_bag"
        }
        return {i for i, ans in enumerate(answer_vocab) if ans in component_answers}
    
    # General questions (default case)
    else:  # query_type == 'general' or unknown
        general_answers = {
            "abdominal_wall_cavity", "adhesion", "aspirate", "bipolar", "blood_vessel", 
            "clip", "clipper", "coagulate", "cut", "cystic_artery", "cystic_duct", 
            "cystic_pedicle", "cystic_plate", "dissect", "fluid", "gallbladder", 
            "grasp", "grasper", "gut", "hook", "irrigate", "irrigator", "liver",
            "omentum", "pack", "peritoneum", "retract", "scissors", "silver", "specimen_bag"
        }
        return {i for i, ans in enumerate(answer_vocab) if ans in general_answers}


def compute_validity_score(predictions, questions_metadata, answer_vocab):
    """
    Compute validity score - percentage of predictions that are valid for their question type
    
    Args:
        predictions (list): Model predictions (answer indices)
        questions_metadata (list): Question metadata with query_type, ana_type, etc.
        answer_vocab (list): List of all possible answers
        
    Returns:
        float: Validity score (0-1)
    """
    if len(predictions) != len(questions_metadata):
        return 0.0
    
    valid_count = 0
    total_count = 0
    
    for pred_idx, metadata in zip(predictions, questions_metadata):
        # Get valid answers based on metadata (query_type, ana_type)
        valid_answers = get_valid_answers_for_question(metadata, answer_vocab)
        
        if pred_idx in valid_answers:
            valid_count += 1
        total_count += 1
    
    return valid_count / total_count if total_count > 0 else 0.0


def compute_distribution_similarity(true_labels, predictions, answer_vocab):
    """
    Compute distribution similarity using chi-square test
    
    Args:
        true_labels (list): Ground truth answer indices
        predictions (list): Model predictions (answer indices) 
        answer_vocab (list): List of all possible answers
        
    Returns:
        dict: Distribution similarity metrics
    """
    # Count answer frequencies
    true_dist = np.bincount(true_labels, minlength=len(answer_vocab))
    pred_dist = np.bincount(predictions, minlength=len(answer_vocab))
    
    # Remove zero counts to avoid division by zero in chi-square
    nonzero_mask = (true_dist > 0) | (pred_dist > 0)
    true_dist_nz = true_dist[nonzero_mask]
    pred_dist_nz = pred_dist[nonzero_mask]
    
    if len(true_dist_nz) == 0:
        return {'chi_square': float('inf'), 'p_value': 0.0, 'similarity_score': 0.0}
    
    # Chi-square test
    # Add small epsilon to avoid zero division
    expected = true_dist_nz + 1e-8
    observed = pred_dist_nz + 1e-8
    
    try:
        chi2_stat, p_value = chisquare(observed, expected)
        # Convert chi-square to similarity score (0-1, higher is better)
        # Use negative exponential to map chi2 to [0,1] range
        similarity_score = np.exp(-chi2_stat / len(expected))
    except (ValueError, ZeroDivisionError):
        chi2_stat, p_value, similarity_score = float('inf'), 0.0, 0.0
    
    return {
        'chi_square': float(chi2_stat),
        'p_value': float(p_value), 
        'similarity_score': float(similarity_score)
    }


def evaluate_baseline_on_test(model_path, batch_size=32):
    """
    Evaluate trained baseline model on test set with analysis by question type
    
    Args:
        model_path (str): Path to trained baseline model checkpoint
        batch_size (int): Batch size for evaluation
    """
    
    print("🧪 Baseline Model Test Set Evaluation")
    print("🔬 Ablation Study: Global Frame Features Only (No GNN)")
    print("=" * 60)
    
    # Setup
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model path: {model_path}")
    
    # Load test dataset
    print("\n📁 Loading baseline test dataset...")
    test_dataset = FullFrameBaselineDataset(mode="test")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=baseline_collate_fn,
        num_workers=2
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Memory usage: ~{test_dataset.get_memory_usage():.1f}MB (frame features)")
    
    # Load model
    print("\n🤖 Loading trained baseline model...")
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        # Use saved model config if available, otherwise use default
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            print("✅ Using saved baseline model configuration")
        else:
            model_config = {
                'frame_feature_dim': 512,
                'hidden_dim': 768,
                'question_dim': 768,
                'num_classes': 50,
                'bert_model_name': "emilyalsentzer/Bio_ClinicalBERT",
                'num_transformer_layers': 3,
                'dropout_prob': 0.1
            }
            print("⚠️ Using default baseline model configuration (saved config not found)")
        
        model = create_full_frame_baseline_model(**model_config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Baseline model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_acc' in checkpoint:
            print(f"✅ Validation accuracy: {checkpoint['val_acc']:.2f}%")
        
        # Verify it's a baseline model
        model_type = checkpoint.get('model_type', 'unknown')
        print(f"🔬 Model type: {model_type}")
        
    else:
        raise FileNotFoundError(f"Baseline model checkpoint not found: {model_path}")
    
    # Load answer vocabulary for validity and distribution analysis
    print("\n📚 Loading answer vocabulary...")
    try:
        with open(cfg.META_INFO_DIR / "label2ans.json") as f:
            answer_vocab = json.load(f)
        print(f"✅ Loaded {len(answer_vocab)} answer classes")
    except Exception as e:
        print(f"⚠️ Could not load answer vocabulary: {e}")
        answer_vocab = [str(i) for i in range(50)]  # Fallback
    
    # Get question metadata for comprehensive analysis
    print("\n📊 Analyzing question types and preparing advanced metrics...")
    ana_type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    query_type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall_stats = {'correct': 0, 'total': 0}
    
    # Evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        test_pbar = create_progress_bar(enumerate(test_loader), desc="Evaluating", leave=True)
        
        for batch_idx, batch in test_pbar:
            
            # Extract data from baseline batch format
            frame_features = batch['frame_features'].to(device)
            questions_batch = {
                'input_ids': batch['questions_batch']['input_ids'].to(device),
                'attention_mask': batch['questions_batch']['attention_mask'].to(device),
                'token_type_ids': batch['questions_batch']['token_type_ids'].to(device)
            }
            vqa_labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(frame_features, questions_batch)
            _, predicted = torch.max(outputs, 1)
            
            # Collect predictions and labels
            predictions = predicted.cpu().numpy()
            labels = vqa_labels.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            
            # Get ana_type for each sample in batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(test_dataset))
            
            for i, (pred, label) in enumerate(zip(predictions, labels)):
                sample_idx = start_idx + i
                if sample_idx < len(test_dataset):
                    # Get question type metadata from dataset
                    if hasattr(test_dataset, 'tokenized_data') and test_dataset.tokenized_data:
                        _, metadata = test_dataset.vqas[sample_idx]
                        ana_type = metadata.get('ana_type', 'general')
                        query_type = metadata.get('query_type', 'general')
                        
                        # Safety check: ensure no None values
                        if ana_type is None:
                            ana_type = 'general'
                        if query_type is None:
                            query_type = 'general'
                    else:
                        # Fallback: try to get from original format
                        ana_type = 'general'
                        query_type = 'general'
                    
                    # Update statistics for both question type categories
                    ana_type_stats[ana_type]['total'] += 1
                    query_type_stats[query_type]['total'] += 1
                    overall_stats['total'] += 1
                    
                    if pred == label:
                        ana_type_stats[ana_type]['correct'] += 1
                        query_type_stats[query_type]['correct'] += 1
                        overall_stats['correct'] += 1
            
            # Update progress bar
            current_acc = 100 * overall_stats['correct'] / overall_stats['total']
            test_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
    
    # Clear GPU cache after evaluation to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Calculate overall metrics
    overall_acc = 100 * overall_stats['correct'] / overall_stats['total']
    
    # Calculate precision, recall, F1-score
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Macro-averaged metrics (average across all classes)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )
    
    # Weighted metrics (weighted by class frequency)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Advanced metrics: Validity and Distribution Similarity
    print("\n🔍 Computing advanced metrics...")
    
    # Collect question metadata for validity analysis
    questions_metadata = []
    for i in range(len(all_predictions)):
        if i < len(test_dataset):
            if hasattr(test_dataset, 'tokenized_data') and test_dataset.tokenized_data:
                _, metadata = test_dataset.vqas[i]
                questions_metadata.append(metadata)
            else:
                # Fallback - try to get question from original format
                try:
                    question_text = test_dataset.vqas[i][1].split("|")[0] if len(test_dataset.vqas[i]) > 1 else ""
                    questions_metadata.append({'question': question_text})
                except:
                    questions_metadata.append({'question': ''})
    
    # Compute validity score
    validity_score = compute_validity_score(all_predictions, questions_metadata, answer_vocab)
    
    # Compute distribution similarity
    distribution_metrics = compute_distribution_similarity(all_labels, all_predictions, answer_vocab)
    
    # Print overall results
    print(f"\n📈 Overall Baseline Test Results:")
    print("=" * 70)
    print(f"🔬 Model Type: Baseline (Global Frame Features Only)")
    print(f"❌ No Graph Structure | No GNN Processing | No Multi-node Representation")
    print("-" * 70)
    print(f"Accuracy:             {overall_acc:.2f}% ({overall_stats['correct']}/{overall_stats['total']})")
    print(f"Precision (macro):    {precision_macro:.4f}")
    print(f"Recall (macro):       {recall_macro:.4f}")
    print(f"F1-Score (macro):     {f1_macro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (weighted):    {recall_weighted:.4f}")
    print(f"F1-Score (weighted):  {f1_weighted:.4f}")
    print()
    print("📊 Advanced Metrics:")
    print("-" * 70)
    print(f"Validity Score:       {validity_score:.4f} ({validity_score*100:.2f}%)")
    print(f"Distribution Similarity: {distribution_metrics['similarity_score']:.4f}")
    print(f"Chi-Square Statistic: {distribution_metrics['chi_square']:.4f}")
    print(f"Chi-Square p-value:   {distribution_metrics['p_value']:.6f}")
    
    # Analyze results by ana_type and query_type
    def analyze_question_type(type_stats, type_name, predictions_dict, labels_dict):
        print(f"\n📋 Baseline Results by {type_name}:")
        print("-" * 85)
        print(f"{type_name.capitalize():<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Samples':<8}")
        print("-" * 85)

        # Filter out None values and sort the rest, handle None separately
        non_none_types = [k for k in type_stats.keys() if k is not None]
        sorted_types = sorted(non_none_types)
        
        # Add None at the end if it exists
        if None in type_stats:
            sorted_types.append(None)
        
        for qtype in sorted_types:
            stats = type_stats[qtype]
            acc = 100 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            
            # Calculate precision, recall, F1 for this question type
            if qtype in predictions_dict and len(predictions_dict[qtype]) > 0:
                type_pred = np.array(predictions_dict[qtype])
                type_label = np.array(labels_dict[qtype])
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    type_label, type_pred, average='macro', zero_division=0
                )
            else:
                precision = recall = f1 = 0.0
            
            # Handle None display properly
            display_name = str(qtype) if qtype is not None else "Unknown/None"
            print(f"{display_name:<20} {acc:>6.2f}%    {precision:>8.4f}   {recall:>8.4f}   {f1:>8.4f}   {stats['total']:>6}")
    
    # Group predictions by both ana_type and query_type
    ana_type_predictions = defaultdict(list)
    ana_type_labels = defaultdict(list)
    query_type_predictions = defaultdict(list)
    query_type_labels = defaultdict(list)
    
    for i, (pred, label) in enumerate(zip(all_predictions, all_labels)):
        if i < len(test_dataset):
            if hasattr(test_dataset, 'tokenized_data') and test_dataset.tokenized_data:
                _, metadata = test_dataset.vqas[i]
                ana_type = metadata.get('ana_type', 'general')
                query_type = metadata.get('query_type', 'general')
                
                # Safety check: ensure no None values
                if ana_type is None:
                    ana_type = 'general'
                if query_type is None:
                    query_type = 'general'
            else:
                ana_type = 'general'
                query_type = 'general'
            
            ana_type_predictions[ana_type].append(pred)
            ana_type_labels[ana_type].append(label)
            query_type_predictions[query_type].append(pred)
            query_type_labels[query_type].append(label)
    
    # Display results for both question type categories
    analyze_question_type(ana_type_stats, "Ana Type", ana_type_predictions, ana_type_labels)
    analyze_question_type(query_type_stats, "Query Type", query_type_predictions, query_type_labels)
    
    # Save detailed results with all metrics
    results = {
        'model_type': 'baseline',
        'overall': {
            'accuracy': overall_acc,
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro), 
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'validity_score': float(validity_score),
            'distribution_similarity': distribution_metrics['similarity_score'],
            'chi_square_statistic': distribution_metrics['chi_square'],
            'chi_square_p_value': distribution_metrics['p_value'],
            'correct': overall_stats['correct'], 
            'total': overall_stats['total']
        },
        'by_ana_type': {},
        'by_query_type': {}
    }
    
    # Add detailed results for both question type categories
    def add_detailed_results(results_dict, type_stats, type_predictions, type_labels, category_key):
        results_dict[category_key] = {}
        
        # Filter out None values and sort the rest, handle None separately
        non_none_types = [k for k in type_stats.keys() if k is not None]
        sorted_types = sorted(non_none_types)
        
        # Add None at the end if it exists
        if None in type_stats:
            sorted_types.append(None)
        
        for qtype in sorted_types:
            stats = type_stats[qtype]
            acc = 100 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            
            # Calculate metrics for this question type
            if qtype in type_predictions and len(type_predictions[qtype]) > 0:
                type_pred = np.array(type_predictions[qtype])
                type_label = np.array(type_labels[qtype])
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    type_label, type_pred, average='macro', zero_division=0
                )
            else:
                precision = recall = f1 = 0.0
            
            # Convert None keys to string for JSON compatibility
            json_key = str(qtype) if qtype is not None else "unknown_none"
            results_dict[category_key][json_key] = {
                'accuracy': acc,
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'correct': stats['correct'],
                'total': stats['total']
            }
    
    # Add results for both ana_type and query_type
    add_detailed_results(results, ana_type_stats, ana_type_predictions, ana_type_labels, 'by_ana_type')
    add_detailed_results(results, query_type_stats, query_type_predictions, query_type_labels, 'by_query_type')
    
    # Save results to file
    results_path = model_path.replace('.pth', '_baseline_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed baseline results saved to: {results_path}")
    
    # Generate classification report for additional insights
    print(f"\n📊 Detailed Classification Report:")
    print("-" * 80)
    report = classification_report(all_labels, all_predictions, zero_division=0)
    print(report)
    
    return results


if __name__ == "__main__":
    # === BASELINE TRAINING WITH MIXED PRECISION ===
    resume_checkpoint = None
    
    # 🚀 Enable mixed precision training
    # Set to False if you want to disable mixed precision
    use_mixed_precision = True
    
    print(f"🔬 Baseline Model Training (Ablation Study)")
    print(f"🎯 Mixed Precision Training: {'ENABLED' if use_mixed_precision else 'DISABLED'}")
    print(f"❌ No Graph | No GNN | Global Frame Features Only")
    
    # Uncomment to train
    train_baseline_vqa(resume_from=resume_checkpoint, use_mixed_precision=use_mixed_precision)
    
    # === EVALUATION ===
    # model_path = "/kaggle/working/checkpoints/best_baseline_vqa_model.pth" 
    # results = evaluate_baseline_on_test(model_path, batch_size=32)
    # print(f"\nFinal baseline test accuracy: {results['overall']['accuracy']:.2f}%")