#!/usr/bin/env python3
"""
VQA-Only Training Script

Simple training script for surgical VQA without triplet prediction

cụ thể xét theo query_type:: 
câu hỏi general trả về ["abdominal_wall_cavity",
    "adhesion",
    "aspirate",
    "bipolar",
    "blood_vessel", "clip",
    "clipper",
    "coagulate",
    "cut",
    "cystic_artery",
    "cystic_duct",
    "cystic_pedicle",
    "cystic_plate",
    "dissect",
    "fluid",
    "gallbladder",
    "grasp",
    "grasper",
    "gut",
    "hook",  "irrigate",
    "irrigator",
    "liver",
    "omentum",
    "pack",
    "peritoneum", "retract",
    "scissors",
    "silver",
    "specimen_bag"]
câu hỏi ana_type = count thì trả về số. 
câu hỏi query_component trả về tập hợp của instrument + anatomy 
["abdominal_wall_cavity",
    "adhesion",
    "aspirate",
    "bipolar",
    "blood_vessel", 
    "clipper",
    "cystic_artery",
    "cystic_duct",
    "cystic_pedicle",
    "cystic_plate",
    "fluid",
    "gallbladder",
    "grasper",
    "gut",
    "hook",  
    "irrigator",
    "liver",
    "omentum",
    "peritoneum", 
    "scissors",
    "silver",
    "specimen_bag"]
câu hỏi query_color trả về tập hợp màu sắc [white, yellow, silver, red, blue, brown]
câu hỏi exist trả về [True, False]
câu hỏi query type trả về [instrument, anatomy]
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# Configure tqdm for Kaggle environment
tqdm.pandas()
os.environ['TQDM_DISABLE'] = '0'  # Ensure tqdm is enabled

# Progress bar settings - change this if tqdm has issues
USE_TQDM = True  # Set to False to use manual progress printing
MANUAL_PROGRESS_INTERVAL = 10  # Print every N batches when not using tqdm

# Local imports
from ssg_dataset import SSGDataset, vqa_only_collate_fn
from model import SSGModel
from collections import defaultdict
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report


def create_progress_bar(iterable, desc, leave=False):
    """Create progress bar with fallback to manual progress"""
    if USE_TQDM:
        return tqdm(iterable, desc=desc, leave=leave, dynamic_ncols=True, 
                   ascii=True, file=sys.stdout, miniters=1, mininterval=0.1)
    else:
        return iterable



from scipy.stats import chisquare
import re


def train_vqa_only(resume_from=None, gnn_type="gcn"):
    """Train VQA-only model without triplet prediction
    
    Args:
        resume_from (str): Path to checkpoint file to resume from
    """
    
    print("🚀 VQA-Only Training")
    print("=" * 40)
    
    # Setup
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Training config
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4  # Lower LR for stable training and reduced overfitting
    WEIGHT_DECAY = 1e-4   # Moderate weight decay (reduced from 1e-2)
    NUM_EPOCHS = 10
    GRAD_CLIP_NORM = 1.0  # Gradient clipping threshold
    
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Gradient clipping: {GRAD_CLIP_NORM}")
    
    if resume_from:
        print(f"Resume from: {resume_from}")
    
    # Load datasets
    print("\n📁 Loading datasets...")
    train_dataset = SSGDataset(mode="train")
    val_dataset = SSGDataset(mode="val")
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=vqa_only_collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=vqa_only_collate_fn,
        num_workers=2
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\n🤖 Creating model...")
    model = SSGModel(gnn_type=gnn_type).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler - Reduce on plateau (adaptive)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-7
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_acc = 0.0
    
    if resume_from and os.path.exists(resume_from):
        print(f"\n📂 Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('val_acc', 0.0)
        
        # Resume scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✅ Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.2f}%")
        
        # Debug: Show scheduler state after resume
        if hasattr(scheduler, 'num_bad_epochs'):
            print(f"📊 Scheduler state: patience={scheduler.num_bad_epochs}/{scheduler.patience}, best={scheduler.best:.4f}")
    elif resume_from:
        print(f"⚠️ Checkpoint file not found: {resume_from}")
        print("Starting training from scratch...")
    
    # Training loop
    print(f"\n🏋️ Starting training from epoch {start_epoch + 1}...")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 20)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = create_progress_bar(enumerate(train_loader), desc="Training", leave=False)
        
        for batch_idx, batch in train_pbar:
            
            graph_batch, questions_batch, vqa_labels = batch
            
            # Move to device
            graph_batch = graph_batch.to(device)
            questions_batch = {k: v.to(device) for k, v in questions_batch.items()}
            vqa_labels = vqa_labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(graph_batch, questions_batch)
            
            # Compute loss
            loss = model.compute_vqa_only_loss(outputs, vqa_labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += vqa_labels.size(0)
            train_correct += (predicted == vqa_labels).sum().item()
            
            # Update progress bar every 10 batches to reduce output noise
            if batch_idx % 10 == 0:
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * train_correct / train_total:.2f}%'
                })
                train_pbar.refresh()  # Force refresh for Kaggle
        
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
                
                graph_batch, questions_batch, vqa_labels = batch
                
                # Move to device
                graph_batch = graph_batch.to(device)
                questions_batch = {k: v.to(device) for k, v in questions_batch.items()}
                vqa_labels = vqa_labels.to(device)
                
                # Forward pass
                outputs = model(graph_batch, questions_batch)
                loss = model.compute_vqa_only_loss(outputs, vqa_labels)
                
                # Statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += vqa_labels.size(0)
                val_correct += (predicted == vqa_labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })
                val_pbar.refresh()  # Force refresh for Kaggle
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Step scheduler with validation loss (ReduceLROnPlateau)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Clear GPU cache after each epoch to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Optional: print GPU memory usage
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_cached = torch.cuda.memory_reserved(0) / 1024**3
        
        # Print epoch results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        if torch.cuda.is_available():
            print(f"GPU Memory: {gpu_allocated:.1f}GB/{gpu_memory:.1f}GB (cached: {gpu_cached:.1f}GB)")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = '/kaggle/working/checkpoints/best_vqa_model.pth'
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Save regular checkpoint every few epochs
        # if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
        checkpoint_path = f'/kaggle/working/checkpoints/checkpoint_epoch_{epoch + 1}.pth'
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'best_val_acc': best_val_acc,
        }, checkpoint_path)
        print(f"📁 Checkpoint saved at epoch {epoch + 1}")
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


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
    ana_type = metadata.get('ana_type', None) if isinstance(metadata, dict) else None
    
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


def evaluate_on_test(model_path, gnn_type="gcn", batch_size=32):
    """
    Evaluate trained model on test set with analysis by question type (ana_type)
    
    Args:
        model_path (str): Path to trained model checkpoint
        gnn_type (str): GNN architecture type
        batch_size (int): Batch size for evaluation
    """
    
    print("🧪 Test Set Evaluation")
    print("=" * 40)
    
    # Setup
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model path: {model_path}")
    
    # Load test dataset
    print("\n📁 Loading test dataset...")
    test_dataset = SSGDataset(mode="test")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=vqa_only_collate_fn,
        num_workers=2
    )
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Load model
    print("\n🤖 Loading trained model...")
    model = SSGModel(gnn_type=gnn_type).to(device)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_acc' in checkpoint:
            print(f"✅ Validation accuracy: {checkpoint['val_acc']:.2f}%")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load answer vocabulary for validity and distribution analysis
    print("\n📚 Loading answer vocabulary...")
    try:
        import config as cfg
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
            
            graph_batch, questions_batch, vqa_labels = batch
            
            # Move to device
            graph_batch = graph_batch.to(device)
            questions_batch = {k: v.to(device) for k, v in questions_batch.items()}
            vqa_labels = vqa_labels.to(device)
            
            # Forward pass
            outputs = model(graph_batch, questions_batch)
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
    print(f"\n📈 Overall Test Results:")
    print("=" * 70)
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
        print(f"\n📋 Results by {type_name}:")
        print("-" * 85)
        print(f"{type_name.capitalize():<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Samples':<8}")
        print("-" * 85)
        
        sorted_types = sorted(type_stats.keys())
        
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
            
            print(f"{qtype:<20} {acc:>6.2f}%    {precision:>8.4f}   {recall:>8.4f}   {f1:>8.4f}   {stats['total']:>6}")
    
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
        'by_ana_type': {}
    }
    
    # Add detailed results for both question type categories
    def add_detailed_results(results_dict, type_stats, type_predictions, type_labels, category_key):
        results_dict[category_key] = {}
        sorted_types = sorted(type_stats.keys())
        
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
            
            results_dict[category_key][qtype] = {
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
    results_path = model_path.replace('.pth', '_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_path}")
    
    # Generate classification report for additional insights
    print(f"\n📊 Detailed Classification Report:")
    print("-" * 80)
    report = classification_report(all_labels, all_predictions, zero_division=0)
    print(report)
    
    return results


if __name__ == "__main__":
    # For Kaggle: Simply call the function with resume path if needed
    
    # === TRAINING ===
    # To resume from checkpoint, change the path below:
    resume_checkpoint = None  # Set to checkpoint path to resume  
    # resume_checkpoint = "/kaggle/working/checkpoints/checkpoint_epoch_5.pth"  # Example
    
    # Uncomment to train
    train_vqa_only(resume_from=resume_checkpoint, gnn_type='gcn')
    
    # === EVALUATION ===
    # To evaluate on test set, uncomment and set model path:
    # model_path = "/kaggle/working/checkpoints/best_vqa_model.pth"
    # results = evaluate_on_test(model_path, gnn_type="gcn", batch_size=32)
    # print(f"\nFinal test accuracy: {results['overall']['accuracy']:.2f}%")