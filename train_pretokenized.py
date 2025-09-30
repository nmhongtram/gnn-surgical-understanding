#!/usr/bin/env python3
"""
Training Script with Pre-tokenized Model

Test training with pre-tokenized questions to eliminate CPU tokenization overhead
"""

import torch
import os
from pathlib import Path
from torch.utils.data import DataLoader
import time

# Fix tokenizers parallelism issue
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("🔧 Tokenizers parallelism disabled")

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)
    print(f"✅ GPU: {torch.cuda.get_device_name()}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("❌ CPU only")

# Import modules with pre-tokenized support
try:
    from ssg_dataset_pretokenized import PreTokenizedSSGVQA as SSGVQA, pretokenized_collate_fn as collate_fn
    from model_pretokenized import PreTokenizedSurgicalSceneGraphVQA as SurgicalSceneGraphVQA
    print("🚀 Using pre-tokenized dataset and model for zero CPU tokenization overhead")
    using_pretokenized = True
except ImportError as e:
    print(f"⚠️ Pre-tokenized modules not found: {e}")
    from ssg_dataset import SSGVQA, multi_triplet_collate_fn as collate_fn
    from model import SurgicalSceneGraphVQA
    print("🔄 Using original dataset (with CPU tokenization)")
    using_pretokenized = False

from training_utils import GNNTrainer

print(f"🎯 Device: {device}")

# Configuration
CONFIG = {
    'gnn_type': 'gcn',
    'num_classes': 51,
    'epochs': 3,  # Short test
    'batch_size': 8,  # Small batch for testing
    'learning_rate': 1e-4,
    'use_uncertainty': True,
    'save_every_n_epochs': 1,
    'checkpoint_dir': 'checkpoints',
    # Ana_type filtering for focused training
    'ana_type': ['zero_hop.json'],  # Start with basic questions only
}

print("Configuration:", CONFIG)
print(f"Ana_type filter: {CONFIG['ana_type'] if CONFIG['ana_type'] else 'All question types'}")

# Check for pre-tokenized files
if using_pretokenized:
    import config as cfg
    
    # Check for ana_type specific files first
    if CONFIG['ana_type']:
        ana_suffix = "_".join(CONFIG['ana_type']).replace('.json', '')
        pretokenized_files = {
            'debug': Path(f"tokenized_debug_{ana_suffix}.h5")
        }
    else:
        pretokenized_files = {
            'debug': Path("tokenized_debug.h5")
        }
    
    missing_files = [mode for mode, path in pretokenized_files.items() if not path.exists()]
    
    if missing_files:
        print(f"⚠️ Missing pre-tokenized files for: {missing_files}")
        print("💡 Run preprocessing command:")
        if CONFIG['ana_type']:
            ana_args = " ".join(CONFIG['ana_type'])
            print(f"   python preprocess_questions.py --modes debug --ana_type {ana_args}")
        else:
            print("   python preprocess_questions.py --modes debug")
        print("🔄 Falling back to original dataset with CPU tokenization")
        
        from ssg_dataset import SSGVQA, multi_triplet_collate_fn as collate_fn
        from model import SurgicalSceneGraphVQA
        using_pretokenized = False
    else:
        print("✅ Pre-tokenized files found!")
        for mode, path in pretokenized_files.items():
            file_size = path.stat().st_size / (1024 * 1024)
            print(f"   {mode}: {path.name} ({file_size:.1f} MB)")

# GPU Monitoring
class GPUMonitor:
    def __init__(self):
        self.enabled = torch.cuda.is_available()
        
    def get_stats(self):
        if not self.enabled:
            return "CPU only"
        
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        
        return f"{allocated:.0f}MB allocated, {reserved:.0f}MB reserved (peak: {max_allocated:.0f}MB)"
    
    def print_stats(self, prefix="🔥 GPU"):
        if self.enabled:
            print(f"{prefix}: {self.get_stats()}")
    
    def cleanup(self):
        if self.enabled:
            torch.cuda.empty_cache()

monitor = GPUMonitor()

def main():
    """Main training function"""
    print("\n🚀 Starting Pre-tokenized Model Training Test")
    print("=" * 50)
    
    # Create datasets
    print("\n📊 Creating datasets...")
    
    try:
        # Test dataset with debug mode for quick validation
        if using_pretokenized:
            test_dataset = SSGVQA(
                ana_type=CONFIG['ana_type'],
                mode="debug", 
                use_pretokenized=True
            )
        else:
            test_dataset = SSGVQA(mode="debug")
        
        print(f"✅ Dataset created: {len(test_dataset)} questions")
        
        # Show first few examples
        print("\n📝 Sample questions:")
        for i in range(min(3, len(test_dataset))):
            graph_data, question, label, triplet_data = test_dataset[i]
            answer = test_dataset.labels[label]
            if isinstance(question, dict):
                print(f"   Q{i+1}: [Pre-tokenized] → A: {answer}")
            else:
                print(f"   Q{i+1}: {question[:50]}... → A: {answer}")
    
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        return
    
    # Create DataLoader
    print(f"\n🔄 Creating DataLoader (batch_size={CONFIG['batch_size']})...")
    
    try:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=torch.cuda.is_available()
        )
        print(f"✅ DataLoader created: {len(test_loader)} batches")
    
    except Exception as e:
        print(f"❌ DataLoader creation failed: {e}")
        return
    
    # Initialize model
    print(f"\n🤖 Initializing model (GNN: {CONFIG['gnn_type']})...")
    monitor.print_stats("Before model")
    
    try:
        model = SurgicalSceneGraphVQA(
            num_classes=CONFIG['num_classes'],
            gnn_type=CONFIG['gnn_type'],
            use_uncertainty_weights=CONFIG['use_uncertainty']
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ Model initialized:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        monitor.print_stats("After model")
    
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return
    
    # Test forward pass
    print(f"\n🧪 Testing forward pass...")
    model.eval()
    
    with torch.no_grad():
        try:
            start_time = time.time()
            
            # Get first batch
            batch = next(iter(test_loader))
            graph_data, questions, labels, triplet_data = batch
            
            # Move to device
            graph_data = graph_data.to(device)
            labels = labels.to(device)
            
            print(f"   Batch shapes:")
            if isinstance(questions, dict):
                print(f"   - Questions (pre-tokenized): {questions['input_ids'].shape}")
                questions = {k: v.to(device) for k, v in questions.items()}
            else:
                print(f"   - Questions (strings): {len(questions)}")
            print(f"   - Labels: {labels.shape}")
            print(f"   - Graph nodes: {graph_data.x.shape}")
            print(f"   - Graph edges: {graph_data.edge_index.shape}")
            
            monitor.print_stats("Before forward")
            
            # Forward pass
            outputs = model(graph_data, questions)
            
            forward_time = time.time() - start_time
            
            if isinstance(outputs, dict):
                print(f"   ✅ Forward pass successful! ({forward_time:.3f}s)")
                for key, value in outputs.items():
                    print(f"     {key}: {value.shape}")
            else:
                print(f"   ✅ Forward pass successful! ({forward_time:.3f}s)")
                print(f"     Output shape: {outputs.shape}")
            
            monitor.print_stats("After forward")
            
        except Exception as e:
            print(f"   ❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Test training step  
    print(f"\n🏋️ Testing training step...")
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    
    try:
        start_time = time.time()
        
        # Training step
        optimizer.zero_grad()
        
        outputs = model(graph_data, questions)
        
        if isinstance(outputs, dict):
            loss = criterion(outputs['vqa'], labels)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        train_time = time.time() - start_time
        
        print(f"   ✅ Training step successful! ({train_time:.3f}s)")
        print(f"     Loss: {loss.item():.4f}")
        
        monitor.print_stats("After training")
        
    except Exception as e:
        print(f"   ❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print(f"\n🎉 Pre-tokenized Model Test Complete!")
    print("=" * 40)
    print(f"✅ Dataset: {len(test_dataset)} questions")
    print(f"✅ Model: {trainable_params:,} parameters")
    print(f"✅ Forward pass: {forward_time:.3f}s")
    print(f"✅ Training step: {train_time:.3f}s")
    
    if using_pretokenized:
        print("🚀 Pre-tokenized benefits:")
        print("   - Zero CPU tokenization overhead")
        print("   - Faster data loading")
        print("   - Better GPU utilization")
    else:
        print("⚠️ Using original dataset with CPU tokenization")
        print("💡 Run preprocessing to enable pre-tokenized benefits")
    
    monitor.print_stats("Final")
    monitor.cleanup()


if __name__ == "__main__":
    main()