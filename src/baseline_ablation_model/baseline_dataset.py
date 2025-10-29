"""
Dataset loader for Full Frame Baseline Model
Loads only global frame features (no graph data) for ablation study
"""

import pathlib
import glob
import h5py
import json
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

import src.config as cfg


class FullFrameBaselineDataset(Dataset):
    """
    Full Frame Baseline Dataset for Ablation Study
    
    Loads only global frame features and questions (no graph data)
    to measure the contribution of GNN to VQA performance.
    
    Args:
        mode: Dataset split (debug, train, val, test)
        frame_feature_path: Path to pre-computed global frame features
    """

    def __init__(self, mode="debug", frame_feature_path=None):
        
        self.mode = mode
        self.vqas = []
        self.tokenized_data = None
        self.frame_features = {}
        
        # Load pre-tokenized questions (same as enhanced model)
        tokenized_path = self._find_best_tokenized_file(mode)
        
        if not tokenized_path or not tokenized_path.exists():
            raise FileNotFoundError(
                f"Pre-tokenized file not found for mode '{mode}'. "
                f"Please run: python preprocess_questions.py --modes {mode}"
            )
        
        print(f"📁 Loading pre-tokenized questions from {tokenized_path}")
        self._load_pretokenized_questions(tokenized_path)
        
        # Load other necessary mappings
        self._load_mappings()

        # 🖼️ PRE-LOAD GLOBAL FRAME FEATURES from cropped_images
        print(f"🖼️ Pre-loading global frame features from cropped_images...")
        if frame_feature_path is None:
            frame_feature_path = cfg.VISUAL_FEATURES_DIR / "cropped_images"
        self._preload_frame_features(frame_feature_path)
        
        print(f"✅ Baseline Dataset ready: {len(self.vqas)} questions with global frame features only")

    def _find_best_tokenized_file(self, mode):
        """Find the best available tokenized file for the mode"""
        tokenized_dir = cfg.TOKENIZED_QUESTIONS_DIR
        if not tokenized_dir.exists():
            return None
            
        # Look for exact mode file first
        exact_file = tokenized_dir / f"tokenized_{mode}.h5"
        if exact_file.exists():
            return exact_file
            
        # Fallback patterns
        pattern_files = list(tokenized_dir.glob(f"tokenized_{mode}*.h5"))
        if pattern_files:
            return pattern_files[0]
            
        return None

    def _load_pretokenized_questions(self, tokenized_path, ana_type=None):
        """Load pre-tokenized questions from HDF5 file"""
        try:
            with h5py.File(tokenized_path, 'r') as f:
                # Load tokenized tensors
                input_ids = torch.from_numpy(f['input_ids'][:])
                attention_mask = torch.from_numpy(f['attention_mask'][:])
                
                token_type_ids = None
                if 'token_type_ids' in f:
                    token_type_ids = torch.from_numpy(f['token_type_ids'][:])
                
                # Load metadata
                questions_metadata = []
                for item in f['questions_metadata'][:]:
                    if isinstance(item, bytes):
                        item = item.decode('utf-8')
                    questions_metadata.append(json.loads(item))
                
                # Store tokenized data
                self.tokenized_data = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'token_type_ids': token_type_ids,
                    'metadata': questions_metadata
                }
                
                print(f"✅ Loaded {len(questions_metadata)} pre-tokenized questions")
                print(f"   Shape: {input_ids.shape}, Max length: {input_ids.shape[1]}")
                
                # Filter by ana_type if specified
                if ana_type:
                    self._filter_by_ana_type(ana_type)
                else:
                    # Create vqas list from metadata
                    for i, meta in enumerate(questions_metadata):
                        self.vqas.append((i, meta))  # (index, metadata)
                
        except Exception as e:
            print(f"❌ Error loading pre-tokenized questions: {e}")
            self.tokenized_data = None

    def _load_mappings(self):
        """Load answer mappings (same as enhanced model)"""
        with open(cfg.META_INFO_DIR / "label2ans.json") as f:
            self.label2ans = json.load(f)
        
        # Create reverse mapping
        # label2ans: list
        self.label2ans = {index: label for index, label in enumerate(self.label2ans)}
        self.ans2label = {label: index for index, label in self.label2ans.items()}

    def _preload_frame_features(self, frame_feature_path=None):
        """Pre-load global frame features from cropped_images directory (same as SSGDataset)"""
        
        # Use cropped_images directory (same as SSGDataset but no roi_coord_gt)
        if frame_feature_path is None:
            frame_feature_path = cfg.VISUAL_FEATURES_DIR / "cropped_images"
        
        # Get unique vidframe_ids
        unique_vidframe_ids = set()
        for item in self.vqas:
            _, metadata = item
            vidframe_id = metadata['file']
            unique_vidframe_ids.add(vidframe_id)
        
        print(f"📊 Pre-loading global frame features from {frame_feature_path}")
        print(f"🖼️ Loading features for {len(unique_vidframe_ids)} unique frames...")
        
        start_time = time.time()
        failed_loads = 0
        
        if not pathlib.Path(frame_feature_path).exists():
            print(f"⚠️ Frame feature directory not found: {frame_feature_path}")
            print("🔧 Creating dummy features for ablation study...")
            
            # Create dummy features for all frames
            for vidframe_id in unique_vidframe_ids:
                self.frame_features[vidframe_id] = torch.randn(512)  # Random features for testing
            return
        
        # Load frame features from HDF5 files (same format as SSGDataset)
        for vidframe_id in tqdm(unique_vidframe_ids, desc="Pre-loading frame features"):
            try:
                # Construct file path (same as SSGDataset)
                visual_path = pathlib.Path(frame_feature_path) / f"{vidframe_id}.hdf5"
                
                if visual_path.exists():
                    # Load the HDF5 file (same as SSGDataset)
                    with h5py.File(visual_path, "r", rdcc_nbytes=1024*1024*8) as visual_data:  # 8MB cache
                        full_frame_visual_features = torch.from_numpy(visual_data["visual_features"][:])
                    
                    # Global average pooling if multiple regions (same as SSGDataset)
                    if full_frame_visual_features.shape[0] > 1:
                        frame_feature = full_frame_visual_features.mean(dim=0)  # Global average pooling
                    else:
                        frame_feature = full_frame_visual_features.squeeze(0)  # Remove batch dimension
                    
                    self.frame_features[vidframe_id] = frame_feature.float()
                    
                else:
                    # Use dummy feature if file not found
                    print(f"⚠️ Feature file not found: {visual_path}")
                    self.frame_features[vidframe_id] = torch.zeros(512)
                    failed_loads += 1
                    
            except Exception as e:
                print(f"⚠️ Failed to load frame feature for {vidframe_id}: {e}")
                self.frame_features[vidframe_id] = torch.zeros(512)
                failed_loads += 1
        
        elapsed_time = time.time() - start_time
        print(f"✅ Frame feature pre-loading completed in {elapsed_time:.1f}s")
        print(f"📊 Memory usage: ~{self.get_memory_usage():.1f}MB for frame features only")
        print(f"📊 Success: {len(unique_vidframe_ids) - failed_loads}/{len(unique_vidframe_ids)} frames loaded")
        
        if failed_loads > 0:
            print(f"⚠️ {failed_loads} frames loaded with dummy features")

    def get_memory_usage(self):
        """Get estimated memory usage of pre-loaded frame features"""
        total_mb = 0
        
        for frame_feature in self.frame_features.values():
            if frame_feature is not None:
                memory_bytes = frame_feature.numel() * frame_feature.element_size()
                total_mb += memory_bytes / (1024 * 1024)
        
        return total_mb

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        """
        Get item for baseline model (only frame features + question)
        
        Returns:
            Dict with:
            - frame_feature: Global frame feature vector
            - input_ids, attention_mask, token_type_ids: Question tokens
            - answer_label: Ground truth answer
        """
        if self.tokenized_data is not None:
            # Use pre-tokenized data
            token_idx, metadata = self.vqas[idx]
            vidframe_id = metadata['file']
            answer = metadata['answer']
            
            # Get pre-tokenized question
            input_ids = self.tokenized_data['input_ids'][token_idx]
            attention_mask = self.tokenized_data['attention_mask'][token_idx] 
            token_type_ids = self.tokenized_data['token_type_ids'][token_idx]
            
        else:
            raise NotImplementedError("Non-pretokenized mode not implemented for baseline")
        
        # Get global frame feature
        frame_feature = self.frame_features.get(vidframe_id, torch.zeros(512))
        
        # Convert answer to label
        answer_label = self.ans2label.get(answer, 0)  # Default to 0 if answer not found
        
        return {
            'frame_feature': frame_feature,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'answer_label': torch.tensor(answer_label, dtype=torch.long),
            'vidframe_id': vidframe_id,
            'answer_text': answer
        }


def baseline_collate_fn(batch):
    """
    Collate function for baseline model
    
    Args:
        batch: List of items from __getitem__
        
    Returns:
        Batched data for baseline model (no graph data)
    """
    frame_features = torch.stack([item['frame_feature'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    answer_labels = torch.stack([item['answer_label'] for item in batch])
    
    return {
        'frame_features': frame_features,      # [batch_size, 512]
        'questions_batch': {
            'input_ids': input_ids,            # [batch_size, seq_len]
            'attention_mask': attention_mask,  # [batch_size, seq_len]
            'token_type_ids': token_type_ids   # [batch_size, seq_len]
        },
        'labels': answer_labels,               # [batch_size]
        'vidframe_ids': [item['vidframe_id'] for item in batch],
        'answer_texts': [item['answer_text'] for item in batch]
    }


if __name__ == "__main__":
    # Test the baseline dataset
    print("🔬 Testing Full Frame Baseline Dataset...")
    
    # Create dataset
    dataset = FullFrameBaselineDataset(mode="debug")
    
    print(f"📊 Dataset size: {len(dataset)}")
    print(f"💾 Memory usage: {dataset.get_memory_usage():.1f}MB")
    
    # Test single item
    item = dataset[0]
    print(f"🔍 Sample item keys: {item.keys()}")
    print(f"🖼️ Frame feature shape: {item['frame_feature'].shape}")
    print(f"❓ Question shape: {item['input_ids'].shape}")
    print(f"🎯 Answer: {item['answer_text']} (label: {item['answer_label'].item()})")
    
    # Test DataLoader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=baseline_collate_fn
    )
    
    batch = next(iter(dataloader))
    print(f"\n📦 Batch shapes:")
    print(f"   Frame features: {batch['frame_features'].shape}")
    print(f"   Questions: {batch['questions_batch']['input_ids'].shape}")
    print(f"   Labels: {batch['labels'].shape}")
    
    print("✅ Baseline dataset test completed successfully!")