import pathlib
import glob
import h5py
import json
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from tqdm import tqdm

import src.config as cfg


class SSGDataset(Dataset):
    """
    Pre-tokenized Scene Graph VQA Dataset with Memory Pre-loading
    
    Uses pre-tokenized questions from HDF5 files and pre-loads all data into memory
    to eliminate I/O overhead during training.
    
    Args:
        ana_type: Analysis types to filter (for compatibility only)
        mode: Dataset split (debug, train, val, test, full_test)
    """

    def __init__(self, ana_type=[], mode="debug"):
        
        self.mode = mode
        self.vqas = []
        self.tokenized_data = None
        
        # Load pre-tokenized questions (required)
        tokenized_path = self._find_best_tokenized_file(mode)
        
        if not tokenized_path or not tokenized_path.exists():
            raise FileNotFoundError(
                f"Pre-tokenized file not found for mode '{mode}'. "
                f"Please run: python preprocess_questions.py --modes {mode}"
            )
        
        print(f"📁 Loading pre-tokenized questions from {tokenized_path}")
        self._load_pretokenized_questions(tokenized_path, ana_type)
        
        # Load other necessary mappings (same as original SSGVQA)
        self._load_mappings()

        # 🚀 PRE-LOAD GRAPH DATA: Load only graph data for VQA-only training
        print(f"🚀 Pre-loading graph data into memory for VQA-only training...")
        self._preload_graph_data_only()
        
        print(f"✅ Dataset ready: {len(self.vqas)} questions with all data pre-loaded in memory")

    def get_memory_usage(self):
        """Get estimated memory usage of pre-loaded graph data (VQA-only)"""
        total_mb = 0
        
        # Graph data memory only
        for graph_data in self.preloaded_graph_data.values():
            if graph_data is not None:
                memory_bytes = (
                    graph_data.x.numel() * graph_data.x.element_size() +
                    graph_data.edge_index.numel() * graph_data.edge_index.element_size() +
                    graph_data.edge_attr.numel() * graph_data.edge_attr.element_size() +
                    graph_data.class_indices.numel() * graph_data.class_indices.element_size()
                )
                total_mb += memory_bytes / (1024 * 1024)
        
        return total_mb

    def _preload_graph_data_only(self):
        """Pre-load only graph data into memory for VQA-only training (no triplet data)"""
        
        self.preloaded_graph_data = {}
        
        # Get unique vidframe_ids to avoid duplicate loading
        unique_vidframe_ids = set()
        
        for item in self.vqas:
            if self.tokenized_data is not None:
                _, metadata = item
                vidframe_id = metadata['file']
            else:
                file_path = item[0]
                vidframe_id = file_path.stem
            unique_vidframe_ids.add(vidframe_id)
        
        print(f"📊 Pre-loading graph data for {len(unique_vidframe_ids)} unique frames...")
        
        start_time = time.time()
        failed_loads = 0
        
        # Pre-load with progress bar
        for vidframe_id in tqdm(unique_vidframe_ids, desc="Pre-loading graph data"):
            try:
                # Pre-load only graph data (no triplet data needed for VQA-only)
                graph_data = self._load_graph_data_init(vidframe_id)
                self.preloaded_graph_data[vidframe_id] = graph_data
                
            except Exception as e:
                print(f"❌ Failed to pre-load graph data for {vidframe_id}: {e}")
                # For VQA-only, we can skip problematic frames
                self.preloaded_graph_data[vidframe_id] = None
                failed_loads += 1
        
        load_time = time.time() - start_time
        
        # Calculate memory usage for graph data only
        total_memory_mb = 0
        successful_loads = 0
        for graph_data in self.preloaded_graph_data.values():
            if graph_data is not None:
                successful_loads += 1
                # Estimate memory usage for graph data
                memory_bytes = (
                    graph_data.x.numel() * graph_data.x.element_size() +
                    graph_data.edge_index.numel() * graph_data.edge_index.element_size() +
                    graph_data.edge_attr.numel() * graph_data.edge_attr.element_size() +
                    graph_data.class_indices.numel() * graph_data.class_indices.element_size()
                )
                total_memory_mb += memory_bytes / (1024 * 1024)
        
        print(f"✅ VQA-only pre-loading completed in {load_time:.1f}s")
        print(f"📊 Memory usage: ~{total_memory_mb:.1f}MB for graph data only")
        print(f"📊 Success: {successful_loads}/{len(unique_vidframe_ids)} frames loaded")
        if failed_loads > 0:
            print(f"⚠️ Skipped {failed_loads} problematic frames (VQA-only can handle this)")

    def _find_best_tokenized_file(self, mode):
        """
        Find pre-tokenized file for given mode
        
        Returns: tokenized_{mode}.h5
        """
        base_path = cfg.TOKENIZED_QUESTIONS_DIR
        tokenized_path = base_path / f"tokenized_{mode}.h5"
        return tokenized_path

    def _load_pretokenized_questions(self, tokenized_path, ana_type):
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

    def _filter_by_ana_type(self, ana_type):
        """Filter pre-tokenized questions by analysis type"""
        if not self.tokenized_data or not ana_type:
            # If no ana_type specified, include all questions
            for i, meta in enumerate(self.tokenized_data['metadata']):
                self.vqas.append((i, meta))
            return
        
        filtered_indices = []
        filtered_metadata = []
        
        print(f"🔍 Filtering by ana_type: {ana_type}")
        
        for i, meta in enumerate(self.tokenized_data['metadata']):
            # Check if this question matches any requested ana_type
            question_ana_type = meta.get('ana_type')
            
            if question_ana_type and question_ana_type in ana_type:
                filtered_indices.append(i)
                filtered_metadata.append(meta)
                self.vqas.append((i, meta))
        
        # Update tokenized data to only include filtered items
        if filtered_indices:
            indices_tensor = torch.tensor(filtered_indices)
            self.tokenized_data['input_ids'] = self.tokenized_data['input_ids'][indices_tensor]
            self.tokenized_data['attention_mask'] = self.tokenized_data['attention_mask'][indices_tensor]
            if self.tokenized_data['token_type_ids'] is not None:
                self.tokenized_data['token_type_ids'] = self.tokenized_data['token_type_ids'][indices_tensor]
            self.tokenized_data['metadata'] = filtered_metadata
            
            print(f"✅ Filtered to {len(filtered_indices)} questions matching ana_type")
        else:
            print(f"⚠️ No questions found matching ana_type: {ana_type}")
            # Keep original data but empty vqas list
            self.vqas = []

    def _load_mappings(self):
        """Load all necessary mappings (same as original SSGVQA)"""
        # labels
        with open(cfg.META_INFO_DIR / "label2ans.json") as f:
            self.labels = json.load(f)

        # object class names mapping
        with open(cfg.META_INFO_DIR / "objects.json") as f:
            self.object_class_names = json.load(f)

        # predicate names mapping
        with open(cfg.META_INFO_DIR / "predicates.json") as f:
            predicate_names = json.load(f)
            self.SPATIAL_REL = predicate_names["SPATIAL_REL"]
            self.ACTION_REL = predicate_names["ACTION_REL"]

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        """Ultra-fast __getitem__ for VQA-only training (no triplet data)"""
        # Determine how to get question and metadata
        if self.tokenized_data is not None:
            # Use pre-tokenized data
            tokenized_idx, metadata = self.vqas[idx]
            vidframe_id = metadata['file']
            answer = metadata['answer']
            
            # Get pre-tokenized question data (already in memory)
            tokenized_question = {
                'input_ids': self.tokenized_data['input_ids'][tokenized_idx],
                'attention_mask': self.tokenized_data['attention_mask'][tokenized_idx]
            }
            if self.tokenized_data['token_type_ids'] is not None:
                tokenized_question['token_type_ids'] = self.tokenized_data['token_type_ids'][tokenized_idx]
        else:
            # Use original method
            file_path = self.vqas[idx][0]
            vidframe_id = file_path.stem
            answer = self.vqas[idx][1].split("|")[1]
            tokenized_question = None  # Will be tokenized in model
        
        # 🚀 ZERO I/O: Get pre-loaded graph data from memory
        graph_data = self.preloaded_graph_data[vidframe_id]
        
        # Skip if graph data failed to load
        if graph_data is None:
            # Return a dummy sample or raise error - you can customize this
            raise ValueError(f"Graph data not available for {vidframe_id}")
        
        # Get label (fast lookup)
        label = self.labels.index(str(answer))
        
        # VQA-only return format (no triplet data)
        if tokenized_question is not None:
            return graph_data, tokenized_question, label
        else:
            return graph_data, self.vqas[idx][1].split("|")[0], label

    def _load_graph_data_init(self, vidframe_id):
        """Load graph data during initialization (optimized for bulk loading)"""
        # 1. Load ROI features (classes + bbox + visual features)
        roi_feature_path = pathlib.Path.joinpath(
            cfg.VISUAL_FEATURES_DIR,
            "roi_coord_gt",
            f"{vidframe_id}.hdf5",
        )
        
        try:
            # 🚀 Optimize HDF5 reading for bulk loading
            with h5py.File(roi_feature_path, "r", rdcc_nbytes=1024*1024*8) as roi_data:  # 8MB cache for bulk loading
                roi_features = torch.from_numpy(roi_data["visual_features"][:])  # [N, 530]
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required ROI file not found: {roi_feature_path}") from e

        # Check if file contains only zeros (no objects detected)
        if roi_features.shape[0] == 1 and torch.all(roi_features == 0):
            object_classes_indices = torch.tensor([])
            object_bboxes = torch.zeros(0, 4)
            object_visual_features = torch.zeros(0, 512)
            num_objects = 0
        else:
            classes_one_hot = roi_features[:, :14]
            classes_indices = classes_one_hot.argmax(dim=1)
            bboxes = roi_features[:, 14:18]
            roi_visual_features = roi_features[:, 18:]

            valid_mask = (classes_one_hot.sum(dim=1) > 0)
            
            object_classes_indices = classes_indices[valid_mask]
            object_bboxes = bboxes[valid_mask]
            object_visual_features = roi_visual_features[valid_mask]
            
            num_objects = valid_mask.sum().item()
        
        # 2. Load full frame visual features
        visual_path = pathlib.Path.joinpath(
            cfg.VISUAL_FEATURES_DIR,
            "cropped_images",
            f"{vidframe_id}.hdf5",
        )
        
        try:
            with h5py.File(visual_path, "r", rdcc_nbytes=1024*1024*8) as visual_data:  # 8MB cache
                full_frame_visual_features = torch.from_numpy(visual_data["visual_features"][:])
            
            if full_frame_visual_features.shape[0] > 1:
                full_frame_visual_features = full_frame_visual_features.mean(dim=0, keepdim=True)
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required visual features file not found: {visual_path}") from e
        
        # 3. Create hierarchical graph
        total_nodes = num_objects + 1
        full_frame_idx = num_objects
        
        if num_objects == 0:
            node_features = torch.cat([
                torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                full_frame_visual_features
            ], dim=1)
            classes_indices = torch.tensor([14])
        else:
            object_node_features = torch.cat([object_bboxes, object_visual_features], dim=1)
            full_frame_node_features = torch.cat([
                torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                full_frame_visual_features
            ], dim=1)
            
            node_features = torch.cat([object_node_features, full_frame_node_features], dim=0)
            classes_indices = torch.cat([object_classes_indices, torch.tensor([14])])

        # 4. Create edges (same as original)
        edge_index, edge_attr = self._create_edges(num_objects, object_bboxes, full_frame_idx)

        # Create PyG graph data
        graph_data = Data(
            x=node_features,
            class_indices=classes_indices,
            edge_index=edge_index,
            edge_attr=edge_attr,
            vidframe_id=vidframe_id,
            num_nodes=total_nodes,
        )

        return graph_data

    def _create_edges(self, num_objects, object_bboxes, full_frame_idx):
        """Create hierarchical edges"""
        edge_index = []
        edge_attr = []
        
        if num_objects == 0:
            edge_index.append([0, 0])
            self_onehot = F.one_hot(torch.tensor(self.SPATIAL_REL['self']), num_classes=10).float()
            zero_continuous = torch.zeros(6, dtype=torch.float32) 
            self_enhanced = torch.cat([self_onehot, zero_continuous], dim=0)
            edge_attr.append(self_enhanced)
        else:
            # Object-Object relationships
            for i in range(num_objects):
                for j in range(num_objects):
                    if i != j:
                        edge_index.append([i, j])
                        enhanced_features = self._compute_enhanced_edge_features(
                            object_bboxes[i], object_bboxes[j]
                        )
                        edge_attr.append(enhanced_features)
            
            # Self-loops
            for i in range(num_objects):
                edge_index.append([i, i])
                self_onehot = F.one_hot(torch.tensor(self.SPATIAL_REL['self']), num_classes=10).float()
                zero_continuous = torch.zeros(6, dtype=torch.float32)
                self_enhanced = torch.cat([self_onehot, zero_continuous], dim=0)
                edge_attr.append(self_enhanced)

            # Hierarchical relationships
            for i in range(num_objects):
                edge_index.append([i, full_frame_idx])
                contributes_onehot = F.one_hot(torch.tensor(self.SPATIAL_REL['contributes_to']), num_classes=10).float()
                zero_continuous = torch.zeros(6, dtype=torch.float32)
                contributes_enhanced = torch.cat([contributes_onehot, zero_continuous], dim=0)
                edge_attr.append(contributes_enhanced)
                
                edge_index.append([full_frame_idx, i])
                influences_onehot = F.one_hot(torch.tensor(self.SPATIAL_REL['influences']), num_classes=10).float()
                influences_enhanced = torch.cat([influences_onehot, zero_continuous], dim=0)
                edge_attr.append(influences_enhanced)

            # Full frame self-loop
            edge_index.append([full_frame_idx, full_frame_idx])
            frame_self_onehot = F.one_hot(torch.tensor(self.SPATIAL_REL['self']), num_classes=10).float()
            zero_continuous = torch.zeros(6, dtype=torch.float32)
            frame_self_enhanced = torch.cat([frame_self_onehot, zero_continuous], dim=0)
            edge_attr.append(frame_self_enhanced)

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.stack(edge_attr, dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 16), dtype=torch.float)

        return edge_index, edge_attr

    def _compute_enhanced_edge_features(self, bbox_i, bbox_j):
        """Compute enhanced edge features (same as original)"""
        
        spatial_rel_idx = self._compute_spatial_relation(bbox_i, bbox_j)
        spatial_rel_onehot = F.one_hot(torch.tensor(spatial_rel_idx), num_classes=10).float()
        
        def compute_bbox_features(bbox):
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = width / (height + 1e-8)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            return {
                'width': width, 'height': height, 'area': area,
                'aspect_ratio': aspect_ratio,
                'center': torch.tensor([center_x, center_y])
            }
        
        features_i = compute_bbox_features(bbox_i)
        features_j = compute_bbox_features(bbox_j)
        
        center_distance = torch.norm(features_j['center'] - features_i['center'])
        bbox_distance = center_distance / math.sqrt(2.0)
        
        x1_i, y1_i, x2_i, y2_i = bbox_i
        x1_j, y1_j, x2_j, y2_j = bbox_j
        
        x1_int = max(x1_i, x1_j)
        y1_int = max(y1_i, y1_j)
        x2_int = min(x2_i, x2_j)
        y2_int = min(y2_i, y2_j)
        
        if x2_int > x1_int and y2_int > y1_int:
            intersection = (x2_int - x1_int) * (y2_int - y1_int)
        else:
            intersection = 0.0
        
        union = features_i['area'] + features_j['area'] - intersection
        bbox_iou = intersection / (union + 1e-8)
        
        size_ratio = torch.log(features_j['area'] / (features_i['area'] + 1e-8) + 1e-8)
        
        x_overlap_length = max(0, min(x2_i, x2_j) - max(x1_i, x1_j))
        x_union_length = max(x2_i, x2_j) - min(x1_i, x1_j)
        x_overlap = x_overlap_length / (x_union_length + 1e-8)
        
        y_overlap_length = max(0, min(y2_i, y2_j) - max(y1_i, y1_j))
        y_union_length = max(y2_i, y2_j) - min(y1_i, y1_j)
        y_overlap = y_overlap_length / (y_union_length + 1e-8)
        
        aspect_ratio_diff = torch.log(features_j['aspect_ratio'] / (features_i['aspect_ratio'] + 1e-8) + 1e-8)
        
        continuous_features = torch.tensor([
            bbox_distance, bbox_iou, size_ratio, x_overlap, y_overlap, aspect_ratio_diff
        ], dtype=torch.float32)
        
        enhanced_features = torch.cat([spatial_rel_onehot, continuous_features], dim=0)
        return enhanced_features

    def _compute_spatial_relation(self, bbox_i, bbox_j):
        """Compute spatial relationship (same as original)"""
        center_i = bbox_i[:2] + bbox_i[2:]/2
        center_j = bbox_j[:2] + bbox_j[2:]/2
        
        dx = center_j[0] - center_i[0]
        dy = center_j[1] - center_i[1]
        
        if (bbox_j[0] >= bbox_i[0] and bbox_j[2] <= bbox_i[2] and
            bbox_j[1] >= bbox_i[1] and bbox_j[3] <= bbox_i[3]):
            return self.SPATIAL_REL['within']
        
        if abs(dx) > abs(dy):
            if abs(dx) > 0.1:
                return self.SPATIAL_REL['right'] if dx > 0 else self.SPATIAL_REL['left']
            else:
                return self.SPATIAL_REL['horizontal']
        else:
            if abs(dy) > 0.1:
                return self.SPATIAL_REL['above'] if dy < 0 else self.SPATIAL_REL['below']
            else:
                return self.SPATIAL_REL['vertical']

def vqa_only_collate_fn(batch):
    """
    Enhanced collate function for full enhanced model training
    
    Handles graph data, questions, and VQA labels - compatible with full enhanced model
    """
    # Fast null check
    if not batch:
        return None
        
    # VQA-only data extraction
    graph_data_list = []
    input_ids_list = []
    attention_mask_list = []
    token_type_ids_list = []
    labels = []
    
    # Handle VQA-only format: (graph_data, question, vqa_label)
    for item in batch:
        if len(item) == 3:  # VQA-only format
            graph_data, tokenized_question, vqa_label = item
            graph_data_list.append(graph_data)
            
            input_ids_list.append(tokenized_question['input_ids'])
            attention_mask_list.append(tokenized_question['attention_mask'])
            
            # Handle token_type_ids if present
            if 'token_type_ids' in tokenized_question:
                token_type_ids_list.append(tokenized_question['token_type_ids'])
            else:
                # Create zeros if not present
                token_type_ids_list.append(torch.zeros_like(tokenized_question['input_ids']))
                
            labels.append(vqa_label)
        else:
            raise ValueError(f"Expected 3 items per batch sample for VQA-only, got {len(item)}")
    
    # Batch graph data using PyTorch Geometric
    batched_graph = Batch.from_data_list(graph_data_list)
    
    # Add tokenized questions to batch (compatible with full enhanced model)
    batched_graph.input_ids = torch.stack(input_ids_list, dim=0)
    batched_graph.attention_mask = torch.stack(attention_mask_list, dim=0)
    batched_graph.token_type_ids = torch.stack(token_type_ids_list, dim=0)
    batched_graph.y = torch.tensor(labels, dtype=torch.long)
    
    return batched_graph
