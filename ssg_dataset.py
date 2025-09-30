"""
Enhanced Scene Graph VQA Dataloader with Hierarchical Graph Structure

Key Features:
- Full frame as central hub node (class index 15)
- Hierarchical edges: object-object + object-fullframe + fullframe-object
- Handles roi_coord_gt files without padding (single row of zeros for empty files)
- Self-connected + bidirectional edges for all nodes (tham khảo GQA + LRTA paper)
- Enhanced spatial relationship encoding

Architecture:
- Object nodes: [bbox(4) + roi_visual(512)] = 516 dims
- Full frame node: [full_bbox(4) + full_visual(512)] = 516 dims  
- Classes: 0-13 (existing) + 15 (full_frame)
- Relations: 0-7 (spatial) + 8 (contributes_to) + 9 (influences)
"""


import pathlib
import glob
import h5py
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
# import torchvision.transforms as transforms
from torch_geometric.data import Data

import config as cfg


class SSGVQA(Dataset):
    """
    ana_type: 
        - zero_hop.json, one_hop.json, single_and.json
        - query_color, query_type, query_location
        - query_component
        - count, exist
    mode: debug, train, val, test
    """

    def __init__(self, ana_type=[], mode="debug"):
        
        self.mode = mode  # Store mode for scene graph loading
        self.vqas = []

        qa_folder_path = cfg.QUESTIONS_DIR / mode
        file_list = list(qa_folder_path.glob("*.txt"))  # Convert generator to list

        if ana_type:
            for file in file_list:
                file_data = open(file, "r")
                lines = [line.strip("\n") for line in file_data if line != "\n"]
                file_data.close()
                for idx, line in enumerate(lines):
                    if idx >= 2 and line.count("|") >= 3:
                        ll = line.split("|")
                        t1 = ll[2]
                        t2 = ll[3]
                        if t1 in ana_type or t2 in ana_type:
                            self.vqas.append([file, line])
        else:
            for file in file_list:
                file_data = open(file, "r")
                lines = [line.strip("\n") for line in file_data if line != "\n"]
                file_data.close()
                for idx, line in enumerate(lines):
                    if idx >= 2:
                        self.vqas.append([file, line])
        print(
            "Total files: %d | Total question: %.d"
            % (len(file_list), len(self.vqas))
        )

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
        
        # Load direct triplet mapping for multi-task learning
        with open(cfg.META_INFO_DIR / "triplet_components.json") as f:
            self.triplet_mapping = json.load(f)
            self.instrument_vocab = self.triplet_mapping["instrument"]
            self.target_vocab = self.triplet_mapping["target"] 
            self.verb_vocab = self.triplet_mapping["verb"]
        
        # Load direct triplet ID mapping from maps.txt
        self.triplet_to_id = {}  # (instrument, verb, target) -> triplet_id
        with open(cfg.META_INFO_DIR / "maps.txt") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        triplet_id, inst_id, verb_id, target_id, _, _ = map(int, line.split(','))
                        self.triplet_to_id[(inst_id, verb_id, target_id)] = triplet_id
                    except ValueError:
                        continue

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        # Get video and frame info
        file_path = self.vqas[idx][0]  # This is a pathlib.Path object
        vidframe_id = file_path.stem  # Get filename without .txt extension

        # 1. Load ROI features (classes + bbox + visual features)
        roi_feature_path = pathlib.Path.joinpath(
            cfg.VISUAL_FEATURES_DIR,
            "roi_coord_gt",
            f"{vidframe_id}.hdf5",
        )
        
        try:
            with h5py.File(roi_feature_path, "r") as roi_data:
                roi_features = torch.from_numpy(roi_data["visual_features"][:])  # [N, 530]
        except FileNotFoundError:
            print(f"⚠️ ROI features not found: {roi_feature_path}")
            # Create single zero row for missing file
            roi_features = torch.zeros(1, 530)

        # Check if file contains only zeros (no objects detected)
        if roi_features.shape[0] == 1 and torch.all(roi_features == 0):
            # Empty file with single zero row - no objects detected
            object_classes_indices = torch.tensor([])
            object_bboxes = torch.zeros(0, 4)
            object_visual_features = torch.zeros(0, 512)
            num_objects = 0
        else:
            # Extract components from ROI features [classes(14) + bbox(4) + visual(512)]
            classes_one_hot = roi_features[:, :14]           # [N, 14]
            classes_indices = classes_one_hot.argmax(dim=1)  # [N]
            bboxes = roi_features[:, 14:18]                  # [N, 4] 
            roi_visual_features = roi_features[:, 18:]       # [N, 512]

            # Filter valid objects (objects with assigned classes)
            valid_mask = (classes_one_hot.sum(dim=1) > 0)
            
            # Apply filtering
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
            with h5py.File(visual_path, "r") as visual_data:
                full_frame_visual_features = torch.from_numpy(visual_data["visual_features"][:])  # [1, 512]
        except FileNotFoundError:
            print(f"⚠️ Full frame features not found: {visual_path}")
            full_frame_visual_features = torch.zeros(1, 512)

        # Ensure we have single feature vector for full frame
        if full_frame_visual_features.shape[0] > 1:
            full_frame_visual_features = full_frame_visual_features.mean(dim=0, keepdim=True)  # [1, 512]
        
        # 3. Create hierarchical graph with full frame as hub node
        # Total nodes = objects + full frame hub
        total_nodes = num_objects + 1
        full_frame_idx = num_objects  # Full frame is the last node
        
        # Create node features
        if num_objects == 0:
            # Only full frame node (test set case)
            node_features = torch.cat([
                torch.tensor([[0.0, 0.0, 1.0, 1.0]]),  # Full frame bbox [1, 4]
                full_frame_visual_features               # Full frame visual [1, 512]
            ], dim=1)  # [1, 516]
            
            classes_indices = torch.tensor([14])  # "full_frame" class
            
        else:
            # Objects + full frame hub node
            # Object node features: [bbox(4) + roi_visual(512)]
            object_node_features = torch.cat([
                object_bboxes,           # [num_objects, 4]
                object_visual_features   # [num_objects, 512]
            ], dim=1)  # [num_objects, 516]
            
            # Full frame node features: [full_bbox(4) + full_visual(512)]
            full_frame_node_features = torch.cat([
                torch.tensor([[0.0, 0.0, 1.0, 1.0]]),  # Full frame bbox [1, 4]
                full_frame_visual_features               # Full frame visual [1, 512]
            ], dim=1)  # [1, 516]
            
            # Combine all node features
            node_features = torch.cat([
                object_node_features,      # [num_objects, 516]
                full_frame_node_features   # [1, 516]
            ], dim=0)  # [total_nodes, 516]
            
            # Combine class indices
            classes_indices = torch.cat([
                object_classes_indices,    # [num_objects]
                torch.tensor([14])         # "full_frame" class [1]
            ])  # [total_nodes]

        # 4. Create hierarchical edges
        edge_index = []
        edge_attr = []
        
        if num_objects == 0:
            # Only full frame node - add self-loop with enhanced features
            edge_index.append([0, 0])
            # Create self-loop enhanced features: one-hot for 'self' + zero continuous
            self_onehot = F.one_hot(torch.tensor(self.SPATIAL_REL['self']), num_classes=10).float()
            zero_continuous = torch.zeros(6, dtype=torch.float32) 
            self_enhanced = torch.cat([self_onehot, zero_continuous], dim=0)
            edge_attr.append(self_enhanced)
            
        else:
            # 1. Object-Object spatial relationships with enhanced features
            for i in range(num_objects):
                for j in range(num_objects):
                    if i != j:
                        edge_index.append([i, j])
                        # NEW: Use enhanced edge features (16-dim)
                        enhanced_features = self._compute_enhanced_edge_features(
                            object_bboxes[i], object_bboxes[j]
                        )
                        edge_attr.append(enhanced_features)
            
            # 2. Object self-loops with enhanced features
            for i in range(num_objects):
                edge_index.append([i, i])
                # Self-loop enhanced features
                self_onehot = F.one_hot(torch.tensor(self.SPATIAL_REL['self']), num_classes=10).float()
                zero_continuous = torch.zeros(6, dtype=torch.float32)
                self_enhanced = torch.cat([self_onehot, zero_continuous], dim=0)
                edge_attr.append(self_enhanced)

            # 3. Object-FullFrame hierarchical relationships with enhanced features
            for i in range(num_objects):
                # Object → Full Frame (object contributes to global context)
                edge_index.append([i, full_frame_idx])
                contributes_onehot = F.one_hot(torch.tensor(self.SPATIAL_REL['contributes_to']), num_classes=10).float()
                zero_continuous = torch.zeros(6, dtype=torch.float32)
                contributes_enhanced = torch.cat([contributes_onehot, zero_continuous], dim=0)
                edge_attr.append(contributes_enhanced)
                
                # Full Frame → Object (global context influences object)  
                edge_index.append([full_frame_idx, i])
                influences_onehot = F.one_hot(torch.tensor(self.SPATIAL_REL['influences']), num_classes=10).float()
                influences_enhanced = torch.cat([influences_onehot, zero_continuous], dim=0)
                edge_attr.append(influences_enhanced)

            # 4. Full Frame self-loop (global consistency) with enhanced features
            edge_index.append([full_frame_idx, full_frame_idx])
            frame_self_onehot = F.one_hot(torch.tensor(self.SPATIAL_REL['self']), num_classes=10).float()
            zero_continuous = torch.zeros(6, dtype=torch.float32)
            frame_self_enhanced = torch.cat([frame_self_onehot, zero_continuous], dim=0)
            edge_attr.append(frame_self_enhanced)

        # 5. Convert lists to tensors
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            # NEW: Stack enhanced edge features (already in tensor format)
            edge_attr = torch.stack(edge_attr, dim=0)  # [num_edges, 16]
        else:
            # Create empty edge tensors if no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 16), dtype=torch.float)  # [0, 16] for enhanced features

        # 6. Load direct triplet IDs for multi-task learning
        scene_graph_path = pathlib.Path.joinpath(
            cfg.SCENE_GRAPHS_DIR, 
            self.mode if hasattr(self, 'mode') else 'debug', 
            f"{vidframe_id}.json"
        )
        
        direct_triplet_ids = []
        
        try:
            with open(scene_graph_path, 'r') as f:
                scene_data = json.load(f)
                
            # Extract triplets from scene graph and convert to direct IDs
            if 'info' in scene_data and 'triplet' in scene_data['info'] and scene_data['info']['triplet']:
                triplets = scene_data['info']['triplet']
                
                for triplet_str in triplets:
                    parts = triplet_str.split(',')
                    if len(parts) == 3:
                        instrument, verb, target = [part.strip() for part in parts]
                        
                        # Map to component indices first
                        instrument_idx = self._map_triplet_component(instrument, self.instrument_vocab, 'null_instrument')
                        verb_idx = self._map_triplet_component(verb, self.verb_vocab, 'null_verb')
                        target_idx = self._map_triplet_component(target, self.target_vocab, 'null_target')
                        
                        # Convert to direct triplet ID
                        triplet_key = (instrument_idx, verb_idx, target_idx)
                        if triplet_key in self.triplet_to_id:
                            direct_triplet_ids.append(self.triplet_to_id[triplet_key])
                        else:
                            # Fallback to null triplet (ID 100)
                            direct_triplet_ids.append(100)
                            
        except FileNotFoundError:
            # Scene graph file not found - frame without annotations
            pass  # Will use default null values below
        except Exception as e:
            print(f"⚠️ Error loading triplets for {vidframe_id}: {e}")
        
        # Ensure at least one triplet ID (null if no annotations)
        if not direct_triplet_ids:
            # No triplets found - use null triplet ID 100
            direct_triplet_ids = [100]
        
        # Convert to tensor
        direct_triplet_labels = torch.tensor(direct_triplet_ids, dtype=torch.long)

        # Create PyG graph data
        # Store class indices for model to handle BERT embeddings
        graph_data = Data(
            x=node_features,              # Node features [num_nodes, 516] (4+512)
            class_indices=classes_indices, # Class indices for trainable BERT embedding [num_nodes]
            edge_index=edge_index,        # Graph connectivity [2, num_edges] 
            edge_attr=edge_attr,          # Edge features [num_edges, 16] (enhanced: 10-dim categorical + 6-dim continuous)
            # Additional attributes
            vidframe_id=vidframe_id,           
            num_nodes=total_nodes,        # Fixed: total nodes including full frame
            # Direct triplet labels
            triplet_labels=direct_triplet_labels
        )

        # question and answer
        question = self.vqas[idx][1].split("|")[0]
        label = self.labels.index(str(self.vqas[idx][1].split("|")[1]))
        
        # Return direct triplet IDs as simple list
        triplet_data = direct_triplet_ids
        
        return graph_data, question, label, triplet_data

    def _compute_spatial_relation(self, bbox_i, bbox_j):
        """Compute categorical spatial relationship between two bounding boxes."""
        # Calculate centers
        center_i = bbox_i[:2] + bbox_i[2:]/2
        center_j = bbox_j[:2] + bbox_j[2:]/2
        
        dx = center_j[0] - center_i[0]
        dy = center_j[1] - center_i[1]
        
        # Check containment (within)
        if (bbox_j[0] >= bbox_i[0] and bbox_j[2] <= bbox_i[2] and
            bbox_j[1] >= bbox_i[1] and bbox_j[3] <= bbox_i[3]):
            return self.SPATIAL_REL['within']
        
        # Check directional relationships
        if abs(dx) > abs(dy):
            if abs(dx) > 0.1:  # Threshold for horizontal relationship
                return self.SPATIAL_REL['right'] if dx > 0 else self.SPATIAL_REL['left']
            else:
                return self.SPATIAL_REL['horizontal']
        else:
            if abs(dy) > 0.1:  # Threshold for vertical relationship
                return self.SPATIAL_REL['above'] if dy < 0 else self.SPATIAL_REL['below']
            else:
                return self.SPATIAL_REL['vertical']
    
    def _compute_enhanced_edge_features(self, bbox_i, bbox_j):
        """
        🔥 NEW: Compute enhanced edge features with categorical + continuous components
        
        Returns:
            enhanced_features: [16] tensor - 10-dim categorical + 6-dim continuous
        """
        import math
        
        # 1. Categorical spatial relation (same as before)
        spatial_rel_idx = self._compute_spatial_relation(bbox_i, bbox_j)
        spatial_rel_onehot = F.one_hot(torch.tensor(spatial_rel_idx), num_classes=10).float()
        
        # 2. Continuous spatial features computation
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
        
        # Distance between centers (normalized)
        center_distance = torch.norm(features_j['center'] - features_i['center'])
        bbox_distance = center_distance / math.sqrt(2.0)  # Normalize by image diagonal
        
        # Intersection over Union (IoU)
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
        
        # Relative size ratio (log scale)
        size_ratio = torch.log(features_j['area'] / (features_i['area'] + 1e-8) + 1e-8)
        
        # Overlap ratios
        x_overlap_length = max(0, min(x2_i, x2_j) - max(x1_i, x1_j))
        x_union_length = max(x2_i, x2_j) - min(x1_i, x1_j)
        x_overlap = x_overlap_length / (x_union_length + 1e-8)
        
        y_overlap_length = max(0, min(y2_i, y2_j) - max(y1_i, y1_j))
        y_union_length = max(y2_i, y2_j) - min(y1_i, y1_j)
        y_overlap = y_overlap_length / (y_union_length + 1e-8)
        
        # Aspect ratio difference (log scale)
        aspect_ratio_diff = torch.log(features_j['aspect_ratio'] / (features_i['aspect_ratio'] + 1e-8) + 1e-8)
        
        # Combine continuous features
        continuous_features = torch.tensor([
            bbox_distance,      # [0, sqrt(2)]
            bbox_iou,          # [0, 1]
            size_ratio,        # log scale
            x_overlap,         # [0, 1]
            y_overlap,         # [0, 1]
            aspect_ratio_diff  # log scale
        ], dtype=torch.float32)
        
        # Combine: [10-dim categorical] + [6-dim continuous] = 16-dim total
        enhanced_features = torch.cat([spatial_rel_onehot, continuous_features], dim=0)
        
        return enhanced_features
    
    def _map_triplet_component(self, component, vocab, null_key):
        """
        NEW: Robust triplet component mapping with null handling
        
        Args:
            component: Raw string component (e.g., 'grasper', 'null_verb')
            vocab: Vocabulary dictionary 
            null_key: Key for null value in vocab
            
        Returns:
            int: Mapped index, always valid (uses null if not found)
        """
        # Handle explicit null values
        if component.startswith('null_') or component == 'null':
            return vocab.get(null_key, vocab.get('null_instrument', vocab.get('null_verb', vocab.get('null_target', 0))))
        
        # Try exact match first
        if component in vocab:
            return vocab[component]
        
        # Try case-insensitive match
        for key, idx in vocab.items():
            if key.lower() == component.lower():
                return idx
        
        # Component not found - use null
        return vocab.get(null_key, vocab.get('null_instrument', vocab.get('null_verb', vocab.get('null_target', 0))))
    
    def _get_null_indices(self):
        """
        NEW: Get null indices for all triplet components
        
        Returns:
            dict: Null indices for instrument, verb, target
        """
        return {
            'instrument': self.instrument_vocab.get('null_instrument', 6),  # Index 6 in triplet.json
            'verb': self.verb_vocab.get('null_verb', 9),                    # Index 9 in triplet.json  
            'target': self.target_vocab.get('null_target', 14)              # Index 14 in triplet.json
        }


def multi_triplet_collate_fn(batch):
    """
    Enhanced collate function for direct triplet ID prediction
    Handles variable number of direct triplet IDs per sample with proper padding
    
    Input: List of (graph_data, question, vqa_label, triplet_ids)
    Output: (batched_graph, questions, vqa_tensor, triplet_tensor)
    """
    from torch_geometric.data import Batch
    
    if not batch or any(item is None for item in batch):
        return None
    
    # Separate components
    graph_data_list = []
    questions = []
    vqa_labels = []
    triplet_labels = []
    
    for item in batch:
        if item is None:
            continue
        graph_data, question, vqa_label, triplet_data = item
        
        graph_data_list.append(graph_data)
        questions.append(question)
        vqa_labels.append(vqa_label)
        triplet_labels.append(triplet_data)
    
    if not graph_data_list:
        return None
    
    # Batch graph data
    try:
        batched_graph = Batch.from_data_list(graph_data_list)
        # Ensure graph data is moved to the correct device (will use cuda if available)
        # Note: Individual components should already be on CPU, device transfer happens in training loop
    except Exception as e:
        print(f"Graph batching error: {e}")
        return None
    
    # Convert VQA labels to tensor
    vqa_tensor = torch.tensor(vqa_labels, dtype=torch.long)
    
    # Handle triplet labels - now direct triplet IDs per sample
    if triplet_labels and len(triplet_labels) > 0:
        # Find max number of triplets in this batch
        max_triplets = max(len(sample_triplets) for sample_triplets in triplet_labels)
        
        if max_triplets > 0:
            batch_size = len(triplet_labels)
            
            # Initialize tensors for direct triplet IDs (use -1 for padding to avoid confusion with triplet ID 0)
            triplet_tensor = torch.full((batch_size, max_triplets), -1, dtype=torch.long)
            triplet_mask = torch.zeros((batch_size, max_triplets), dtype=torch.bool)
            
            for i, sample_triplets in enumerate(triplet_labels):
                num_triplets = len(sample_triplets)
                if num_triplets > 0:
                    triplet_tensor[i, :num_triplets] = torch.tensor(sample_triplets, dtype=torch.long)
                    triplet_mask[i, :num_triplets] = True
            
            # Create triplet batch data - simplified for direct IDs
            triplet_batch = triplet_tensor
        else:
            # No triplets in batch - use null triplet ID 100
            batch_size = len(triplet_labels)
            triplet_batch = torch.full((batch_size, 1), 100, dtype=torch.long)
    else:
        # No triplet data - use null triplet ID 100
        batch_size = len(graph_data_list)
        triplet_batch = torch.full((batch_size, 1), 100, dtype=torch.long)
    
    return batched_graph, questions, vqa_tensor, triplet_batch