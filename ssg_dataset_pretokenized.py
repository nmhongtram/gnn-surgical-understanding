"""
Pre-tokenized Scene Graph VQA Dataset

This dataset loads pre-tokenized questions from HDF5 files to eliminate
CPU tokenization overhead during training.

Key Features:
- Pre-tokenized questions stored in HDF5 format
- Zero CPU tokenization during training
- Direct GPU tensor loading
- Maintains all original dataset functionality
"""

import pathlib
import glob
import h5py
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data

import config as cfg


class PreTokenizedSSGVQA(Dataset):
    """
    Pre-tokenized Scene Graph VQA Dataset
    
    Uses pre-tokenized questions from HDF5 files to eliminate CPU overhead
    during training. Falls back to original SSGVQA if tokenized files not found.
    
    Args:
        ana_type: Analysis types to filter
        mode: Dataset split (debug, train, val, test)
        use_pretokenized: Whether to use pre-tokenized questions (default: True)
    """

    def __init__(self, ana_type=[], mode="debug", use_pretokenized=True):
        
        self.mode = mode
        self.use_pretokenized = use_pretokenized
        self.vqas = []
        self.tokenized_data = None
        
        # Try to load pre-tokenized questions first
        if use_pretokenized:
            # Try to find the most specific pre-tokenized file
            tokenized_path = self._find_best_tokenized_file(mode, ana_type)
            
            if tokenized_path and tokenized_path.exists():
                print(f"📁 Loading pre-tokenized questions from {tokenized_path}")
                self._load_pretokenized_questions(tokenized_path, ana_type)
            else:
                print(f"⚠️ No suitable pre-tokenized file found for mode={mode}, ana_type={ana_type}")
                print("💡 Run preprocessing:")
                if ana_type:
                    ana_args = " ".join([f'"{at}"' for at in ana_type])
                    print(f"   python preprocess_questions.py --ana_type {ana_args} --modes {mode}")
                else:
                    print(f"   python preprocess_questions.py --modes {mode}")
                use_pretokenized = False
        
        # Fall back to original loading if pre-tokenized not available
        if not use_pretokenized or self.tokenized_data is None:
            print(f"📁 Loading questions dynamically from {mode}")
            self._load_questions_original(ana_type, mode)
        
        # Load other necessary mappings (same as original SSGVQA)
        self._load_mappings()
        
        print(f"✅ Dataset ready: {len(self.vqas)} questions")

    def _find_best_tokenized_file(self, mode, ana_type):
        """
        Find the best matching pre-tokenized file for given mode and ana_type
        
        Priority:
        1. Exact ana_type match: tokenized_{mode}_{ana_type}.h5
        2. All questions: tokenized_{mode}.h5
        3. Any existing file for the mode
        """
        base_path = cfg.TOKENIZED_QUESTION_DIR
        
        if ana_type:
            # Try exact ana_type match
            ana_suffix = "_".join(ana_type).replace('.json', '')
            specific_path = base_path / f"tokenized_{mode}_{ana_suffix}.h5"
            if specific_path.exists():
                return specific_path
            
            # Try individual ana_types
            for at in ana_type:
                single_suffix = at.replace('.json', '')
                single_path = base_path / f"tokenized_{mode}_{single_suffix}.h5"
                if single_path.exists():
                    print(f"🔍 Found partial match for ana_type: {single_path}")
                    return single_path
        
        # Try general file (all questions)
        general_path = base_path / f"tokenized_{mode}.h5"
        if general_path.exists():
            return general_path
        
        # Try any file matching the mode
        matching_files = list(base_path.glob(f"tokenized_{mode}*.h5"))
        if matching_files:
            print(f"🔍 Found alternative pre-tokenized file: {matching_files[0]}")
            return matching_files[0]
        
        return None

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

    def _load_questions_original(self, ana_type, mode):
        """Fallback to original question loading method with improved ana_type filtering"""
        qa_folder_path = cfg.QUESTIONS_DIR / mode
        file_list = list(qa_folder_path.glob("*.txt"))
        
        print(f"📁 Loading questions from {mode} mode (fallback method)")
        if ana_type:
            print(f"🔍 Filtering by ana_type: {ana_type}")
        
        total_loaded = 0
        ana_type_counts = {}
        
        for file in file_list:
            try:
                with open(file, "r", encoding='utf-8') as file_data:
                    lines = [line.strip("\n") for line in file_data if line.strip() != ""]
                    
                for idx, line in enumerate(lines):
                    if idx >= 2 and "|" in line:  # Skip header lines
                        parts = line.split("|")
                        if len(parts) >= 2:
                            # Extract ana_type from line if available
                            line_ana_type = None
                            if len(parts) >= 3:
                                line_ana_type = parts[2].strip()
                            
                            # Count ana_types for statistics
                            ana_key = line_ana_type if line_ana_type else 'unknown'
                            ana_type_counts[ana_key] = ana_type_counts.get(ana_key, 0) + 1
                            
                            # Filter by ana_type if specified
                            if ana_type:
                                if line_ana_type and line_ana_type in ana_type:
                                    self.vqas.append([file, line])
                                    total_loaded += 1
                                # Also check legacy format (column 3 and 4)
                                elif len(parts) >= 4:
                                    t1 = parts[2].strip()
                                    t2 = parts[3].strip()
                                    if t1 in ana_type or t2 in ana_type:
                                        self.vqas.append([file, line])
                                        total_loaded += 1
                            else:
                                # Load all questions
                                self.vqas.append([file, line])
                                total_loaded += 1
                                
            except Exception as e:
                print(f"⚠️ Error reading {file}: {e}")
                continue
        
        print(f"✅ Loaded {total_loaded} questions using fallback method")
        
        # Print ana_type distribution
        if ana_type_counts:
            print("📊 Ana_type distribution (fallback):")
            for at, count in sorted(ana_type_counts.items()):
                print(f"   {at}: {count} questions")

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
        # Determine how to get question and metadata
        if self.tokenized_data is not None:
            # Use pre-tokenized data
            tokenized_idx, metadata = self.vqas[idx]
            vidframe_id = metadata['file']
            question = metadata['question']
            answer = metadata['answer']
            
            # Get pre-tokenized question data
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
            question = self.vqas[idx][1].split("|")[0]
            answer = self.vqas[idx][1].split("|")[1]
            tokenized_question = None  # Will be tokenized in model
        
        # Load graph data (same as original)
        graph_data = self._load_graph_data(vidframe_id)
        
        # Load triplet data (same as original)
        triplet_data = self._load_triplet_data(vidframe_id)
        
        # Get label
        label = self.labels.index(str(answer))
        
        # Return format depends on whether we have pre-tokenized data
        if tokenized_question is not None:
            return graph_data, tokenized_question, label, triplet_data
        else:
            return graph_data, question, label, triplet_data

    def _load_graph_data(self, vidframe_id):
        """Load graph data (same as original SSGVQA.__getitem__)"""
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
            roi_features = torch.zeros(1, 530)

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
            with h5py.File(visual_path, "r") as visual_data:
                full_frame_visual_features = torch.from_numpy(visual_data["visual_features"][:])
        except FileNotFoundError:
            print(f"⚠️ Full frame features not found: {visual_path}")
            full_frame_visual_features = torch.zeros(1, 512)

        if full_frame_visual_features.shape[0] > 1:
            full_frame_visual_features = full_frame_visual_features.mean(dim=0, keepdim=True)
        
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
        import math
        
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

    def _load_triplet_data(self, vidframe_id):
        """Load triplet data (same as original)"""
        scene_graph_path = pathlib.Path.joinpath(
            cfg.SCENE_GRAPHS_DIR, 
            self.mode, 
            f"{vidframe_id}.json"
        )
        
        direct_triplet_ids = []
        
        try:
            with open(scene_graph_path, 'r') as f:
                scene_data = json.load(f)
                
            if 'info' in scene_data and 'triplet' in scene_data['info'] and scene_data['info']['triplet']:
                triplets = scene_data['info']['triplet']
                
                for triplet_str in triplets:
                    parts = triplet_str.split(',')
                    if len(parts) == 3:
                        instrument, verb, target = [part.strip() for part in parts]
                        
                        instrument_idx = self._map_triplet_component(instrument, self.instrument_vocab, 'null_instrument')
                        verb_idx = self._map_triplet_component(verb, self.verb_vocab, 'null_verb')
                        target_idx = self._map_triplet_component(target, self.target_vocab, 'null_target')
                        
                        triplet_key = (instrument_idx, verb_idx, target_idx)
                        if triplet_key in self.triplet_to_id:
                            direct_triplet_ids.append(self.triplet_to_id[triplet_key])
                        else:
                            direct_triplet_ids.append(100)
                            
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ Error loading triplets for {vidframe_id}: {e}")
        
        if not direct_triplet_ids:
            direct_triplet_ids = [100]
        
        return direct_triplet_ids

    def _map_triplet_component(self, component, vocab, null_key):
        """Map triplet component (same as original)"""
        if component.startswith('null_') or component == 'null':
            return vocab.get(null_key, vocab.get('null_instrument', vocab.get('null_verb', vocab.get('null_target', 0))))
        
        if component in vocab:
            return vocab[component]
        
        for key, idx in vocab.items():
            if key.lower() == component.lower():
                return idx
        
        return vocab.get(null_key, vocab.get('null_instrument', vocab.get('null_verb', vocab.get('null_target', 0))))


def pretokenized_collate_fn(batch):
    """
    Collate function for pre-tokenized dataset
    
    Handles both pre-tokenized and original question formats
    """
    from torch_geometric.data import Batch
    
    if not batch or any(item is None for item in batch):
        return None
    
    graph_data_list = []
    questions_or_tokens = []
    vqa_labels = []
    triplet_labels = []
    
    has_pretokenized = False
    
    for item in batch:
        if item is None:
            continue
        graph_data, question_or_token, vqa_label, triplet_data = item
        
        graph_data_list.append(graph_data)
        questions_or_tokens.append(question_or_token)
        vqa_labels.append(vqa_label)
        triplet_labels.append(triplet_data)
        
        # Check if we have pre-tokenized data
        if isinstance(question_or_token, dict) and 'input_ids' in question_or_token:
            has_pretokenized = True
    
    if not graph_data_list:
        return None
    
    # Batch graph data
    try:
        batched_graph = Batch.from_data_list(graph_data_list)
    except Exception as e:
        print(f"⚠️ Graph batching error: {e}")
        return None
    
    # Convert VQA labels
    vqa_tensor = torch.tensor(vqa_labels, dtype=torch.long)
    
    # Handle question data
    if has_pretokenized:
        # Stack pre-tokenized tensors
        input_ids = torch.stack([q['input_ids'] for q in questions_or_tokens])
        attention_mask = torch.stack([q['attention_mask'] for q in questions_or_tokens])
        
        tokenized_batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Add token_type_ids if available
        if 'token_type_ids' in questions_or_tokens[0]:
            token_type_ids = torch.stack([q['token_type_ids'] for q in questions_or_tokens])
            tokenized_batch['token_type_ids'] = token_type_ids
        
        questions_data = tokenized_batch
    else:
        # Use original question strings
        questions_data = questions_or_tokens
    
    # Handle triplet labels
    if triplet_labels and len(triplet_labels) > 0:
        max_triplets = max(len(sample_triplets) for sample_triplets in triplet_labels)
        
        if max_triplets > 0:
            batch_size = len(triplet_labels)
            triplet_tensor = torch.full((batch_size, max_triplets), -1, dtype=torch.long)
            
            for i, sample_triplets in enumerate(triplet_labels):
                num_triplets = len(sample_triplets)
                if num_triplets > 0:
                    triplet_tensor[i, :num_triplets] = torch.tensor(sample_triplets, dtype=torch.long)
            
            triplet_batch = triplet_tensor
        else:
            batch_size = len(triplet_labels)
            triplet_batch = torch.full((batch_size, 1), 100, dtype=torch.long)
    else:
        batch_size = len(graph_data_list)
        triplet_batch = torch.full((batch_size, 1), 100, dtype=torch.long)
    
    return batched_graph, questions_data, vqa_tensor, triplet_batch


# Backward compatibility - alias to original dataset
SSGVQA = PreTokenizedSSGVQA
multi_triplet_collate_fn = pretokenized_collate_fn