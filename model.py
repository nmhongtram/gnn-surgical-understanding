#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, MessagePassing
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, degree
from transformers import AutoTokenizer, AutoModel
import json
import config as cfg
from collections import defaultdict
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, f1_score

# Graph normalization dependencies
from torch_geometric.utils import degree
from torch_scatter import scatter


class GraphLayerNorm(nn.Module):
    """
    Graph-aware Layer Normalization adapted from GraphVQA
    Normalizes features within each graph separately for variable graph sizes
    """
    
    def __init__(self, in_channels, eps=1e-5, affine=True):
        super(GraphLayerNorm, self).__init__()
        self.in_channels = in_channels
        self.eps = eps
        
        if affine:
            self.weight = nn.Parameter(torch.ones(in_channels))
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x, batch=None):
        """
        Args:
            x: Node features [N, D]
            batch: Batch assignment for each node [N] (optional)
        Returns:
            Normalized features [N, D]
        """
        if batch is None:
            # Standard layer normalization
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            out = (x - mean) / (var + self.eps).sqrt()
        else:
            # Graph-aware normalization using GraphVQA approach
            batch_size = int(batch.max()) + 1
            
            # Calculate normalization factor (nodes per graph * feature dimensions)
            norm = degree(batch, batch_size, dtype=x.dtype).clamp_(min=1)
            norm = norm.mul_(x.size(-1)).view(-1, 1)
            
            # Calculate mean per graph
            mean = scatter(x, batch, dim=0, dim_size=batch_size, 
                         reduce='add').sum(dim=-1, keepdim=True) / norm
            
            # Center the data by subtracting graph-specific mean from each node
            x = x - mean[batch]
            
            # Calculate variance per graph using centered data
            var = scatter(x * x, batch, dim=0, dim_size=batch_size,
                        reduce='add').sum(dim=-1, keepdim=True) / norm
            
            # Normalize using graph-specific variance
            out = x / (var.sqrt()[batch] + self.eps)
        
        # Apply learnable parameters
        if self.weight is not None and self.bias is not None:
            out = out * self.weight + self.bias
        
        return out


class SurgicalSceneGraphVQA(nn.Module):
    def __init__(self, num_classes, node_dim=516, hidden_dim=768,  # Increased from 512 to 768
                 question_dim=768, freeze_bert=True, object_class_embed_dim=256,
                 gnn_type='gcn', enable_triplet_detection=True, use_uncertainty_weights=True):
        super(SurgicalSceneGraphVQA, self).__init__()
        
        self.num_classes = num_classes
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.question_dim = question_dim
        self.gnn_type = gnn_type
        self.enable_triplet = enable_triplet_detection
        self.use_uncertainty_weights = use_uncertainty_weights
        
        # Object class encoder - integrated directly
        self.object_class_embed_dim = object_class_embed_dim
        self.num_object_classes = 15
        
        # Load class names for object embedding initialization
        with open(cfg.META_INFO_DIR / "objects.json") as f:
            class_name_to_idx = json.load(f)
            self.class_names = [''] * self.num_object_classes
            for class_name, idx in class_name_to_idx.items():
                if idx < self.num_object_classes:
                    self.class_names[idx] = class_name
        
        # Wait to initialize class embeddings after BERT is loaded
        
        combined_node_dim = node_dim + object_class_embed_dim
        
        # Input projection
        self.node_projection = nn.Linear(combined_node_dim, hidden_dim)
        
        # GNN architectures - Pure implementations for comparison
        if gnn_type == 'gcn':
            # Pure GCN layers
            self.gcn_layers = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim),
                GCNConv(hidden_dim, hidden_dim),
                GCNConv(hidden_dim, hidden_dim)
            ])
            # Graph-specific normalization for GCN (restored GraphLayerNorm)
            self.gcn_graph_norms = nn.ModuleList([
                GraphLayerNorm(hidden_dim) for _ in range(3)
            ])
            
        elif gnn_type == 'gat':
            # Pure GAT layers - updated for hidden_dim=768
            head_dim = hidden_dim // 8  # 768 // 8 = 96
            self.gat_layers = nn.ModuleList([
                GATConv(hidden_dim, head_dim, heads=8, edge_dim=16, dropout=0.1, concat=True),
                GATConv(hidden_dim, head_dim, heads=8, edge_dim=16, dropout=0.1, concat=True),
                GATConv(hidden_dim, hidden_dim, heads=1, edge_dim=16, dropout=0.1, concat=False)
            ])
            # Graph-specific normalization for GAT (restored GraphLayerNorm)
            self.gat_graph_norms = nn.ModuleList([
                GraphLayerNorm(hidden_dim) for _ in range(3)
            ])
            
        elif gnn_type == 'gin':
            # Pure GIN layers  
            self.gin_layers = nn.ModuleList([
                GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )),
                GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(), 
                    nn.Linear(hidden_dim, hidden_dim)
                )),
                GINConv(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                ))
            ])
            # Graph-specific normalization for GIN (restored GraphLayerNorm)
            self.gin_graph_norms = nn.ModuleList([
                GraphLayerNorm(hidden_dim) for _ in range(3)
            ])
            
        elif gnn_type == 'none':
            # No GNN - Baseline with simple MLP processing
            self.baseline_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # Standard normalization for baseline
            self.baseline_norm = nn.LayerNorm(hidden_dim)
            
        else:
            raise ValueError(f"Unsupported gnn_type: {gnn_type}. Choose from 'gcn', 'gat', 'gin', 'none'")
            
        # BERT for question encoding
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.bert_model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        
        if freeze_bert:
            for param in self.bert_model.parameters():
                param.requires_grad = False
            print("BERT parameters frozen - using as feature extractor only")
        
        # Initialize object class embeddings now that BERT is available
        self.class_embeddings = nn.Embedding(self.num_object_classes, object_class_embed_dim)
        self.class_projection = nn.Linear(object_class_embed_dim, object_class_embed_dim)
        self._initialize_class_embeddings()
        
        # Visual Token Type Embeddings (VisualBERT component)
        self.visual_token_type_embeddings = nn.Embedding(3, hidden_dim)  # 0: text, 1: visual, 2: special
        self.visual_position_embeddings = nn.Embedding(100, hidden_dim)  # For visual spatial positions
        
        # Enhanced Visual Feature Preprocessing (VisualBERT style)
        self.visual_preprocessing = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
            
        # Question projection to match graph feature dimension
        self.question_projection = nn.Linear(question_dim, hidden_dim)
        
        # VisualBERT-style cross-modal attention fusion
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Self-attention for refined features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Specialized VisualBERT Scene-Text Attention Layers
        self.scene_to_text_attention = nn.ModuleList([
            self._create_scene_text_attention_layer(hidden_dim) for _ in range(2)
        ])
        
        self.text_to_scene_attention = nn.ModuleList([
            self._create_text_scene_attention_layer(hidden_dim) for _ in range(2)
        ])
        
        # Lightweight fusion layers (replace expensive bilinear)
        self.graph_projection = nn.Linear(hidden_dim, hidden_dim)
        self.question_projection = nn.Linear(question_dim, hidden_dim)
        self.fusion_norm1 = nn.LayerNorm(hidden_dim)
        self.fusion_norm2 = nn.LayerNorm(hidden_dim)
        
        # Multi-task heads with stronger regularization
        # VQA head
        self.vqa_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased from 0.1 to 0.3
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # Additional layer
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, num_classes)
        )
        
        # Direct triplet classification head with stronger regularization
        if self.enable_triplet:
            self.direct_triplet_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),  # Increased from 0.1 to 0.3
                nn.Linear(hidden_dim // 2, hidden_dim // 4),  # Additional layer
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 4, 101)  # 101 valid triplet combinations
            )
        
        # Uncertainty-based loss weighting parameters
        if self.use_uncertainty_weights:
            # Learnable log(sigma^2) parameters for the 2 main tasks
            # Following "Multi-Task Learning Using Uncertainty to Weigh Losses"
            # Both tasks are complex and deserve similar initial uncertainty:
            # 
            # VQA Task Complexity:
            # - Multi-modal fusion (vision + text)
            # - Natural language understanding
            # - 51 answer classes with varying difficulty
            #
            # Triplet Task Complexity: 
            # - 101 triplet classes with extreme imbalance
            # - Multi-label classification (multiple triplets per scene)
            # - Graph structure reasoning (instrument-verb-target relationships)
            # - Scene understanding from surgical context
            #
            # Initialize with moderate uncertainty for both (balanced approach)
            self.log_var_vqa = nn.Parameter(torch.tensor([0.5]))        # VQA task uncertainty 
            self.log_var_triplet = nn.Parameter(torch.tensor([0.5]))    # Triplet task uncertainty
            
            print("🎯 Uncertainty parameters initialized (balanced approach):")
            print(f"   VQA: log_var={self.log_var_vqa.item():.1f} → sigma²={torch.exp(self.log_var_vqa).item():.2f} → precision={1.0/torch.exp(self.log_var_vqa).item():.2f}")
            print(f"   Triplet: log_var={self.log_var_triplet.item():.1f} → sigma²={torch.exp(self.log_var_triplet).item():.2f} → precision={1.0/torch.exp(self.log_var_triplet).item():.2f}")
            print(f"   Both tasks start with equal uncertainty and will adapt during training")
        
        # Load triplet vocabulary if triplet detection is enabled
        self._load_triplet_vocab()
        
        # Load triplet mapping for direct classification
        self._load_triplet_mapping()
        
        # Load and validate class distributions
        self._validate_class_distributions()
        
        # Initialize class-balanced weights for extreme class imbalance
        self._initialize_class_balanced_weights()
    
    def _load_triplet_mapping(self):
        """Load mapping between triplet components and direct triplet IDs"""
        try:
            # Load triplet components mapping
            with open(cfg.META_INFO_DIR / "triplet_components.json") as f:
                self.triplet_components = json.load(f)
            
            # Load valid triplets using maps.txt (the authoritative encoding file)
            self.valid_triplets = {}  # triplet_id -> (instrument, verb, target)
            self.triplet_to_id = {}   # (instrument, verb, target) -> triplet_id
            self.triplet_names = {}   # triplet_id -> (instrument_name, verb_name, target_name)
            
            # First load triplet names from triplets.txt
            with open(cfg.META_INFO_DIR / "triplets.txt") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(':')
                        if len(parts) == 2:
                            triplet_id = int(parts[0])
                            triplet_str = parts[1]
                            components = triplet_str.split(',')
                            if len(components) == 3:
                                self.triplet_names[triplet_id] = tuple(components)
            
            # Then load the correct ID mappings from maps.txt
            with open(cfg.META_INFO_DIR / "maps.txt") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(',')
                        if len(parts) >= 4:  # triplet_id, instrument_id, verb_id, target_id, ...
                            triplet_id = int(parts[0])
                            instrument_id = int(parts[1])
                            verb_id = int(parts[2]) 
                            target_id = int(parts[3])
                            
                            if triplet_id >= 0:
                                # All valid triplets (including null triplet with ID 100)
                                self.valid_triplets[triplet_id] = (instrument_id, verb_id, target_id)
                                self.triplet_to_id[(instrument_id, verb_id, target_id)] = triplet_id
            
            print(f"✅ Loaded {len(self.valid_triplets)} valid triplet combinations")
            
        except Exception as e:
            print(f"Error loading triplet mapping: {e}")
            # Fallback - create empty mappings
            self.valid_triplets = {}
            self.triplet_to_id = {}
    
    def _initialize_class_embeddings(self):
        """Initialize object class embeddings using BERT text embeddings"""
        embeddings = []
        
        with torch.no_grad():
            for i in range(self.num_object_classes):
                if i < len(self.class_names):
                    class_name = self.class_names[i]
                    text = class_name.replace('_', ' ')
                else:
                    text = "unknown object"
                
                inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                outputs = self.bert_model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                
                if cls_embedding.shape[1] > self.object_class_embed_dim:
                    cls_embedding = cls_embedding[:, :self.object_class_embed_dim]
                else:
                    padding = torch.zeros(1, self.object_class_embed_dim - cls_embedding.shape[1], 
                                        device=cls_embedding.device, dtype=cls_embedding.dtype)
                    cls_embedding = torch.cat([cls_embedding, padding], dim=1)
                
                embeddings.append(cls_embedding.squeeze(0))
        
        initial_embeddings = torch.stack(embeddings)
        self.class_embeddings.weight.data.copy_(initial_embeddings)
    
    def _create_scene_text_attention_layer(self, hidden_dim):
        """Create Scene-to-Text attention layer like VisualBERT"""
        class SceneToTextAttention(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True
                )
                self.self_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True
                )
                self.layer_norm1 = nn.LayerNorm(hidden_dim)
                self.layer_norm2 = nn.LayerNorm(hidden_dim)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, text_features, scene_features):
                # Cross attention: text attends to scene (correct direction)
                attended_text, _ = self.cross_attention(
                    text_features, scene_features, scene_features
                )
                attended_text = self.dropout(attended_text)
                attended_text = self.layer_norm1(attended_text + text_features)
                
                # Self attention on attended features
                self_attended, _ = self.self_attention(
                    attended_text, attended_text, attended_text
                )
                self_attended = self.dropout(self_attended)
                return self.layer_norm2(self_attended + attended_text)
                
        return SceneToTextAttention(hidden_dim)
    
    def _create_text_scene_attention_layer(self, hidden_dim):
        """Create Text-to-Scene attention layer like VisualBERT"""
        class TextToSceneAttention(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.cross_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True
                )
                self.self_attention = nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True
                )
                self.layer_norm1 = nn.LayerNorm(hidden_dim)
                self.layer_norm2 = nn.LayerNorm(hidden_dim)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, text_features, scene_features):
                # Cross attention: scene attends to text (correct direction)
                attended_scene, _ = self.cross_attention(
                    scene_features, text_features, text_features
                )
                attended_scene = self.dropout(attended_scene)
                attended_scene = self.layer_norm1(attended_scene + scene_features)
                
                # Self attention on attended features
                self_attended, _ = self.self_attention(
                    attended_scene, attended_scene, attended_scene
                )
                self_attended = self.dropout(self_attended)
                return self.layer_norm2(self_attended + attended_scene)
                
        return TextToSceneAttention(hidden_dim)
                
    def _load_triplet_vocab(self):
        """Load triplet vocabulary for analysis"""
        if self.enable_triplet:
            # Use existing triplet_components.json instead of triplet.json
            with open(cfg.META_INFO_DIR / "triplet_components.json") as f:
                triplet_data = json.load(f)
                self.triplet_vocab = triplet_data
    
    def _validate_class_distributions(self):
        """Validate that class distributions match triplet vocabulary order"""
        if not self.enable_triplet:
            return
        
        # Get expected order from triplet.json
        expected_orders = {}
        for component, vocab in self.triplet_vocab.items():
            # Sort by index to get correct order
            sorted_classes = sorted(vocab.items(), key=lambda x: x[1])
            expected_orders[component] = [class_name for class_name, idx in sorted_classes]
        
        # Print validation info
        print("Class Distribution Validation:")
        for component in ['instrument', 'verb', 'target']:
            expected_order = expected_orders[component]
            print(f"   {component.capitalize()}: {expected_order}")
        
        # Store expected orders for reference
        self.expected_class_orders = expected_orders
    
    def _initialize_class_balanced_weights(self):
        """Initialize class-balanced weights for extreme class imbalance AND label imbalance"""
        if not self.enable_triplet:
            return
            
        # Load class frequencies from triplet_id2count.json
        import json
        import os
        import numpy as np
        
        json_path = cfg.META_INFO_DIR / 'triplet_id2count.json'
        with open(json_path, 'r') as f:
            class_count_dict = json.load(f)
        
        # Convert string keys to int and create frequency list for all classes
        class_counts = {}
        total_positives = 0
        for class_id in range(101):
            count = class_count_dict.get(str(class_id), 0)
            # Handle classes with 0 count by setting minimum count of 1
            class_counts[class_id] = max(count, 1)
            total_positives += count
        
        # Calculate total samples and negatives
        # Assuming roughly 3000 total training samples with ~2-3 triplets per sample
        total_samples = 3000  # Approximate from your dataset
        total_possible_labels = total_samples * 101  # 3000 * 101 = 303,000
        total_negatives = total_possible_labels - total_positives
        
        # Global positive-negative ratio for base pos_weight
        global_pos_neg_ratio = total_negatives / total_positives if total_positives > 0 else 50.0
        
        print(f"📊 Dataset statistics:")
        print(f"   Total positive labels: {total_positives}")
        print(f"   Total negative labels: {total_negatives}")
        print(f"   Global pos/neg ratio: 1:{global_pos_neg_ratio:.1f}")
        
        # Create class-balanced weights using effective number (for class imbalance)
        beta = 0.99  # Use beta = 0.99 as requested
        effective_nums = []
        for i in range(101):
            count = class_counts[i]
            effective_num = 1.0 - np.power(beta, count)
            effective_nums.append(max(effective_num, 1e-8))
        
        # Calculate class-balanced weights (inverse of effective number)
        class_weights = [(1.0 - beta) / en for en in effective_nums]
        class_weights = np.array(class_weights)
        
        # Normalize class weights but ensure all weights >= 1.0
        class_weights = class_weights / np.min(class_weights)
        
        # COMBINE class imbalance weights with global label imbalance
        # Final pos_weight = global_pos_neg_ratio * class_rebalancing_factor
        final_pos_weights = global_pos_neg_ratio * class_weights
        
        # Ensure minimum pos_weight to handle severe label imbalance
        min_pos_weight = max(10.0, global_pos_neg_ratio * 0.5)  # At least 10x or half of global ratio
        final_pos_weights = np.maximum(final_pos_weights, min_pos_weight)
        
        # Store as model parameter (will be moved to correct device with model)
        self.register_buffer('class_balanced_weights', 
                           torch.tensor(final_pos_weights, dtype=torch.float32))
        
        print(f"✅ Multi-level imbalance weights initialized:")
        print(f"   Global pos/neg ratio: 1:{global_pos_neg_ratio:.1f}")
        print(f"   Class weight range: {class_weights.min():.2f} - {class_weights.max():.2f}")
        print(f"   Final pos_weight range: {final_pos_weights.min():.1f} - {final_pos_weights.max():.1f}")
        print(f"   Common classes: pos_weight = {final_pos_weights.min():.1f}x")
        print(f"   Rare classes: pos_weight up to {final_pos_weights.max():.1f}x")
        print(f"   This handles both class imbalance AND label imbalance (neg >> pos)")
        print(f"   Classes with 0 count (set to 1): {[i for i in range(101) if class_count_dict.get(str(i), 0) == 0]}")
                
    def forward(self, graph_data, questions):
        batch_size = len(questions)
        
        # Process questions - using sequence output for richer representation
        tokenized = self.tokenizer(questions, return_tensors='pt', padding=True, truncation=True)
        # Move tokenized inputs to model device
        tokenized = {key: value.to(next(self.parameters()).device) for key, value in tokenized.items()}
        bert_outputs = self.bert_model(**tokenized)
        question_sequence = bert_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        question_pooled = bert_outputs.pooler_output  # [batch, hidden_dim] - keep for fallback
        
        # Encode object classes directly (no separate forward pass needed)
        class_embeddings = self.class_embeddings(graph_data.class_indices)
        class_embeddings = self.class_projection(class_embeddings)
        
        # Combine node features
        combined_features = torch.cat([graph_data.x, class_embeddings], dim=1)
        node_features = self.node_projection(combined_features)
        
        # GNN processing with batch information
        batch = getattr(graph_data, 'batch', None)
        
        if self.gnn_type == 'gcn':
            graph_features = self._process_gcn(node_features, graph_data.edge_index, graph_data.edge_attr, batch)
        elif self.gnn_type == 'gat':
            graph_features = self._process_gat(node_features, graph_data.edge_index, graph_data.edge_attr, batch)
        elif self.gnn_type == 'gin':
            graph_features = self._process_gin(node_features, graph_data.edge_index, graph_data.edge_attr, batch)
        elif self.gnn_type == 'none':
            graph_features = self._process_baseline(node_features)
        else:
            raise ValueError(f"Unsupported gnn_type: {self.gnn_type}")
            
        # Global pooling for each sample in batch
        if batch is not None:
            # Batch-wise pooling: pool nodes per graph
            batch_size = int(batch.max()) + 1
            pooled_graphs = []
            for i in range(batch_size):
                mask = (batch == i)
                if mask.sum() > 0:
                    graph_pool = graph_features[mask].mean(dim=0, keepdim=True)
                else:
                    graph_pool = torch.zeros(1, self.hidden_dim, device=graph_features.device)
                pooled_graphs.append(graph_pool)
            pooled_graph_batch = torch.cat(pooled_graphs, dim=0)  # [batch_size, hidden_dim]
        else:
            # Single graph case
            pooled_graph_batch = torch.mean(graph_features, dim=0, keepdim=True)  # [1, hidden_dim]
            pooled_graph_batch = pooled_graph_batch.repeat(batch_size, 1)  # [batch_size, hidden_dim]
        
        outputs = {}
        
        # Multi-task: Direct triplet classification (scene understanding only)
        if self.enable_triplet:
            # Direct triplet classification - 101 valid surgical triplet combinations
            outputs['direct_triplets'] = self.direct_triplet_head(pooled_graph_batch)
        
        # Project question sequence to match graph dimension
        question_sequence_proj = self.question_projection(question_sequence)  # [batch, seq_len, hidden_dim]
        
        # VisualBERT-style cross-modal fusion with specialized attention
        # 1. Project graph features to batch dimension
        graph_features_batch = graph_features.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, num_nodes, hidden_dim]
        
        # 2. Add visual token type embeddings
        visual_token_type_ids = torch.ones(batch_size, graph_features_batch.shape[1], dtype=torch.long, device=graph_features_batch.device)
        text_token_type_ids = torch.zeros(batch_size, question_sequence_proj.shape[1], dtype=torch.long, device=question_sequence_proj.device)
        
        # Enhanced visual preprocessing
        graph_preprocessed = self.visual_preprocessing(graph_features_batch)
        graph_with_type = graph_preprocessed + self.visual_token_type_embeddings(visual_token_type_ids)
        question_with_type = question_sequence_proj + self.visual_token_type_embeddings(text_token_type_ids)
        
        # 3. Apply specialized Scene-to-Text attention layers
        scene_attended_text = question_with_type
        for layer in self.scene_to_text_attention:
            scene_attended_text = layer(scene_attended_text, graph_with_type)
        
        # 4. Apply specialized Text-to-Scene attention layers  
        text_attended_scene = graph_with_type
        for layer in self.text_to_scene_attention:
            text_attended_scene = layer(text_attended_scene, scene_attended_text)
        
        # 5. Pool representations after specialized attention
        question_attended = torch.mean(scene_attended_text, dim=1)  # [batch, hidden_dim]
        graph_attended = torch.mean(text_attended_scene, dim=1)  # [batch, hidden_dim]
        
        # 4. Self-attention on combined features with residual connections
        combined = question_attended + graph_attended  # Element-wise addition
        combined_norm = self.fusion_norm1(combined)
        
        fused_attended, _ = self.self_attention(
            combined_norm.unsqueeze(1), 
            combined_norm.unsqueeze(1), 
            combined_norm.unsqueeze(1)
        )
        fused_features = fused_attended.squeeze(1)
        fused_features = self.fusion_norm2(fused_features + combined_norm)  # Residual connection
        
        # VQA prediction
        vqa_logits = self.vqa_head(fused_features)
        
        if self.enable_triplet:
            outputs['vqa'] = vqa_logits
            outputs['graph_features'] = pooled_graph_batch
            return outputs
        else:
            return vqa_logits
            
    def _process_gcn(self, x, edge_index, edge_attr, batch=None):
        """Pure GCN processing - edge attributes not used by GCNConv"""
        for i, (layer, norm) in enumerate(zip(self.gcn_layers, self.gcn_graph_norms)):
            x = layer(x, edge_index)  # GCNConv doesn't use edge_attr
            x = norm(x, batch)  # Graph-aware normalization (restored)
            if i < len(self.gcn_layers) - 1:  # Don't apply ReLU to final layer
                x = F.relu(x)
        return x
        
    def _process_gat(self, x, edge_index, edge_attr, batch=None):
        """Pure GAT processing with edge attributes"""
        for i, (layer, norm) in enumerate(zip(self.gat_layers, self.gat_graph_norms)):
            x = layer(x, edge_index, edge_attr)
            x = norm(x, batch)  # Graph-aware normalization (restored)
            if i < len(self.gat_layers) - 1:  # Don't apply ReLU to final layer
                x = F.relu(x)
        return x
        
    def _process_gin(self, x, edge_index, edge_attr, batch=None):
        """Pure GIN processing - edge attributes not used by standard GINConv"""
        for i, (layer, norm) in enumerate(zip(self.gin_layers, self.gin_graph_norms)):
            x = layer(x, edge_index)  # GINConv doesn't use edge_attr in standard form
            x = norm(x, batch)  # Graph-aware normalization (restored)
            if i < len(self.gin_layers) - 1:  # Don't apply ReLU to final layer
                x = F.relu(x)
        return x
        
    def _process_baseline(self, x):
        """Baseline processing - No GNN, just MLP on node features"""
        x = self.baseline_mlp(x)
        x = self.baseline_norm(x)
        return x
    
    def compute_multi_task_loss(self, outputs, vqa_labels, triplet_labels,
                                  vqa_weight=0.7, direct_triplet_weight=0.3,
                                  use_multi_label=True):
        """
        Compute loss using direct triplet classification (101 classes)
        
        Args:
            outputs: Model outputs including 'direct_triplets'
            vqa_labels: VQA ground truth labels
            triplet_labels: Triplet ground truth labels
            vqa_weight: Weight for VQA loss
            direct_triplet_weight: Weight for direct triplet loss
            use_multi_label: Whether to use multi-label (BCEWithLogits) or single-label (CrossEntropy)
            
        Returns:
            Dictionary with loss components
        """
        losses = {}
        batch_size = outputs['vqa'].size(0)
        
        # 1. VQA loss
        vqa_loss = self._focal_loss(outputs['vqa'], vqa_labels, alpha=1.0, gamma=2.0)
        losses['vqa'] = vqa_loss
        
        # 2. Direct triplet loss
        if 'direct_triplets' in outputs and hasattr(self, 'triplet_to_id'):
            direct_triplet_logits = outputs['direct_triplets']  # [batch_size, 101]
            
            # Convert triplet_labels to direct triplet targets
            if use_multi_label:
                # Multi-label: multiple triplets per sample
                triplet_targets = torch.zeros(batch_size, 101, device=outputs['vqa'].device)
                
                for batch_idx in range(batch_size):
                    if batch_idx < len(triplet_labels):
                        sample_triplet_ids = triplet_labels[batch_idx]
                        
                        # Handle different formats
                        if isinstance(sample_triplet_ids, torch.Tensor):
                            sample_triplet_ids = sample_triplet_ids.tolist()
                        elif not isinstance(sample_triplet_ids, list):
                            sample_triplet_ids = [sample_triplet_ids]
                        
                        # Set targets for each direct triplet ID (ignore padding value -1)
                        for triplet_id in sample_triplet_ids:
                            triplet_id = int(triplet_id)
                            if 0 <= triplet_id < 101:  # Skip padding value -1
                                triplet_targets[batch_idx, triplet_id] = 1.0
                
                # Multi-label BCE loss with DOUBLE imbalance handling:
                # 1. Class imbalance: Some triplet classes are rare vs common
                # 2. Label imbalance: Each sample has ~98 negatives vs ~2-3 positives
                if hasattr(self, 'class_balanced_weights'):
                    # pos_weight = global_pos_neg_ratio * class_rebalancing_factor
                    # This addresses both types of imbalance simultaneously
                    direct_triplet_loss = F.binary_cross_entropy_with_logits(
                        direct_triplet_logits, triplet_targets, 
                        pos_weight=self.class_balanced_weights
                    )
                    
                    # DEBUG: Check if loss is reasonable given pos_weight
                    if hasattr(self, 'training') and self.training:
                        print(f"🔍 Multi-task Triplet Loss Debug:")
                        print(f"   Raw triplet loss: {direct_triplet_loss.item():.6f}")
                        print(f"   Positive samples in batch: {(triplet_targets > 0.5).sum().item()}")
                        print(f"   Total possible labels: {triplet_targets.numel()}")
                        print(f"   Pos_weight applied: {self.class_balanced_weights.min().item():.1f}-{self.class_balanced_weights.max().item():.1f}x")
                        
                else:
                    # Fallback: fixed pos_weight for extreme imbalance (~60:1 ratio)
                    pos_weight = torch.tensor([60.0], device=direct_triplet_logits.device)
                    direct_triplet_loss = F.binary_cross_entropy_with_logits(
                        direct_triplet_logits, triplet_targets, pos_weight=pos_weight
                    )
                
            else:
                # Single-label: one primary triplet per sample
                triplet_targets = torch.full((batch_size,), 100, device=outputs['vqa'].device)  # Default to null class
                
                for batch_idx in range(batch_size):
                    if batch_idx < len(triplet_labels):
                        sample_triplet_ids = triplet_labels[batch_idx]
                        
                        # Handle different formats
                        if isinstance(sample_triplet_ids, torch.Tensor):
                            sample_triplet_ids = sample_triplet_ids.tolist()
                        elif not isinstance(sample_triplet_ids, list):
                            sample_triplet_ids = [sample_triplet_ids]
                        
                        # Take the first valid triplet as primary (ignore padding -1)
                        for triplet_id in sample_triplet_ids:
                            triplet_id = int(triplet_id)
                            if 0 <= triplet_id < 101:  # Skip padding value -1
                                triplet_targets[batch_idx] = triplet_id
                                break  # Take first valid triplet
                
                # Single-label CrossEntropy loss
                direct_triplet_loss = F.cross_entropy(direct_triplet_logits, triplet_targets)
            
            losses['direct_triplets'] = direct_triplet_loss
        else:
            losses['direct_triplets'] = torch.tensor(0.0, device=vqa_loss.device)
        
        # 3. Total loss
        total_loss = vqa_weight * vqa_loss + direct_triplet_weight * losses['direct_triplets']
        losses['total'] = total_loss
        
        return losses
    

    
    def _focal_loss(self, predictions, targets, alpha=1.0, gamma=2.0):
        """
        Focal Loss for addressing class imbalance without prior class weights
        
        Paper: "Focal Loss for Dense Object Detection" (Lin et al.)
        
        Args:
            predictions: Logits [batch_size, num_classes]
            targets: Class labels [batch_size] (not one-hot)
            alpha: Weighting factor for rare class (default: 1.0 - no weighting)
            gamma: Focusing parameter (default: 2.0)
        
        Returns:
            Focal loss value
        """
        # Compute softmax probabilities
        log_probs = F.log_softmax(predictions, dim=-1)
        probs = torch.exp(log_probs)
        
        # Get probabilities for true classes
        log_probs_for_targets = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        probs_for_targets = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = torch.pow(1 - probs_for_targets, gamma)
        
        # Apply alpha weighting (optional)
        alpha_weight = alpha
        
        # Final focal loss
        focal_loss = -alpha_weight * focal_weight * log_probs_for_targets
        
        return focal_loss.mean()
    
    def compute_uncertainty_weighted_loss(self, outputs, vqa_labels, triplet_labels):
        """
        Compute multi-task loss using uncertainty-based weighting for 2 main tasks
        
        Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
        Each task has a learnable noise parameter (log variance) that automatically balances the losses.
        
        The 2 main tasks are:
        1. VQA Task: Visual Question Answering
        2. Triplet Prediction Task: Complete triplet (instrument, verb, target) prediction
        
        Args:
            outputs: Model outputs
            vqa_labels: VQA ground truth labels
            triplet_labels: Triplet ground truth labels
            
        Returns:
            Dictionary with loss components and uncertainty weights
        """
        if not self.use_uncertainty_weights:
            # Fallback to direct triplet loss computation
            return self.compute_multi_task_loss(outputs, vqa_labels, triplet_labels)
        
        losses = {}
        batch_size = outputs['vqa'].size(0)
        
        # Task 1: VQA loss (no weighting applied yet)
        vqa_loss = self._focal_loss(outputs['vqa'], vqa_labels, alpha=1.0, gamma=2.0)
        
        # Task 2: Direct triplet loss (101 valid combinations)
        if 'direct_triplets' in outputs and hasattr(self, 'triplet_to_id'):
            triplet_logits = outputs['direct_triplets']  # [batch_size, 101]
            
            # Convert direct triplet IDs to targets (multi-label)
            triplet_targets = torch.zeros(batch_size, 101, device=outputs['vqa'].device)
            
            # triplet_labels is now a list of direct triplet IDs for each sample
            for batch_idx in range(batch_size):
                if batch_idx < len(triplet_labels):
                    sample_triplet_ids = triplet_labels[batch_idx]
                    
                    # Handle different formats
                    if isinstance(sample_triplet_ids, torch.Tensor):
                        sample_triplet_ids = sample_triplet_ids.tolist()
                    elif not isinstance(sample_triplet_ids, list):
                        sample_triplet_ids = [sample_triplet_ids]
                    
                    # Set targets for each direct triplet ID (ignore padding value -1)
                    for triplet_id in sample_triplet_ids:
                        triplet_id = int(triplet_id)
                        if 0 <= triplet_id < 101:  # Skip padding value -1
                            triplet_targets[batch_idx, triplet_id] = 1.0
            
            # Multi-label BCE Loss handling DOUBLE imbalance:
            # 1. Class imbalance: Rare vs common triplet classes  
            # 2. Label imbalance: ~98 negatives vs ~2-3 positives per sample
            if hasattr(self, 'class_balanced_weights'):
                # pos_weight combines global pos/neg ratio with class-specific rebalancing
                # Ensures both rare classes and positive labels get proper emphasis
                triplet_loss = F.binary_cross_entropy_with_logits(
                    triplet_logits, triplet_targets, 
                    pos_weight=self.class_balanced_weights
                )
                
                # DEBUG: Analyze why triplet loss is small despite high pos_weight
                if hasattr(self, 'training') and self.training:
                    with torch.no_grad():
                        # Check prediction distribution
                        probs = torch.sigmoid(triplet_logits)
                        positive_mask = triplet_targets > 0.5
                        negative_mask = triplet_targets < 0.5
                        
                        if positive_mask.sum() > 0:
                            pos_preds = probs[positive_mask]
                            pos_targets = triplet_targets[positive_mask]
                            avg_pos_pred = pos_preds.mean().item()
                            
                            # Check if model is too confident (predictions near 0 or 1)
                            confident_wrong = ((pos_preds < 0.1) & (pos_targets > 0.5)).sum().item()
                            confident_right = ((pos_preds > 0.9) & (pos_targets > 0.5)).sum().item()
                            
                            print(f"🔍 Triplet Loss Debug:")
                            print(f"   Positive samples: {positive_mask.sum().item()}")
                            print(f"   Avg positive prediction: {avg_pos_pred:.4f}")
                            print(f"   Confident wrong (pred<0.1): {confident_wrong}")
                            print(f"   Confident right (pred>0.9): {confident_right}")
                            print(f"   Raw triplet loss: {triplet_loss.item():.6f}")
                            print(f"   Pos_weight range: {self.class_balanced_weights.min().item():.1f}-{self.class_balanced_weights.max().item():.1f}")
                
            else:
                # Fallback: fixed pos_weight for extreme imbalance (~60:1 ratio)
                pos_weight = torch.tensor([60.0], device=triplet_logits.device)
                triplet_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    triplet_logits, triplet_targets, pos_weight=pos_weight
                )
        else:
            triplet_loss = torch.tensor(0.0, device=vqa_loss.device)
        
        # ADAPTIVE LOSS SCALING: Dynamically balance loss magnitudes
        # Current observation: VQA ~3.0-3.5, Triplet ~0.03-0.1 (ratio ~100:1)
        # Goal: Make both losses have similar magnitude for stable uncertainty weighting
        
        # Calculate dynamic scaling based on current loss values
        vqa_magnitude = vqa_loss.detach().item()
        triplet_magnitude = triplet_loss.detach().item()
        
        if triplet_magnitude > 1e-6:  # Avoid division by zero
            # Target: triplet_scaled ≈ 0.5-1.0 * vqa_loss for balanced learning
            target_triplet_magnitude = vqa_magnitude * 0.7  # 70% of VQA loss magnitude
            scale_factor = target_triplet_magnitude / triplet_magnitude
            # Clamp scale factor to reasonable range
            scale_factor = torch.clamp(torch.tensor(scale_factor), min=10.0, max=200.0).item()
        else:
            scale_factor = 100.0  # Fallback for very small triplet loss
        
        vqa_loss_scaled = vqa_loss  # Keep VQA loss as is
        triplet_loss_scaled = triplet_loss * scale_factor  # Adaptive scaling
        
        # Log scaling info for monitoring
        if hasattr(self, 'training') and self.training:
            print(f"🔧 Loss scaling: VQA={vqa_magnitude:.4f}, Triplet={triplet_magnitude:.4f}×{scale_factor:.1f}={triplet_loss_scaled.item():.4f}")
        
        # Apply uncertainty-based weighting to the normalized losses
        # Formula: L_total = 1/(2*sigma_vqa^2) * L_vqa + log(sigma_vqa) + 
        #                   1/(2*sigma_triplet^2) * L_triplet + log(sigma_triplet)
        
        # VQA task uncertainty weighting
        # Clamp log_var to prevent extreme negative values that cause negative loss
        log_var_vqa_clamped = torch.clamp(self.log_var_vqa, min=-3.0, max=3.0)
        log_var_triplet_clamped = torch.clamp(self.log_var_triplet, min=-3.0, max=3.0)
        
        precision_vqa = torch.exp(-log_var_vqa_clamped)  # 1/sigma_vqa^2
        vqa_weighted = precision_vqa * vqa_loss_scaled + 0.5 * log_var_vqa_clamped
        
        # Triplet task uncertainty weighting
        precision_triplet = torch.exp(-log_var_triplet_clamped)  # 1/sigma_triplet^2
        triplet_weighted = precision_triplet * triplet_loss_scaled + 0.5 * log_var_triplet_clamped
        
        # Total uncertainty-weighted loss
        total_loss = vqa_weighted + triplet_weighted
        
        # Safety check: Ensure loss is never negative
        if total_loss < 0:
            # Fallback to simple weighted sum if uncertainty weighting goes wrong
            total_loss = 0.7 * vqa_loss_scaled + 0.3 * triplet_loss_scaled
            print(f"⚠️ Negative loss detected ({total_loss.item():.4f}), using fallback weighting")
        
        # Store all loss components
        losses['vqa'] = vqa_loss  # Raw VQA loss
        losses['triplet'] = triplet_loss  # Raw direct triplet loss
        losses['vqa_scaled'] = vqa_loss_scaled  # Scaled VQA loss
        losses['triplet_scaled'] = triplet_loss_scaled  # Scaled triplet loss
        losses['vqa_weighted'] = vqa_weighted  # Uncertainty-weighted VQA loss
        losses['triplet_weighted'] = triplet_weighted  # Uncertainty-weighted triplet loss
        losses['total'] = total_loss
        
        # Store uncertainty information for monitoring
        vqa_sigma2 = torch.exp(self.log_var_vqa).item()
        triplet_sigma2 = torch.exp(self.log_var_triplet).item()
        
        losses['uncertainty_params'] = {
            'vqa_sigma2': vqa_sigma2,
            'triplet_sigma2': triplet_sigma2,
            'vqa_precision': 1.0 / vqa_sigma2,
            'triplet_precision': 1.0 / triplet_sigma2,
            'scale_factor': scale_factor
        }
        
        # Detailed logging for debugging (every 10 batches to avoid spam)
        if hasattr(self, 'training') and self.training and hasattr(self, '_batch_count'):
            self._batch_count = getattr(self, '_batch_count', 0) + 1
            if self._batch_count % 10 == 0:
                print(f"📊 Uncertainty Weighting (batch {self._batch_count}):")
                print(f"   Raw losses: VQA={vqa_loss.item():.4f}, Triplet={triplet_loss.item():.6f}")
                print(f"   Scaled losses: VQA={vqa_loss_scaled.item():.4f}, Triplet={triplet_loss_scaled.item():.4f}")
                print(f"   Uncertainties: VQA_σ²={vqa_sigma2:.3f}, Triplet_σ²={triplet_sigma2:.3f}")
                print(f"   Weighted losses: VQA={vqa_weighted.item():.4f}, Triplet={triplet_weighted.item():.4f}")
                print(f"   Total loss: {total_loss.item():.4f}")
        elif not hasattr(self, '_batch_count'):
            self._batch_count = 0
        
        return losses