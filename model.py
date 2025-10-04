import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, MessagePassing
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
from transformers import AutoTokenizer, AutoModel
import json
from collections import defaultdict
import numpy as np

# Graph normalization dependencies
from torch_geometric.utils import degree
from torch_scatter import scatter

import config as cfg


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


class SSGModel(nn.Module):
    def __init__(self, num_classes=50, node_dim=516, hidden_dim=512,  # 512 or 768
                 question_dim=768, freeze_bert=True, object_class_embed_dim=256,
                 gnn_type='gcn'):
        super(SSGModel, self).__init__()

        self.num_classes = num_classes
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.question_dim = question_dim
        self.gnn_type = gnn_type
        
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
            # Pure GAT layers - optimized for hidden_dim=768
            if hidden_dim == 768:
                head_dim = 96  # 768 // 8 = 96
                num_heads = 8
            elif hidden_dim == 512:
                head_dim = 64  # 512 // 8 = 64  
                num_heads = 8
            else:
                head_dim = hidden_dim // 8
                num_heads = 8
                
            self.gat_layers = nn.ModuleList([
                GATConv(hidden_dim, head_dim, heads=num_heads, edge_dim=16, dropout=0.1, concat=True),
                GATConv(hidden_dim, head_dim, heads=num_heads, edge_dim=16, dropout=0.1, concat=True),
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
            
        # # Question projection to match graph feature dimension
        # self.question_projection = nn.Linear(question_dim, hidden_dim)

        # Enhanced Question Feature Preprocessing (symmetric với visual)
        self.question_preprocessing = nn.Sequential(
            nn.Linear(question_dim, hidden_dim),        # 768 → hidden_dim
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),          # hidden_dim → hidden_dim
            nn.LayerNorm(hidden_dim)
        )
        
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
                
                
    def forward(self, graph_data, questions):
        device = next(self.parameters()).device
        tokenized = questions
        # Move pre-tokenized tensors to GPU
        tokenized = {key: value.to(device, non_blocking=True) for key, value in tokenized.items()}
        
        batch_size = tokenized['input_ids'].size(0)
    
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
        
        # # Project question sequence to match graph dimension
        # question_sequence_proj = self.question_projection(question_sequence)  # [batch, seq_len, hidden_dim]

        question_sequence_proj = self.question_preprocessing(question_sequence)  # [batch, seq_len, hidden_dim]

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
        
        # VQA-only prediction
        vqa_logits = self.vqa_head(fused_features)
        
        # VQA-only: Always return just the VQA logits
        return vqa_logits
            
    def _process_gcn(self, x, edge_index, edge_attr, batch=None):
        """Pure GCN processing - GCNConv does not support edge attributes"""
        for i, (layer, norm) in enumerate(zip(self.gcn_layers, self.gcn_graph_norms)):
            x = layer(x, edge_index)  # GCNConv: edge_attr not supported
            x = norm(x, batch)  # Graph-aware normalization
            if i < len(self.gcn_layers) - 1:  # Don't apply ReLU to final layer
                x = F.relu(x)
        return x
        
    def _process_gat(self, x, edge_index, edge_attr, batch=None):
        """Pure GAT processing - GATConv supports and uses edge attributes"""
        for i, (layer, norm) in enumerate(zip(self.gat_layers, self.gat_graph_norms)):
            x = layer(x, edge_index, edge_attr)  # GATConv: edge_attr supported
            x = norm(x, batch)  # Graph-aware normalization
            if i < len(self.gat_layers) - 1:  # Don't apply ReLU to final layer
                x = F.relu(x)
        return x
        
    def _process_gin(self, x, edge_index, edge_attr, batch=None):
        """Pure GIN processing - Standard GINConv does not support edge attributes"""
        for i, (layer, norm) in enumerate(zip(self.gin_layers, self.gin_graph_norms)):
            x = layer(x, edge_index)  # GINConv: edge_attr not supported in standard form
            x = norm(x, batch)  # Graph-aware normalization
            if i < len(self.gin_layers) - 1:  # Don't apply ReLU to final layer
                x = F.relu(x)
        return x
        
    def _process_baseline(self, x):
        """Baseline processing - No GNN, just MLP on node features"""
        x = self.baseline_mlp(x)
        x = self.baseline_norm(x)
        return x

    def _focal_loss(self, predictions, targets, alpha=1.0, gamma=2.0):
        """
        Focal Loss for addressing class imbalance without prior class weights
        
        Paper: "Focal Loss for Dense Object Detection" (Lin et al.)
        NOTE: Currently not used - kept for potential ablation studies
        
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
    
    def compute_vqa_only_loss(self, outputs, vqa_labels):
        """
        Standard VQA-only loss computation using cross entropy
        
        Args:
            outputs: VQA logits [batch_size, num_classes]
            vqa_labels: VQA ground truth labels [batch_size]
            
        Returns:
            VQA loss (scalar tensor)
        """
        # Use standard cross entropy loss (VQA community standard)
        return F.cross_entropy(outputs, vqa_labels)