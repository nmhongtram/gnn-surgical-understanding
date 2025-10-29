"""
Full Enhanced Model with Focal Loss - Fixed cross-modal attention issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoTokenizer, AutoModel
from transformers.activations import ACT2FN
import json
from pathlib import Path
import src.config as cfg
import math


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance
    
    Original paper: "Focal Loss for Dense Object Detection" by Lin et al.
    
    Args:
        alpha (float or tensor): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter to down-weight easy examples (default: 2.0)
        reduction (str): Specifies the reduction to apply to the output
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] - Raw logits from model
            targets: [batch_size] - Ground truth class indices
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute p_t
        pt = torch.exp(-ce_loss)
        
        # Compute alpha_t (class weighting)
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            alpha_t = self.alpha.gather(0, targets)
        
        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class GraphLayerNorm(nn.Module):
    """Graph-aware Layer Normalization for variable-size graphs"""
    def __init__(self, in_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(in_channels))
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x, batch=None):
        if batch is None:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            out = (x - mean) / (var + self.eps).sqrt()
        else:
            # Graph-aware normalization
            unique_batch = torch.unique(batch)
            normalized_parts = []
            
            for b in unique_batch:
                mask = (batch == b)
                x_batch = x[mask]
                
                mean = x_batch.mean(dim=0, keepdim=True)
                var = x_batch.var(dim=0, keepdim=True, unbiased=False)
                x_norm = (x_batch - mean) / (var + self.eps).sqrt()
                normalized_parts.append(x_norm)
            
            out = torch.cat(normalized_parts, dim=0)
        
        if self.weight is not None:
            out = out * self.weight + self.bias
        
        return out


class BERTStyleIntermediate(nn.Module):
    """BERT-style intermediate feed-forward layer with GELU activation"""
    def __init__(self, hidden_size, intermediate_size=None, hidden_act="gelu", dropout_prob=0.1):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size * 4  # Standard BERT ratio
            
        self.dense = nn.Linear(hidden_size, intermediate_size)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class BERTStyleOutput(nn.Module):
    """BERT-style output layer with residual connection and layer norm"""
    def __init__(self, intermediate_size, hidden_size, dropout_prob=0.1, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class EnhancedFeedForward(nn.Module):
    """Enhanced feed-forward network with BERT-style architecture"""
    def __init__(self, hidden_size, intermediate_size=None, hidden_act="gelu", dropout_prob=0.1):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
            
        self.intermediate = BERTStyleIntermediate(hidden_size, intermediate_size, hidden_act, dropout_prob)
        self.output = BERTStyleOutput(intermediate_size, hidden_size, dropout_prob)
        
    def forward(self, hidden_states):
        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output(intermediate_output, hidden_states)
        return layer_output


class MultiHeadAttention(nn.Module):
    """Fixed Multi-head attention layer compatible with our dimensions"""
    def __init__(self, hidden_dim, num_heads=8, dropout_prob=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query, key, value, attention_mask=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        # Linear projections
        Q = self.query(query)  # [batch_size, seq_len_q, hidden_dim]
        K = self.key(key)      # [batch_size, seq_len_k, hidden_dim]
        V = self.value(value)  # [batch_size, seq_len_k, hidden_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and combine heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.hidden_dim
        )
        
        # Final output projection
        output = self.output(context)
        
        return output, attention_weights


class SceneToTextAttentionLayer(nn.Module):
    """Fixed Scene-to-Text attention layer"""
    def __init__(self, hidden_dim, num_heads=3, dropout_prob=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, text_features, scene_features):
        """
        text_features: [batch_size, text_seq_len, hidden_dim]  
        scene_features: [batch_size, scene_seq_len, hidden_dim]
        """
        # Text attends to scene
        attended_text, attention_weights = self.attention(
            query=text_features,
            key=scene_features, 
            value=scene_features
        )
        
        # Residual connection and layer norm
        output = self.layer_norm(text_features + self.dropout(attended_text))
        
        return output


class TextToSceneAttentionLayer(nn.Module):
    """Fixed Text-to-Scene attention layer"""
    def __init__(self, hidden_dim, num_heads=3, dropout_prob=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, scene_features, text_features):
        """
        scene_features: [batch_size, scene_seq_len, hidden_dim]
        text_features: [batch_size, text_seq_len, hidden_dim]  
        """
        # Scene attends to text
        attended_scene, attention_weights = self.attention(
            query=scene_features,
            key=text_features,
            value=text_features
        )
        
        # Residual connection and layer norm
        output = self.layer_norm(scene_features + self.dropout(attended_scene))
        
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer like VisualBERT"""
    def __init__(self, hidden_dim, num_heads=8, intermediate_size=None, dropout_prob=0.1):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_dim * 4
            
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(hidden_dim, num_heads, dropout_prob)
        
        # Feed-forward network (BERT style)
        self.feed_forward = EnhancedFeedForward(
            hidden_dim, 
            intermediate_size=intermediate_size,
            dropout_prob=dropout_prob
        )
        
        # Layer norms
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, x, attention_mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, attention_mask)
        attn_output = self.dropout(attn_output)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x


class TransformerEncoder(nn.Module):
    """Multi-layer transformer encoder like VisualBERT"""
    def __init__(self, hidden_dim, num_layers=6, num_heads=8, dropout_prob=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads, 
                dropout_prob=dropout_prob
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class FullEnhancedModelFocalLoss(nn.Module):
    """Full Enhanced Model with cross-modal attention and Focal Loss"""
    
    def __init__(
        self, 
        gnn_type='gcn',
        hidden_dim=1024,
        question_dim=768,
        num_object_classes=15,
        object_class_embed_dim=384,  # Increased from 256 -> 384 (divisible by 3 for attention)
        num_classes=50,
        bert_model_name="emilyalsentzer/Bio_ClinicalBERT",
        num_gnn_layers=3,
        num_transformer_layers=6,
        num_cross_modal_layers=2,
        dropout_prob=0.1,
        add_cross_modal_attention=True,
        scene_nodes_count=8,  # Number of scene nodes like VisualBERT objects
        focal_alpha=2.5,      # Focal Loss alpha parameter
        focal_gamma=3.0       # Focal Loss gamma parameter
    ):
        super().__init__()
        
        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.num_object_classes = num_object_classes
        self.object_class_embed_dim = object_class_embed_dim
        self.num_classes = num_classes
        self.add_cross_modal_attention = add_cross_modal_attention
        self.scene_nodes_count = scene_nodes_count
        
        # BERT for question encoding (frozen)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        
        # Object class embeddings with BERT initialization
        self.class_embeddings = nn.Embedding(num_object_classes + 1, object_class_embed_dim)
        
        # Advanced: Attention-based dimension reduction (learns what's important)
        self.bert_to_class_projection = nn.Sequential(
            # First stage: Feature extraction
            nn.Linear(question_dim, object_class_embed_dim * 2),  # 768 -> 768
            nn.GELU(),
            nn.Dropout(dropout_prob),
            
            # Second stage: Attention-based compression
            nn.Linear(object_class_embed_dim * 2, object_class_embed_dim),  # 768 -> 384
            nn.LayerNorm(object_class_embed_dim)
        )
        
        # # Optional: Attention mechanism for selective dimension reduction
        # self.class_embed_attention = nn.MultiheadAttention(
        #     embed_dim=question_dim,
        #     num_heads=8,
        #     dropout=dropout_prob,
        #     batch_first=True
        # )
        
        self._initialize_class_embeddings()
        
        # Enhanced projections with GELU
        self.class_projection = nn.Sequential(
            nn.Linear(object_class_embed_dim, object_class_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(object_class_embed_dim)
        )
        
        self.node_projection = nn.Sequential(
            nn.Linear(516 + object_class_embed_dim, hidden_dim),  # 516 + 384 = 900 -> hidden_dim
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(hidden_dim)
        )
        
        # Question sequence projection (for full sequence processing like VisualBERT)
        self.question_projection = nn.Sequential(
            nn.Linear(question_dim, hidden_dim),  # Project each token in sequence
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(hidden_dim)
        )
        
        # VisualBERT-style pooler (only for final classification if needed)
        self.question_pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()  # Like VisualBERT pooler
        )
        
        # GNN layers
        self._build_gnn_layers(num_gnn_layers, dropout_prob)
        
        # Cross-modal attention layers (fixed)
        if self.add_cross_modal_attention:
            self.scene_to_text_layers = nn.ModuleList([
                SceneToTextAttentionLayer(hidden_dim, num_heads=3, dropout_prob=dropout_prob)
                for _ in range(num_cross_modal_layers)
            ])
            
            self.text_to_scene_layers = nn.ModuleList([
                TextToSceneAttentionLayer(hidden_dim, num_heads=3, dropout_prob=dropout_prob)  
                for _ in range(num_cross_modal_layers)
            ])
        
        # Transformer encoder for final fusion
        self.fusion_transformer = TransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_transformer_layers,
            num_heads=8,
            dropout_prob=dropout_prob
        )
        
        # Enhanced classification head
        self.vqa_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob * 2),
            nn.LayerNorm(hidden_dim // 2),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_prob * 1.5),
            nn.LayerNorm(hidden_dim // 4),
            
            nn.Linear(hidden_dim // 4, self.num_classes)
        )
        
        # Loss function - FOCAL LOSS instead of Cross Entropy
        self.criterion = FocalLoss(
            alpha=focal_alpha, 
            gamma=focal_gamma, 
            reduction='mean'
        )
    
    def _build_gnn_layers(self, num_layers, dropout_prob):
        """Build GNN layers with enhanced feed-forward"""
        if self.gnn_type == 'gcn':
            self.gnn_layers = nn.ModuleList([
                GCNConv(self.hidden_dim, self.hidden_dim) for _ in range(num_layers)
            ])
        elif self.gnn_type == 'gat':
            self.gnn_layers = nn.ModuleList([
                GATConv(
                    self.hidden_dim, self.hidden_dim // 8, heads=8, 
                    edge_dim=16, dropout=dropout_prob
                ) for _ in range(num_layers)
            ])
        elif self.gnn_type == 'gin':
            self.gnn_layers = nn.ModuleList([
                GINConv(
                    nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                        nn.GELU(),
                        nn.Dropout(dropout_prob),
                        nn.Linear(self.hidden_dim * 2, self.hidden_dim)
                    )
                ) for _ in range(num_layers)
            ])
        elif self.gnn_type == 'none':
            self.baseline_mlp = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_prob * 3),
                nn.Linear(self.hidden_dim // 2, self.hidden_dim)
            )
            self.baseline_norm = nn.LayerNorm(self.hidden_dim)
        
        # Enhanced normalization and feed-forward for each GNN layer
        if self.gnn_type != 'none':
            self.gnn_norms = nn.ModuleList([
                GraphLayerNorm(self.hidden_dim) for _ in range(num_layers)
            ])
            self.gnn_feed_forwards = nn.ModuleList([
                EnhancedFeedForward(self.hidden_dim, dropout_prob=dropout_prob)
                for _ in range(num_layers)
            ])
    
    def _initialize_class_embeddings(self):
        """Initialize class embeddings using BERT"""
        # Load object class names
        with open(cfg.META_INFO_DIR / "objects.json") as f:
            class_name_to_idx = json.load(f)
        
        class_names = [''] * (self.num_object_classes + 1)
        for class_name, idx in class_name_to_idx.items():
            if idx < self.num_object_classes:
                class_names[idx] = class_name
        
        # Generate BERT embeddings
        embeddings = []
        for i in range(self.num_object_classes + 1):
            class_name = class_names[i] if i < len(class_names) else "unknown_object"
            text = class_name.replace('_', ' ')
            
            inputs = self.tokenizer(
                text, return_tensors='pt', padding=True, 
                truncation=True, max_length=32
            )
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                
                # Multi-scale BERT features for richer representation
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768] - CLS token
                
                # Optional: Use mean pooling as additional signal
                # sequence_mean = outputs.last_hidden_state.mean(dim=1)  # [1, 768] - Mean of all tokens
                # combined_embedding = 0.8 * cls_embedding + 0.2 * sequence_mean  # Weighted combination
                
                # For now, use CLS token with learned projection
                combined_embedding = cls_embedding
            
            # Option 1: Standard learned projection
            projected_embedding = self.bert_to_class_projection(combined_embedding)  # [1, 384]
            
            # Option 2: Attention-based selective compression (experimental)
            # full_sequence = outputs.last_hidden_state  # [1, seq_len, 768]
            # attended_embedding, _ = self.class_embed_attention(
            #     combined_embedding.unsqueeze(1),  # Query: CLS token
            #     full_sequence,                     # Key: Full sequence  
            #     full_sequence                      # Value: Full sequence
            # )
            # projected_embedding = self.bert_to_class_projection(attended_embedding.squeeze(1))
            
            embeddings.append(projected_embedding.squeeze(0))
        
        initial_embeddings = torch.stack(embeddings)
        self.class_embeddings.weight.data.copy_(initial_embeddings)
    
    def _extract_multi_node_scene_features(self, node_features, batch, top_k=8):
        """
        Extract multi-node scene representation like VisualBERT visual objects
        
        Args:
            node_features: [total_nodes, hidden_dim] - All node features in batch
            batch: [total_nodes] - Batch assignment for each node
            top_k: Number of top nodes to select per graph
            
        Returns:
            scene_features: [batch_size, top_k, hidden_dim] - Multi-node scene representation
        """
        batch_size = int(batch.max()) + 1
        scene_representations = []
        
        for b in range(batch_size):
            # Get nodes for this graph
            mask = (batch == b)
            graph_nodes = node_features[mask]  # [num_nodes_in_graph, hidden_dim]
            num_nodes = graph_nodes.size(0)
            
            if num_nodes <= top_k:
                # If fewer nodes than top_k, use all nodes + padding
                if num_nodes < top_k:
                    # Pad with mean pooled representation
                    mean_node = graph_nodes.mean(dim=0, keepdim=True)  # [1, hidden_dim]
                    padding = mean_node.repeat(top_k - num_nodes, 1)   # [top_k - num_nodes, hidden_dim]
                    graph_representation = torch.cat([graph_nodes, padding], dim=0)
                else:
                    graph_representation = graph_nodes
            else:
                # Select top-k most representative nodes using multiple strategies
                
                # Strategy 1: Attention-based selection (most informative nodes)
                # Compute attention scores based on feature magnitude and diversity
                node_attention_scores = torch.sum(graph_nodes ** 2, dim=1)  # [num_nodes] - Feature magnitude
                attention_weights = F.softmax(node_attention_scores, dim=0)
                
                # Strategy 2: Combine with diversity (avoid selecting similar nodes)
                # Use feature magnitude + ensure diversity
                node_norms = torch.norm(graph_nodes, dim=1)  # [num_nodes]
                
                # Select top nodes with diversity constraint
                selected_indices = []
                remaining_indices = list(range(num_nodes))
                
                # First, select the most activated node
                first_idx = torch.argmax(node_norms).item()
                selected_indices.append(first_idx)
                remaining_indices.remove(first_idx)
                
                # Select remaining nodes ensuring diversity
                for _ in range(top_k - 1):
                    if not remaining_indices:
                        break
                        
                    # Compute distances to already selected nodes
                    selected_features = graph_nodes[selected_indices]  # [num_selected, hidden_dim]
                    remaining_features = graph_nodes[remaining_indices]  # [num_remaining, hidden_dim]
                    
                    # Distance to closest selected node
                    distances = torch.cdist(remaining_features.unsqueeze(0), 
                                          selected_features.unsqueeze(0)).squeeze(0)  # [num_remaining, num_selected]
                    min_distances = torch.min(distances, dim=1)[0]  # [num_remaining]
                    
                    # Balance between activation strength and diversity
                    remaining_norms = node_norms[remaining_indices]
                    combined_scores = 0.7 * remaining_norms + 0.3 * min_distances  # Weighted combination
                    
                    # Select node with highest combined score
                    best_remaining_idx = torch.argmax(combined_scores).item()
                    actual_idx = remaining_indices[best_remaining_idx]
                    selected_indices.append(actual_idx)
                    remaining_indices.remove(actual_idx)
                
                graph_representation = graph_nodes[selected_indices]  # [top_k, hidden_dim]
            
            scene_representations.append(graph_representation)
        
        # Stack all graph representations
        scene_features = torch.stack(scene_representations)  # [batch_size, top_k, hidden_dim]
        
        return scene_features
    
    def forward(self, graph_data, questions_batch):
        batch_size = questions_batch['input_ids'].shape[0]
        
        # Question encoding - Use FULL SEQUENCE like VisualBERT (not just CLS)
        with torch.no_grad():
            question_outputs = self.bert_model(**questions_batch)
            # Get FULL question sequence: [batch_size, seq_len, hidden_dim]
            question_sequence = question_outputs.last_hidden_state
        
        # Project full question sequence to model's hidden_dim
        # question_sequence: [batch_size, seq_len, 768] -> [batch_size, seq_len, hidden_dim]
        question_projected = self.question_projection(question_sequence)
        
        # Graph processing
        class_embeddings = self.class_embeddings(graph_data.class_indices)
        class_embeddings = self.class_projection(class_embeddings)
        
        combined_features = torch.cat([graph_data.x, class_embeddings], dim=1)
        node_features = self.node_projection(combined_features)
        
        # GNN processing
        if self.gnn_type == 'none':
            x = self.baseline_mlp(node_features)
            x = self.baseline_norm(x)
        else:
            x = node_features
            for i, (gnn_layer, norm_layer, ff_layer) in enumerate(
                zip(self.gnn_layers, self.gnn_norms, self.gnn_feed_forwards)
            ):
                residual = x
                if self.gnn_type == 'gat':
                    x = gnn_layer(x, graph_data.edge_index, graph_data.edge_attr)
                else:
                    x = gnn_layer(x, graph_data.edge_index)
                
                x = norm_layer(x, graph_data.batch)
                x = F.gelu(x)
                x = ff_layer(x) + residual
        
        # Multi-node scene representation (like VisualBERT with multiple visual objects)
        scene_features_0 = self._extract_multi_node_scene_features(x, graph_data.batch, top_k=self.scene_nodes_count)
        
        # Prepare features for cross-modal attention - VisualBERT style
        # Text: Use FULL sequence [batch_size, text_seq_len, hidden_dim]
        text_features_0 = question_projected  # Full question sequence
        # Scene: Multi-node sequence [batch_size, top_k, hidden_dim] like VisualBERT objects
        
        # Cross-modal attention layers (fixed dimensions)
        if self.add_cross_modal_attention:
            for scene_to_text_layer, text_to_scene_layer in zip(
                self.scene_to_text_layers, self.text_to_scene_layers
            ):
                # Text attends to scene
                text_features = scene_to_text_layer(text_features_0, scene_features_0)
                # Scene attends to text
                scene_features = text_to_scene_layer(scene_features_0, text_features_0)
        
        # Combine for transformer: [batch_size, text_seq_len + top_k, hidden_dim] like VisualBERT
        combined_sequence = torch.cat([text_features, scene_features], dim=1)
        
        # Apply transformer encoder
        transformer_output = self.fusion_transformer(combined_sequence)
        
        # Pool transformer output (use first token like BERT CLS)
        pooled_output = transformer_output[:, 0, :]  # [batch_size, hidden_dim]
        
        # VQA prediction
        logits = self.vqa_head(pooled_output)
        
        return logits
    
    def compute_loss(self, logits, labels):
        """Compute Focal Loss instead of Cross Entropy"""
        return self.criterion(logits, labels)


def create_full_enhanced_model(**kwargs):
    """Factory function to create full enhanced model with Focal Loss"""
    return FullEnhancedModelFocalLoss(**kwargs)