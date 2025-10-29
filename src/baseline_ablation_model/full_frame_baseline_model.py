"""
Full Frame Baseline Model - Ablation Study
No Graph Representation, No GNN Processing

This model uses only global frame features (no graph structure) as an ablation study
to measure the contribution of graph neural networks to VQA performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.activations import ACT2FN
import json
from pathlib import Path
import config as cfg
import math


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
    """Multi-head attention layer compatible with our dimensions"""
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


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer for fusion"""
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
    """Multi-layer transformer encoder"""
    def __init__(self, hidden_dim, num_layers=3, num_heads=8, dropout_prob=0.1):
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


class FullFrameBaselineModel(nn.Module):
    """
    Full Frame Baseline Model - Ablation Study
    
    This model removes all graph processing and uses only global frame features
    to measure the contribution of GNN to VQA performance.
    
    Architecture:
    1. Global frame feature extraction (pre-computed)
    2. Question encoding with BERT
    3. Cross-modal fusion with transformer
    4. VQA classification
    
    NO: Graph construction, GNN processing, multi-node representation
    """
    
    def __init__(
        self, 
        frame_feature_dim=512,  # Global frame feature dimension
        hidden_dim=768,
        question_dim=768,
        num_classes=50,
        bert_model_name="emilyalsentzer/Bio_ClinicalBERT",
        num_transformer_layers=3,
        dropout_prob=0.1
    ):
        super().__init__()
        
        self.frame_feature_dim = frame_feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # BERT for question encoding (frozen)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        
        # # Frame feature projection
        # self.frame_projection = nn.Sequential(
        #     nn.Linear(frame_feature_dim, hidden_dim * 2),  # 512 -> 1536
        #     nn.GELU(),
        #     nn.Dropout(dropout_prob),
        #     nn.Linear(hidden_dim * 2, hidden_dim),        # 1536 -> 768
        #     nn.LayerNorm(hidden_dim)
        # )
        
        # Frame feature projection
        self.frame_projection = nn.Sequential(
            nn.Linear(frame_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(hidden_dim)
        )

        # Question sequence projection
        self.question_projection = nn.Sequential(
            nn.Linear(question_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.LayerNorm(hidden_dim)
        )
        
        # Transformer fusion encoder
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
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, frame_features, questions_batch):
        """
        Forward pass using only global frame features
        
        Args:
            frame_features: [batch_size, frame_feature_dim] - Global frame features
            questions_batch: Dict with 'input_ids', 'attention_mask', 'token_type_ids'
            
        Returns:
            logits: [batch_size, num_classes] - VQA answer predictions
        """
        batch_size = questions_batch['input_ids'].shape[0]
        
        # Question encoding - Use FULL sequence like enhanced model
        with torch.no_grad():
            question_outputs = self.bert_model(**questions_batch)
            # Get FULL question sequence: [batch_size, seq_len, hidden_dim]
            question_sequence = question_outputs.last_hidden_state
        
        # Project full question sequence to model's hidden_dim
        question_projected = self.question_projection(question_sequence)  # [B, seq_len, hidden_dim]
        
        # Frame feature processing - single global representation
        frame_projected = self.frame_projection(frame_features)  # [B, hidden_dim]
        frame_features_seq = frame_projected.unsqueeze(1)  # [B, 1, hidden_dim] - Single frame token
        
        # Combine for transformer: [batch_size, seq_len + 1, hidden_dim]
        # Text sequence + single frame token (no multi-node like enhanced model)
        combined_sequence = torch.cat([question_projected, frame_features_seq], dim=1)
        
        # Apply transformer encoder for fusion
        transformer_output = self.fusion_transformer(combined_sequence)
        
        # Pool transformer output (use first token like BERT CLS)
        pooled_output = transformer_output[:, 0, :]  # [batch_size, hidden_dim]
        
        # VQA prediction
        logits = self.vqa_head(pooled_output)
        
        return logits
    
    def compute_loss(self, logits, labels):
        return self.criterion(logits, labels)


def create_full_frame_baseline_model(**kwargs):
    """Factory function to create full frame baseline model"""
    return FullFrameBaselineModel(**kwargs)


if __name__ == "__main__":
    # Test the baseline model
    model = create_full_frame_baseline_model(
        frame_feature_dim=512,
        hidden_dim=768,
        num_transformer_layers=3
    )
    
    print(f"🎯 Full Frame Baseline Model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    print("📊 Architecture: Global frame features only (no graph, no GNN)")
    print("🔬 Purpose: Ablation study to measure GNN contribution")
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    
    # Dummy inputs
    frame_features = torch.randn(batch_size, 512)  # Global frame features
    questions_batch = {
        'input_ids': torch.randint(0, 30000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len),
        'token_type_ids': torch.zeros(batch_size, seq_len, dtype=torch.long)
    }
    
    # Forward pass
    with torch.no_grad():
        logits = model(frame_features, questions_batch)
        print(f"✅ Forward pass successful: {logits.shape}")
        print(f"📈 Output shape: [batch_size={batch_size}, num_classes=50]")