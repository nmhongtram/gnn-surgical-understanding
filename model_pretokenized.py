#!/usr/bin/env python3
"""
Modified SurgicalSceneGraphVQA Model for Pre-tokenized Questions

This model handles both pre-tokenized questions (from HDF5 files) and 
original string questions, eliminating CPU tokenization overhead during training.

Key Changes:
- Handles pre-tokenized question dictionaries
- Falls back to original tokenization if needed
- Zero CPU tokenization overhead when using pre-tokenized data
"""

# Import all components from original model
from model import *


class PreTokenizedSurgicalSceneGraphVQA(SurgicalSceneGraphVQA):
    """
    Modified model that handles pre-tokenized questions
    
    Inherits all functionality from original SurgicalSceneGraphVQA but with
    optimized question processing for pre-tokenized data.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("✅ Pre-tokenized model initialized - supports both tokenized and string questions")
    
    def forward(self, graph_data, questions):
        """
        Forward pass with support for pre-tokenized questions
        
        Args:
            graph_data: PyTorch Geometric graph data
            questions: Either:
                - List of question strings (original format)
                - Dict with pre-tokenized data: {'input_ids': tensor, 'attention_mask': tensor, ...}
        
        Returns:
            Model outputs (same as original)
        """
        device = next(self.parameters()).device
        
        # Handle both pre-tokenized and string questions
        if isinstance(questions, dict) and 'input_ids' in questions:
            # 🚀 ZERO CPU OVERHEAD: Use pre-tokenized questions directly
            tokenized = questions
            
            # Move pre-tokenized tensors to GPU
            tokenized = {key: value.to(device, non_blocking=True) for key, value in tokenized.items()}
            
            batch_size = tokenized['input_ids'].size(0)
            
            print(f"🔥 Using pre-tokenized questions: {tokenized['input_ids'].shape}")
            
        else:
            # Fallback: Original tokenization for string questions
            batch_size = len(questions)
            
            print(f"⚠️ Falling back to CPU tokenization for {batch_size} string questions")
            
            # Process questions - using sequence output for richer representation
            tokenized = self.tokenizer(questions, return_tensors='pt', padding=True, truncation=True)
            # Move tokenized inputs to model device
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
        
        # Rest of forward pass is identical to original
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


# For backward compatibility - use as drop-in replacement
SurgicalSceneGraphVQA = PreTokenizedSurgicalSceneGraphVQA