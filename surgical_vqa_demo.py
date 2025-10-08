#!/usr/bin/env python3
"""
Surgical VQA Demo Web UI with Gradio

This application creates a comprehensive web demo for Surgical Visual Question Answering (VQA) that integrates:
- YOLOv8n object detection for surgical instrument identification
- VQA Model (GCN/GAT/GIN variants) for question answering
- Gradio Interface for user-friendly web interaction

The demo allows users to upload surgical images, ask questions, and get intelligent answers
based on visual scene understanding using Graph Neural Networks.
"""

import os
import sys
import torch
import torch.nn.functional as F
import gradio as gr
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import List, Dict, Tuple, Optional
import warnings
import time
warnings.filterwarnings('ignore')

# Deep Learning Libraries
try:
    from ultralytics import YOLO
except ImportError:
    print("⚠️ ultralytics not installed. Run: pip install ultralytics")
    YOLO = None

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("⚠️ transformers not installed. Run: pip install transformers")
    AutoTokenizer = None

try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
except ImportError:
    print("⚠️ torch_geometric not installed. Run: pip install torch_geometric")
    Data = None
    Batch = None

try:
    import torchvision.models as models
    import torchvision.transforms as transforms
except ImportError:
    print("⚠️ torchvision not installed. Run: pip install torchvision")
    models = None
    transforms = None

# Local imports
try:
    import config as cfg
    from full_enhanced_model import create_full_enhanced_model
except ImportError as e:
    print(f"⚠️ Some local imports failed: {e}")
    print("Make sure you're running from the correct directory")

print("✅ Libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


class ModelManager:
    """Manages all pre-trained models and weights"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")
        
        # Model paths
        self.yolo_path = Path("detection_model/runs/detect/train3/weights/best.pt")
        self.vqa_checkpoint_dir = Path("checkpoints")
        
        # Models
        self.yolo_model = None
        self.vqa_models = {}
        self.tokenizer = None
        
        # Load all models
        self._load_models()
    
    def _load_models(self):
        """Load all required models"""
        print("📥 Loading models...")
        
        # 1. Load YOLOv8n Detection Model
        try:
            if self.yolo_path.exists() and YOLO is not None:
                self.yolo_model = YOLO(str(self.yolo_path))
                print(f"✅ YOLOv8n loaded from {self.yolo_path}")
            else:
                if YOLO is not None:
                    print(f"⚠️ YOLOv8n weight not found at {self.yolo_path}")
                    print("📥 Loading default YOLOv8n model...")
                    self.yolo_model = YOLO('yolov8n.pt')  # Fallback to default model
                else:
                    print("❌ YOLO not available - object detection disabled")
        except Exception as e:
            print(f"❌ Error loading YOLOv8n: {e}")
            self.yolo_model = None
        
        # 2. Load BioClinicalBERT Tokenizer
        try:
            if AutoTokenizer is not None:
                self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
                print("✅ BioClinicalBERT tokenizer loaded")
            else:
                print("❌ Transformers not available - text processing disabled")
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
        
        # 3. Load VQA Models (GAT, GCN, NoGNN)
        self._load_vqa_models()
        
        # 4. Load answer mappings
        self._load_answer_mappings()
    
    def _load_vqa_models(self):
        """Load VQA model weights"""
        model_types = ['gat', 'gcn', 'nognn']
        
        for model_type in model_types:
            try:
                # Model configuration based on actual trained models
                model_config = {
                    'gnn_type': 'gat' if model_type == 'gat' 
                               else 'gcn' if model_type == 'gcn' 
                               else 'none',
                    'hidden_dim': 768,
                    'num_transformer_layers': 4,  # Changed to 4 based on error (layers.3 exists)
                    'num_cross_modal_layers': 2,
                    'add_cross_modal_attention': True,
                    'scene_nodes_count': 8,
                    'num_object_classes': 15,
                    'object_class_embed_dim': 384
                }
                
                # Create model
                model = create_full_enhanced_model(**model_config)
                
                # Load weights if available
                checkpoint_path = self.vqa_checkpoint_dir / model_type / "best_model.pth"
                if not checkpoint_path.exists():
                    # Try alternative naming
                    alt_paths = [
                        self.vqa_checkpoint_dir / model_type / "best_gcn_model.pth",
                        self.vqa_checkpoint_dir / model_type / "best_gat_model.pth",
                        self.vqa_checkpoint_dir / model_type / "best_enhanced_model.pth",
                        self.vqa_checkpoint_dir / f"best_{model_type}_model.pth"
                    ]
                    
                    for alt_path in alt_paths:
                        if alt_path.exists():
                            checkpoint_path = alt_path
                            break
                
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    try:
                        if 'model_state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        else:
                            model.load_state_dict(checkpoint, strict=False)
                        print(f"✅ {model_type.upper()} VQA model loaded from {checkpoint_path} (with compatibility)")
                    except Exception as e:
                        print(f"⚠️ {model_type.upper()} model loaded with warnings: {str(e)[:100]}...")
                        print(f"    Using partially loaded {model_type.upper()} model for demo")
                else:
                    print(f"⚠️ {model_type.upper()} checkpoint not found")
                    print(f"    Searched: {self.vqa_checkpoint_dir / model_type}")
                    print(f"    Using randomly initialized {model_type.upper()} model for demo")
                
                model.to(self.device)
                model.eval()
                self.vqa_models[model_type] = model
                
            except Exception as e:
                print(f"❌ Error loading {model_type} model: {e}")
        
        print(f"📊 Loaded {len(self.vqa_models)} VQA models: {list(self.vqa_models.keys())}")
    
    def _load_answer_mappings(self):
        """Load answer label mappings"""
        try:
            label2ans_path = Path("meta_info/label2ans.json")
            if label2ans_path.exists():
                with open(label2ans_path) as f:
                    data = json.load(f)
                    
                # Handle different formats of label2ans.json
                if isinstance(data, dict):
                    self.label2ans = data
                elif isinstance(data, list):
                    # If it's a list, convert to dict with indices
                    self.label2ans = {str(i): str(data[i]) for i in range(len(data))}
                else:
                    # Fallback
                    self.label2ans = {str(i): f"answer_{i}" for i in range(50)}
                    
                print(f"✅ Answer mappings loaded: {len(self.label2ans)} classes")
            else:
                # Fallback answer mapping for demo
                self.label2ans = {str(i): f"answer_{i}" for i in range(50)}
                print("⚠️ Using fallback answer mappings")
                
            # Reverse mapping
            try:
                self.ans2label = {v: int(k) for k, v in self.label2ans.items()}
            except:
                self.ans2label = {f"answer_{i}": i for i in range(len(self.label2ans))}
            
        except Exception as e:
            print(f"❌ Error loading answer mappings: {e}")
            self.label2ans = {str(i): f"answer_{i}" for i in range(50)}
            self.ans2label = {f"answer_{i}": i for i in range(50)}


class FeatureExtractor:
    """
    Feature Extractor for full frame features
    Based on the original feature_extract.py logic
    """
    
    def __init__(self, patch_size=1, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.patch_size = patch_size
        
        # Load pre-trained ResNet for feature extraction
        try:
            if models is None or transforms is None:
                raise ImportError("torchvision not available")
                
            # Use ResNet18 like in original feature_extract.py
            self.model = models.resnet18(pretrained=True)
            # Remove final layers (same as original)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-2])
            
            # Add adaptive pooling layer like original
            self.resize_dim = torch.nn.AdaptiveAvgPool2d((patch_size, patch_size))
            
            self.model.to(self.device)
            self.resize_dim.to(self.device)
            self.model.eval()
            
            # Image preprocessing - exactly like feature_extract.py
            self.transform = transforms.Compose([
                transforms.Resize((300, 256)),  # Original preprocessing size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self.feature_dim = 512  # ResNet18 feature dimension
            
            print(f"✅ FeatureExtractor initialized with ResNet18, patch_size={patch_size}")
            
        except Exception as e:
            print(f"❌ Error initializing FeatureExtractor: {e}")
            self.model = None
            self.feature_dim = 512
    
    def extract_features(self, image):
        """Extract global frame features from image - following feature_extract.py"""
        if self.model is None:
            return torch.randn(self.feature_dim)  # Dummy features
        
        try:
            # Preprocess image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif isinstance(image, str):
                image = Image.open(image)
            
            # Apply transforms and extract features
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Forward through ResNet backbone
                features = self.model(input_tensor)
                # Apply adaptive pooling
                features = self.resize_dim(features)
                
                # Flatten and permute like original code
                features = torch.flatten(features, start_dim=2)  # [B, C, H*W]
                features = features.permute((0, 2, 1))          # [B, H*W, C]
                features = features.squeeze(0)                  # [H*W, C] or [1, C] if patch_size=1
                
                # If patch_size=1, we get [1, 512], squeeze to [512]
                if features.shape[0] == 1:
                    features = features.squeeze(0)
                else:
                    # For multiple patches, take mean
                    features = features.mean(dim=0)
                
                features = features.cpu()
            
            return features
            
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return torch.randn(self.feature_dim)


class FeatureROIExtractor:
    """
    ROI Feature Extractor
    Based on the original feature_extract_roi.py logic
    """
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_bbox = 20
        
        # Initialize ResNet18 backbone like feature_extract_roi.py
        try:
            if models is None or transforms is None:
                raise ImportError("torchvision not available")
                
            # Use ResNet18 with latest weights
            self.img_feature_extractor = models.resnet18(pretrained=True)
            # Remove final layers (same as original)
            self.img_feature_extractor = torch.nn.Sequential(
                *list(self.img_feature_extractor.children())[:-2]
            )
            
            self.img_feature_extractor.to(self.device)
            self.img_feature_extractor.eval()
            
            # Image preprocessing - exactly like feature_extract_roi.py  
            self.transform = transforms.Compose([
                transforms.Resize((300, 256)),  # Same as feature_extract.py
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            print("✅ FeatureROIExtractor initialized with ResNet18")
            
        except Exception as e:
            print(f"❌ Error initializing FeatureROIExtractor: {e}")
            self.img_feature_extractor = None
    
    def extract_roi_features(self, image, bboxes, class_ids):
        """Extract ROI features from detected objects - following feature_extract_roi.py"""
        try:
            if self.img_feature_extractor is None:
                # Return dummy features if model not available
                dummy_features = torch.zeros(530)
                return dummy_features.unsqueeze(0), torch.zeros(1, 4)
                
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # If no detections, return empty features
            if len(bboxes) == 0:
                # Return single zero feature for "no objects detected"
                dummy_features = torch.zeros(530)  # 14 (classes) + 4 (bbox) + 512 (visual)
                return dummy_features.unsqueeze(0), torch.zeros(1, 4)
            
            # Preprocess image like feature_extract_roi.py
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Prepare bboxes and classes like original code
            boxes = []
            classes = []
            
            # Object class mapping (correct mapping from feature_extract_roi.py)
            object_dict = {
                "cystic_plate": 0,
                "gallbladder": 1,
                "abdominal_wall_cavity": 2,
                "omentum": 3,
                "liver": 4,
                "cystic_duct": 5,
                "gut": 6,
                "bipolar": 7,
                "clipper": 8,
                "grasper": 9,
                "hook": 10,
                "irrigator": 11,
                "scissors": 12,
                "specimenbag": 13,
            }
            
            for bbox, class_id in zip(bboxes, class_ids):
                x1, y1, x2, y2 = bbox
                
                # Scale bboxes by 2 (like in original preprocessing)
                scaled_bbox = [x1 * 2, y1 * 2, x2 * 2, y2 * 2]
                boxes.append(scaled_bbox)
                
                # Create one-hot class encoding (14 classes)
                class_onehot = torch.zeros(14)
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    mapped_id = object_dict.get(class_name, -1)
                    if mapped_id >= 0 and mapped_id < 14:
                        class_onehot[mapped_id] = 1.0
                
                classes.append(class_onehot.unsqueeze(0))
            
            # Convert to tensors like original code
            boxes = [torch.tensor(boxes, dtype=torch.float32).to(self.device)]
            if len(classes) > 0:
                classes = torch.cat(classes, 0).to(self.device)
            else:
                classes = torch.zeros(0, 14).to(self.device)
            
            with torch.no_grad():
                # Extract backbone features
                backbone_features = self.img_feature_extractor(img_tensor)
                
                # Apply ROI Align like original (spatial_scale=0.031 for 300x256 -> feature map)
                try:
                    from torchvision.ops import roi_align
                    roi_features = roi_align(
                        backbone_features, 
                        boxes, 
                        spatial_scale=0.031,  # Same as original
                        output_size=1
                    ).squeeze(-1).squeeze(-1)
                except ImportError:
                    # Fallback if torchvision.ops not available
                    print("⚠️ ROI align not available, using global pooling")
                    roi_features = backbone_features.mean(dim=(2, 3)).repeat(len(boxes[0]), 1)
                
                # Normalize bboxes like original (relative to 860x480)
                img_width, img_height = 860, 480  # Target size from YOLO preprocessing
                boxes_norm = torch.tensor([
                    [bbox[0] / img_width, bbox[1] / img_height, 
                     bbox[2] / img_width, bbox[3] / img_height]
                    for bbox in boxes[0].cpu().numpy()  
                ], dtype=torch.float32).to(self.device)
                
                # Combine features: [classes + normalized_bbox + visual_features] = [14 + 4 + 512] = 530
                combined_features = torch.cat([classes, boxes_norm, roi_features], dim=1)
                
                return combined_features.cpu(), boxes_norm.cpu()
            
        except Exception as e:
            print(f"❌ Error extracting ROI features: {e}")
            # Return dummy features
            dummy_features = torch.zeros(530)
            return dummy_features.unsqueeze(0), torch.zeros(1, 4)


class VQAPipeline:
    """Complete VQA pipeline for surgical scene understanding"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.device = model_manager.device
        
        # Object class names (surgical instruments) - matching YOLO training classes
        self.class_names = [
            'cystic_plate', 'gallbladder', 'abdominal_wall_cavity', 'omentum',
            'liver', 'cystic_duct', 'gut', 'bipolar', 'clipper', 'grasper', 
            'hook', 'irrigator', 'scissors', 'specimenbag'
        ]
        
        # Initialize feature extractors
        self.feature_extractor = FeatureExtractor(device=self.device)
        self.roi_extractor = FeatureROIExtractor(device=self.device)
        
        # Pass class names to ROI extractor
        self.roi_extractor.class_names = self.class_names
        
        print("✅ VQA Pipeline initialized")
    
    def detect_objects(self, image):
        """Run YOLOv8n object detection with proper preprocessing"""
        try:
            if self.model_manager.yolo_model is None:
                return [], [], []
            
            # Preprocess image for YOLO like in yolov8n-ssgvqa.ipynb
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Resize image to 860x480 (same as training preprocessing)
            original_size = image.size
            yolo_image = image.resize((860, 480))
            
            # Run detection on preprocessed image
            results = self.model_manager.yolo_model(yolo_image, conf=0.3, iou=0.5)
            
            bboxes = []
            class_ids = []
            confidences = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                # Scale factors to map back to original image size
                scale_x = original_size[0] / 860
                scale_y = original_size[1] / 480
                
                for i in range(len(boxes)):
                    # Get bbox coordinates [x1, y1, x2, y2] on 860x480 image
                    bbox = boxes.xyxy[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())
                    confidence = float(boxes.conf[i].cpu().numpy())
                    
                    # Scale bbox back to original image size
                    x1, y1, x2, y2 = bbox
                    scaled_bbox = [
                        x1 * scale_x,
                        y1 * scale_y, 
                        x2 * scale_x,
                        y2 * scale_y
                    ]
                    
                    bboxes.append(scaled_bbox)
                    class_ids.append(class_id)
                    confidences.append(confidence)
            
            print(f"🔍 Detected {len(bboxes)} objects")
            return bboxes, class_ids, confidences
            
        except Exception as e:
            print(f"❌ Error in object detection: {e}")
            return [], [], []
    
    def create_graph_data(self, image, bboxes, class_ids):
        """Create graph data for VQA model - following ssg_dataset.py approach"""
        try:
            if Data is None:
                raise ImportError("torch_geometric not available")
            
            # For demo, we need to simulate the features since we don't have pre-computed HDF5 files
            # In production, this would load from roi_coord_gt/*.hdf5 and cropped_images/*.hdf5
            
            if len(bboxes) == 0:
                # No objects detected - create single frame node like ssg_dataset.py
                frame_features = self.feature_extractor.extract_features(image)
                
                # Ensure frame features are 512-dimensional
                if frame_features.numel() < 512:
                    padding = torch.zeros(512 - frame_features.numel())
                    frame_features = torch.cat([frame_features, padding])
                elif frame_features.numel() > 512:
                    frame_features = frame_features[:512]
                
                # Create single node: [bbox + visual_features] = [4 + 512] = 516
                node_features = torch.cat([
                    torch.tensor([0.0, 0.0, 1.0, 1.0]),  # Full frame bbox
                    frame_features
                ]).unsqueeze(0)
                
                class_indices = torch.tensor([14])  # Background class
                
                # Self-loop edge
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                edge_attr = torch.zeros(1, 16)  # Dummy edge attributes
                
                total_nodes = 1
                
            else:
                # Extract features for objects and frame
                frame_features = self.feature_extractor.extract_features(image)
                roi_features, _ = self.roi_extractor.extract_roi_features(image, bboxes, class_ids)
                
                # Process ROI features like in ssg_dataset.py
                classes_one_hot = roi_features[:, :14]  # First 14 dims are class one-hot
                classes_indices = classes_one_hot.argmax(dim=1)  # Get class indices
                bboxes_norm = roi_features[:, 14:18]  # Next 4 dims are normalized bboxes
                roi_visual_features = roi_features[:, 18:]  # Remaining dims are visual features
                
                # Filter valid objects (like ssg_dataset.py)
                valid_mask = (classes_one_hot.sum(dim=1) > 0)
                object_classes_indices = classes_indices[valid_mask]
                object_bboxes = bboxes_norm[valid_mask]
                object_visual_features = roi_visual_features[valid_mask]
                num_objects = valid_mask.sum().item()
                
                # Ensure frame features are correct size
                if frame_features.numel() < 512:
                    padding = torch.zeros(512 - frame_features.numel())
                    frame_features = torch.cat([frame_features, padding])
                elif frame_features.numel() > 512:
                    frame_features = frame_features[:512]
                
                if num_objects == 0:
                    # All objects filtered out - use single frame node
                    node_features = torch.cat([
                        torch.tensor([0.0, 0.0, 1.0, 1.0]),
                        frame_features
                    ]).unsqueeze(0)
                    class_indices = torch.tensor([14])
                    total_nodes = 1
                    
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                    edge_attr = torch.zeros(1, 16)
                    
                else:
                    # Create object nodes: [bbox + visual_features] = [4 + 512] = 516
                    object_node_features = torch.cat([object_bboxes, object_visual_features], dim=1)
                    
                    # Create frame node: [bbox + visual_features] = [4 + 512] = 516  
                    frame_node_features = torch.cat([
                        torch.tensor([[0.0, 0.0, 1.0, 1.0]]),  # Full frame bbox
                        frame_features.unsqueeze(0)
                    ], dim=1)
                    
                    # Combine all nodes
                    node_features = torch.cat([object_node_features, frame_node_features], dim=0)
                    
                    # Class indices: objects + frame (class 14)
                    class_indices = torch.cat([object_classes_indices, torch.tensor([14])])
                    
                    total_nodes = num_objects + 1
                    
                    # Create edges (all-to-all connectivity)
                    edge_list = []
                    for i in range(total_nodes):
                        for j in range(total_nodes):
                            edge_list.append([i, j])
                    
                    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                    edge_attr = torch.zeros(len(edge_list), 16)  # Dummy edge attributes
            
            # Create PyTorch Geometric Data object (same structure as ssg_dataset.py)
            graph_data = Data(
                x=node_features,
                class_indices=class_indices,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=total_nodes
            )
            
            return graph_data
            
        except Exception as e:
            print(f"❌ Error creating graph data: {e}")
            # Return dummy graph data
            dummy_features = torch.randn(1, 516)  # 4 bbox + 512 visual
            dummy_classes = torch.tensor([14])    # Background class
            dummy_edges = torch.tensor([[0], [0]], dtype=torch.long)
            dummy_edge_attr = torch.zeros(1, 16)
            
            return Data(
                x=dummy_features,
                class_indices=dummy_classes,
                edge_index=dummy_edges, 
                edge_attr=dummy_edge_attr,
                num_nodes=1
            ) if Data is not None else None
    
    def tokenize_question(self, question):
        """Tokenize question using BioClinicalBERT"""
        try:
            if self.model_manager.tokenizer is None:
                return None
            
            # Tokenize with max length 128
            tokens = self.model_manager.tokenizer(
                question,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=128
            )
            
            return tokens
            
        except Exception as e:
            print(f"❌ Error tokenizing question: {e}")
            return None
    
    def predict_vqa(self, image, question, model_type='gat', top_k=3):
        """Complete VQA prediction pipeline"""
        try:
            print(f"🔮 Running VQA prediction with {model_type.upper()} model...")
            
            # 1. Object Detection
            bboxes, class_ids, confidences = self.detect_objects(image)
            
            # 2. Create Graph Data
            graph_data = self.create_graph_data(image, bboxes, class_ids)
            if graph_data is None:
                return [], "Error: Graph data creation failed"
            
            # 3. Tokenize Question
            question_tokens = self.tokenize_question(question)
            if question_tokens is None:
                return [], "Error tokenizing question"
            
            # 4. Move to device
            graph_data = graph_data.to(self.device)
            question_batch = {k: v.to(self.device) for k, v in question_tokens.items()}
            
            # 5. VQA Model Inference
            if model_type not in self.model_manager.vqa_models:
                return [], f"Model {model_type} not available"
            
            model = self.model_manager.vqa_models[model_type]
            
            with torch.no_grad():
                # Create batch for single sample
                if Batch is not None:
                    batch_graph = Batch.from_data_list([graph_data])
                else:
                    return [], "Error: torch_geometric not available"
                
                # Forward pass
                logits = model(batch_graph, question_batch)
                probabilities = F.softmax(logits, dim=1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
                
                predictions = []
                for i in range(top_k):
                    idx = top_indices[0][i].item()
                    prob = top_probs[0][i].item()
                    answer = self.model_manager.label2ans.get(str(idx), f"answer_{idx}")
                    
                    predictions.append({
                        'answer': answer,
                        'confidence': prob,
                        'rank': i + 1
                    })
            
            # 6. Format detection results
            detection_info = {
                'num_objects': len(bboxes),
                'detected_classes': [self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}" 
                                   for cls_id in class_ids],
                'bboxes': bboxes,
                'confidences': confidences
            }
            
            return predictions, detection_info
            
        except Exception as e:
            print(f"❌ Error in VQA prediction: {e}")
            return [], f"Error: {str(e)}"


def draw_bounding_boxes(image, bboxes, class_ids, confidences, class_names):
    """Draw bounding boxes on image"""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Create a copy for drawing
        img_with_boxes = image.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        # Colors for different classes
        colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', 
            '#00FFFF', '#800000', '#008000', '#000080', '#808000',
            '#800080', '#008080', '#FFA500', '#FFC0CB', '#A52A2A'
        ]
        
        for bbox, class_id, conf in zip(bboxes, class_ids, confidences):
            x1, y1, x2, y2 = map(int, bbox)
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            label = f"{class_name}: {conf:.2f}"
            
            # Try to load font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Get text size and draw background
            bbox_text = draw.textbbox((x1, y1-25), label, font=font)
            draw.rectangle(bbox_text, fill=color)
            draw.text((x1, y1-25), label, fill='white', font=font)
        
        return img_with_boxes
        
    except Exception as e:
        print(f"Error drawing bounding boxes: {e}")
        return image


# Question templates for surgical VQA
QUESTION_TEMPLATES = [
    "What is the main surgical instrument visible in this image?",
    "How many surgical tools are present in the scene?",
    "Is the gallbladder visible in this surgical view?",
    "What type of surgical procedure is being performed?",
    "Are there any clips or clippers visible?",
    "Is the liver visible in this laparoscopic view?",
    "What is the surgeon grasping with the instrument?",
    "Is there any bleeding or irrigation visible?",
    "How many graspers are being used?",
    "Is the cystic artery visible in the image?",
    "What anatomical structure is most prominent?",
    "Are scissors being used in this procedure?",
    "Is there a specimen bag visible?",
    "What is the current surgical step?",
    "Is suction being applied in this view?"
]


def format_predictions(predictions, detection_info):
    """Format VQA predictions and detection info for display"""
    try:
        # Format VQA results
        vqa_results = "🎯 **VQA Predictions:**\n\n"
        if predictions:
            for pred in predictions:
                vqa_results += f"**{pred['rank']}.** {pred['answer']} " \
                              f"(Confidence: {pred['confidence']:.3f})\n"
        else:
            vqa_results += "No predictions available\n"
        
        # Format detection results
        detection_results = "\n🔍 **Object Detection Results:**\n\n"
        if isinstance(detection_info, dict):
            detection_results += f"**Objects detected:** {detection_info['num_objects']}\n"
            if detection_info['detected_classes']:
                detection_results += "**Detected classes:**\n"
                for i, (cls, conf) in enumerate(zip(detection_info['detected_classes'], 
                                                  detection_info['confidences'])):
                    detection_results += f"  • {cls} ({conf:.3f})\n"
        else:
            detection_results += f"Detection info: {detection_info}\n"
        
        return vqa_results + detection_results
        
    except Exception as e:
        return f"Error formatting results: {e}"


# Global variable for VQA pipeline
vqa_pipeline = None

def surgical_vqa_predict(image, question, model_type, predict_bbox, top_k):
    """
    Main prediction function for Gradio interface
    
    Args:
        image: PIL Image uploaded by user
        question: Text question from user
        model_type: Selected VQA model (gat/gcn/nognn)  
        predict_bbox: Boolean to show bounding boxes
        top_k: Number of top predictions to return
    
    Returns:
        Tuple of (image_with_boxes, formatted_results)
    """
    global vqa_pipeline
    
    try:
        print(f"🚀 Starting surgical VQA prediction...")
        print(f"   Model: {model_type}, Question: '{question[:50]}...'")
        
        # Validate inputs
        if image is None:
            return None, "❌ Please upload an image"
        
        if not question or question.strip() == "":
            return image, "❌ Please enter a question"
        
        if model_type not in ['gat', 'gcn', 'nognn']:
            return image, "❌ Invalid model type selected"
            
        if vqa_pipeline is None:
            return image, "❌ VQA pipeline not initialized"
        
        # Run VQA prediction
        predictions, detection_info = vqa_pipeline.predict_vqa(
            image=image,
            question=question,
            model_type=model_type,
            top_k=min(max(1, top_k), 5)  # Clamp between 1-5
        )
        
        # Prepare output image
        output_image = image
        if predict_bbox and isinstance(detection_info, dict) and detection_info['num_objects'] > 0:
            # Draw bounding boxes on image
            output_image = draw_bounding_boxes(
                image=image,
                bboxes=detection_info['bboxes'],
                class_ids=[vqa_pipeline.class_names.index(cls) if cls in vqa_pipeline.class_names else 0 
                          for cls in detection_info['detected_classes']],
                confidences=detection_info['confidences'],
                class_names=vqa_pipeline.class_names
            )
        
        # Format results for display
        formatted_results = format_predictions(predictions, detection_info)
        
        print(f"✅ Prediction completed successfully!")
        return output_image, formatted_results
        
    except Exception as e:
        error_msg = f"❌ Error during prediction: {str(e)}"
        print(error_msg)
        return image, error_msg


def load_sample_image(sample_name):
    """Load sample images from VID22 directory"""
    try:
        vid22_dir = Path("VID22")
        if vid22_dir.exists():
            # Get list of sample images
            image_files = list(vid22_dir.glob("*.png")) + list(vid22_dir.glob("*.jpg"))
            if image_files:
                # Load first few images as samples
                sample_images = sorted(image_files)[:5]  # First 5 images
                
                if sample_name == "Sample 1" and len(sample_images) > 0:
                    return Image.open(sample_images[0])
                elif sample_name == "Sample 2" and len(sample_images) > 1:
                    return Image.open(sample_images[1])
                elif sample_name == "Sample 3" and len(sample_images) > 2:
                    return Image.open(sample_images[2])
        
        return None
        
    except Exception as e:
        print(f"Error loading sample image: {e}")
        return None


def create_surgical_vqa_demo():
    """Create comprehensive Gradio demo for Surgical VQA"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(45deg, #2196F3, #21CBF3);
        border: none;
    }
    .gr-button-secondary {
        background: linear-gradient(45deg, #FF9800, #FF5722);
        border: none;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Surgical VQA Demo") as demo:
        
        # Header
        gr.Markdown("""
        # 🏥 Surgical VQA Demo
        
        **Intelligent Visual Question Answering for Surgical Scenes**
        
        Upload a surgical image, ask questions about the procedure, and get AI-powered answers using advanced Graph Neural Networks and Computer Vision.
        
        ---
        """)
        
        with gr.Row():
            # Left Column - Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 **Input**")
                
                # Image upload
                image_input = gr.Image(
                    type="pil",
                    label="Upload Surgical Image",
                    height=300,
                    sources=["upload", "clipboard"],
                    interactive=True
                )
                
                # Sample image buttons
                with gr.Row():
                    sample_btn1 = gr.Button("📷 Sample 1", size="sm")
                    sample_btn2 = gr.Button("📷 Sample 2", size="sm") 
                    sample_btn3 = gr.Button("📷 Sample 3", size="sm")
                
                # Question input
                question_input = gr.Textbox(
                    label="Ask a Question",
                    placeholder="What surgical instrument is visible?",
                    lines=2
                )
                
                # Question templates dropdown
                template_dropdown = gr.Dropdown(
                    choices=QUESTION_TEMPLATES,
                    label="Or select a template question:",
                    value=None
                )
                
                # Model selection
                model_select = gr.Radio(
                    choices=["gat", "gcn", "nognn"],
                    value="gat",
                    label="VQA Model Type",
                    info="Select the Graph Neural Network architecture"
                )
                
                # Options
                with gr.Row():
                    bbox_checkbox = gr.Checkbox(
                        label="Show Bounding Boxes",
                        value=True,
                        info="Display object detection results"
                    )
                    
                    topk_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Top-K Answers",
                        info="Number of top predictions to show"
                    )
                
                # Predict button
                predict_btn = gr.Button(
                    "🔮 Analyze Image & Answer Question",
                    variant="primary",
                    size="lg"
                )
            
            # Right Column - Output
            with gr.Column(scale=1):
                gr.Markdown("### 📊 **Results**")
                
                # Output image
                image_output = gr.Image(
                    label="Analyzed Image",
                    height=300
                )
                
                # Results text
                results_output = gr.Markdown(
                    label="VQA Results",
                    value="Upload an image and ask a question to see results here..."
                )
        
        # Footer info
        gr.Markdown("""
        ---
        ### 🔧 **Model Information**
        
        - **Object Detection**: YOLOv8n fine-tuned on surgical instruments
        - **VQA Models**: GCN/GAT/GIN Graph Neural Networks with cross-modal attention
        - **Text Encoder**: BioClinicalBERT for medical domain understanding
        - **Feature Extraction**: ResNet-based visual feature extraction
        
        ### 📋 **Supported Question Types**
        - Instrument identification and counting
        - Anatomical structure recognition  
        - Surgical procedure understanding
        - Scene description and analysis
        """)
        
        # Event handlers
        def update_question_from_template(template):
            return template if template else ""
        
        # Connect template dropdown to question input
        template_dropdown.change(
            fn=update_question_from_template,
            inputs=[template_dropdown],
            outputs=[question_input]
        )
        
        # Connect sample buttons to image input
        sample_btn1.click(
            fn=lambda: load_sample_image("Sample 1"),
            outputs=[image_input]
        )
        sample_btn2.click(
            fn=lambda: load_sample_image("Sample 2"),
            outputs=[image_input]
        )
        sample_btn3.click(
            fn=lambda: load_sample_image("Sample 3"),
            outputs=[image_input]
        )
        
        # Main prediction function
        predict_btn.click(
            fn=surgical_vqa_predict,
            inputs=[
                image_input,
                question_input, 
                model_select,
                bbox_checkbox,
                topk_slider
            ],
            outputs=[image_output, results_output]
        )
        
        # Example interactions (fixed - removed image examples that cause errors)
        # gr.Examples(
        #     examples=[
        #         [None, "What surgical instrument is being used?", "gat", True, 3],
        #         [None, "How many tools are visible in the image?", "gcn", True, 2],
        #         [None, "Is the gallbladder visible in this view?", "gat", False, 1],
        #     ],
        #     inputs=[image_input, question_input, model_select, bbox_checkbox, topk_slider]
        # )
    
    return demo


def main():
    """Main function to initialize and launch the demo"""
    global vqa_pipeline
    
    print("🚀 Initializing Surgical VQA Demo...")
    
    # Initialize model manager
    print("📥 Loading models...")
    model_manager = ModelManager()
    
    # Initialize VQA pipeline
    print("🔄 Initializing VQA Pipeline...")
    vqa_pipeline = VQAPipeline(model_manager)
    
    # Create and launch the demo
    print("🎭 Creating Gradio demo interface...")
    demo = create_surgical_vqa_demo()
    
    print("🚀 Launching Surgical VQA Demo...")
    print("🌐 The demo will be available at: http://localhost:7860")
    print("📱 Access from mobile/other devices using your machine's IP address")
    
    # Launch the demo with minimal configuration to avoid JSON schema issues
    try:
        demo.launch(
            server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0
            server_port=7860,
            share=False,  # Disable share to avoid complex API processing
            debug=False,  # Disable debug to reduce API info processing
            show_error=True,
            inbrowser=True,  # Automatically open browser
            quiet=True  # Reduce logging output
        )
    except ValueError as e:
        if "shareable link" in str(e):
            print("⚠️ Localhost access issue detected. Trying with share enabled...")
            demo.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=True,  # Enable sharing to bypass localhost issues
                debug=False,
                show_error=True,
                inbrowser=True,
                quiet=True
            )
        else:
            raise e


if __name__ == "__main__":
    main()