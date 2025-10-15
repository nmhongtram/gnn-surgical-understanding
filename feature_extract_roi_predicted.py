"""
Project: Advancing Surgical VQA with Scene Graph Knowledge
-----
ROI Feature Extraction using YOLO Predicted Bounding Boxes for Test Set
"""
import os
import sys
import h5py
import json
import argparse

import torch
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
from glob import glob

import torch
from torch import nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision.ops import roi_pool, roi_align
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path


class FeatureROIExtractor(nn.Module):
    def __init__(self):
        super(FeatureROIExtractor, self).__init__()
        # visual feature extraction
        self.img_feature_extractor = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.img_feature_extractor = torch.nn.Sequential(
            *(list(self.img_feature_extractor.children())[:-2])
        )
        self.max_bbox = 20

    def forward(self, img, boxes, classes):
        if len(boxes[0]) == 0:
            final_outputs = torch.zeros((1, 530)).cuda()
            return final_outputs

        outputs = self.img_feature_extractor(img)
        # Use ROI align for better feature extraction
        outputs = (
            roi_align(outputs, boxes, spatial_scale=0.031, output_size=1)
            .squeeze(-1)
            .squeeze(-1)
        )
        
        # Normalize bounding boxes coordinates
        boxes_norm = torch.FloatTensor(
            [[i[0] / 864, i[1] / 480, i[2] / 864, i[3] / 480] for i in boxes[0]]
        ).cuda()
        
        # Concatenate class info, normalized bbox, and visual features
        outputs = torch.cat([classes, boxes_norm, outputs], 1)

        return outputs


class PredictedROIFeatureExtractor:
    def __init__(self, predictions_file, test_images_dir, output_dir):
        """
        Initialize ROI feature extractor using predicted bounding boxes
        
        Args:
            predictions_file (str): Path to JSON file containing YOLO predictions
            test_images_dir (str): Directory containing test images
            output_dir (str): Directory to save extracted features
        """
        self.predictions_file = predictions_file
        self.test_images_dir = test_images_dir
        self.output_dir = output_dir
        
        # Load predictions
        with open(predictions_file, 'r', encoding='utf-8') as f:
            self.predictions = json.load(f)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Object mapping (same as original)
        self.object_dict = {
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
        
        # Image transforms (same as original)
        self.transform = transforms.Compose([
            transforms.Resize((480, 864)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Initialize feature extraction model
        self.feature_network = FeatureROIExtractor()
        
        # Set up GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            num_gpu = torch.cuda.device_count()
            if num_gpu > 1:
                device_ids = np.arange(num_gpu).tolist()
                self.feature_network = nn.DataParallel(self.feature_network, device_ids=device_ids)
            self.feature_network = self.feature_network.cuda()
        
        self.feature_network.eval()
        
        print(f"🎯 Initialized ROI Feature Extractor")
        print(f"📁 Predictions file: {predictions_file}")
        print(f"📁 Test images dir: {test_images_dir}")
        print(f"📁 Output dir: {output_dir}")
        print(f"🔧 Device: {self.device}")
        print(f"📊 Total predictions: {len(self.predictions)}")
    
    def get_image_predictions(self, image_name):
        """Get predictions for a specific image"""
        for pred in self.predictions:
            if pred['image_name'] == image_name:
                return pred['predictions']
        return []
    
    def prepare_boxes_and_classes(self, predictions):
        """
        Prepare bounding boxes and class vectors from predictions
        
        Args:
            predictions (list): List of prediction dictionaries
            
        Returns:
            tuple: (boxes, classes) ready for feature extraction
        """
        if not predictions:
            return [torch.empty(0, 4)], torch.empty(0, 14)
        
        boxes = []
        classes = []
        
        for pred in predictions:
            # Get bounding box coordinates (already in pixel coordinates from prediction)
            bbox = [
                pred['bbox_x1'],
                pred['bbox_y1'], 
                pred['bbox_x2'],
                pred['bbox_y2']
            ]
            boxes.append(bbox)
            
            # Create one-hot class vector
            classes_vector = torch.zeros(14)
            class_id = pred['class_id']
            if 0 <= class_id < 14:
                classes_vector[class_id] = 1
            classes.append(classes_vector.unsqueeze(0))
        
        # Convert to tensors
        boxes = [torch.Tensor(boxes)]
        if classes:
            classes = torch.cat(classes, 0)
        else:
            classes = torch.empty(0, 14)
        
        return boxes, classes
    
    def extract_features_for_image(self, image_path, image_name):
        """
        Extract ROI features for a single image
        
        Args:
            image_path (str): Path to the image file
            image_name (str): Name of the image file
            
        Returns:
            numpy.ndarray: Extracted visual features
        """
        try:
            # Get predictions for this image
            predictions = self.get_image_predictions(image_name)
            
            # Prepare boxes and classes
            boxes, classes = self.prepare_boxes_and_classes(predictions)
            
            # Load and transform image
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)
            img = torch.unsqueeze(img, 0)
            
            # Move to device
            if torch.cuda.is_available():
                img = img.cuda()
                if len(classes) > 0:
                    classes = classes.cuda()
                if len(boxes[0]) > 0:
                    boxes = [boxes[0].cuda()]
            
            # Extract features
            with torch.no_grad():
                visual_features = self.feature_network(img, boxes, classes)
                visual_features = visual_features.data.cpu().numpy()
            
            return visual_features
            
        except Exception as e:
            print(f"❌ Error extracting features for {image_name}: {str(e)}")
            # Return empty features in case of error
            return np.zeros((1, 530))
    
    def generate_save_filename(self, image_name):
        """
        Generate save filename based on image name
        Following the pattern from original code: VIDxx_yyyy -> xx_yyyy.hdf5
        """
        # Remove extension
        base_name = os.path.splitext(image_name)[0]
        
        # Handle VIDxx_yyyy format
        if base_name.startswith('VID'):
            # Extract xx and yyyy parts
            parts = base_name.split('_')
            if len(parts) >= 2:
                vid_num = parts[0].replace('VID', '')  # Remove VID prefix
                frame_num = parts[1]
                filename = f"{vid_num}_{frame_num}.hdf5"
            else:
                filename = f"{base_name}.hdf5"
        else:
            filename = f"{base_name}.hdf5"
        
        return filename
    
    def extract_all_features(self):
        """Extract ROI features for all images with predictions"""
        print("🚀 Starting ROI feature extraction for all test images...")
        
        successful_count = 0
        error_count = 0
        
        for i, pred_data in enumerate(tqdm(self.predictions, desc="Extracting ROI features")):
            image_name = pred_data['image_name']
            image_path = os.path.join(self.test_images_dir, image_name)
            
            try:
                # Check if image file exists
                if not os.path.exists(image_path):
                    print(f"⚠️ Image not found: {image_path}")
                    error_count += 1
                    continue
                
                # Extract features
                visual_features = self.extract_features_for_image(image_path, image_name)
                
                # Generate save filename
                save_filename = self.generate_save_filename(image_name)
                save_path = os.path.join(self.output_dir, save_filename)
                
                # Save to HDF5 file
                with h5py.File(save_path, 'w') as hdf5_file:
                    hdf5_file.create_dataset('visual_features', data=visual_features)
                
                successful_count += 1
                
                # Progress update
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(self.predictions)} images")
                
            except Exception as e:
                print(f"❌ Error processing {image_name}: {str(e)}")
                error_count += 1
                continue
        
        print(f"\n✅ ROI feature extraction completed!")
        print(f"📊 Successfully processed: {successful_count}")
        print(f"📊 Errors encountered: {error_count}")
        print(f"📁 Features saved to: {self.output_dir}")
        
        return successful_count, error_count


def main():
    """Main function to run ROI feature extraction"""
    
    # Configuration - UPDATE THESE PATHS
    predictions_file = r"D:\\KLTN\\gnn-surgical-understanding\\detection_model\\prediction_results\\prediction_summary_20241013_*.json"
    test_images_dir = r"D:\\KLTN\\gnn-surgical-understanding\\detection_model\\dataset\\images\\test"
    output_dir = r"D:\\KLTN\\gnn-surgical-understanding\\roi_features_predicted"

    # Find the most recent predictions file
    prediction_files = glob(predictions_file)
    if not prediction_files:
        print(f"❌ No prediction files found matching: {predictions_file}")
        print("Please run the prediction script first or update the predictions_file path")
        return
    
    # Use the most recent file
    predictions_file = max(prediction_files, key=os.path.getctime)
    print(f"📁 Using predictions file: {predictions_file}")
    
    # Check if test images directory exists
    if not os.path.exists(test_images_dir):
        print(f"❌ Test images directory not found: {test_images_dir}")
        return
    
    # Initialize and run feature extractor
    extractor = PredictedROIFeatureExtractor(
        predictions_file=predictions_file,
        test_images_dir=test_images_dir,
        output_dir=output_dir
    )
    
    # Extract features for all images
    successful, errors = extractor.extract_all_features()
    
    print(f"\n🎉 Feature extraction summary:")
    print(f"✅ Success: {successful}")
    print(f"❌ Errors: {errors}")
    print(f"📁 Output directory: {output_dir}")


if __name__ == "__main__":
    main()