"""
Script để predict và lưu thông tin bounding box và label cho tập test
"""

import os
import json
import cv2
import torch
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
from pathlib import Path

# Mapping labels
label2id = {
    "abdominal_wall_cavity": 2,
    "cystic_duct": 5,
    "cystic_plate": 0,
    "gallbladder": 1,
    "gut": 6,
    "liver": 4,
    "omentum": 3,
    "bipolar": 7,
    "clipper": 8,
    "grasper": 9,
    "hook": 10,
    "irrigator": 11,
    "scissors": 12,
    "specimenbag": 13,
}

id2label = {v: k for k, v in label2id.items()}

class YOLOPredictor:
    def __init__(self, model_path, confidence_threshold=0.25):
        """
        Khởi tạo YOLO predictor
        
        Args:
            model_path (str): Đường dẫn đến model weights
            confidence_threshold (float): Ngưỡng confidence cho detection
        """
        self.model = YOLO(model_path)
        self.conf_threshold = confidence_threshold
        self.results_data = []
        
    def predict_single_image(self, image_path):
        """
        Predict một ảnh duy nhất
        
        Args:
            image_path (str): Đường dẫn đến ảnh
            
        Returns:
            dict: Thông tin prediction cho ảnh
        """
        # Thực hiện prediction
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            imgsz=(480, 860),
            save=False,  # Không lưu ảnh prediction
            verbose=False
        )
        
        result = results[0]
        image_name = os.path.basename(image_path)
        
        # Lấy thông tin ảnh với xử lý lỗi
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img_height, img_width = img.shape[:2]
        
        predictions = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                # Normalize coordinates (0-1)
                x1_norm = x1 / img_width
                y1_norm = y1 / img_height
                x2_norm = x2 / img_width
                y2_norm = y2 / img_height
                width_norm = width / img_width
                height_norm = height / img_height
                
                prediction_info = {
                    'image_name': image_name,
                    'detection_id': i,
                    'class_id': int(cls),
                    'class_name': id2label.get(int(cls), f'unknown_{cls}'),
                    'confidence': float(score),
                    'bbox_x1': float(x1),
                    'bbox_y1': float(y1),
                    'bbox_x2': float(x2),
                    'bbox_y2': float(y2),
                    'bbox_width': float(width),
                    'bbox_height': float(height),
                    'bbox_x1_norm': float(x1_norm),
                    'bbox_y1_norm': float(y1_norm),
                    'bbox_x2_norm': float(x2_norm),
                    'bbox_y2_norm': float(y2_norm),
                    'bbox_width_norm': float(width_norm),
                    'bbox_height_norm': float(height_norm),
                    'image_width': img_width,
                    'image_height': img_height
                }
                predictions.append(prediction_info)
        
        return {
            'image_name': image_name,
            'image_path': image_path,
            'image_width': img_width,
            'image_height': img_height,
            'num_detections': len(predictions),
            'predictions': predictions
        }
    
    def predict_test_set(self, test_images_dir, output_dir='prediction_results'):
        """
        Predict toàn bộ tập test và lưu kết quả
        
        Args:
            test_images_dir (str): Thư mục chứa ảnh test
            output_dir (str): Thư mục lưu kết quả
        """
        print(f"🔍 Starting prediction on test set...")
        print(f"📂 Test images directory: {test_images_dir}")
        print(f"💾 Output directory: {output_dir}")
        
        # Tạo thư mục output
        os.makedirs(output_dir, exist_ok=True)
        
        # Lấy danh sách tất cả ảnh trong thư mục test
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(test_images_dir).glob(f'*{ext}'))
        
        # Loại bỏ duplicates nếu có
        image_files = list(set(image_files))
        
        if not image_files:
            print("❌ No images found in test directory!")
            return
        
        print(f"🖼 Found {len(image_files)} images to process")
        
        all_predictions = []
        all_detections = []
        
        # Process từng ảnh
        successful_count = 0
        error_count = 0
        
        for i, image_path in enumerate(image_files):
            if i % 100 == 0 or i == len(image_files) - 1:  # Progress update mỗi 100 ảnh hoặc ảnh cuối
                progress = (i + 1) / len(image_files) * 100
                print(f"Progress: {i+1}/{len(image_files)} ({progress:.1f}%) - Processing: {image_path.name}")
            elif i % 10 == 0:  # Simple counter mỗi 10 ảnh
                print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            try:
                # Kiểm tra file có tồn tại và có thể đọc được không
                if not os.path.exists(str(image_path)):
                    print(f"⚠️ File not found: {image_path.name}")
                    error_count += 1
                    continue
                    
                # Thử đọc ảnh trước khi predict
                test_img = cv2.imread(str(image_path))
                if test_img is None:
                    print(f"⚠️ Cannot read image: {image_path.name}")
                    error_count += 1
                    continue
                
                result = self.predict_single_image(str(image_path))
                all_predictions.append(result)
                
                # Thêm các detection vào danh sách chung
                for pred in result['predictions']:
                    all_detections.append(pred)
                
                successful_count += 1
                    
            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {str(e)}")
                error_count += 1
                continue
        
        # Lưu kết quả
        self._save_results(all_predictions, all_detections, output_dir)
        
        print(f"✅ Prediction completed!")
        print(f"📊 Total images found: {len(image_files)}")
        print(f"📊 Successfully processed: {successful_count}")
        print(f"📊 Errors encountered: {error_count}")
        print(f"📊 Total detections: {len(all_detections)}")
        
        return all_predictions, all_detections
    
    def _save_results(self, all_predictions, all_detections, output_dir):
        """
        Lưu kết quả prediction ra các file khác nhau
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Lưu tổng quan kết quả mỗi ảnh (JSON)
        summary_file = os.path.join(output_dir, f'prediction_summary_{timestamp}.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_predictions, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved prediction summary to: {summary_file}")
        
        # 2. Lưu tất cả detections (CSV)
        if all_detections:
            df_detections = pd.DataFrame(all_detections)
            csv_file = os.path.join(output_dir, f'all_detections_{timestamp}.csv')
            df_detections.to_csv(csv_file, index=False)
            print(f"💾 Saved all detections to: {csv_file}")
            
            # 3. Lưu thống kê theo class
            class_stats = df_detections.groupby('class_name').agg({
                'confidence': ['count', 'mean', 'std', 'min', 'max'],
                'bbox_width_norm': 'mean',
                'bbox_height_norm': 'mean'
            }).round(4)
            
            stats_file = os.path.join(output_dir, f'class_statistics_{timestamp}.csv')
            class_stats.to_csv(stats_file)
            print(f"💾 Saved class statistics to: {stats_file}")
        
        # 4. Lưu file YOLO format cho mỗi ảnh (để có thể so sánh với ground truth)
        yolo_dir = os.path.join(output_dir, 'yolo_predictions')
        os.makedirs(yolo_dir, exist_ok=True)
        
        for pred_result in all_predictions:
            if pred_result['predictions']:
                image_name = pred_result['image_name']
                txt_name = os.path.splitext(image_name)[0] + '.txt'
                txt_path = os.path.join(yolo_dir, txt_name)
                
                with open(txt_path, 'w') as f:
                    for pred in pred_result['predictions']:
                        # Convert to YOLO format: class_id center_x center_y width height
                        center_x = (pred['bbox_x1_norm'] + pred['bbox_x2_norm']) / 2
                        center_y = (pred['bbox_y1_norm'] + pred['bbox_y2_norm']) / 2
                        width = pred['bbox_width_norm']
                        height = pred['bbox_height_norm']
                        
                        f.write(f"{pred['class_id']} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"💾 Saved YOLO format predictions to: {yolo_dir}")

def main():
    """
    Main function để chạy prediction
    """
    # Cấu hình paths - CẬP NHẬT THEO DỰ ÁN CỦA BẠN
    model_path = r"D:\\KLTN\\gnn-surgical-understanding\\detection_model\\runs\\detect\\train3\\weights\\best.pt"  # Đường dẫn đến model weights
    test_images_dir = r"D:\\KLTN\\gnn-surgical-understanding\\detection_model\\dataset\\images\\test"  # Thư mục chứa ảnh test
    output_dir = r"D:\\KLTN\\gnn-surgical-understanding\\detection_model\\prediction_results"  # Thư mục lưu kết quả
    confidence_threshold = 0.25  # Ngưỡng confidence
    
    # Kiểm tra file model có tồn tại không
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please update the model_path variable to point to your trained model")
        return
    
    # Kiểm tra thư mục test có tồn tại không
    if not os.path.exists(test_images_dir):
        print(f"❌ Test images directory not found: {test_images_dir}")
        print("Please update the test_images_dir variable to point to your test images")
        return
    
    print("🚀 Starting YOLO Prediction and Results Saving...")
    print(f"🎯 Model: {model_path}")
    print(f"📁 Test images: {test_images_dir}")
    print(f"🎚 Confidence threshold: {confidence_threshold}")
    
    # Khởi tạo predictor
    predictor = YOLOPredictor(model_path, confidence_threshold)
    
    # Thực hiện prediction
    predictions, detections = predictor.predict_test_set(test_images_dir, output_dir)
    
    print("\n📈 PREDICTION SUMMARY:")
    print(f"Total images: {len(predictions)}")
    print(f"Total detections: {len(detections)}")
    
    if detections:
        df = pd.DataFrame(detections)
        print(f"Average confidence: {df['confidence'].mean():.3f}")
        print(f"Classes detected: {df['class_name'].nunique()}")
        print("\nClass distribution:")
        print(df['class_name'].value_counts())

if __name__ == "__main__":
    main()