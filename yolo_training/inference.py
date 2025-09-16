"""
YOLOv8 Inference Script for People Detection
This script performs inference on images using a trained YOLOv8 model.
"""

import os
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import argparse

# Fix for PyTorch 2.6+ compatibility
import torch.serialization
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
    torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
    torch.serialization.add_safe_globals([torch.nn.modules.conv.Conv2d])
    torch.serialization.add_safe_globals([torch.nn.modules.batchnorm.BatchNorm2d])
    torch.serialization.add_safe_globals([torch.nn.modules.activation.SiLU])
    torch.serialization.add_safe_globals([torch.nn.modules.pooling.AdaptiveAvgPool2d])
    torch.serialization.add_safe_globals([torch.nn.modules.linear.Linear])
    torch.serialization.add_safe_globals([torch.nn.modules.dropout.Dropout])
except ImportError:
    pass

class PeopleDetector:
    def __init__(self, model_path="runs/detect/people_detection/weights/best.pt"):
        """
        Initialize the people detector.
        
        Args:
            model_path: Path to the trained YOLO model
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained YOLO model."""
        # Set weights_only=False for PyTorch 2.6+ compatibility
        import torch
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
        
        if os.path.exists(self.model_path):
            print(f"Loading model from: {self.model_path}")
            self.model = YOLO(self.model_path)
        else:
            print(f"Model not found at: {self.model_path}")
            print("Using pre-trained YOLOv8n model instead...")
            self.model = YOLO('yolov8n.pt')
    
    def detect_people(self, image_path, conf_threshold=0.5, save_result=True, output_dir="results"):
        """
        Detect people in an image.
        
        Args:
            image_path: Path to the input image
            conf_threshold: Confidence threshold for detections
            save_result: Whether to save the result image
            output_dir: Directory to save results
            
        Returns:
            List of detection results
        """
        if self.model is None:
            print("Model not loaded!")
            return []
        
        # Run inference
        results = self.model(image_path, conf=conf_threshold)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(confidence),
                        'class': 'person'
                    })
        
        # Save result if requested
        if save_result:
            self.save_result(image_path, results, output_dir)
        
        return detections
    
    def save_result(self, image_path, results, output_dir):
        """Save the detection result image."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the result image from YOLO
        result_image = results[0].plot()
        
        # Save the image
        image_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{image_name}_detected.jpg")
        
        cv2.imwrite(output_path, result_image)
        print(f"Result saved to: {output_path}")
    
    def detect_batch(self, image_dir, conf_threshold=0.5, output_dir="results"):
        """
        Detect people in a batch of images.
        
        Args:
            image_dir: Directory containing images
            conf_threshold: Confidence threshold for detections
            output_dir: Directory to save results
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"Image directory not found: {image_dir}")
            return
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
            image_files.extend(image_dir.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No images found in: {image_dir}")
            return
        
        print(f"Processing {len(image_files)} images...")
        
        # Process each image
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            detections = self.detect_people(str(image_path), conf_threshold, True, output_dir)
            print(f"Found {len(detections)} people")
    
    def evaluate_on_test_set(self, test_dir="yolo_dataset/test/images", conf_threshold=0.5):
        """
        Evaluate the model on the test set.
        
        Args:
            test_dir: Directory containing test images
            conf_threshold: Confidence threshold for detections
        """
        if self.model is None:
            print("Model not loaded!")
            return
        
        print(f"Evaluating on test set: {test_dir}")
        
        # Run validation
        results = self.model.val(data='dataset.yaml', split='test', conf=conf_threshold)
        
        print("Test set evaluation results:")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"Precision: {results.box.mp:.4f}")
        print(f"Recall: {results.box.mr:.4f}")
        
        return results

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='People Detection Inference')
    parser.add_argument('--image', type=str, help='Path to single image for detection')
    parser.add_argument('--batch', type=str, help='Directory containing images for batch detection')
    parser.add_argument('--model', type=str, default='runs/detect/people_detection/weights/best.pt',
                       help='Path to trained model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate on test set')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = PeopleDetector(args.model)
    
    if args.evaluate:
        # Evaluate on test set
        detector.evaluate_on_test_set()
    
    elif args.image:
        # Single image detection
        if os.path.exists(args.image):
            print(f"Detecting people in: {args.image}")
            detections = detector.detect_people(args.image, args.conf, True, args.output)
            print(f"Found {len(detections)} people")
            for i, det in enumerate(detections):
                print(f"Person {i+1}: confidence={det['confidence']:.3f}, bbox={det['bbox']}")
        else:
            print(f"Image not found: {args.image}")
    
    elif args.batch:
        # Batch detection
        detector.detect_batch(args.batch, args.conf, args.output)
    
    else:
        # Default: detect on a sample image from test set
        test_dir = Path("yolo_dataset/test/images")
        if test_dir.exists():
            sample_images = list(test_dir.glob("*.jpg"))
            if sample_images:
                sample_image = sample_images[0]
                print(f"Detecting people in sample image: {sample_image}")
                detections = detector.detect_people(str(sample_image), args.conf, True, args.output)
                print(f"Found {len(detections)} people")
            else:
                print("No sample images found in test directory")
        else:
            print("Test directory not found. Please provide an image with --image or --batch")

if __name__ == "__main__":
    main()
