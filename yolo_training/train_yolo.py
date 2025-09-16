"""
YOLOv8 Training Script for People Detection
This script trains a YOLOv8 model on the people detection dataset.
"""

import os
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path

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

def check_dataset():
    """Check if dataset is properly set up."""
    dataset_path = Path("yolo_dataset")
    yaml_path = Path("dataset.yaml")
    
    if not dataset_path.exists():
        print("Error: yolo_dataset directory not found!")
        print("Please run convert_annotations.py first.")
        return False
    
    if not yaml_path.exists():
        print("Error: dataset.yaml not found!")
        print("Please run convert_annotations.py first.")
        return False
    
    # Check if all splits exist
    for split in ['train', 'valid', 'test']:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Error: {split} directory structure incomplete!")
            return False
        
        # Count files
        img_count = len(list(images_dir.glob('*.jpg')))
        label_count = len(list(labels_dir.glob('*.txt')))
        
        print(f"{split}: {img_count} images, {label_count} labels")
        
        if img_count == 0:
            print(f"Warning: No images found in {images_dir}")
            return False
    
    return True

def train_model():
    """Train YOLOv8 model on people detection dataset."""
    
    # Check dataset
    if not check_dataset():
        return
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load YOLOv8 model
    print("Loading YOLOv8n model...")
    # Set weights_only=False for PyTorch 2.6+ compatibility
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, 'weights_only': False})
    model = YOLO('yolov8n.pt')  # nano version for faster training
    
    # Training parameters
    training_args = {
        'data': 'dataset.yaml',
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': device,
        'project': 'runs/detect',
        'name': 'people_detection',
        'save': True,
        'save_period': 10,
        'cache': True,
        'workers': 4,
        'patience': 20,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'crop_fraction': 1.0,
        'val': True,
        'plots': True,
        'verbose': True
    }
    
    print("Starting training...")
    print(f"Training arguments: {training_args}")
    
    try:
        # Train the model
        results = model.train(**training_args)
        
        print("Training completed successfully!")
        print(f"Best model saved at: {results.save_dir}")
        
        # Validate the model
        print("Running validation...")
        val_results = model.val()
        
        print("Validation completed!")
        print(f"mAP50: {val_results.box.map50:.4f}")
        print(f"mAP50-95: {val_results.box.map:.4f}")
        
        return model, results
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        return None, None

def main():
    """Main training function."""
    print("=== YOLOv8 People Detection Training ===")
    print("This script will train a YOLOv8 model on your people detection dataset.")
    print()
    
    # Train the model
    model, results = train_model()
    
    if model is not None:
        print("\n=== Training Summary ===")
        print("✅ Training completed successfully!")
        print("📁 Results saved in: runs/detect/people_detection/")
        print("🎯 Best model: runs/detect/people_detection/weights/best.pt")
        print("📊 Training plots: runs/detect/people_detection/")
        print()
        print("Next steps:")
        print("1. Check the training plots in runs/detect/people_detection/")
        print("2. Use the trained model for inference with inference.py")
        print("3. Evaluate performance on test set")
    else:
        print("❌ Training failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
