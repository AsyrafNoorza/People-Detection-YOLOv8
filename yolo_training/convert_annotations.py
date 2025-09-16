"""
Convert CSV annotations to YOLO format for people detection.
This script converts the Roboflow CSV format to YOLO txt format.
"""

import pandas as pd
import os
from pathlib import Path
import shutil

def convert_csv_to_yolo(csv_path, images_dir, output_dir):
    """
    Convert CSV annotations to YOLO format.
    
    Args:
        csv_path: Path to the CSV annotation file
        images_dir: Directory containing the images
        output_dir: Directory to save YOLO format annotations
    """
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by filename to handle multiple objects per image
    grouped = df.groupby('filename')
    
    for filename, group in grouped:
        # Get image dimensions (should be same for all annotations of same image)
        width = group.iloc[0]['width']
        height = group.iloc[0]['height']
        
        # Create YOLO annotation file
        yolo_filename = filename.replace('.jpg', '.txt')
        yolo_path = os.path.join(output_dir, yolo_filename)
        
        with open(yolo_path, 'w') as f:
            for _, row in group.iterrows():
                # Convert bounding box coordinates to YOLO format
                # YOLO format: class_id center_x center_y width height (all normalized)
                
                # Calculate center coordinates and dimensions
                center_x = (row['xmin'] + row['xmax']) / 2.0
                center_y = (row['ymin'] + row['ymax']) / 2.0
                bbox_width = row['xmax'] - row['xmin']
                bbox_height = row['ymax'] - row['ymin']
                
                # Normalize coordinates
                center_x_norm = center_x / width
                center_y_norm = center_y / height
                width_norm = bbox_width / width
                height_norm = bbox_height / height
                
                # Class ID for person (0 for single class)
                class_id = 0
                
                # Write YOLO format line
                f.write(f"{class_id} {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        print(f"Converted: {filename} -> {yolo_filename}")

def setup_yolo_dataset():
    """
    Set up the complete YOLO dataset structure.
    """
    # Define paths
    data_root = Path("data")
    yolo_data_root = Path("yolo_dataset")
    
    # Create YOLO dataset structure
    for split in ['train', 'valid', 'test']:
        # Create directories
        images_dir = yolo_data_root / split / 'images'
        labels_dir = yolo_data_root / split / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert annotations
        csv_path = data_root / split / '_annotations.csv'
        if csv_path.exists():
            print(f"Converting {split} annotations...")
            convert_csv_to_yolo(csv_path, data_root / split, labels_dir)
            
            # Copy images to YOLO structure
            source_images_dir = data_root / split
            for img_file in source_images_dir.glob('*.jpg'):
                dest_path = images_dir / img_file.name
                shutil.copy2(img_file, dest_path)
            
            print(f"Copied {len(list(images_dir.glob('*.jpg')))} images to {images_dir}")
        else:
            print(f"Warning: {csv_path} not found")

def create_dataset_yaml():
    """
    Create dataset.yaml file for YOLO training.
    """
    yaml_content = """# People Detection Dataset Configuration
path: yolo_dataset  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images    # val images (relative to 'path')
test: test/images    # test images (relative to 'path')

# Classes
nc: 1  # number of classes
names: ['person']  # class names

# Additional info
task: detect
"""
    
    with open('dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("Created dataset.yaml configuration file")

if __name__ == "__main__":
    print("Setting up YOLO dataset structure...")
    setup_yolo_dataset()
    create_dataset_yaml()
    print("Dataset conversion completed!")
    print("\nDataset structure:")
    print("yolo_dataset/")
    print("├── train/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("├── valid/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("└── test/")
    print("    ├── images/")
    print("    └── labels/")
