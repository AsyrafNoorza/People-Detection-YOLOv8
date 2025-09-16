# API Documentation

## People Detection with YOLOv8

This document provides API documentation for the People Detection with YOLOv8 project.

**Repository:** [https://github.com/AsyrafNoorza/People-Detection-YOLOv8](https://github.com/AsyrafNoorza/People-Detection-YOLOv8)

## Core Modules

### Data Processing (`src.data`)

#### `convert_annotations.py`
Converts CSV annotations to YOLO format.

**Functions:**
- `convert_csv_to_yolo(csv_path, images_dir, output_dir)`: Convert CSV annotations to YOLO format
- `setup_yolo_dataset()`: Set up complete YOLO dataset structure
- `create_dataset_yaml()`: Create dataset.yaml configuration file

### Model Training (`src.models`)

#### `train.py`
Main training script for YOLOv8 people detection.

**Functions:**
- `check_dataset()`: Validate dataset structure and files
- `train_model()`: Train YOLOv8 model on people detection dataset
- `main()`: Main training function with CLI interface

**Classes:**
- `PeopleDetectionTrainer`: Main training class

### Inference (`src.inference`)

#### `predict.py`
Inference script for people detection.

**Classes:**
- `PeopleDetector`: Main inference class
  - `__init__(model_path)`: Initialize detector with model
  - `load_model()`: Load trained YOLO model
  - `detect_people(image_path, conf_threshold, save_result, output_dir)`: Detect people in image
  - `detect_batch(image_dir, conf_threshold, output_dir)`: Batch processing
  - `evaluate_on_test_set(test_dir, conf_threshold)`: Model evaluation

### Utilities (`src.utils`)

#### `config.py`
Configuration management utilities.

**Functions:**
- `load_config(config_path)`: Load YAML configuration
- `validate_config(config)`: Validate configuration parameters
- `get_default_config()`: Get default configuration

#### `visualization.py`
Visualization utilities.

**Functions:**
- `plot_training_curves(results)`: Plot training curves
- `visualize_predictions(image, detections)`: Visualize detection results
- `create_confusion_matrix(y_true, y_pred)`: Create confusion matrix

## Configuration Files

### Training Configuration (`configs/training_config.yaml`)
Contains all training parameters:
- Model configuration
- Training hyperparameters
- Data augmentation settings
- Output configuration

### Inference Configuration (`configs/inference_config.yaml`)
Contains inference parameters:
- Model settings
- Detection thresholds
- Input/output configuration
- Visualization settings

## Command Line Interface

### Training
```bash
python -m src.models.train --config configs/training_config.yaml
```

### Inference
```bash
python -m src.inference.predict --image path/to/image.jpg --config configs/inference_config.yaml
```

### Batch Processing
```bash
python -m src.inference.predict --batch path/to/images/ --config configs/inference_config.yaml
```

## Data Formats

### Input Data
- **Images**: JPG, PNG, BMP, TIFF formats
- **Annotations**: CSV format with columns: filename, width, height, xmin, ymin, xmax, ymax

### Output Data
- **Detections**: JSON format with bounding boxes and confidence scores
- **Visualizations**: Annotated images with bounding boxes
- **Models**: PyTorch .pt format

## Error Handling

The API includes comprehensive error handling for:
- Missing files and directories
- Invalid configuration parameters
- Model loading failures
- GPU/CUDA availability issues
- Data format validation

## Examples

### Basic Training
```python
from src.models.train import train_model

# Train with default configuration
model, results = train_model()
```

### Basic Inference
```python
from src.inference.predict import PeopleDetector

# Initialize detector
detector = PeopleDetector("models/best.pt")

# Detect people in image
detections = detector.detect_people("image.jpg", conf_threshold=0.5)
```

### Custom Configuration
```python
from src.utils.config import load_config

# Load custom configuration
config = load_config("configs/custom_config.yaml")

# Use configuration in training
model, results = train_model(config=config)
```
