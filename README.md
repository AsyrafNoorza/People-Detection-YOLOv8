# People Detection with YOLOv8

A complete YOLOv8 implementation for detecting people in images using a curated dataset from Roboflow. This project provides a robust, production-ready people detection system with training, inference, and evaluation capabilities.

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)](https://github.com/AsyrafNoorza/People-Detection-YOLOv8)

## 🚨 IMPORTANT: Virtual Environment Required

**This project REQUIRES the virtual environment to be activated before running any code.**

### Quick Start:
```bash
# Activate environment
ai_image_env\Scripts\activate

# Verify setup
python verify_environment.py

# Run YOLO training
cd yolo_training
python train_yolo.py

# Run inference on images
python inference.py --image path/to/image.jpg
```

## 📊 Project Status

- ✅ **YOLO Training**: Fully functional and tested
- ✅ **Inference System**: Complete with batch processing
- ✅ **Environment Setup**: Complete with virtual environment isolation
- ✅ **Dataset**: Ready (17,401 images with annotations)
- ✅ **Documentation**: Comprehensive guides and examples

## 🎯 YOLO People Detection Features

**Best for:** Object detection with bounding boxes, real-time applications, surveillance systems

**Key Features:**
- Detects and localizes people with precise bounding boxes
- Can detect multiple people in a single image
- Fast inference speed (real-time capable)
- Uses YOLOv8 architecture (nano, small, medium, large variants)
- Provides confidence scores and coordinates
- Batch processing capabilities
- Comprehensive evaluation metrics

## 📁 Project Structure

```
People-Detection-YOLOv8/
├── ai_image_env/              # Virtual environment (REQUIRED)
├── data/                      # Original dataset (shared)
│   ├── train/                 # 15,210 training images
│   ├── valid/                 # 1,431 validation images
│   └── test/                  # 760 test images
├── yolo_training/             # YOLO implementation (COMPLETE)
│   ├── train_yolo.py         # YOLOv8 training script
│   ├── inference.py          # YOLO inference script
│   ├── convert_annotations.py # CSV to YOLO format converter
│   ├── dataset.yaml          # YOLO dataset configuration
│   ├── yolo_dataset/         # YOLO format dataset
│   │   ├── train/            # Training images and labels
│   │   ├── valid/            # Validation images and labels
│   │   └── test/             # Test images and labels
│   ├── results/              # Detection output images
│   └── yolov8n.pt           # Pre-trained YOLOv8 nano model
├── requirements.txt           # Python dependencies
├── verify_environment.py     # Environment verification tool
├── start_project.bat         # Easy environment activation
├── ENVIRONMENT_SETUP.md      # Complete setup guide
└── README.md                 # This file
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.13.7 (or compatible)
- Windows PowerShell or Command Prompt
- Internet connection
- CUDA-compatible GPU (optional, for faster training)

### Step 1: Create Virtual Environment
```bash
python -m venv ai_image_env
```

### Step 2: Activate Environment
**PowerShell:**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\ai_image_env\Scripts\Activate.ps1
```

**Command Prompt:**
```bash
ai_image_env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python verify_environment.py
```

## 📋 Dataset Information

- **Total Images**: 17,401
- **Training Set**: 15,210 images
- **Validation Set**: 1,431 images
- **Test Set**: 760 images
- **Classes**: 1 (person)
- **Format**: CSV annotations with bounding box coordinates
- **Source**: Roboflow People Detection Dataset

## 🎮 Usage Examples

### Training Your Own Model
```bash
# Activate environment
ai_image_env\Scripts\activate

# Navigate to YOLO training
cd yolo_training

# Convert dataset to YOLO format (if not already done)
python convert_annotations.py

# Train the model
python train_yolo.py
```

### Running Inference

#### Single Image Detection
```bash
# Detect people in a single image
python inference.py --image "path/to/image.jpg" --conf 0.5

# Use default test image
python inference.py

# Detect people in sample image from project
python inference.py --image "../HAUM kiiikiii.jpg"
```

#### Batch Processing
```bash
# Process multiple images
python inference.py --batch "path/to/image/directory" --conf 0.5

# Process test set
python inference.py --batch "../data/test" --conf 0.5
```

#### Model Evaluation
```bash
# Evaluate on test set
python inference.py --evaluate

# Use custom trained model
python inference.py --model "runs/detect/people_detection/weights/best.pt" --image "image.jpg"
```

## ⚙️ Training Configuration

The training script (`train_yolo.py`) uses these optimized parameters:
- **Epochs**: 100
- **Batch Size**: 16
- **Image Size**: 640x640
- **Model**: YOLOv8n (nano) - fast and efficient
- **Device**: Auto-detect (CUDA if available, CPU fallback)
- **Data Augmentation**: Comprehensive augmentation pipeline
- **Optimization**: AdamW optimizer with learning rate scheduling

## 📈 Model Performance

### YOLO Model Metrics
- **mAP50**: Mean Average Precision at IoU 0.5
- **mAP50-95**: Mean Average Precision across IoU 0.5-0.95
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Expected Performance
- **Inference Speed**: 30-60 FPS (depending on hardware)
- **Accuracy**: 85-95% mAP50 on people detection
- **Model Size**: ~6MB (YOLOv8n)
- **Memory Usage**: ~1-2GB during training

### Model Outputs

After training, models are saved in:
- `runs/detect/people_detection/weights/best.pt` - Best model
- `runs/detect/people_detection/weights/last.pt` - Last epoch

## 🔧 Environment Management

### Daily Workflow
```bash
# 1. Navigate to project
cd C:\Users\Acap\Documents\People-Detection-YOLOv8

# 2. Activate environment
ai_image_env\Scripts\activate

# 3. Work on project
cd yolo_training
python train_yolo.py

# 4. Deactivate when done
deactivate
```

### Using Helper Scripts
```bash
# Easy activation
start_project.bat

# Verify environment
python verify_environment.py
```

## 📦 Dependencies

### Core ML Libraries
- **PyTorch 2.8.0** - Deep learning framework
- **Ultralytics 8.0.196** - YOLOv8 implementation
- **OpenCV 4.12.0** - Computer vision library

### Computer Vision & Processing
- **Pillow 11.3.0** - Image processing
- **NumPy 2.2.6** - Numerical computing
- **Pandas 2.3.2** - Data manipulation
- **Matplotlib 3.10.6** - Plotting and visualization

### Utilities
- **tqdm 4.67.1** - Progress bars
- **PyYAML 6.0.2** - YAML parsing
- **Seaborn 0.13.2** - Statistical visualization

## 🎨 Detection Results

The inference script provides:
- **Bounding boxes** around detected people
- **Confidence scores** for each detection
- **Annotated images** saved to `results/` folder
- **Console output** with detection details

## 🎯 Use Cases

### Real-time Applications
- **Surveillance Systems**: Monitor areas for people presence
- **Security Cameras**: Detect intruders or unauthorized access
- **Smart Buildings**: Count people for occupancy management
- **Retail Analytics**: Track customer movement and behavior

### Batch Processing
- **Image Analysis**: Process large datasets of images
- **Data Mining**: Extract people information from image collections
- **Research**: Analyze human presence in various environments
- **Content Moderation**: Detect people in user-uploaded content

## 🐛 Troubleshooting

### Common Issues

1. **"ModuleNotFoundError"**
   - Solution: Activate virtual environment first
   ```bash
   ai_image_env\Scripts\activate
   ```

2. **"Model not found"**
   - Solution: Train the model first or check file paths
   ```bash
   cd yolo_training
   python train_yolo.py
   ```

3. **"CUDA Out of Memory"**
   - Solution: Reduce batch size in training script
   - Edit `train_yolo.py` and change `'batch': 16` to `'batch': 8`

4. **"Dataset not found"**
   - Solution: Ensure `data/` directory exists with proper structure
   - Run `python convert_annotations.py` to set up YOLO dataset

### Performance Tips

1. **For Faster Training**:
   - Use YOLOv8n (nano) model
   - Reduce image size to 416
   - Use fewer epochs for initial testing
   - Enable GPU acceleration

2. **For Better Accuracy**:
   - Use YOLOv8m or YOLOv8l models
   - Increase training epochs to 200+
   - Use larger image size (832)
   - Enable data augmentation

3. **For Production Deployment**:
   - Use YOLOv8n for speed
   - Optimize with TensorRT (NVIDIA GPUs)
   - Use ONNX format for cross-platform deployment

## 📊 Example Output

### Detection Results
```
Found 2 people
Person 1: confidence=0.856, bbox=[245, 123, 456, 789]
Person 2: confidence=0.743, bbox=[567, 234, 678, 890]
Result saved to: results/image_detected.jpg
```

### Training Progress
```
Epoch    GPU_mem   box_loss   obj_loss   cls_loss   Instances   Size
  1/100     1.2G     0.1234     0.0567     0.0123        1234    640
  2/100     1.2G     0.1156     0.0543     0.0118        1234    640
  ...
```

## 🚨 Important Notes

- **Virtual Environment Required:** Always activate `ai_image_env` before running scripts
- **PyTorch Compatibility:** Scripts include fixes for PyTorch 2.6+ compatibility
- **GPU Support:** Automatically uses CUDA if available, falls back to CPU
- **Model Weights:** Uses pre-trained YOLOv8n model by default

## 🎯 Performance

The YOLOv8n model provides:
- **Fast inference** (real-time capable)
- **High accuracy** for person detection
- **Multiple object detection** in single images
- **Robust performance** across different lighting and poses

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Project Repository](https://github.com/AsyrafNoorza/People-Detection-YOLOv8)

## 🚀 Next Steps

1. **Train your model**: Run `python train_yolo.py` to create a custom model
2. **Test inference**: Use `python inference.py` to detect people in your images
3. **Evaluate performance**: Run evaluation on the test set
4. **Deploy**: Use the trained model in your applications

## 📝 Notes

- **Virtual environment is REQUIRED** - Project will not work without it
- **First run takes longer** - PyTorch needs to initialize and download models
- **GPU acceleration** - Automatically used if CUDA is available
- **Model weights** - Pre-trained YOLOv8n model included for immediate use
- **Test images** - Sample images available in the `data/test/` folder

---

**Ready to detect people in your images with state-of-the-art YOLOv8!** 🎉