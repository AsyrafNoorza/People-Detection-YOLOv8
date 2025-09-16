# 🐍 Virtual Environment Setup Guide

## ⚠️ IMPORTANT: This Project REQUIRES Virtual Environment

**All code in this project MUST be run within the virtual environment.** The project uses PyTorch and YOLOv8 which require specific package versions for optimal performance.

## 🚀 Quick Start

### 1. Activate Virtual Environment
**For PowerShell:**
```bash
# First, allow script execution (one-time setup)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate the environment
.\ai_image_env\Scripts\Activate.ps1
```

**For Command Prompt:**
```bash
# Windows
ai_image_env\Scripts\activate

# You should see (ai_image_env) in your prompt
```

### 2. Verify Environment
```bash
# Check that packages are available
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

### 3. Run Project Code
```bash
# YOLO Training and Inference
cd yolo_training
python train_yolo.py
python inference.py --image "path/to/image.jpg"
```

### 4. Deactivate When Done
```bash
deactivate
```

## 📋 Environment Contents

The virtual environment contains all required packages for YOLOv8 people detection:

### Core ML Libraries
- **PyTorch 2.8.0** - Deep learning framework (YOLOv8 backend)
- **Ultralytics 8.0.196** - YOLOv8 implementation and training
- **OpenCV 4.12.0** - Computer vision library for image processing

### Computer Vision & Processing
- **Pillow 11.3.0** - Image processing and manipulation
- **NumPy 2.2.6** - Numerical computing foundation
- **Pandas 2.3.2** - Data manipulation and CSV processing
- **Matplotlib 3.10.6** - Plotting and visualization
- **Seaborn 0.13.2** - Statistical visualization

### Utilities
- **tqdm 4.67.1** - Progress bars for training
- **PyYAML 6.0.2** - YAML parsing for configuration
- **scikit-learn 1.7.2** - Machine learning utilities
- **scipy 1.16.2** - Scientific computing

### GPU Support (Optional)
- **nvidia-cublas-cu12** - CUDA BLAS library
- **nvidia-cuda-nvrtc-cu12** - CUDA runtime
- **nvidia-cuda-runtime-cu12** - CUDA runtime
- **nvidia-cudnn-cu12** - CUDA Deep Neural Network library

## 🔧 Environment Management

### Install New Packages
```bash
# Activate environment first
ai_image_env\Scripts\activate

# Install package
pip install package_name

# Update requirements.txt
pip freeze > requirements.txt
```

### Recreate Environment
```bash
# Create new environment
python -m venv ai_image_env

# Activate
ai_image_env\Scripts\activate

# Install all packages
pip install -r requirements.txt
```

### Check Environment Status
```bash
# See installed packages
pip list

# Check specific package
pip show package_name

# Verify Python path
python -c "import sys; print(sys.executable)"

# Check GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## 🚨 Common Issues & Solutions

### Issue: "ModuleNotFoundError"
**Solution:** You're not in the virtual environment
```bash
# Activate environment
ai_image_env\Scripts\activate
# You should see (ai_image_env) in prompt
```

### Issue: "Command not found"
**Solution:** Use full path to activate
```bash
# Windows
.\ai_image_env\Scripts\activate

# Or use full path
C:\Users\Acap\Documents\People-Detection-YOLOv8\ai_image_env\Scripts\activate
```

### Issue: "Package not found"
**Solution:** Install in virtual environment
```bash
# Activate first
ai_image_env\Scripts\activate

# Then install
pip install package_name
```

### Issue: "CUDA not available"
**Solution:** Check GPU setup
```bash
# Check if CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# If False, training will use CPU (slower but functional)
```

### Issue: "Out of memory during training"
**Solution:** Reduce batch size
```bash
# Edit yolo_training/train_yolo.py
# Change 'batch': 16 to 'batch': 8 or 'batch': 4
```

## 📁 Project Structure with Environment

```
People-Detection-YOLOv8/
├── ai_image_env/              # Virtual environment (DO NOT DELETE)
│   ├── Scripts/
│   │   ├── activate          # Activate script
│   │   └── python.exe        # Environment Python
│   └── Lib/site-packages/    # All project packages
├── yolo_training/            # YOLO implementation
│   ├── train_yolo.py        # Training script
│   ├── inference.py         # Inference script
│   └── yolo_dataset/        # YOLO format dataset
├── data/                     # Original dataset
├── requirements.txt          # Package list
└── ENVIRONMENT_SETUP.md      # This file
```

## 🎯 Best Practices

1. **Always activate environment before coding**
2. **Never install packages globally for this project**
3. **Keep requirements.txt updated**
4. **Use environment-specific Python interpreter in IDE**
5. **Deactivate when switching to other projects**
6. **Check GPU availability before training**
7. **Monitor memory usage during training**

## 🔄 Daily Workflow

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

## 🚀 Performance Optimization

### GPU Acceleration
```bash
# Check GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# If available, training will automatically use GPU
# If not, training will use CPU (slower but functional)
```

### Memory Management
```bash
# For systems with limited RAM/VRAM:
# 1. Reduce batch size in train_yolo.py
# 2. Use smaller image size (416 instead of 640)
# 3. Use YOLOv8n (nano) model instead of larger variants
```

### Training Speed
```bash
# For faster training:
# 1. Use GPU if available
# 2. Increase batch size (if memory allows)
# 3. Use YOLOv8n model
# 4. Reduce image size
# 5. Use fewer epochs for initial testing
```

## 📞 Support

If you encounter issues:

1. **Check that virtual environment is activated**
2. **Verify packages are installed**: `pip list`
3. **Check GPU availability**: `python -c "import torch; print(torch.cuda.is_available())"`
4. **Recreate environment if needed**
5. **Check this guide for solutions**

## 🔍 Environment Verification

Run the verification script to check everything:
```bash
python verify_environment.py
```

This will check:
- ✅ Virtual environment activation
- ✅ All required packages
- ✅ Package versions
- ✅ GPU availability
- ✅ Project structure

**Remember: This project will NOT work without the virtual environment!**

---

**Your YOLOv8 people detection environment is ready to go!** 🎉