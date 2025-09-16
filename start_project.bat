@echo off
echo ========================================
echo   YOLOv8 People Detection Project
echo ========================================
echo.
echo Activating virtual environment...
call ai_image_env\Scripts\activate
echo.
echo ✅ Virtual environment activated!
echo You should see (ai_image_env) in your prompt.
echo.
echo 📋 Available commands:
echo   cd yolo_training      - Go to YOLO training directory
echo   python verify_environment.py - Check environment setup
echo   deactivate           - Exit virtual environment
echo.
echo 🚀 YOLOv8 People Detection Commands:
echo   cd yolo_training
echo   python train_yolo.py          - Train YOLOv8 model
echo   python inference.py           - Run inference on images
echo   python convert_annotations.py - Convert dataset to YOLO format
echo.
echo 🎯 Quick Start:
echo   1. cd yolo_training
echo   2. python inference.py --image "path/to/image.jpg"
echo   3. python train_yolo.py (to train your own model)
echo.
echo Ready to detect people with YOLOv8! 🎉
echo.