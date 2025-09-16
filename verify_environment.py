#!/usr/bin/env python3
"""
Script to verify that the virtual environment is properly set up
and all required packages for YOLOv8 people detection are available.
"""

import sys
import importlib

def check_package(package_name, display_name=None):
    """Check if a package is available and print its version"""
    if display_name is None:
        display_name = package_name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'Unknown version')
        print(f"✅ {display_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {display_name}: Not found")
        return False

def check_gpu_support():
    """Check if GPU support is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU Support: {gpu_count} GPU(s) available - {gpu_name}")
            return True
        else:
            print("⚠️  GPU Support: CUDA not available (will use CPU)")
            return False
    except ImportError:
        print("❌ GPU Support: PyTorch not available")
        return False

def check_project_structure():
    """Check if project structure is correct"""
    import os
    from pathlib import Path
    
    required_dirs = [
        'yolo_training',
        'data',
        'ai_image_env'
    ]
    
    required_files = [
        'yolo_training/train_yolo.py',
        'yolo_training/inference.py',
        'yolo_training/convert_annotations.py',
        'yolo_training/dataset.yaml'
    ]
    
    print("\n📁 Checking project structure:")
    print("-" * 30)
    
    all_good = True
    
    # Check directories
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory: {dir_name}/")
        else:
            print(f"❌ Directory: {dir_name}/ (missing)")
            all_good = False
    
    # Check files
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✅ File: {file_name}")
        else:
            print(f"❌ File: {file_name} (missing)")
            all_good = False
    
    return all_good

def main():
    """Main verification function"""
    print("🔍 YOLOv8 PEOPLE DETECTION - ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    # Check Python path
    print(f"🐍 Python executable: {sys.executable}")
    print(f"📁 Python path: {sys.path[0]}")
    
    # Check if we're in virtual environment
    if 'ai_image_env' in sys.executable:
        print("✅ Running in virtual environment")
    else:
        print("⚠️  WARNING: Not running in virtual environment!")
        print("   Please run: ai_image_env\\Scripts\\activate")
        return
    
    print("\n📦 Checking required packages:")
    print("-" * 30)
    
    # Core ML packages for YOLOv8
    packages = [
        ('torch', 'PyTorch'),
        ('ultralytics', 'Ultralytics (YOLOv8)'),
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('tqdm', 'tqdm'),
        ('yaml', 'PyYAML'),
        ('sklearn', 'scikit-learn'),
        ('scipy', 'SciPy'),
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package, display_name in packages:
        if check_package(package, display_name):
            success_count += 1
    
    # Check GPU support
    print("\n🖥️  Checking GPU support:")
    print("-" * 30)
    gpu_available = check_gpu_support()
    
    # Check project structure
    structure_ok = check_project_structure()
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {success_count}/{total_count} packages available")
    
    if success_count == total_count:
        print("🎉 All packages are available!")
        print("✅ Virtual environment is properly set up")
        print("🚀 You can now run the YOLOv8 people detection project")
        
        if gpu_available:
            print("⚡ GPU acceleration is available for faster training")
        else:
            print("💻 Training will use CPU (slower but functional)")
        
        if structure_ok:
            print("📁 Project structure is correct")
        else:
            print("⚠️  Some project files are missing")
        
    else:
        print("❌ Some packages are missing")
        print("💡 Try running: pip install -r requirements.txt")
    
    print("\n📋 Next steps:")
    print("1. Navigate to yolo_training/ directory")
    print("2. Run: python convert_annotations.py (if not done)")
    print("3. Run: python train_yolo.py (to train model)")
    print("4. Run: python inference.py --image path/to/image.jpg (to detect people)")
    print("5. Use deactivate when done")
    
    print("\n🎯 Quick test commands:")
    print("cd yolo_training")
    print("python inference.py  # Test with default image")
    print("python train_yolo.py  # Train your model")

if __name__ == "__main__":
    main()