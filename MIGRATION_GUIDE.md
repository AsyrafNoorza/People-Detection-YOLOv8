# Migration Guide: From Current to Professional Structure

This guide helps you migrate your current project structure to a professional ML project layout.

## 🎯 **Current vs Professional Structure**

### **Current Structure:**
```
People-Detection-YOLOv8/
├── ai_image_env/              # Virtual environment
├── data/                      # Raw dataset
├── yolo_training/             # All code mixed together
├── requirements.txt           # Dependencies
├── verify_environment.py     # Environment check
├── start_project.bat         # Windows script
├── ENVIRONMENT_SETUP.md      # Setup guide
└── README.md                 # Documentation
```

### **Professional Structure:**
```
people-detection-yolov8/
├── data/
│   ├── raw/                  # Original, immutable data
│   ├── processed/            # Cleaned/processed data
│   └── external/             # Third-party data
├── notebooks/                # Jupyter notebooks for exploration
├── src/
│   ├── data/                 # Data processing scripts
│   ├── models/               # Model training scripts
│   ├── inference/            # Inference scripts
│   ├── utils/                # Utility functions
│   └── visualization/        # Plotting scripts
├── models/                   # Saved models and checkpoints
├── reports/                  # Generated reports and figures
├── tests/                    # Unit tests
├── configs/                  # Configuration files
├── scripts/                  # Utility scripts
├── docs/                     # Additional documentation
├── .gitignore               # Git ignore file
├── requirements.txt         # Dependencies
├── setup.py                 # Package setup
├── pyproject.toml           # Modern Python packaging
└── README.md                # Main documentation
```

## 📋 **Migration Steps**

### **Step 1: Create Professional Structure**
```bash
# Run the setup script to create the new structure
python scripts/setup_project.py
```

### **Step 2: Move Your Data**
```bash
# Move your current data to the new structure
mv data/* data/raw/
```

### **Step 3: Reorganize Your Code**
```bash
# Move training code
mv yolo_training/train_yolo.py src/models/train.py
mv yolo_training/convert_annotations.py src/data/convert_annotations.py
mv yolo_training/inference.py src/inference/predict.py
mv yolo_training/dataset.yaml configs/dataset.yaml

# Move processed dataset
mv yolo_training/yolo_dataset data/processed/
```

### **Step 4: Update Import Statements**
Update all Python files to use the new import structure:

**Before:**
```python
# In yolo_training/train_yolo.py
from ultralytics import YOLO
```

**After:**
```python
# In src/models/train.py
from ultralytics import YOLO
from ..utils.config import load_config
```

### **Step 5: Update Configuration**
Use the new configuration files:
- `configs/training_config.yaml` - Training parameters
- `configs/inference_config.yaml` - Inference parameters

### **Step 6: Update Documentation**
- Update README.md with new structure
- Update all file paths in documentation
- Update command examples

## 🔧 **Code Migration Examples**

### **Training Script Migration**

**Before (`yolo_training/train_yolo.py`):**
```python
def train_model():
    # Hardcoded parameters
    training_args = {
        'data': 'dataset.yaml',
        'epochs': 100,
        'batch': 16,
        # ... more hardcoded values
    }
```

**After (`src/models/train.py`):**
```python
from ..utils.config import load_config

def train_model(config_path="configs/training_config.yaml"):
    # Load configuration
    config = load_config(config_path)
    
    # Use configuration parameters
    training_args = {
        'data': config['dataset']['path'],
        'epochs': config['training']['epochs'],
        'batch': config['training']['batch_size'],
        # ... from config
    }
```

### **Inference Script Migration**

**Before (`yolo_training/inference.py`):**
```python
class PeopleDetector:
    def __init__(self, model_path="runs/detect/people_detection/weights/best.pt"):
        self.model_path = model_path
```

**After (`src/inference/predict.py`):**
```python
from ..utils.config import load_config

class PeopleDetector:
    def __init__(self, model_path=None, config_path="configs/inference_config.yaml"):
        config = load_config(config_path)
        self.model_path = model_path or config['model']['path']
```

## 🧪 **Testing Migration**

### **Run Tests**
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### **Test Your Migration**
```bash
# Test data processing
python -m src.data.convert_annotations

# Test training
python -m src.models.train --config configs/training_config.yaml

# Test inference
python -m src.inference.predict --image data/raw/test.jpg
```

## 📦 **Package Installation**

### **Install in Development Mode**
```bash
# Install the package in development mode
pip install -e .

# This allows you to import your modules from anywhere
python -c "from src.models.train import train_model"
```

### **Install with Extras**
```bash
# Install with development dependencies
pip install -e .[dev]

# Install with notebook dependencies
pip install -e .[notebooks]
```

## 🔄 **Gradual Migration Strategy**

If you want to migrate gradually without breaking your current setup:

### **Phase 1: Add Professional Files**
1. Add `.gitignore`, `setup.py`, `pyproject.toml`
2. Add `configs/` directory with configuration files
3. Add `tests/` directory with test files

### **Phase 2: Create New Structure**
1. Create `src/` directory structure
2. Copy (don't move) your existing code to `src/`
3. Update imports in the new files

### **Phase 3: Test New Structure**
1. Test the new structure works
2. Update documentation
3. Gradually switch to using the new structure

### **Phase 4: Clean Up**
1. Remove old files once new structure is working
2. Update all references
3. Final testing

## ⚠️ **Common Migration Issues**

### **Import Errors**
**Problem:** `ModuleNotFoundError` after moving files
**Solution:** Update all import statements and ensure `__init__.py` files exist

### **Path Issues**
**Problem:** File paths no longer work
**Solution:** Use relative paths from project root or absolute paths

### **Configuration Issues**
**Problem:** Hardcoded parameters no longer work
**Solution:** Move all parameters to configuration files

### **Testing Issues**
**Problem:** Tests fail after migration
**Solution:** Update test imports and file paths

## 🎉 **Benefits After Migration**

1. **Professional Structure** - Industry-standard organization
2. **Better Maintainability** - Clear separation of concerns
3. **Easier Collaboration** - Standard structure for team members
4. **Better Testing** - Comprehensive test coverage
5. **Configuration Management** - Centralized configuration
6. **Package Installation** - Install as proper Python package
7. **Documentation** - Professional API documentation
8. **Version Control** - Proper Git setup with .gitignore

## 📚 **Additional Resources**

- [Python Packaging User Guide](https://packaging.python.org/)
- [Pytest Documentation](https://docs.pytest.org/)
- [YAML Configuration](https://yaml.org/)
- [Git Best Practices](https://git-scm.com/doc)
- [Project Repository](https://github.com/AsyrafNoorza/People-Detection-YOLOv8)

---

**Ready to make your project professional!** 🚀
