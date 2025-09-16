#!/usr/bin/env python3
"""
Setup script for People Detection with YOLOv8 project.
This script helps set up the project structure and environment.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def create_directory_structure():
    """Create the professional directory structure."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "notebooks",
        "src/data",
        "src/models",
        "src/inference",
        "src/utils",
        "src/visualization",
        "models",
        "reports",
        "tests",
        "configs",
        "scripts",
        "docs",
        "logs"
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}/")
    
    # Create __init__.py files for Python packages
    init_files = [
        "src/__init__.py",
        "src/data/__init__.py", 
        "src/models/__init__.py",
        "src/inference/__init__.py",
        "src/utils/__init__.py",
        "src/visualization/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ Created: {init_file}")


def setup_git():
    """Initialize Git repository if not already done."""
    if not Path(".git").exists():
        print("Initializing Git repository...")
        subprocess.run(["git", "init"], check=True)
        print("✅ Git repository initialized")
    else:
        print("✅ Git repository already exists")


def install_dependencies():
    """Install project dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    return True


def setup_pre_commit():
    """Set up pre-commit hooks if available."""
    try:
        subprocess.run(["pre-commit", "install"], check=True, capture_output=True)
        print("✅ Pre-commit hooks installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Pre-commit not available (install with: pip install pre-commit)")


def create_sample_notebook():
    """Create a sample Jupyter notebook."""
    notebook_content = """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# People Detection with YOLOv8 - Data Exploration\\n",
    "\\n",
    "This notebook explores the people detection dataset and provides initial analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "from pathlib import Path\\n",
    "\\n",
    "# Set up plotting style\\n",
    "plt.style.use('seaborn-v0_8')\\n",
    "sns.set_palette('husl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load and explore dataset\\n",
    "data_path = Path('../data/raw')\\n",
    "print(f'Dataset path: {data_path}')\\n",
    "print(f'Dataset exists: {data_path.exists()}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}"""
    
    notebook_path = Path("notebooks/01_data_exploration.ipynb")
    with open(notebook_path, 'w') as f:
        f.write(notebook_content)
    print(f"✅ Created sample notebook: {notebook_path}")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup People Detection YOLOv8 project")
    parser.add_argument("--skip-deps", action="store_true", 
                       help="Skip dependency installation")
    parser.add_argument("--skip-git", action="store_true",
                       help="Skip Git setup")
    parser.add_argument("--skip-notebook", action="store_true",
                       help="Skip sample notebook creation")
    
    args = parser.parse_args()
    
    print("🚀 Setting up People Detection with YOLOv8 project...")
    print("=" * 60)
    
    # Create directory structure
    create_directory_structure()
    print()
    
    # Setup Git
    if not args.skip_git:
        setup_git()
        print()
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            print("❌ Setup incomplete due to dependency installation failure")
            return 1
        print()
    
    # Setup pre-commit
    setup_pre_commit()
    print()
    
    # Create sample notebook
    if not args.skip_notebook:
        create_sample_notebook()
        print()
    
    print("=" * 60)
    print("🎉 Project setup complete!")
    print()
    print("Next steps:")
    print("1. Activate your virtual environment")
    print("2. Move your existing data to data/raw/")
    print("3. Move your existing code to src/")
    print("4. Update configuration files in configs/")
    print("5. Run tests: pytest tests/")
    print("6. Start development!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
