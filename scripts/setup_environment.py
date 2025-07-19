#!/usr/bin/env python3
"""
Environment setup script for clickbait detection project
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8+ is required")
        return False
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            logger.warning("CUDA not available - will use CPU")
            return False
    except ImportError:
        logger.warning("PyTorch not installed yet")
        return False

def install_requirements():
    """Install required packages"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        logger.error("requirements.txt not found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "results/experiments",
        "results/models", 
        "results/logs",
        "data/processed",
        "docs",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def check_data():
    """Check if data files exist"""
    data_files = [
        "data/simple_dataset/train/train.json",
        "data/simple_dataset/val/val.json", 
        "data/simple_dataset/test/test.json"
    ]
    
    missing_files = []
    for file_path in data_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning("Missing data files:")
        for file_path in missing_files:
            logger.warning(f"  - {file_path}")
        return False
    else:
        logger.info("All data files found")
        return True

def setup_huggingface():
    """Setup Hugging Face authentication"""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        logger.info(f"Hugging Face authenticated as: {user['name']}")
        return True
    except Exception as e:
        logger.warning(f"Hugging Face not authenticated: {e}")
        logger.info("Run 'huggingface-cli login' to authenticate")
        return False

def main():
    """Main setup function"""
    logger.info("=== Setting up Clickbait Detection Environment ===")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        logger.error("Failed to install requirements")
        sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    # Check data
    check_data()
    
    # Setup Hugging Face
    setup_huggingface()
    
    logger.info("=== Setup completed successfully! ===")
    logger.info("Next steps:")
    logger.info("1. Run 'python scripts/run_experiment.py' to test the setup")
    logger.info("2. Run experiments with scripts in the scripts/ directory")

if __name__ == "__main__":
    main()
