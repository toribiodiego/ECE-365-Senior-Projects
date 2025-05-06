#!/usr/bin/env python3
"""
Simple test script to verify that all necessary components of the application are working.
Run this before pushing to Git to ensure everything is functioning correctly.
"""

import os
import sys
import importlib
import traceback

def test_imports():
    """Test that all necessary modules can be imported."""
    required_modules = [
        "numpy", "torch", "torchvision", "cv2", "PIL", 
        "gradio", "matplotlib", "sklearn.preprocessing"
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            missing_modules.append(module)
            print(f"❌ {module} import failed: {e}")
    
    if missing_modules:
        print(f"\n⚠️ Missing modules: {', '.join(missing_modules)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def test_internal_modules():
    """Test that our internal modules can be imported."""
    internal_modules = [
        "find_similar_enhanced",
        "face_align"
    ]
    
    missing_modules = []
    
    for module in internal_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            missing_modules.append(module)
            print(f"❌ {module} import failed: {e}")
    
    if missing_modules:
        print(f"\n⚠️ Missing internal modules: {', '.join(missing_modules)}")
        print("Please ensure all project files are in the correct locations.")
        return False
    
    return True

def test_directory_structure():
    """Test that required directories exist."""
    required_dirs = [
        "embeddings_cache",
        "aligned_faces",
        "models"
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if not os.path.isdir(directory):
            missing_dirs.append(directory)
            print(f"❌ {directory} directory not found")
        else:
            print(f"✅ {directory} directory exists")
    
    if missing_dirs:
        print(f"\n⚠️ Missing directories: {', '.join(missing_dirs)}")
        print("Please create these directories.")
        try:
            for directory in missing_dirs:
                os.makedirs(directory, exist_ok=True)
                print(f"Created directory: {directory}")
        except Exception as e:
            print(f"Error creating directories: {e}")
        return False
    
    return True

def test_torch_cuda():
    """Test if CUDA is available for PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA is not available. The application will run on CPU only.")
        return True
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def run_tests():
    """Run all tests and return overall result."""
    print("=" * 50)
    print("Running application tests...")
    print("=" * 50)
    
    passed = True
    
    print("\n--- Testing imports ---")
    if not test_imports():
        passed = False
    
    print("\n--- Testing internal modules ---")
    if not test_internal_modules():
        passed = False
    
    print("\n--- Testing directory structure ---")
    if not test_directory_structure():
        passed = False
    
    print("\n--- Testing PyTorch CUDA ---")
    if not test_torch_cuda():
        passed = False
    
    print("\n" + "=" * 50)
    if passed:
        print("✅ All tests passed! The application should work correctly.")
    else:
        print("⚠️ Some tests failed. Please fix the issues before pushing to Git.")
    print("=" * 50)
    
    return passed

if __name__ == "__main__":
    try:
        success = run_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print("❌ Error during testing:")
        traceback.print_exc()
        sys.exit(1) 