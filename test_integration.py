#!/usr/bin/env python3
"""
HunyuanWorld Integration Test Script
Tests the real model integration vs placeholder behavior
"""

import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_integration():
    """Test HunyuanWorld integration"""
    print("🔍 Testing HunyuanWorld Integration")
    print("=" * 50)
    
    # Test 1: Import integration module
    print("\n1. Testing integration module import...")
    try:
        from core.hunyuan_integration import HUNYUAN_AVAILABLE, get_hunyuan_model_class
        print(f"✅ Integration module imported successfully")
        print(f"HunyuanWorld Available: {'✅ Yes' if HUNYUAN_AVAILABLE else '❌ No'}")
    except ImportError as e:
        print(f"❌ Failed to import integration module: {e}")
        return False
    
    # Test 2: Test model manager integration
    print("\n2. Testing model manager integration...")
    try:
        from core.model_manager import ModelManager
        manager = ModelManager()
        print(f"✅ Model manager created successfully")
        print(f"Device: {manager.device}")
        print(f"Precision: {manager.precision}")
    except Exception as e:
        print(f"❌ Failed to create model manager: {e}")
        return False
    
    # Test 3: Test model loading (will use placeholder if HunyuanWorld not available)
    print("\n3. Testing model loading...")
    try:
        # Test text-to-panorama model loading
        model = manager.load_model(
            model_path="models/hunyuan_world",
            model_type="text_to_panorama",
            precision="fp16"
        )
        print(f"✅ Text-to-panorama model loaded: {type(model.model).__name__}")
        
        # Test image-to-panorama model loading  
        model2 = manager.load_model(
            model_path="models/hunyuan_world",
            model_type="image_to_panorama", 
            precision="fp16"
        )
        print(f"✅ Image-to-panorama model loaded: {type(model2.model).__name__}")
        
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return False
    
    # Test 4: Test inference (will be placeholder if HunyuanWorld not available)
    print("\n4. Testing inference...")
    try:
        import torch
        
        # Test text-to-panorama generation
        result = model.model.generate_panorama(
            prompt="Test mountain landscape",
            width=1920,
            height=960,
            num_inference_steps=5  # Low steps for testing
        )
        print(f"✅ Text-to-panorama generated: {result.shape}")
        
        # Test image-to-panorama generation
        test_image = torch.randn(512, 512, 3)  # Dummy image
        result2 = model2.model.generate_panorama(
            image=test_image,
            width=1920,
            height=960
        )
        print(f"✅ Image-to-panorama generated: {result2.shape}")
        
    except Exception as e:
        print(f"❌ Failed inference test: {e}")
        return False
    
    # Test 5: Memory usage check
    print("\n5. Checking memory usage...")
    try:
        memory_info = manager.get_memory_usage()
        print(f"✅ Memory info retrieved:")
        print(f"   Device: {memory_info['device']}")
        print(f"   Loaded models: {memory_info['loaded_models']}")
        print(f"   Model types: {memory_info['model_types']}")
    except Exception as e:
        print(f"❌ Failed to get memory info: {e}")
        return False
    
    print("\n" + "=" * 50)
    if HUNYUAN_AVAILABLE:
        print("🎉 INTEGRATION TEST PASSED - Real HunyuanWorld inference available!")
    else:
        print("⚠️ FRAMEWORK TEST PASSED - Using placeholder inference")
        print("   To enable real inference:")
        print("   1. Follow INTEGRATION_GUIDE.md")
        print("   2. Clone HunyuanWorld-1.0 repository")
        print("   3. Install dependencies: pip install -r requirements_hunyuan.txt")
    
    return True

def test_dependencies():
    """Test if all required dependencies are available"""
    print("\n🔍 Testing Dependencies")
    print("=" * 30)
    
    required_packages = [
        'torch',
        'torchvision', 
        'numpy',
        'PIL',
        'cv2'
    ]
    
    optional_packages = [
        'diffusers',
        'transformers',
        'accelerate',
        'safetensors'
    ]
    
    print("\nRequired packages:")
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - REQUIRED")
    
    print("\nOptional packages (for real inference):")
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"⚠️ {package} - optional")

def test_directory_structure():
    """Test if directory structure is correct"""
    print("\n🔍 Testing Directory Structure")
    print("=" * 35)
    
    required_files = [
        'core/model_manager.py',
        'core/data_types.py',
        'core/hunyuan_integration.py',
        'nodes/generation_nodes.py',
        '__init__.py'
    ]
    
    optional_files = [
        'HunyuanWorld-1.0/hy3dworld/__init__.py',
        'requirements_hunyuan.txt',
        'INTEGRATION_GUIDE.md'
    ]
    
    print("\nRequired files:")
    for file_path in required_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
    
    print("\nOptional files (for integration):")
    for file_path in optional_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"⚠️ {file_path} - not found")

if __name__ == "__main__":
    print("HunyuanWorld ComfyUI Integration Test")
    print("=====================================")
    
    # Run all tests
    test_directory_structure()
    test_dependencies()
    success = test_integration()
    
    if success:
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)