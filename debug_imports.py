#!/usr/bin/env python3
"""
Debug script to test imports step by step
"""

import sys
import os
import traceback

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_import_step(step_name, import_func):
    """Test a single import step"""
    print(f"\n--- {step_name} ---")
    try:
        result = import_func()
        print(f"✅ {step_name}: SUCCESS")
        return result
    except Exception as e:
        print(f"❌ {step_name}: FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        return None

def main():
    print("=== ComfyUI HunyuanWorld Import Debug ===")
    
    # Test 1: Core data types
    def test_data_types():
        from core.data_types import PanoramaImage, Scene3D, WorldMesh, HUNYUAN_DATA_TYPES
        return len(HUNYUAN_DATA_TYPES)
    
    result1 = test_import_step("Core Data Types", test_data_types)
    
    # Test 2: Model manager
    def test_model_manager():
        from core.model_manager import ModelManager, model_manager
        return model_manager
    
    result2 = test_import_step("Model Manager", test_model_manager)
    
    # Test 3: Input nodes
    def test_input_nodes():
        from nodes.input_nodes import HunyuanTextInput, HunyuanImageInput
        return [HunyuanTextInput, HunyuanImageInput]
    
    result3 = test_import_step("Input Nodes", test_input_nodes)
    
    # Test 4: Generation nodes
    def test_generation_nodes():
        from nodes.generation_nodes import HunyuanLoader, HunyuanTextToPanorama
        return [HunyuanLoader, HunyuanTextToPanorama]
    
    result4 = test_import_step("Generation Nodes", test_generation_nodes)
    
    # Test 5: Output nodes
    def test_output_nodes():
        from nodes.output_nodes import HunyuanViewer, HunyuanMeshExporter
        return [HunyuanViewer, HunyuanMeshExporter]
    
    result5 = test_import_step("Output Nodes", test_output_nodes)
    
    # Test 6: Main module registration
    def test_main_init():
        # Import the main __init__ module
        import __init__ as main_module
        return {
            'NODE_CLASS_MAPPINGS': getattr(main_module, 'NODE_CLASS_MAPPINGS', {}),
            'NODE_DISPLAY_NAME_MAPPINGS': getattr(main_module, 'NODE_DISPLAY_NAME_MAPPINGS', {}),
            'WEB_DIRECTORY': getattr(main_module, 'WEB_DIRECTORY', None)
        }
    
    result6 = test_import_step("Main Init Module", test_main_init)
    
    # Summary
    print("\n=== SUMMARY ===")
    tests = [
        ("Core Data Types", result1 is not None),
        ("Model Manager", result2 is not None),
        ("Input Nodes", result3 is not None),
        ("Generation Nodes", result4 is not None),
        ("Output Nodes", result5 is not None),
        ("Main Init", result6 is not None)
    ]
    
    for test_name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    if result6:
        print(f"\nRegistered nodes: {len(result6.get('NODE_CLASS_MAPPINGS', {}))}")
        print(f"Display names: {len(result6.get('NODE_DISPLAY_NAME_MAPPINGS', {}))}")
        print(f"Web directory: {result6.get('WEB_DIRECTORY', 'None')}")

if __name__ == "__main__":
    main()