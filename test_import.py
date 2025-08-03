#!/usr/bin/env python3
"""
Test script to validate all node imports
"""

import sys
import traceback

def test_import(module_name, class_names):
    """Test importing specific classes from a module"""
    try:
        module = __import__(module_name, fromlist=class_names)
        for class_name in class_names:
            if hasattr(module, class_name):
                print(f"✅ {class_name} imported successfully")
            else:
                print(f"❌ {class_name} not found in {module_name}")
        return True
    except Exception as e:
        print(f"❌ Error importing {module_name}: {e}")
        traceback.print_exc()
        return False

def main():
    print("Testing HunyuanWorld ComfyUI Node Imports...")
    print("=" * 50)
    
    # Test data types first
    print("\n1. Testing Data Types:")
    test_import("core.data_types", ["HUNYUAN_DATA_TYPES", "PanoramaImage", "Scene3D", "WorldMesh"])
    
    # Test input nodes
    print("\n2. Testing Input Nodes:")
    test_import("nodes.input_nodes", [
        "HunyuanTextInput", 
        "HunyuanImageInput", 
        "HunyuanPromptProcessor",
        "HunyuanObjectLabeler",
        "HunyuanMaskCreator"
    ])
    
    # Test generation nodes
    print("\n3. Testing Generation Nodes:")
    test_import("nodes.generation_nodes", [
        "HunyuanLoader",
        "HunyuanTextToPanorama", 
        "HunyuanImageToPanorama",
        "HunyuanSceneGenerator",
        "HunyuanWorldReconstructor",
        "HunyuanSceneInpainter",
        "HunyuanSkyInpainter", 
        "HunyuanLayeredSceneGenerator"
    ])
    
    # Test output nodes
    print("\n4. Testing Output Nodes:")
    test_import("nodes.output_nodes", [
        "HunyuanViewer",
        "HunyuanMeshExporter", 
        "HunyuanDataInfo",
        "HunyuanDracoExporter",
        "HunyuanLayeredMeshExporter"
    ])
    
    # Test main module import
    print("\n5. Testing Main Module:")
    try:
        import __init__
        print("✅ Main __init__.py imported successfully")
        
        if hasattr(__init__, 'NODE_CLASS_MAPPINGS'):
            print(f"✅ Found {len(__init__.NODE_CLASS_MAPPINGS)} nodes in NODE_CLASS_MAPPINGS")
        else:
            print("❌ NODE_CLASS_MAPPINGS not found")
            
        if hasattr(__init__, 'NODE_DISPLAY_NAME_MAPPINGS'):
            print(f"✅ Found {len(__init__.NODE_DISPLAY_NAME_MAPPINGS)} display names")
        else:
            print("❌ NODE_DISPLAY_NAME_MAPPINGS not found")
            
    except Exception as e:
        print(f"❌ Error importing main module: {e}")
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Import test completed!")

if __name__ == "__main__":
    main()