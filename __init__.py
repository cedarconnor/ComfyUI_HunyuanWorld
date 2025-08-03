"""
HunyuanWorld ComfyUI Integration

Provides ComfyUI nodes for Tencent's HunyuanWorld-1.0 framework,
enabling text-to-world and image-to-world generation capabilities.
"""

import sys
import os

print("🚀 Starting HunyuanWorld ComfyUI import...")

# Get current directory and ensure it's in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Clear any cached modules
modules_to_clear = [name for name in sys.modules.keys() if 'HunyuanWorld' in name or 'hunyuan' in name.lower()]
for module_name in modules_to_clear:
    if module_name in sys.modules:
        del sys.modules[module_name]
        print(f"🧹 Cleared cached module: {module_name}")

# Initialize ComfyUI mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"

print(f"📁 Working directory: {current_dir}")

# Step 1: Load core modules with explicit path manipulation
print("📚 Loading core modules...")
core_path = os.path.join(current_dir, "core")
if core_path not in sys.path:
    sys.path.insert(0, core_path)

try:
    # Import data types first
    with open(os.path.join(core_path, "data_types.py"), 'r', encoding='utf-8') as f:
        data_types_code = f.read()
    exec(data_types_code, globals())
    print("✅ Data types loaded")
    
    # Import hunyuan_integration next
    with open(os.path.join(core_path, "hunyuan_integration.py"), 'r', encoding='utf-8') as f:
        integration_code = f.read()
    
    integration_namespace = {
        '__builtins__': __builtins__,
        '__name__': 'hunyuan_integration',
        '__file__': os.path.join(core_path, "hunyuan_integration.py"),
        'os': os,
        'sys': sys,
        'torch': __import__('torch'),
        'typing': __import__('typing'),
        'Dict': __import__('typing').Dict,
        'Any': __import__('typing').Any,
        'Optional': __import__('typing').Optional,
        'Union': __import__('typing').Union,
        'Path': __import__('pathlib').Path,
    }
    
    exec(integration_code, integration_namespace)
    print("✅ Hunyuan integration loaded")
    
    # Import model manager with fixed imports
    with open(os.path.join(core_path, "model_manager.py"), 'r', encoding='utf-8') as f:
        model_manager_code = f.read()
    
    # Fix the relative imports in model_manager.py
    model_manager_code = model_manager_code.replace("from .data_types import ModelHunyuan", "# from .data_types import ModelHunyuan")
    # Replace the entire multi-line import block
    import_block = """from .hunyuan_integration import (
    get_hunyuan_model_class, 
    HUNYUAN_AVAILABLE
)"""
    replacement_block = """# from .hunyuan_integration import (
#     get_hunyuan_model_class, 
#     HUNYUAN_AVAILABLE
# )"""
    model_manager_code = model_manager_code.replace(import_block, replacement_block)
    
    # Create namespace with required objects
    model_manager_namespace = {
        '__builtins__': __builtins__,
        '__name__': 'model_manager',
        '__file__': os.path.join(core_path, "model_manager.py"),
        'os': os,
        'torch': __import__('torch'),
        'gc': __import__('gc'),
        'typing': __import__('typing'),
        'Dict': __import__('typing').Dict,
        'Any': __import__('typing').Any,
        'Optional': __import__('typing').Optional,
        'Union': __import__('typing').Union,
        'ModelHunyuan': ModelHunyuan,  # From data_types loaded above
        # From hunyuan_integration
        'get_hunyuan_model_class': integration_namespace['get_hunyuan_model_class'],
        'HUNYUAN_AVAILABLE': integration_namespace['HUNYUAN_AVAILABLE'],
    }
    
    exec(model_manager_code, model_manager_namespace)
    
    # Extract the classes we need
    ModelManager = model_manager_namespace['ModelManager']
    model_manager = model_manager_namespace['model_manager']
    
    print("✅ Model manager loaded")
    
except Exception as e:
    print(f"❌ Core module error: {e}")
    import traceback
    traceback.print_exc()
    # Create fallback variables to prevent NameError in node loading
    model_manager = None
    ModelManager = None

# Step 2: Load nodes with direct file execution
print("🔧 Loading node classes...")

def load_node_classes():
    """Load all node classes using direct file execution"""
    loaded_classes = {}
    
    node_files = {
        "input_nodes.py": [
            'HunyuanTextInput', 'HunyuanImageInput', 'HunyuanPromptProcessor',
            'HunyuanObjectLabeler', 'HunyuanMaskCreator'
        ],
        "generation_nodes.py": [
            'HunyuanLoader', 'HunyuanTextToPanorama', 'HunyuanImageToPanorama',
            'HunyuanSceneGenerator', 'HunyuanWorldReconstructor', 'HunyuanSceneInpainter',
            'HunyuanSkyInpainter', 'HunyuanLayeredSceneGenerator', 'HunyuanFluxGenerator'
        ],
        "output_nodes.py": [
            'HunyuanViewer', 'HunyuanMeshExporter', 'HunyuanDataInfo',
            'HunyuanDracoExporter', 'HunyuanLayeredMeshExporter'
        ]
    }
    
    nodes_path = os.path.join(current_dir, "nodes")
    
    for file_name, class_names in node_files.items():
        file_path = os.path.join(nodes_path, file_name)
        print(f"🔍 Loading {file_name}...")
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace problematic relative imports
            content = content.replace("from ..core.data_types import", "# from ..core.data_types import")
            content = content.replace("from ..core.model_manager import model_manager", "# from ..core.model_manager import model_manager")
            
            # Create execution namespace with all required objects
            namespace = {
                '__builtins__': __builtins__,
                '__file__': file_path,
                '__name__': file_name[:-3],  # Remove .py extension
                # Standard imports
                'torch': __import__('torch'),
                'numpy': __import__('numpy'),
                'os': os,
                'json': __import__('json'),
                'typing': __import__('typing'),
                'PIL': __import__('PIL'),
                'Image': __import__('PIL.Image', fromlist=['Image']).Image,
                # Our data types (from globals)
                'PanoramaImage': PanoramaImage,
                'Scene3D': Scene3D,
                'WorldMesh': WorldMesh,
                'ModelHunyuan': ModelHunyuan,
                'LayeredScene3D': LayeredScene3D,
                'ObjectLabels': ObjectLabels,
                'SceneMask': SceneMask,
                'LayerMesh': LayerMesh,
                'model_manager': model_manager,
                # Additional typing imports
                'Dict': __import__('typing').Dict,
                'Any': __import__('typing').Any,
                'List': __import__('typing').List,
                'Optional': __import__('typing').Optional,
                'Tuple': __import__('typing').Tuple,
            }
            
            # Execute the file
            exec(content, namespace)
            
            # Extract node classes
            for class_name in class_names:
                if class_name in namespace and hasattr(namespace[class_name], 'INPUT_TYPES'):
                    loaded_classes[class_name] = namespace[class_name]
                    print(f"  ✅ {class_name}")
                else:
                    print(f"  ❌ {class_name} not found or invalid")
                    
        except Exception as e:
            print(f"  ❌ Error loading {file_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return loaded_classes

# Load all node classes only if core modules loaded successfully
if model_manager is not None:
    all_classes = load_node_classes()
else:
    print("⚠️ Skipping node loading due to core module errors")
    all_classes = {}

# Set up ComfyUI mappings
NODE_CLASS_MAPPINGS = all_classes
NODE_DISPLAY_NAME_MAPPINGS = {
    name: name.replace('Hunyuan', 'Hunyuan ') for name in all_classes.keys()
}

# Final status
print(f"🎯 Final Results:")
print(f"   📊 Total nodes loaded: {len(NODE_CLASS_MAPPINGS)}")
print(f"   📋 Available nodes: {list(NODE_CLASS_MAPPINGS.keys())}")

if len(NODE_CLASS_MAPPINGS) > 0:
    print("✅ HunyuanWorld ComfyUI: Import successful!")
else:
    print("❌ HunyuanWorld ComfyUI: Import failed!")

# Package metadata
__version__ = "1.0.0"
__author__ = "HunyuanWorld ComfyUI Integration"
__description__ = "ComfyUI nodes for HunyuanWorld-1.0 3D world generation"

# Export for ComfyUI
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS", 
    "WEB_DIRECTORY",
]

print("🏁 HunyuanWorld import process complete.")