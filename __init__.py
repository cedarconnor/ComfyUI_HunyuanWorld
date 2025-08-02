"""
HunyuanWorld ComfyUI Integration

Provides ComfyUI nodes for Tencent's HunyuanWorld-1.0 framework,
enabling text-to-world and image-to-world generation capabilities.
"""

from .nodes.input_nodes import (
    HunyuanTextInput,
    HunyuanImageInput,
    HunyuanPromptProcessor
)

from .nodes.generation_nodes import (
    HunyuanLoader,
    HunyuanTextToPanorama,
    HunyuanImageToPanorama,
    HunyuanSceneGenerator,
    HunyuanWorldReconstructor
)

from .nodes.output_nodes import (
    HunyuanViewer,
    HunyuanMeshExporter,
    HunyuanDataInfo
)

from .core.data_types import HUNYUAN_DATA_TYPES

# ComfyUI Node Mappings
NODE_CLASS_MAPPINGS = {
    # Input Nodes
    "HunyuanTextInput": HunyuanTextInput,
    "HunyuanImageInput": HunyuanImageInput,
    "HunyuanPromptProcessor": HunyuanPromptProcessor,
    
    # Generation Nodes
    "HunyuanLoader": HunyuanLoader,
    "HunyuanTextToPanorama": HunyuanTextToPanorama,
    "HunyuanImageToPanorama": HunyuanImageToPanorama,
    "HunyuanSceneGenerator": HunyuanSceneGenerator,
    "HunyuanWorldReconstructor": HunyuanWorldReconstructor,
    
    # Output Nodes
    "HunyuanViewer": HunyuanViewer,
    "HunyuanMeshExporter": HunyuanMeshExporter,
    "HunyuanDataInfo": HunyuanDataInfo,
}

# Display Names for ComfyUI Interface
NODE_DISPLAY_NAME_MAPPINGS = {
    # Input Nodes
    "HunyuanTextInput": "Hunyuan Text Input",
    "HunyuanImageInput": "Hunyuan Image Input",
    "HunyuanPromptProcessor": "Hunyuan Prompt Processor",
    
    # Generation Nodes
    "HunyuanLoader": "Hunyuan Model Loader",
    "HunyuanTextToPanorama": "Hunyuan Text to Panorama",
    "HunyuanImageToPanorama": "Hunyuan Image to Panorama",
    "HunyuanSceneGenerator": "Hunyuan Scene Generator",
    "HunyuanWorldReconstructor": "Hunyuan World Reconstructor",
    
    # Output Nodes
    "HunyuanViewer": "Hunyuan Viewer",
    "HunyuanMeshExporter": "Hunyuan Mesh Exporter",
    "HunyuanDataInfo": "Hunyuan Data Info",
}

# Web Extensions for ComfyUI Frontend
WEB_DIRECTORY = "./web"

# Custom Data Types for ComfyUI
def register_custom_types():
    """Register custom data types with ComfyUI"""
    try:
        # Try to register custom types if ComfyUI supports it
        import comfy.model_management as model_management
        
        # Register our custom data types
        for type_name, type_class in HUNYUAN_DATA_TYPES.items():
            # This is a placeholder - actual registration depends on ComfyUI's API
            pass
            
    except ImportError:
        # ComfyUI might not expose this API, that's okay
        pass

# Initialize custom types when module is imported
register_custom_types()

# Package metadata
__version__ = "1.0.0"
__author__ = "HunyuanWorld ComfyUI Integration"
__description__ = "ComfyUI nodes for HunyuanWorld-1.0 3D world generation"

# Export main components
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "HUNYUAN_DATA_TYPES",
]