import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

class PanoramaImage:
    """Custom data type for 360-degree panoramic images"""
    
    def __init__(self, image: torch.Tensor, metadata: Optional[Dict[str, Any]] = None):
        self.image = image  # Shape: (H, W, C) or (B, H, W, C)
        self.metadata = metadata or {}
        
        # Validate panoramic format (2:1 aspect ratio for equirectangular)
        if len(image.shape) >= 2:
            h, w = image.shape[-2:]
            if abs(w / h - 2.0) > 0.1:  # Allow some tolerance
                print(f"Warning: Image aspect ratio {w/h:.2f} may not be proper equirectangular format (expected ~2.0)")
    
    @property
    def shape(self):
        return self.image.shape
    
    @property
    def device(self):
        return self.image.device
    
    def clone(self):
        return PanoramaImage(self.image.clone(), self.metadata.copy())

class Scene3D:
    """Custom data type for 3D scene data with depth maps and semantic layers"""
    
    def __init__(self, 
                 panorama: PanoramaImage,
                 depth_map: Optional[torch.Tensor] = None,
                 semantic_masks: Optional[Dict[str, torch.Tensor]] = None,
                 object_layers: Optional[List[Dict[str, Any]]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        
        self.panorama = panorama
        self.depth_map = depth_map  # Shape: (H, W) or (B, H, W)
        self.semantic_masks = semantic_masks or {}  # Dict of mask_name -> tensor
        self.object_layers = object_layers or []  # List of object information
        self.metadata = metadata or {}
    
    @property
    def device(self):
        return self.panorama.device
    
    def to(self, device):
        """Move all tensors to specified device"""
        new_panorama = PanoramaImage(self.panorama.image.to(device), self.panorama.metadata)
        new_depth = self.depth_map.to(device) if self.depth_map is not None else None
        new_masks = {k: v.to(device) for k, v in self.semantic_masks.items()}
        
        return Scene3D(new_panorama, new_depth, new_masks, self.object_layers, self.metadata)

class WorldMesh:
    """Custom data type for 3D world geometry with textures and materials"""
    
    def __init__(self,
                 vertices: torch.Tensor,
                 faces: torch.Tensor,
                 texture_coords: Optional[torch.Tensor] = None,
                 textures: Optional[Dict[str, torch.Tensor]] = None,
                 materials: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        
        self.vertices = vertices  # Shape: (N, 3)
        self.faces = faces  # Shape: (M, 3) - triangle indices
        self.texture_coords = texture_coords  # Shape: (N, 2) - UV coordinates
        self.textures = textures or {}  # Dict of texture_name -> tensor
        self.materials = materials or {}  # Dict of material properties
        self.metadata = metadata or {}
    
    @property
    def device(self):
        return self.vertices.device
    
    @property
    def num_vertices(self):
        return self.vertices.shape[0]
    
    @property
    def num_faces(self):
        return self.faces.shape[0]
    
    def to(self, device):
        """Move all tensors to specified device"""
        new_vertices = self.vertices.to(device)
        new_faces = self.faces.to(device)
        new_tex_coords = self.texture_coords.to(device) if self.texture_coords is not None else None
        new_textures = {k: v.to(device) for k, v in self.textures.items()}
        
        return WorldMesh(new_vertices, new_faces, new_tex_coords, new_textures, 
                        self.materials, self.metadata)

class ModelHunyuan:
    """Custom data type for loaded HunyuanWorld model reference"""
    
    def __init__(self, 
                 model: Any,
                 model_type: str,
                 device: str,
                 precision: str = "fp32",
                 metadata: Optional[Dict[str, Any]] = None):
        
        self.model = model
        self.model_type = model_type  # "text_to_panorama", "scene_generator", etc.
        self.device = device
        self.precision = precision
        self.metadata = metadata or {}
        self.is_loaded = True
    
    def unload(self):
        """Unload model to free memory"""
        if hasattr(self.model, 'cpu'):
            self.model.cpu()
        self.is_loaded = False
    
    def reload(self):
        """Reload model to device"""
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        self.is_loaded = True

# ComfyUI data type registration
HUNYUAN_DATA_TYPES = {
    "PANORAMA_IMAGE": PanoramaImage,
    "SCENE_3D": Scene3D,
    "WORLD_MESH": WorldMesh,
    "MODEL_HUNYUAN": ModelHunyuan,
}