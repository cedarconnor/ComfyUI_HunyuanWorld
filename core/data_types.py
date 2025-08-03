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

class LayeredScene3D:
    """Advanced data type for multi-layer 3D scene decomposition"""
    
    def __init__(self,
                 panorama: PanoramaImage,
                 background_scene: Scene3D,
                 foreground_layers: List[Dict[str, Any]],
                 layer_depth_maps: Dict[str, torch.Tensor],
                 layer_masks: Dict[str, torch.Tensor],
                 object_labels: Optional['ObjectLabels'] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        
        self.panorama = panorama
        self.background_scene = background_scene
        self.foreground_layers = foreground_layers  # List of layer info dicts
        self.layer_depth_maps = layer_depth_maps  # Dict: layer_name -> depth_tensor
        self.layer_masks = layer_masks  # Dict: layer_name -> mask_tensor
        self.object_labels = object_labels
        self.metadata = metadata or {}
    
    @property
    def device(self):
        return self.panorama.device
    
    @property
    def num_layers(self):
        return len(self.foreground_layers)
    
    def get_layer_by_name(self, layer_name: str) -> Optional[Dict[str, Any]]:
        """Get layer information by name"""
        for layer in self.foreground_layers:
            if layer.get('name') == layer_name:
                return layer
        return None

class ObjectLabels:
    """Data type for foreground object labeling and classification"""
    
    def __init__(self,
                 fg_labels_1: List[str],
                 fg_labels_2: List[str],
                 scene_class: str = "outdoor",
                 confidence_threshold: float = 0.5,
                 label_weights: Optional[Dict[str, float]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        
        self.fg_labels_1 = fg_labels_1  # Primary foreground objects
        self.fg_labels_2 = fg_labels_2  # Secondary foreground objects
        self.scene_class = scene_class  # Scene classification
        self.confidence_threshold = confidence_threshold
        self.label_weights = label_weights or {}  # Object importance weights
        self.metadata = metadata or {}
    
    def get_all_labels(self) -> List[str]:
        """Get all foreground labels combined"""
        return self.fg_labels_1 + self.fg_labels_2
    
    def get_weighted_labels(self) -> Dict[str, float]:
        """Get labels with their weights"""
        weighted = {}
        for label in self.fg_labels_1:
            weighted[label] = self.label_weights.get(label, 1.0)
        for label in self.fg_labels_2:
            weighted[label] = self.label_weights.get(label, 0.8)
        return weighted

class SceneMask:
    """Data type for panorama inpainting masks"""
    
    def __init__(self,
                 mask: torch.Tensor,
                 mask_type: str = "scene",  # "scene", "sky", "object", "custom"
                 invert: bool = False,
                 feather: float = 0.0,
                 target_regions: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        
        self.mask = mask  # Shape: (H, W) or (B, H, W) - binary or soft mask
        self.mask_type = mask_type
        self.invert = invert
        self.feather = feather  # Feathering amount for soft edges
        self.target_regions = target_regions or []  # Named regions to target
        self.metadata = metadata or {}
    
    @property
    def shape(self):
        return self.mask.shape
    
    @property
    def device(self):
        return self.mask.device
    
    def get_processed_mask(self) -> torch.Tensor:
        """Get the processed mask with inversion and feathering applied"""
        processed = self.mask.clone()
        
        if self.invert:
            processed = 1.0 - processed
        
        if self.feather > 0:
            # Apply gaussian blur for feathering (simplified)
            # In real implementation, would use proper gaussian blur
            processed = torch.clamp(processed, 0.0, 1.0)
        
        return processed

class LayerMesh:
    """Data type for layered 3D mesh with separate layer geometry"""
    
    def __init__(self,
                 base_mesh: WorldMesh,
                 layer_meshes: Dict[str, WorldMesh],
                 layer_hierarchy: List[str],
                 layer_transforms: Optional[Dict[str, torch.Tensor]] = None,
                 layer_visibility: Optional[Dict[str, bool]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        
        self.base_mesh = base_mesh  # Background/base layer
        self.layer_meshes = layer_meshes  # Dict: layer_name -> WorldMesh
        self.layer_hierarchy = layer_hierarchy  # Rendering order (back to front)
        self.layer_transforms = layer_transforms or {}  # Layer transformations
        self.layer_visibility = layer_visibility or {}  # Layer visibility
        self.metadata = metadata or {}
    
    @property
    def device(self):
        return self.base_mesh.device
    
    @property
    def total_vertices(self):
        """Total vertex count across all layers"""
        total = self.base_mesh.num_vertices
        for mesh in self.layer_meshes.values():
            total += mesh.num_vertices
        return total
    
    @property
    def total_faces(self):
        """Total face count across all layers"""
        total = self.base_mesh.num_faces
        for mesh in self.layer_meshes.values():
            total += mesh.num_faces
        return total
    
    def get_layer_mesh(self, layer_name: str) -> Optional[WorldMesh]:
        """Get mesh for specific layer"""
        return self.layer_meshes.get(layer_name)
    
    def set_layer_visibility(self, layer_name: str, visible: bool):
        """Set visibility for a specific layer"""
        self.layer_visibility[layer_name] = visible

# ComfyUI data type registration
HUNYUAN_DATA_TYPES = {
    "PANORAMA_IMAGE": PanoramaImage,
    "SCENE_3D": Scene3D,
    "WORLD_MESH": WorldMesh,
    "MODEL_HUNYUAN": ModelHunyuan,
    "LAYERED_SCENE_3D": LayeredScene3D,
    "OBJECT_LABELS": ObjectLabels,
    "SCENE_MASK": SceneMask,
    "LAYER_MESH": LayerMesh,
}