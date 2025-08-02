import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union

def validate_panorama_aspect_ratio(width: int, height: int, tolerance: float = 0.1) -> bool:
    """Validate that image has correct panoramic aspect ratio (2:1)"""
    aspect_ratio = width / height
    expected_ratio = 2.0
    return abs(aspect_ratio - expected_ratio) <= tolerance

def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], allow_batch: bool = True) -> bool:
    """Validate tensor has expected shape, optionally allowing batch dimension"""
    if allow_batch and len(tensor.shape) == len(expected_shape) + 1:
        # Check without batch dimension
        return tensor.shape[1:] == expected_shape
    return tensor.shape == expected_shape

def validate_image_tensor(tensor: torch.Tensor, min_size: int = 64, max_size: int = 4096) -> Dict[str, Any]:
    """Validate image tensor and return validation info"""
    result = {
        "valid": True,
        "issues": [],
        "info": {}
    }
    
    # Check tensor type
    if not isinstance(tensor, torch.Tensor):
        result["valid"] = False
        result["issues"].append(f"Expected torch.Tensor, got {type(tensor)}")
        return result
    
    # Check dimensions
    if len(tensor.shape) < 3 or len(tensor.shape) > 4:
        result["valid"] = False
        result["issues"].append(f"Expected 3D or 4D tensor, got {len(tensor.shape)}D")
        return result
    
    # Extract dimensions
    if len(tensor.shape) == 4:
        batch, height, width, channels = tensor.shape
        result["info"]["has_batch"] = True
        result["info"]["batch_size"] = batch
    else:
        height, width, channels = tensor.shape
        result["info"]["has_batch"] = False
        result["info"]["batch_size"] = 1
    
    result["info"]["height"] = height
    result["info"]["width"] = width
    result["info"]["channels"] = channels
    
    # Validate size constraints
    if height < min_size or height > max_size:
        result["issues"].append(f"Height {height} outside valid range [{min_size}, {max_size}]")
    
    if width < min_size or width > max_size:
        result["issues"].append(f"Width {width} outside valid range [{min_size}, {max_size}]")
    
    # Validate channels
    if channels not in [1, 3, 4]:
        result["issues"].append(f"Invalid channel count: {channels} (expected 1, 3, or 4)")
    
    # Check data range
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    result["info"]["value_range"] = (min_val, max_val)
    
    if min_val < -0.1 or max_val > 1.1:
        result["issues"].append(f"Values outside expected range [0, 1]: [{min_val:.3f}, {max_val:.3f}]")
    
    # Check for NaN or Inf
    if torch.isnan(tensor).any():
        result["issues"].append("Tensor contains NaN values")
    
    if torch.isinf(tensor).any():
        result["issues"].append("Tensor contains infinite values")
    
    # Set overall validity
    result["valid"] = len(result["issues"]) == 0
    
    return result

def validate_mesh_data(vertices: torch.Tensor, faces: torch.Tensor, texture_coords: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Validate 3D mesh data"""
    result = {
        "valid": True,
        "issues": [],
        "info": {}
    }
    
    # Validate vertices
    if not isinstance(vertices, torch.Tensor):
        result["valid"] = False
        result["issues"].append(f"Vertices must be torch.Tensor, got {type(vertices)}")
        return result
    
    if len(vertices.shape) != 2 or vertices.shape[1] != 3:
        result["valid"] = False
        result["issues"].append(f"Vertices must have shape (N, 3), got {vertices.shape}")
        return result
    
    num_vertices = vertices.shape[0]
    result["info"]["num_vertices"] = num_vertices
    
    if num_vertices < 3:
        result["issues"].append(f"Too few vertices: {num_vertices} (minimum 3)")
    
    # Validate faces
    if not isinstance(faces, torch.Tensor):
        result["valid"] = False
        result["issues"].append(f"Faces must be torch.Tensor, got {type(faces)}")
        return result
    
    if len(faces.shape) != 2 or faces.shape[1] != 3:
        result["valid"] = False
        result["issues"].append(f"Faces must have shape (M, 3), got {faces.shape}")
        return result
    
    num_faces = faces.shape[0]
    result["info"]["num_faces"] = num_faces
    
    if num_faces < 1:
        result["issues"].append("No faces provided")
    
    # Check face indices are valid
    max_face_idx = faces.max().item()
    min_face_idx = faces.min().item()
    
    if min_face_idx < 0:
        result["issues"].append(f"Negative face index found: {min_face_idx}")
    
    if max_face_idx >= num_vertices:
        result["issues"].append(f"Face index {max_face_idx} exceeds vertex count {num_vertices}")
    
    # Validate texture coordinates if provided
    if texture_coords is not None:
        if not isinstance(texture_coords, torch.Tensor):
            result["issues"].append(f"Texture coords must be torch.Tensor, got {type(texture_coords)}")
        elif len(texture_coords.shape) != 2 or texture_coords.shape[1] != 2:
            result["issues"].append(f"Texture coords must have shape (N, 2), got {texture_coords.shape}")
        elif texture_coords.shape[0] != num_vertices:
            result["issues"].append(f"Texture coords count {texture_coords.shape[0]} != vertex count {num_vertices}")
        else:
            # Check UV range
            uv_min = texture_coords.min().item()
            uv_max = texture_coords.max().item()
            result["info"]["uv_range"] = (uv_min, uv_max)
            
            if uv_min < -0.1 or uv_max > 1.1:
                result["issues"].append(f"UV coordinates outside [0, 1] range: [{uv_min:.3f}, {uv_max:.3f}]")
    
    # Check for degenerate faces
    degenerate_faces = []
    for i, face in enumerate(faces):
        if len(set(face.tolist())) < 3:
            degenerate_faces.append(i)
    
    if degenerate_faces:
        result["issues"].append(f"Found {len(degenerate_faces)} degenerate faces")
        result["info"]["degenerate_faces"] = degenerate_faces[:10]  # First 10
    
    # Check for isolated vertices
    used_vertices = set(faces.flatten().tolist())
    isolated_vertices = set(range(num_vertices)) - used_vertices
    
    if isolated_vertices:
        result["issues"].append(f"Found {len(isolated_vertices)} isolated vertices")
        result["info"]["isolated_vertices"] = list(isolated_vertices)[:10]  # First 10
    
    result["valid"] = len(result["issues"]) == 0
    
    return result

def validate_prompt(prompt: str, max_length: int = 1000, min_length: int = 1) -> Dict[str, Any]:
    """Validate text prompt"""
    result = {
        "valid": True,
        "issues": [],
        "info": {}
    }
    
    if not isinstance(prompt, str):
        result["valid"] = False
        result["issues"].append(f"Prompt must be string, got {type(prompt)}")
        return result
    
    prompt_length = len(prompt.strip())
    result["info"]["length"] = prompt_length
    result["info"]["word_count"] = len(prompt.strip().split())
    
    if prompt_length < min_length:
        result["issues"].append(f"Prompt too short: {prompt_length} chars (minimum {min_length})")
    
    if prompt_length > max_length:
        result["issues"].append(f"Prompt too long: {prompt_length} chars (maximum {max_length})")
    
    # Check for potentially problematic content
    if not prompt.strip():
        result["issues"].append("Empty prompt")
    
    # Check for excessive special characters
    special_char_ratio = sum(1 for c in prompt if not c.isalnum() and not c.isspace()) / len(prompt)
    if special_char_ratio > 0.3:
        result["issues"].append(f"High special character ratio: {special_char_ratio:.2f}")
    
    result["valid"] = len(result["issues"]) == 0
    
    return result

def validate_generation_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate generation parameters"""
    result = {
        "valid": True,
        "issues": [],
        "info": {}
    }
    
    # Define parameter constraints
    constraints = {
        "width": {"type": int, "min": 256, "max": 4096, "multiple": 8},
        "height": {"type": int, "min": 128, "max": 2048, "multiple": 8},
        "num_inference_steps": {"type": int, "min": 1, "max": 200},
        "guidance_scale": {"type": (int, float), "min": 0.1, "max": 30.0},
        "strength": {"type": (int, float), "min": 0.0, "max": 1.0},
        "seed": {"type": int, "min": -1, "max": 2**32 - 1},
        "mesh_resolution": {"type": int, "min": 64, "max": 2048, "multiple": 64},
        "texture_resolution": {"type": int, "min": 128, "max": 4096, "multiple": 128},
    }
    
    for param, value in params.items():
        if param not in constraints:
            continue
            
        constraint = constraints[param]
        
        # Type check
        if not isinstance(value, constraint["type"]):
            result["issues"].append(f"{param}: Expected {constraint['type']}, got {type(value)}")
            continue
        
        # Range check
        if "min" in constraint and value < constraint["min"]:
            result["issues"].append(f"{param}: {value} below minimum {constraint['min']}")
        
        if "max" in constraint and value > constraint["max"]:
            result["issues"].append(f"{param}: {value} above maximum {constraint['max']}")
        
        # Multiple check
        if "multiple" in constraint and value % constraint["multiple"] != 0:
            result["issues"].append(f"{param}: {value} not multiple of {constraint['multiple']}")
    
    # Cross-parameter validation
    if "width" in params and "height" in params:
        aspect_ratio = params["width"] / params["height"]
        if not validate_panorama_aspect_ratio(params["width"], params["height"]):
            result["issues"].append(f"Non-standard panoramic aspect ratio: {aspect_ratio:.2f} (expected ~2.0)")
    
    result["valid"] = len(result["issues"]) == 0
    
    return result

def validate_device_compatibility(device: str) -> Dict[str, Any]:
    """Validate device compatibility"""
    result = {
        "valid": True,
        "issues": [],
        "info": {}
    }
    
    if device == "auto":
        # Auto-detect best device
        if torch.cuda.is_available():
            result["info"]["detected_device"] = "cuda"
            result["info"]["cuda_devices"] = torch.cuda.device_count()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            result["info"]["detected_device"] = "mps"
        else:
            result["info"]["detected_device"] = "cpu"
    
    elif device.startswith("cuda"):
        if not torch.cuda.is_available():
            result["valid"] = False
            result["issues"].append("CUDA not available")
        else:
            if ":" in device:
                try:
                    device_id = int(device.split(":")[1])
                    if device_id >= torch.cuda.device_count():
                        result["issues"].append(f"CUDA device {device_id} not available")
                except ValueError:
                    result["issues"].append(f"Invalid CUDA device specification: {device}")
    
    elif device == "mps":
        if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            result["valid"] = False
            result["issues"].append("MPS not available")
    
    elif device == "cpu":
        # CPU always available
        pass
    
    else:
        result["issues"].append(f"Unknown device: {device}")
    
    result["valid"] = len(result["issues"]) == 0
    
    return result

def validate_memory_requirements(operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate and validate memory requirements"""
    result = {
        "valid": True,
        "issues": [],
        "info": {}
    }
    
    # Rough memory estimates (in GB)
    base_memory = {
        "text_to_panorama": 2.0,
        "scene_generation": 1.5,
        "world_reconstruction": 3.0,
    }
    
    estimated_memory = base_memory.get(operation, 1.0)
    
    # Scale based on parameters
    if "width" in params and "height" in params:
        resolution_factor = (params["width"] * params["height"]) / (1024 * 512)
        estimated_memory *= max(1.0, resolution_factor)
    
    if "mesh_resolution" in params:
        mesh_factor = (params["mesh_resolution"] / 512) ** 2
        estimated_memory *= max(1.0, mesh_factor)
    
    result["info"]["estimated_memory_gb"] = estimated_memory
    
    # Check available memory
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated_memory = torch.cuda.memory_allocated(0) / 1e9
        available_memory = total_memory - allocated_memory
        
        result["info"]["total_gpu_memory_gb"] = total_memory
        result["info"]["available_gpu_memory_gb"] = available_memory
        
        if estimated_memory > available_memory:
            result["issues"].append(
                f"Insufficient GPU memory: need {estimated_memory:.1f}GB, "
                f"have {available_memory:.1f}GB available"
            )
    
    result["valid"] = len(result["issues"]) == 0
    
    return result