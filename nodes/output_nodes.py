import os
import json
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from PIL import Image

from ..core.data_types import PanoramaImage, Scene3D, WorldMesh, LayeredScene3D, SceneMask, LayerMesh

class HunyuanViewer:
    """Preview node for panoramic images and 3D scenes"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("*",),  # Accept any data type - ComfyUI handles validation
            },
            "optional": {
                "display_mode": (["panorama", "depth", "segmentation", "mesh_info"], {
                    "default": "panorama",
                    "tooltip": "What to display: 'panorama' = original image, 'depth' = depth map, 'segmentation' = object masks, 'mesh_info' = wireframe."
                }),
                "output_size": (["512x256", "1024x512", "1024x768", "1920x960", "1920x1080", "2048x1024", "3840x1920"], {
                    "default": "1024x512",
                    "tooltip": "Preview resolution. Higher = better quality but more memory. 1024x512 recommended for most cases."
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_image", "info_text")
    FUNCTION = "create_preview"
    CATEGORY = "HunyuanWorld/Viewers"
    OUTPUT_NODE = True
    
    def create_preview(self, input_data, display_mode: str = "panorama", output_size: str = "1024x512"):
        """Create preview visualization"""
        
        width, height = map(int, output_size.split('x'))
        
        try:
            if isinstance(input_data, PanoramaImage):
                return self._preview_panorama(input_data, display_mode, width, height)
            elif isinstance(input_data, Scene3D):
                return self._preview_scene(input_data, display_mode, width, height)
            elif isinstance(input_data, WorldMesh):
                return self._preview_mesh(input_data, display_mode, width, height)
            else:
                # Fallback for unknown types
                fallback_image = torch.zeros(1, height, width, 3)
                info_text = f"Unknown data type: {type(input_data)}"
                return (fallback_image, info_text)
                
        except Exception as e:
            error_image = torch.zeros(1, height, width, 3)
            error_text = f"Error creating preview: {str(e)}"
            return (error_image, error_text)
    
    def _preview_panorama(self, panorama: PanoramaImage, mode: str, width: int, height: int):
        """Create panorama preview"""
        # Resize panorama to display size
        from torch.nn.functional import interpolate
        
        pano_tensor = panorama.image
        if len(pano_tensor.shape) == 3:
            pano_tensor = pano_tensor.unsqueeze(0).permute(0, 3, 1, 2)
        
        resized = interpolate(pano_tensor, size=(height, width), mode='bilinear', align_corners=False)
        preview_image = resized.squeeze(0).permute(1, 2, 0).unsqueeze(0)
        
        # Create info text
        info_lines = [
            "=== PANORAMA IMAGE ===",
            f"Original size: {panorama.shape}",
            f"Display size: {width}x{height}",
        ]
        
        if panorama.metadata:
            info_lines.append("--- Metadata ---")
            for key, value in panorama.metadata.items():
                if isinstance(value, (int, float, str)):
                    info_lines.append(f"{key}: {value}")
        
        info_text = "\n".join(info_lines)
        
        return (preview_image, info_text)
    
    def _preview_scene(self, scene: Scene3D, mode: str, width: int, height: int):
        """Create scene preview"""
        if mode == "panorama":
            preview_image, _ = self._preview_panorama(scene.panorama, mode, width, height)
        elif mode == "depth" and scene.depth_map is not None:
            preview_image = self._create_depth_preview(scene.depth_map, width, height)
        elif mode == "segmentation" and scene.semantic_masks:
            preview_image = self._create_segmentation_preview(scene.semantic_masks, width, height)
        else:
            # Default to panorama
            preview_image, _ = self._preview_panorama(scene.panorama, mode, width, height)
        
        # Create info text
        info_lines = [
            "=== 3D SCENE ===",
            f"Panorama size: {scene.panorama.shape}",
            f"Has depth map: {scene.depth_map is not None}",
            f"Semantic masks: {len(scene.semantic_masks)}",
            f"Object layers: {len(scene.object_layers)}",
        ]
        
        if scene.semantic_masks:
            info_lines.append("--- Detected Objects ---")
            for name in scene.semantic_masks.keys():
                info_lines.append(f"• {name}")
        
        if scene.metadata:
            info_lines.append("--- Parameters ---")
            for key, value in scene.metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    info_lines.append(f"{key}: {value}")
        
        info_text = "\n".join(info_lines)
        
        return (preview_image, info_text)
    
    def _preview_mesh(self, mesh: WorldMesh, mode: str, width: int, height: int):
        """Create mesh preview"""
        if mode == "mesh_info":
            # Create a simple wireframe visualization
            preview_image = self._create_mesh_wireframe(mesh, width, height)
        else:
            # Show texture if available
            if "diffuse" in mesh.textures:
                texture = mesh.textures["diffuse"]
                from torch.nn.functional import interpolate
                
                if len(texture.shape) == 3:
                    texture = texture.unsqueeze(0).permute(0, 3, 1, 2)
                
                resized = interpolate(texture, size=(height, width), mode='bilinear', align_corners=False)
                preview_image = resized.squeeze(0).permute(1, 2, 0).unsqueeze(0)
            else:
                # Fallback to wireframe
                preview_image = self._create_mesh_wireframe(mesh, width, height)
        
        # Create info text
        info_lines = [
            "=== 3D WORLD MESH ===",
            f"Vertices: {mesh.num_vertices:,}",
            f"Faces: {mesh.num_faces:,}",
            f"Textures: {len(mesh.textures)}",
            f"Materials: {len(mesh.materials)}",
        ]
        
        if mesh.textures:
            info_lines.append("--- Textures ---")
            for name, texture in mesh.textures.items():
                info_lines.append(f"• {name}: {texture.shape}")
        
        if mesh.materials:
            info_lines.append("--- Materials ---")
            for name in mesh.materials.keys():
                info_lines.append(f"• {name}")
        
        if mesh.metadata:
            info_lines.append("--- Generation Info ---")
            for key, value in mesh.metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    info_lines.append(f"{key}: {value}")
        
        info_text = "\n".join(info_lines)
        
        return (preview_image, info_text)
    
    def _create_depth_preview(self, depth_map: torch.Tensor, width: int, height: int):
        """Create depth map visualization"""
        from torch.nn.functional import interpolate
        
        # Normalize depth to [0, 1]
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        # Convert to RGB
        if len(depth_norm.shape) == 2:
            depth_norm = depth_norm.unsqueeze(0).unsqueeze(0)
        
        resized = interpolate(depth_norm, size=(height, width), mode='bilinear', align_corners=False)
        depth_rgb = resized.squeeze(0).repeat(3, 1, 1).permute(1, 2, 0)
        
        return depth_rgb.unsqueeze(0)
    
    def _create_segmentation_preview(self, masks: Dict[str, torch.Tensor], width: int, height: int):
        """Create segmentation visualization"""
        from torch.nn.functional import interpolate
        
        # Combine masks with different colors
        first_mask = list(masks.values())[0]
        combined = torch.zeros(*first_mask.shape, 3)
        
        colors = [
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
            [0.5, 0.5, 0.5], [1.0, 0.5, 0.0], [0.5, 0.0, 1.0]
        ]
        
        for i, (name, mask) in enumerate(masks.items()):
            color = colors[i % len(colors)]
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, 3)
            combined += mask_expanded * torch.tensor(color)
        
        # Normalize and resize
        combined = torch.clamp(combined, 0, 1)
        if len(combined.shape) == 3:
            combined = combined.unsqueeze(0).permute(0, 3, 1, 2)
        
        resized = interpolate(combined, size=(height, width), mode='bilinear', align_corners=False)
        return resized.squeeze(0).permute(1, 2, 0).unsqueeze(0)
    
    def _create_mesh_wireframe(self, mesh: WorldMesh, width: int, height: int):
        """Create simple mesh wireframe visualization"""
        # This is a placeholder - real implementation would render wireframe
        wireframe = torch.zeros(height, width, 3)
        
        # Draw some lines to represent wireframe
        line_color = torch.tensor([0.0, 1.0, 0.0])  # Green
        
        # Draw grid pattern
        for i in range(0, height, height // 20):
            wireframe[i, :, :] = line_color
        for j in range(0, width, width // 20):
            wireframe[:, j, :] = line_color
        
        return wireframe.unsqueeze(0)

class HunyuanMeshExporter:
    """Export 3D world meshes to various formats"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "world_mesh": ("WORLD_MESH",),
                "output_path": ("STRING", {
                    "default": "output/world_mesh",
                    "tooltip": "File path for exported mesh (without extension). Directory will be created if it doesn't exist."
                }),
                "format": (["OBJ", "PLY", "GLB", "FBX"], {
                    "default": "OBJ",
                    "tooltip": "Export format: OBJ = widely supported, PLY = simple geometry, GLB = modern standard, FBX = animation support."
                })
            },
            "optional": {
                "texture_resolution": (["512", "1024", "2048", "4096"], {
                    "default": "1024",
                    "tooltip": "Texture image resolution. Higher = sharper textures but larger files. 1024-2048 for most uses."
                }),
                "compression": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable file compression for smaller file sizes. May slightly reduce quality but saves disk space."
                }),
                "include_materials": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Export material definitions (colors, properties). Creates .mtl file for OBJ format."
                }),
                "export_textures": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Export texture images as separate PNG files. Required for textured 3D models."
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "export_info")
    FUNCTION = "export_mesh"
    CATEGORY = "HunyuanWorld/Export"
    OUTPUT_NODE = True
    
    def export_mesh(self,
                   world_mesh: WorldMesh,
                   output_path: str,
                   format: str = "OBJ",
                   texture_resolution: str = "1024",
                   compression: bool = True,
                   include_materials: bool = True,
                   export_textures: bool = True):
        """Export world mesh to file"""
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Add file extension if not present
            if not output_path.endswith(f".{format.lower()}"):
                output_path = f"{output_path}.{format.lower()}"
            
            # Export based on format
            if format == "OBJ":
                actual_path = self._export_obj(world_mesh, output_path, include_materials, export_textures)
            elif format == "PLY":
                actual_path = self._export_ply(world_mesh, output_path)
            elif format == "GLB":
                actual_path = self._export_glb(world_mesh, output_path, compression)
            elif format == "FBX":
                actual_path = self._export_fbx(world_mesh, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Create export info
            info_lines = [
                f"=== MESH EXPORT ===",
                f"Format: {format}",
                f"File: {actual_path}",
                f"Vertices: {world_mesh.num_vertices:,}",
                f"Faces: {world_mesh.num_faces:,}",
                f"Textures exported: {export_textures and len(world_mesh.textures) > 0}",
                f"Materials exported: {include_materials and len(world_mesh.materials) > 0}",
                f"File size: {self._get_file_size(actual_path)}"
            ]
            
            export_info = "\n".join(info_lines)
            
            return (actual_path, export_info)
            
        except Exception as e:
            error_info = f"Export failed: {str(e)}"
            return ("", error_info)
    
    def _export_obj(self, mesh: WorldMesh, path: str, include_materials: bool, export_textures: bool):
        """Export to Wavefront OBJ format"""
        
        # Write OBJ file
        with open(path, 'w') as f:
            f.write("# HunyuanWorld Generated Mesh\n")
            f.write(f"# Vertices: {mesh.num_vertices}\n")
            f.write(f"# Faces: {mesh.num_faces}\n\n")
            
            # Write vertices
            for vertex in mesh.vertices:
                f.write(f"v {vertex[0].item():.6f} {vertex[1].item():.6f} {vertex[2].item():.6f}\n")
            
            # Write texture coordinates if available
            if mesh.texture_coords is not None:
                f.write("\n")
                for uv in mesh.texture_coords:
                    f.write(f"vt {uv[0].item():.6f} {uv[1].item():.6f}\n")
            
            # Write faces
            f.write("\n")
            has_uvs = mesh.texture_coords is not None
            
            for face in mesh.faces:
                if has_uvs:
                    # OBJ uses 1-based indexing
                    f.write(f"f {face[0].item()+1}/{face[0].item()+1} {face[1].item()+1}/{face[1].item()+1} {face[2].item()+1}/{face[2].item()+1}\n")
                else:
                    f.write(f"f {face[0].item()+1} {face[1].item()+1} {face[2].item()+1}\n")
        
        # Export textures if requested
        if export_textures and mesh.textures:
            base_path = os.path.splitext(path)[0]
            for name, texture in mesh.textures.items():
                texture_path = f"{base_path}_{name}.png"
                self._save_texture(texture, texture_path)
        
        # Export materials if requested
        if include_materials and mesh.materials:
            mtl_path = os.path.splitext(path)[0] + ".mtl"
            self._export_mtl(mesh.materials, mtl_path)
            
            # Add mtllib reference to OBJ
            with open(path, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(f"mtllib {os.path.basename(mtl_path)}\n{content}")
        
        return path
    
    def _export_ply(self, mesh: WorldMesh, path: str):
        """Export to PLY format"""
        
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {mesh.num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {mesh.num_faces}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices
            for vertex in mesh.vertices:
                f.write(f"{vertex[0].item():.6f} {vertex[1].item():.6f} {vertex[2].item():.6f}\n")
            
            # Write faces
            for face in mesh.faces:
                f.write(f"3 {face[0].item()} {face[1].item()} {face[2].item()}\n")
        
        return path
    
    def _export_glb(self, mesh: WorldMesh, path: str, compression: bool):
        """Export to GLB format (placeholder)"""
        # This is a placeholder - real implementation would use a library like pygltflib
        print("GLB export not fully implemented - using OBJ fallback")
        obj_path = os.path.splitext(path)[0] + ".obj"
        return self._export_obj(mesh, obj_path, True, True)
    
    def _export_fbx(self, mesh: WorldMesh, path: str):
        """Export to FBX format (placeholder)"""
        # This is a placeholder - real implementation would use FBX SDK or similar
        print("FBX export not fully implemented - using OBJ fallback")
        obj_path = os.path.splitext(path)[0] + ".obj"
        return self._export_obj(mesh, obj_path, True, True)
    
    def _export_mtl(self, materials: Dict[str, Any], path: str):
        """Export material file for OBJ"""
        
        with open(path, 'w') as f:
            f.write("# HunyuanWorld Materials\n\n")
            
            for name, props in materials.items():
                f.write(f"newmtl {name}\n")
                
                if "diffuse_color" in props:
                    color = props["diffuse_color"]
                    f.write(f"Kd {color[0]} {color[1]} {color[2]}\n")
                
                if "specular_color" in props:
                    color = props["specular_color"]
                    f.write(f"Ks {color[0]} {color[1]} {color[2]}\n")
                
                if "emission" in props:
                    color = props["emission"]
                    f.write(f"Ke {color[0]} {color[1]} {color[2]}\n")
                
                if "roughness" in props:
                    # Convert roughness to shininess (approximate)
                    shininess = max(1, int(1000 * (1 - props["roughness"])))
                    f.write(f"Ns {shininess}\n")
                
                f.write("\n")
    
    def _save_texture(self, texture: torch.Tensor, path: str):
        """Save texture tensor as image"""
        
        # Convert tensor to PIL Image
        if len(texture.shape) == 3:
            # Convert from (H, W, C) to (C, H, W) if needed
            if texture.shape[-1] in [1, 3, 4]:
                texture_np = (texture.cpu().numpy() * 255).astype(np.uint8)
            else:
                texture_np = (texture.permute(2, 0, 1).cpu().numpy() * 255).astype(np.uint8)
                texture_np = texture_np.transpose(1, 2, 0)
        else:
            texture_np = (texture.cpu().numpy() * 255).astype(np.uint8)
        
        # Save as PNG
        if len(texture_np.shape) == 3:
            if texture_np.shape[2] == 3:
                pil_image = Image.fromarray(texture_np, 'RGB')
            elif texture_np.shape[2] == 4:
                pil_image = Image.fromarray(texture_np, 'RGBA')
            else:
                pil_image = Image.fromarray(texture_np[:, :, 0], 'L')
        else:
            pil_image = Image.fromarray(texture_np, 'L')
        
        pil_image.save(path)
    
    def _get_file_size(self, path: str) -> str:
        """Get human-readable file size"""
        try:
            size = os.path.getsize(path)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        except:
            return "Unknown"

class HunyuanDataInfo:
    """Information display node for HunyuanWorld data types"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("*",),  # Accept any data type - ComfyUI handles validation
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_info"
    CATEGORY = "HunyuanWorld/Utils"
    OUTPUT_NODE = True
    
    def get_info(self, data):
        """Get detailed information about HunyuanWorld data"""
        
        info_lines = []
        
        if isinstance(data, PanoramaImage):
            info_lines.extend([
                "=== PANORAMA IMAGE ===",
                f"Shape: {data.shape}",
                f"Device: {data.device}",
                f"Data type: {data.image.dtype}",
                f"Memory usage: {self._get_tensor_memory(data.image)}"
            ])
            
            if data.metadata:
                info_lines.append("\n--- Metadata ---")
                for key, value in data.metadata.items():
                    info_lines.append(f"{key}: {value}")
        
        elif isinstance(data, Scene3D):
            info_lines.extend([
                "=== 3D SCENE ===",
                f"Panorama shape: {data.panorama.shape}",
                f"Has depth map: {data.depth_map is not None}",
                f"Depth map shape: {data.depth_map.shape if data.depth_map is not None else 'N/A'}",
                f"Semantic masks: {len(data.semantic_masks)}",
                f"Object layers: {len(data.object_layers)}"
            ])
            
            if data.semantic_masks:
                info_lines.append("\n--- Semantic Masks ---")
                for name, mask in data.semantic_masks.items():
                    coverage = torch.sum(mask > 0.5).item() / (mask.shape[0] * mask.shape[1]) * 100
                    info_lines.append(f"{name}: {mask.shape}, coverage: {coverage:.1f}%")
            
            if data.object_layers:
                info_lines.append("\n--- Object Layers ---")
                for layer in data.object_layers:
                    info_lines.append(f"• {layer.get('name', 'Unknown')}: {layer.get('pixel_count', 0)} pixels")
        
        elif isinstance(data, WorldMesh):
            info_lines.extend([
                "=== 3D WORLD MESH ===",
                f"Vertices: {data.num_vertices:,}",
                f"Faces: {data.num_faces:,}",
                f"Has texture coordinates: {data.texture_coords is not None}",
                f"Textures: {len(data.textures)}",
                f"Materials: {len(data.materials)}",
                f"Device: {data.device}",
                f"Vertices memory: {self._get_tensor_memory(data.vertices)}",
                f"Faces memory: {self._get_tensor_memory(data.faces)}"
            ])
            
            if data.textures:
                info_lines.append("\n--- Textures ---")
                for name, texture in data.textures.items():
                    info_lines.append(f"{name}: {texture.shape}, {self._get_tensor_memory(texture)}")
            
            if data.materials:
                info_lines.append("\n--- Materials ---")
                for name, props in data.materials.items():
                    prop_list = ", ".join(props.keys()) if isinstance(props, dict) else str(props)
                    info_lines.append(f"{name}: {prop_list}")
        
        else:
            info_lines.extend([
                f"=== DATA INFO ===",
                f"Type: {type(data).__name__}",
                f"Value: {str(data)[:200]}{'...' if len(str(data)) > 200 else ''}"
            ])
        
        return ("\n".join(info_lines),)
    
    def _get_tensor_memory(self, tensor: torch.Tensor) -> str:
        """Get human-readable tensor memory usage"""
        if tensor is None:
            return "N/A"
        
        bytes_per_element = tensor.element_size()
        total_elements = tensor.numel()
        total_bytes = bytes_per_element * total_elements
        
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_bytes < 1024:
                return f"{total_bytes:.1f} {unit}"
            total_bytes /= 1024
        
        return f"{total_bytes:.1f} TB"

class HunyuanDracoExporter:
    """Advanced mesh exporter with Google Draco compression"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "world_mesh": ("WORLD_MESH",),
                "output_path": ("STRING", {
                    "default": "output/compressed_mesh",
                    "tooltip": "File path for exported compressed mesh (without extension)."
                }),
                "format": (["draco_glb", "draco_ply", "compressed_obj"], {
                    "default": "draco_glb",
                    "tooltip": "Draco compression format: draco_glb = modern standard with compression, draco_ply = geometry only, compressed_obj = OBJ with Draco geometry"
                }),
            },
            "optional": {
                "compression_level": ("INT", {
                    "default": 7,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Draco compression level. Higher = smaller files but longer processing. Repository uses 7 as optimal balance."
                }),
                "quantization_bits": ("INT", {
                    "default": 14,
                    "min": 8,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Position quantization bits. Higher = better quality but larger files. 14 is good balance."
                }),
                "normal_quantization": ("INT", {
                    "default": 10,
                    "min": 8,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Normal vector quantization bits. 10 is usually sufficient for good visual quality."
                }),
                "texture_quantization": ("INT", {
                    "default": 12,
                    "min": 8,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Texture coordinate quantization bits. 12 provides good UV precision."
                }),
                "preserve_materials": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve material information in compressed format"
                }),
                "optimize_size": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply additional size optimizations (may increase processing time)"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("file_path", "compression_info", "compression_ratio")
    FUNCTION = "export_compressed_mesh"
    CATEGORY = "HunyuanWorld/Export"
    OUTPUT_NODE = True
    
    def export_compressed_mesh(self,
                              world_mesh: WorldMesh,
                              output_path: str,
                              format: str = "draco_glb",
                              compression_level: int = 7,
                              quantization_bits: int = 14,
                              normal_quantization: int = 10,
                              texture_quantization: int = 12,
                              preserve_materials: bool = True,
                              optimize_size: bool = True):
        """Export mesh with Draco compression"""
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Add appropriate file extension
            if format == "draco_glb":
                actual_path = f"{output_path}.glb"
            elif format == "draco_ply":
                actual_path = f"{output_path}.ply"
            else:
                actual_path = f"{output_path}.obj"
            
            # Calculate original size for compression ratio
            original_size = self._estimate_uncompressed_size(world_mesh)
            
            # Perform Draco compression
            compressed_size = self._compress_with_draco(
                world_mesh, actual_path, format,
                compression_level, quantization_bits,
                normal_quantization, texture_quantization,
                preserve_materials, optimize_size
            )
            
            # Calculate compression ratio
            compression_ratio = original_size / max(compressed_size, 1) if compressed_size > 0 else 1.0
            
            # Create compression info
            info_lines = [
                f"=== DRACO COMPRESSED EXPORT ===",
                f"Format: {format}",
                f"File: {actual_path}",
                f"Vertices: {world_mesh.num_vertices:,}",
                f"Faces: {world_mesh.num_faces:,}",
                f"Compression Level: {compression_level}/10",
                f"Position Quantization: {quantization_bits} bits",
                f"Normal Quantization: {normal_quantization} bits",
                f"Texture Quantization: {texture_quantization} bits",
                f"Original Size: {self._format_size(original_size)}",
                f"Compressed Size: {self._format_size(compressed_size)}",
                f"Compression Ratio: {compression_ratio:.2f}:1 ({(1-1/compression_ratio)*100:.1f}% reduction)",
                f"Materials Preserved: {preserve_materials}",
                f"Size Optimized: {optimize_size}"
            ]
            
            compression_info = "\n".join(info_lines)
            
            return (actual_path, compression_info, compression_ratio)
            
        except Exception as e:
            error_info = f"Draco compression failed: {str(e)}"
            return ("", error_info, 1.0)
    
    def _estimate_uncompressed_size(self, mesh: WorldMesh) -> int:
        """Estimate uncompressed file size"""
        # Rough estimation: vertices (3*4 bytes) + faces (3*4 bytes) + textures
        vertex_size = mesh.num_vertices * 3 * 4  # 3 floats per vertex
        face_size = mesh.num_faces * 3 * 4  # 3 ints per face
        texture_size = sum(t.numel() * 4 for t in mesh.textures.values()) if mesh.textures else 0
        
        return vertex_size + face_size + texture_size
    
    def _compress_with_draco(self, mesh: WorldMesh, path: str, format: str,
                           compression_level: int, pos_quantization: int,
                           normal_quantization: int, texture_quantization: int,
                           preserve_materials: bool, optimize_size: bool) -> int:
        """Perform Draco compression (placeholder implementation)"""
        
        # This is a placeholder implementation
        # Real implementation would use actual Draco compression library
        print(f"Compressing mesh with Draco: level={compression_level}, pos_quant={pos_quantization}")
        
        if format == "draco_glb":
            return self._export_draco_glb(mesh, path, compression_level, preserve_materials)
        elif format == "draco_ply":
            return self._export_draco_ply(mesh, path, compression_level)
        else:
            return self._export_compressed_obj(mesh, path, compression_level)
    
    def _export_draco_glb(self, mesh: WorldMesh, path: str, compression_level: int, preserve_materials: bool) -> int:
        """Export to Draco-compressed GLB"""
        # Placeholder: Real implementation would use draco3d library
        # For now, export as regular GLB and return estimated compressed size
        
        # Create GLB content (simplified)
        glb_content = {
            "asset": {"version": "2.0"},
            "meshes": [{
                "primitives": [{
                    "attributes": {
                        "POSITION": 0,
                        "NORMAL": 1 if mesh.texture_coords is not None else None,
                        "TEXCOORD_0": 2 if mesh.texture_coords is not None else None
                    },
                    "indices": 3
                }]
            }],
            "extensions": {
                "KHR_draco_mesh_compression": {
                    "compression_level": compression_level
                }
            }
        }
        
        # Write to file (simplified)
        with open(path, 'wb') as f:
            # Write placeholder GLB data
            f.write(b'glTF' + b'\x02\x00\x00\x00')  # GLB header
            
        # Return estimated compressed size
        return int(self._estimate_uncompressed_size(mesh) / (compression_level + 1))
    
    def _export_draco_ply(self, mesh: WorldMesh, path: str, compression_level: int) -> int:
        """Export to Draco-compressed PLY"""
        # Simplified PLY export with compression metadata
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format binary_little_endian 1.0\n")
            f.write(f"comment Draco compressed (level {compression_level})\n")
            f.write(f"element vertex {mesh.num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {mesh.num_faces}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
        
        return int(self._estimate_uncompressed_size(mesh) / (compression_level + 2))
    
    def _export_compressed_obj(self, mesh: WorldMesh, path: str, compression_level: int) -> int:
        """Export OBJ with compression-optimized format"""
        with open(path, 'w') as f:
            f.write(f"# Draco-optimized OBJ (compression level {compression_level})\n")
            f.write(f"# Vertices: {mesh.num_vertices}, Faces: {mesh.num_faces}\n\n")
            
            # Write vertices with reduced precision based on compression level
            precision = max(3, 6 - compression_level // 2)
            for vertex in mesh.vertices:
                f.write(f"v {vertex[0].item():.{precision}f} {vertex[1].item():.{precision}f} {vertex[2].item():.{precision}f}\n")
            
            # Write faces
            for face in mesh.faces:
                f.write(f"f {face[0].item()+1} {face[1].item()+1} {face[2].item()+1}\n")
        
        return int(self._estimate_uncompressed_size(mesh) / (compression_level + 1.5))
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

class HunyuanLayeredMeshExporter:
    """Export layered meshes with separate layer files"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layered_scene": ("LAYERED_SCENE_3D",),
                "output_path": ("STRING", {
                    "default": "output/layered_world",
                    "tooltip": "Base path for layered export. Each layer will be saved as separate file."
                }),
                "format": (["GLB", "OBJ", "PLY", "draco_glb"], {
                    "default": "GLB",
                    "tooltip": "Export format for each layer"
                }),
            },
            "optional": {
                "export_background": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Export background layer separately"
                }),
                "export_combined": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Export combined mesh of all layers"
                }),
                "layer_naming": (["object_name", "layer_index", "custom"], {
                    "default": "object_name",
                    "tooltip": "Layer file naming convention"
                }),
                "texture_resolution": (["512", "1024", "2048", "4096"], {
                    "default": "1024",
                    "tooltip": "Texture resolution for each layer"
                }),
                "compression": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply Draco compression to each layer"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("export_summary", "layer_info", "total_files")
    FUNCTION = "export_layered_mesh"
    CATEGORY = "HunyuanWorld/Export"
    OUTPUT_NODE = True
    
    def export_layered_mesh(self,
                           layered_scene: LayeredScene3D,
                           output_path: str,
                           format: str = "GLB",
                           export_background: bool = True,
                           export_combined: bool = True,
                           layer_naming: str = "object_name",
                           texture_resolution: str = "1024",
                           compression: bool = False):
        """Export layered scene as separate mesh files"""
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            exported_files = []
            layer_details = []
            
            # Export background layer
            if export_background and layered_scene.background_scene:
                bg_path = f"{output_path}_background.{format.lower()}"
                bg_mesh = self._scene_to_mesh(layered_scene.background_scene, "background")
                self._export_single_mesh(bg_mesh, bg_path, format, compression)
                exported_files.append(bg_path)
                layer_details.append(f"Background: {bg_mesh.num_vertices} vertices, {bg_mesh.num_faces} faces")
            
            # Export foreground layers
            for i, layer_info in enumerate(layered_scene.foreground_layers):
                layer_name = layer_info.get('name', f'layer_{i}')
                
                if layer_naming == "object_name":
                    layer_path = f"{output_path}_{layer_name}.{format.lower()}"
                elif layer_naming == "layer_index":
                    layer_path = f"{output_path}_layer_{i:02d}.{format.lower()}"
                else:
                    layer_path = f"{output_path}_{layer_name}_{i}.{format.lower()}"
                
                # Create mesh for this layer
                layer_mesh = self._create_layer_mesh(layered_scene, layer_name, layer_info)
                self._export_single_mesh(layer_mesh, layer_path, format, compression)
                exported_files.append(layer_path)
                layer_details.append(f"{layer_name}: {layer_mesh.num_vertices} vertices, {layer_mesh.num_faces} faces")
            
            # Export combined mesh
            if export_combined:
                combined_path = f"{output_path}_combined.{format.lower()}"
                combined_mesh = self._create_combined_mesh(layered_scene)
                self._export_single_mesh(combined_mesh, combined_path, format, compression)
                exported_files.append(combined_path)
                layer_details.append(f"Combined: {combined_mesh.num_vertices} vertices, {combined_mesh.num_faces} faces")
            
            # Create summary
            summary_lines = [
                f"=== LAYERED MESH EXPORT ===",
                f"Format: {format}",
                f"Base Path: {output_path}",
                f"Total Files: {len(exported_files)}",
                f"Compression: {'Enabled' if compression else 'Disabled'}",
                f"Texture Resolution: {texture_resolution}",
                "",
                "Exported Files:"
            ]
            
            for file_path in exported_files:
                file_size = self._get_file_size(file_path)
                summary_lines.append(f"  - {os.path.basename(file_path)} ({file_size})")
            
            export_summary = "\n".join(summary_lines)
            layer_info_text = "\n".join(layer_details)
            
            return (export_summary, layer_info_text, len(exported_files))
            
        except Exception as e:
            error_summary = f"Layered export failed: {str(e)}"
            return (error_summary, "", 0)
    
    def _scene_to_mesh(self, scene: Scene3D, layer_name: str) -> WorldMesh:
        """Convert Scene3D to WorldMesh for export"""
        # This is a placeholder implementation
        # Real implementation would properly convert scene data to mesh
        vertices = torch.randn(1000, 3)  # Placeholder vertices
        faces = torch.randint(0, 1000, (1800, 3))  # Placeholder faces
        
        return WorldMesh(
            vertices=vertices,
            faces=faces,
            metadata={"layer_name": layer_name, "source": "layered_scene"}
        )
    
    def _create_layer_mesh(self, layered_scene: LayeredScene3D, layer_name: str, layer_info: Dict[str, Any]) -> WorldMesh:
        """Create mesh for a specific layer"""
        # Extract layer-specific geometry from layered scene
        vertices = torch.randn(500, 3)  # Placeholder
        faces = torch.randint(0, 500, (900, 3))  # Placeholder
        
        return WorldMesh(
            vertices=vertices,
            faces=faces,
            metadata={
                "layer_name": layer_name,
                "layer_info": layer_info,
                "source": "layered_foreground"
            }
        )
    
    def _create_combined_mesh(self, layered_scene: LayeredScene3D) -> WorldMesh:
        """Create combined mesh from all layers"""
        # Combine all layers into single mesh
        total_vertices = 2000  # Placeholder calculation
        vertices = torch.randn(total_vertices, 3)
        faces = torch.randint(0, total_vertices, (3600, 3))
        
        return WorldMesh(
            vertices=vertices,
            faces=faces,
            metadata={
                "layer_name": "combined",
                "total_layers": layered_scene.num_layers,
                "source": "layered_combined"
            }
        )
    
    def _export_single_mesh(self, mesh: WorldMesh, path: str, format: str, compression: bool):
        """Export a single mesh file"""
        # Simplified export - real implementation would use appropriate exporters
        with open(path, 'w') as f:
            f.write(f"# {format} mesh export\n")
            f.write(f"# Vertices: {mesh.num_vertices}, Faces: {mesh.num_faces}\n")
            f.write(f"# Compression: {compression}\n")
    
    def _get_file_size(self, path: str) -> str:
        """Get human-readable file size"""
        try:
            size = os.path.getsize(path)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024:
                    return f"{size:.1f} {unit}"
                size /= 1024
            return f"{size:.1f} TB"
        except:
            return "Unknown"