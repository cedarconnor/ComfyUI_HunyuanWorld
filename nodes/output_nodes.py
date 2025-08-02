import os
import json
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from PIL import Image

from ..core.data_types import PanoramaImage, Scene3D, WorldMesh

class HunyuanViewer:
    """Preview node for panoramic images and 3D scenes"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_data": ("*",),  # Accept any HunyuanWorld data type
            },
            "optional": {
                "display_mode": (["panorama", "depth", "segmentation", "mesh_info"], {
                    "default": "panorama"
                }),
                "output_size": (["512x256", "1024x512", "2048x1024"], {
                    "default": "1024x512"
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
                    "default": "output/world_mesh"
                }),
                "format": (["OBJ", "PLY", "GLB", "FBX"], {
                    "default": "OBJ"
                })
            },
            "optional": {
                "texture_resolution": (["512", "1024", "2048", "4096"], {
                    "default": "1024"
                }),
                "compression": ("BOOLEAN", {"default": True}),
                "include_materials": ("BOOLEAN", {"default": True}),
                "export_textures": ("BOOLEAN", {"default": True})
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
                "data": ("*",),  # Accept any data type
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