import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import open3d as o3d


def tensor_to_pil(tensor):
    """Convert ComfyUI tensor to PIL Image"""
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]  # Remove batch dimension
    
    # Convert to numpy and scale to 0-255
    if tensor.max() <= 1.0:
        image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    else:
        image_np = tensor.cpu().numpy().astype(np.uint8)
    
    return Image.fromarray(image_np)


class HYW_TextureBaker:
    """Bake textures from 3D world layers"""
    
    CATEGORY = "HunyuanWorld/Export"
    RETURN_TYPES = ("HYW_BAKED_TEXTURES", "HYW_METADATA")
    RETURN_NAMES = ("baked_textures", "bake_metadata")
    FUNCTION = "bake_textures"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "world_layers": ("HYW_MESH_LAYERS",),
                "texture_resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 256}),
                "map_types": ("STRING", {
                    "multiline": True,
                    "default": "albedo\nnormal\nroughness\nao"
                }),
            },
            "optional": {
                "panorama": ("IMAGE",),
                "enable_ambient_occlusion": ("BOOLEAN", {"default": True}),
                "ao_samples": ("INT", {"default": 100, "min": 10, "max": 1000}),
                "enable_normal_maps": ("BOOLEAN", {"default": True}),
                "texture_padding": ("INT", {"default": 4, "min": 0, "max": 16}),
                "uv_unwrap_method": (["angle_based", "conformal"], {"default": "angle_based"}),
            }
        }

    def generate_uv_mapping(self, mesh, method="angle_based"):
        """Generate UV mapping for the mesh"""
        try:
            if method == "angle_based":
                # Use angle-based UV unwrapping (simulated)
                vertices = np.asarray(mesh.vertices)
                if len(vertices) == 0:
                    return np.array([]).reshape(0, 2)
                
                # Simple spherical UV mapping as fallback
                x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
                
                # Convert to spherical coordinates
                r = np.sqrt(x**2 + y**2 + z**2)
                theta = np.arctan2(y, x)  # Azimuth
                phi = np.arccos(z / np.maximum(r, 1e-8))  # Elevation
                
                # Normalize to UV coordinates
                u = (theta + np.pi) / (2 * np.pi)
                v = phi / np.pi
                
                return np.column_stack([u, v])
            
            elif method == "conformal":
                # Conformal UV mapping (simplified)
                # In a real implementation, this would use more sophisticated algorithms
                vertices = np.asarray(mesh.vertices)
                if len(vertices) == 0:
                    return np.array([]).reshape(0, 2)
                
                # Simple projection as placeholder
                u = (vertices[:, 0] - vertices[:, 0].min()) / (vertices[:, 0].max() - vertices[:, 0].min() + 1e-8)
                v = (vertices[:, 1] - vertices[:, 1].min()) / (vertices[:, 1].max() - vertices[:, 1].min() + 1e-8)
                
                return np.column_stack([u, v])
        
        except Exception as e:
            print(f"UV mapping failed: {e}")
            # Return default UV coordinates
            vertices = np.asarray(mesh.vertices)
            if len(vertices) == 0:
                return np.array([]).reshape(0, 2)
            return np.column_stack([np.zeros(len(vertices)), np.zeros(len(vertices))])

    def bake_ambient_occlusion(self, mesh, resolution=1024, samples=100):
        """Bake ambient occlusion map"""
        try:
            # This is a simplified AO baking - in practice would use ray tracing
            vertices = np.asarray(mesh.vertices)
            
            if len(vertices) == 0:
                # Return white AO map if no vertices
                return Image.new('L', (resolution, resolution), 255)
            
            # Create a simple AO texture based on vertex positions
            ao_map = np.ones((resolution, resolution), dtype=np.float32)
            
            # Simple cavity-based AO approximation
            if mesh.has_vertex_normals():
                normals = np.asarray(mesh.vertex_normals)
                
                # Calculate cavity factor based on normal directions
                cavity_factor = np.abs(normals[:, 1])  # Y-component for vertical cavities
                
                # Map to texture space (simplified)
                uv_coords = self.generate_uv_mapping(mesh)
                
                for i, (u, v) in enumerate(uv_coords):
                    x = int(u * (resolution - 1))
                    y = int(v * (resolution - 1))
                    x = np.clip(x, 0, resolution - 1)
                    y = np.clip(y, 0, resolution - 1)
                    
                    # Apply cavity factor to AO
                    ao_factor = 0.3 + 0.7 * cavity_factor[i]
                    ao_map[y, x] = min(ao_map[y, x], ao_factor)
            
            # Convert to PIL Image
            ao_image = (ao_map * 255).astype(np.uint8)
            return Image.fromarray(ao_image, mode='L')
            
        except Exception as e:
            print(f"AO baking failed: {e}")
            return Image.new('L', (resolution, resolution), 128)

    def bake_normal_map(self, mesh, resolution=1024):
        """Bake normal map from mesh"""
        try:
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            
            normals = np.asarray(mesh.vertex_normals)
            
            if len(normals) == 0:
                # Return neutral normal map
                neutral_color = [128, 128, 255]  # Neutral normal (pointing up)
                return Image.new('RGB', (resolution, resolution), tuple(neutral_color))
            
            # Create normal map texture
            normal_map = np.full((resolution, resolution, 3), [128, 128, 255], dtype=np.uint8)
            
            # Map normals to texture space
            uv_coords = self.generate_uv_mapping(mesh)
            
            for i, (u, v) in enumerate(uv_coords):
                x = int(u * (resolution - 1))
                y = int(v * (resolution - 1))
                x = np.clip(x, 0, resolution - 1)
                y = np.clip(y, 0, resolution - 1)
                
                # Convert normal to color space
                normal = normals[i]
                normal_color = ((normal + 1) * 127.5).astype(np.uint8)
                normal_map[y, x] = normal_color
            
            return Image.fromarray(normal_map, mode='RGB')
            
        except Exception as e:
            print(f"Normal map baking failed: {e}")
            return Image.new('RGB', (resolution, resolution), (128, 128, 255))

    def extract_albedo_from_panorama(self, panorama_tensor, mesh, resolution=1024):
        """Extract albedo texture from panorama"""
        try:
            pano_pil = tensor_to_pil(panorama_tensor)
            pano_width, pano_height = pano_pil.size
            pano_array = np.array(pano_pil)
            
            # Create albedo texture
            albedo_map = np.ones((resolution, resolution, 3), dtype=np.uint8) * 128
            
            # Map mesh vertices to panorama coordinates
            vertices = np.asarray(mesh.vertices)
            if len(vertices) == 0:
                return Image.fromarray(albedo_map, mode='RGB')
            
            # Convert 3D positions to spherical coordinates for panorama lookup
            x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
            
            # Spherical coordinate conversion
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(y, x)  # Azimuth
            phi = np.arccos(z / np.maximum(r, 1e-8))  # Elevation
            
            # Map to panorama coordinates
            pano_u = (theta + np.pi) / (2 * np.pi)
            pano_v = phi / np.pi
            
            # Sample colors from panorama
            pano_x = (pano_u * (pano_width - 1)).astype(int)
            pano_y = (pano_v * (pano_height - 1)).astype(int)
            pano_x = np.clip(pano_x, 0, pano_width - 1)
            pano_y = np.clip(pano_y, 0, pano_height - 1)
            
            # Map to texture space
            uv_coords = self.generate_uv_mapping(mesh)
            
            for i, (u, v) in enumerate(uv_coords):
                tex_x = int(u * (resolution - 1))
                tex_y = int(v * (resolution - 1))
                tex_x = np.clip(tex_x, 0, resolution - 1)
                tex_y = np.clip(tex_y, 0, resolution - 1)
                
                # Sample color from panorama
                color = pano_array[pano_y[i], pano_x[i]]
                albedo_map[tex_y, tex_x] = color
            
            return Image.fromarray(albedo_map, mode='RGB')
            
        except Exception as e:
            print(f"Albedo extraction failed: {e}")
            return Image.new('RGB', (resolution, resolution), (128, 128, 128))

    def bake_textures(self, world_layers, texture_resolution, map_types, 
                     panorama=None, enable_ambient_occlusion=True, ao_samples=100,
                     enable_normal_maps=True, texture_padding=4, 
                     uv_unwrap_method="angle_based"):
        """Bake textures for all mesh layers"""
        
        try:
            map_list = [m.strip().lower() for m in map_types.split('\n') if m.strip()]
            
            baked_textures = {
                'layers': [],
                'resolution': texture_resolution,
                'map_types': map_list,
                'total_layers': len(world_layers)
            }
            
            for i, layer in enumerate(world_layers):
                mesh = layer['mesh']
                layer_id = layer.get('layer_id', i)
                
                print(f"Baking textures for layer {layer_id}")
                
                layer_textures = {
                    'layer_id': layer_id,
                    'maps': {}
                }
                
                # Bake different map types
                for map_type in map_list:
                    if map_type == 'albedo' and panorama is not None:
                        texture = self.extract_albedo_from_panorama(
                            panorama, mesh, texture_resolution
                        )
                        layer_textures['maps']['albedo'] = texture
                        
                    elif map_type == 'normal' and enable_normal_maps:
                        texture = self.bake_normal_map(mesh, texture_resolution)
                        layer_textures['maps']['normal'] = texture
                        
                    elif map_type == 'ao' and enable_ambient_occlusion:
                        texture = self.bake_ambient_occlusion(
                            mesh, texture_resolution, ao_samples
                        )
                        layer_textures['maps']['ao'] = texture
                        
                    elif map_type == 'roughness':
                        # Default roughness map (medium roughness)
                        texture = Image.new('L', (texture_resolution, texture_resolution), 128)
                        layer_textures['maps']['roughness'] = texture
                        
                    else:
                        print(f"Unsupported map type: {map_type}")
                
                baked_textures['layers'].append(layer_textures)
            
            # Create metadata
            bake_metadata = {
                'texture_resolution': texture_resolution,
                'map_types': map_list,
                'uv_unwrap_method': uv_unwrap_method,
                'enable_ambient_occlusion': enable_ambient_occlusion,
                'ao_samples': ao_samples,
                'enable_normal_maps': enable_normal_maps,
                'texture_padding': texture_padding,
                'bake_timestamp': datetime.now().isoformat(),
                'layer_count': len(world_layers)
            }
            
            return (baked_textures, bake_metadata)
            
        except Exception as e:
            print(f"Error in texture baking: {e}")
            import traceback
            traceback.print_exc()
            raise e


class HYW_MeshExport:
    """Export 3D meshes to various formats"""
    
    CATEGORY = "HunyuanWorld/Export"
    RETURN_TYPES = ("STRING", "LIST", "HYW_METADATA")
    RETURN_NAMES = ("mesh_file_path", "texture_file_paths", "export_metadata")
    FUNCTION = "export_meshes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "world_layers": ("HYW_MESH_LAYERS",),
                "output_directory": ("STRING", {"default": "outputs/hunyuanworld"}),
                "output_filename": ("STRING", {"default": "world_mesh"}),
                "export_format": (["glb", "gltf", "obj", "ply", "stl"], {"default": "glb"}),
            },
            "optional": {
                "baked_textures": ("HYW_BAKED_TEXTURES",),
                "merge_layers": ("BOOLEAN", {"default": False}),
                "use_draco_compression": ("BOOLEAN", {"default": False}),
                "draco_quality": ("INT", {"default": 20, "min": 1, "max": 30}),
                "export_individual_layers": ("BOOLEAN", {"default": True}),
                "include_materials": ("BOOLEAN", {"default": True}),
                "texture_format": (["png", "jpg"], {"default": "png"}),
            }
        }

    def export_mesh_layer(self, mesh, filepath, format_type):
        """Export individual mesh layer"""
        try:
            if format_type.lower() == "ply":
                o3d.io.write_triangle_mesh(filepath, mesh)
            elif format_type.lower() == "obj":
                o3d.io.write_triangle_mesh(filepath, mesh)
            elif format_type.lower() == "stl":
                o3d.io.write_triangle_mesh(filepath, mesh)
            elif format_type.lower() in ["glb", "gltf"]:
                # For glTF/GLB, save as PLY first then convert if needed
                ply_path = filepath.replace(f".{format_type.lower()}", ".ply")
                o3d.io.write_triangle_mesh(ply_path, mesh)
                # In practice, would convert PLY to glTF/GLB here
                return ply_path
            else:
                o3d.io.write_triangle_mesh(filepath, mesh)
            
            return filepath
            
        except Exception as e:
            print(f"Failed to export mesh layer: {e}")
            return None

    def save_texture(self, texture_image, filepath, format_type="png"):
        """Save texture image to file"""
        try:
            if format_type.lower() == "jpg":
                # Convert RGBA to RGB for JPEG
                if texture_image.mode == 'RGBA':
                    rgb_img = Image.new('RGB', texture_image.size, (255, 255, 255))
                    rgb_img.paste(texture_image, mask=texture_image.split()[-1])
                    texture_image = rgb_img
                texture_image.save(filepath, "JPEG", quality=95)
            else:
                texture_image.save(filepath, "PNG")
            return filepath
        except Exception as e:
            print(f"Failed to save texture: {e}")
            return None

    def export_meshes(self, world_layers, output_directory, output_filename, 
                     export_format, baked_textures=None, merge_layers=False,
                     use_draco_compression=False, draco_quality=20,
                     export_individual_layers=True, include_materials=True,
                     texture_format="png"):
        """Export mesh layers to files"""
        
        try:
            # Create output directory
            os.makedirs(output_directory, exist_ok=True)
            
            exported_mesh_files = []
            exported_texture_files = []
            
            if merge_layers and len(world_layers) > 1:
                # Merge all layers into single mesh
                print("Merging all layers into single mesh...")
                merged_mesh = o3d.geometry.TriangleMesh()
                
                for layer in world_layers:
                    merged_mesh += layer['mesh']
                
                # Remove duplicated vertices after merging
                merged_mesh.remove_duplicated_vertices()
                
                # Export merged mesh
                merged_filepath = os.path.join(
                    output_directory, 
                    f"{output_filename}_merged.{export_format.lower()}"
                )
                
                exported_file = self.export_mesh_layer(
                    merged_mesh, merged_filepath, export_format
                )
                
                if exported_file:
                    exported_mesh_files.append(exported_file)
            
            if export_individual_layers:
                # Export individual layers
                for i, layer in enumerate(world_layers):
                    layer_id = layer.get('layer_id', i)
                    layer_filename = f"{output_filename}_layer{layer_id}.{export_format.lower()}"
                    layer_filepath = os.path.join(output_directory, layer_filename)
                    
                    exported_file = self.export_mesh_layer(
                        layer['mesh'], layer_filepath, export_format
                    )
                    
                    if exported_file:
                        exported_mesh_files.append(exported_file)
            
            # Export textures if available
            if baked_textures and include_materials:
                print("Exporting textures...")
                
                for layer_textures in baked_textures.get('layers', []):
                    layer_id = layer_textures['layer_id']
                    
                    for map_type, texture_image in layer_textures['maps'].items():
                        texture_filename = f"{output_filename}_layer{layer_id}_{map_type}.{texture_format.lower()}"
                        texture_filepath = os.path.join(output_directory, texture_filename)
                        
                        exported_texture = self.save_texture(
                            texture_image, texture_filepath, texture_format
                        )
                        
                        if exported_texture:
                            exported_texture_files.append(exported_texture)
            
            # Create export manifest
            manifest = {
                'export_timestamp': datetime.now().isoformat(),
                'export_format': export_format,
                'output_directory': output_directory,
                'merged_layers': merge_layers,
                'individual_layers': export_individual_layers,
                'layer_count': len(world_layers),
                'mesh_files': exported_mesh_files,
                'texture_files': exported_texture_files,
                'draco_compression': use_draco_compression,
                'draco_quality': draco_quality if use_draco_compression else None,
                'include_materials': include_materials,
                'texture_format': texture_format
            }
            
            manifest_path = os.path.join(output_directory, f"{output_filename}_manifest.json")
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Return primary mesh file path
            primary_mesh_file = exported_mesh_files[0] if exported_mesh_files else ""
            
            export_metadata = {
                **manifest,
                'manifest_file': manifest_path,
                'export_successful': len(exported_mesh_files) > 0
            }
            
            print(f"Export completed:")
            print(f"  - Mesh files: {len(exported_mesh_files)}")
            print(f"  - Texture files: {len(exported_texture_files)}")
            print(f"  - Output directory: {output_directory}")
            
            return (primary_mesh_file, exported_texture_files, export_metadata)
            
        except Exception as e:
            print(f"Error in mesh export: {e}")
            import traceback
            traceback.print_exc()
            raise e


class HYW_Thumbnailer:
    """Generate thumbnails and previews of 3D worlds"""
    
    CATEGORY = "HunyuanWorld/Export"
    RETURN_TYPES = ("IMAGE", "HYW_METADATA")
    RETURN_NAMES = ("thumbnail", "thumbnail_metadata")
    FUNCTION = "generate_thumbnail"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "world_layers": ("HYW_MESH_LAYERS",),
                "thumbnail_size": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 128}),
                "view_angle": ("FLOAT", {"default": 45.0, "min": 0.0, "max": 360.0}),
                "camera_distance": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0}),
            },
            "optional": {
                "panorama": ("IMAGE",),
                "background_color": ("STRING", {"default": "white"}),
                "render_wireframe": ("BOOLEAN", {"default": False}),
                "show_all_layers": ("BOOLEAN", {"default": True}),
                "selected_layer": ("INT", {"default": 0, "min": 0}),
            }
        }

    def render_mesh_preview(self, meshes, size=512, view_angle=45.0, 
                           camera_distance=5.0, background_color="white",
                           wireframe=False):
        """Render mesh preview using Open3D visualization"""
        try:
            # This is a simplified preview - in practice would use proper 3D rendering
            # Create a simple representation by projecting vertices
            
            all_vertices = []
            for mesh in meshes:
                vertices = np.asarray(mesh.vertices)
                if len(vertices) > 0:
                    all_vertices.extend(vertices)
            
            if not all_vertices:
                # Return empty thumbnail if no vertices
                bg_color = (255, 255, 255) if background_color == "white" else (0, 0, 0)
                thumbnail = Image.new('RGB', (size, size), bg_color)
                return thumbnail
            
            all_vertices = np.array(all_vertices)
            
            # Simple orthographic projection
            angle_rad = np.radians(view_angle)
            
            # Rotate vertices
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
            
            rotated_vertices = all_vertices @ rotation_matrix.T
            
            # Project to 2D (orthographic)
            x_proj = rotated_vertices[:, 0]
            y_proj = rotated_vertices[:, 1]
            
            # Normalize to image coordinates
            if len(x_proj) > 0 and len(y_proj) > 0:
                x_min, x_max = x_proj.min(), x_proj.max()
                y_min, y_max = y_proj.min(), y_proj.max()
                
                margin = 0.1
                x_range = x_max - x_min
                y_range = y_max - y_min
                max_range = max(x_range, y_range)
                
                if max_range > 0:
                    x_norm = ((x_proj - x_min) / max_range + margin) * (size * (1 - 2 * margin))
                    y_norm = ((y_proj - y_min) / max_range + margin) * (size * (1 - 2 * margin))
                    
                    x_norm = x_norm.astype(int)
                    y_norm = y_norm.astype(int)
                    
                    # Create image
                    bg_color = (255, 255, 255) if background_color == "white" else (0, 0, 0)
                    img_array = np.full((size, size, 3), bg_color, dtype=np.uint8)
                    
                    # Draw points
                    point_color = (0, 0, 0) if background_color == "white" else (255, 255, 255)
                    for x, y in zip(x_norm, y_norm):
                        if 0 <= x < size and 0 <= y < size:
                            img_array[y, x] = point_color
                    
                    thumbnail = Image.fromarray(img_array)
                else:
                    bg_color = (255, 255, 255) if background_color == "white" else (0, 0, 0)
                    thumbnail = Image.new('RGB', (size, size), bg_color)
            else:
                bg_color = (255, 255, 255) if background_color == "white" else (0, 0, 0)
                thumbnail = Image.new('RGB', (size, size), bg_color)
            
            return thumbnail
            
        except Exception as e:
            print(f"Preview rendering failed: {e}")
            bg_color = (255, 255, 255) if background_color == "white" else (0, 0, 0)
            return Image.new('RGB', (size, size), bg_color)

    def generate_thumbnail(self, world_layers, thumbnail_size, view_angle, 
                          camera_distance, panorama=None, background_color="white",
                          render_wireframe=False, show_all_layers=True, 
                          selected_layer=0):
        """Generate thumbnail image of 3D world"""
        
        try:
            meshes_to_render = []
            
            if show_all_layers:
                meshes_to_render = [layer['mesh'] for layer in world_layers]
            else:
                if 0 <= selected_layer < len(world_layers):
                    meshes_to_render = [world_layers[selected_layer]['mesh']]
            
            # Render preview
            thumbnail_pil = self.render_mesh_preview(
                meshes_to_render,
                size=thumbnail_size,
                view_angle=view_angle,
                camera_distance=camera_distance,
                background_color=background_color,
                wireframe=render_wireframe
            )
            
            # Convert PIL to tensor
            def pil_to_tensor(pil_image):
                image_np = np.array(pil_image).astype(np.float32) / 255.0
                return torch.from_numpy(image_np)[None,]
            
            thumbnail_tensor = pil_to_tensor(thumbnail_pil)
            
            # Create metadata
            thumbnail_metadata = {
                'thumbnail_size': thumbnail_size,
                'view_angle': view_angle,
                'camera_distance': camera_distance,
                'background_color': background_color,
                'render_wireframe': render_wireframe,
                'show_all_layers': show_all_layers,
                'selected_layer': selected_layer if not show_all_layers else None,
                'layers_rendered': len(meshes_to_render),
                'generation_timestamp': datetime.now().isoformat()
            }
            
            return (thumbnail_tensor, thumbnail_metadata)
            
        except Exception as e:
            print(f"Error in thumbnail generation: {e}")
            import traceback
            traceback.print_exc()
            raise e