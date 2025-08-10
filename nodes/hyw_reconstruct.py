import os
import tempfile
import torch
import numpy as np
from PIL import Image
import open3d as o3d
from typing import List, Dict, Tuple, Optional


def pil_to_tensor(pil_image):
    """Convert PIL Image to ComfyUI tensor format"""
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    if len(image_np.shape) == 3:
        image_tensor = torch.from_numpy(image_np)[None,]  # Add batch dimension
    else:
        image_tensor = torch.from_numpy(image_np)[None, None,]  # Add batch and channel dimension
    return image_tensor


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


class HYW_WorldReconstructor:
    """HunyuanWorld 3D World Reconstruction node for ComfyUI"""
    
    CATEGORY = "HunyuanWorld/Reconstruct"
    RETURN_TYPES = ("HYW_MESH_LAYERS", "HYW_METADATA")
    RETURN_NAMES = ("world_layers", "reconstruction_metadata")
    FUNCTION = "reconstruct_world"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hyw_runtime": ("HYW_RUNTIME", {
                    "tooltip": "HunyuanWorld runtime with 3D reconstruction pipeline loaded. Must have LayerDecomposition and WorldComposer components."
                }),
                "panorama": ("IMAGE", {
                    "tooltip": "Input 360° panorama image for 3D world reconstruction. Higher resolution panoramas produce more detailed 3D geometry."
                }),
                "classes": ("STRING", {
                    "default": "outdoor",
                    "tooltip": "Scene classification for reconstruction algorithm. 'outdoor': landscapes/exteriors. 'indoor': interior spaces. Affects depth estimation."
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 2**31-1,
                    "tooltip": "Random seed for reproducible 3D reconstruction. Same seed produces consistent geometry from the same panorama."
                }),
                "target_size": ("INT", {
                    "default": 3840, "min": 1024, "max": 8192, "step": 256,
                    "tooltip": "Target processing resolution. Higher values = more detail but longer processing time and more memory usage. 3840 is standard 4K."
                }),
            },
            "optional": {
                "labels_fg1": ("STRING", {
                    "multiline": True,
                    "default": "tree, building, car",
                    "tooltip": "Comma-separated labels for primary foreground objects to extract into separate layers. Examples: tree, building, car, rock, monument."
                }),
                "labels_fg2": ("STRING", {
                    "multiline": True,
                    "default": "person, object, furniture",
                    "tooltip": "Comma-separated labels for secondary foreground objects. Usually smaller/detailed items: person, furniture, sign, vehicle details."
                }),
                "quality": (["preview", "standard", "high"], {
                    "default": "standard",
                    "tooltip": "Reconstruction quality preset. Preview: fast/low-poly. Standard: balanced quality/speed. High: maximum detail but slower."
                }),
                "max_triangles": ("INT", {
                    "default": 200000, "min": 10000, "max": 1000000,
                    "tooltip": "Maximum triangles per mesh layer before simplification. Higher values = more detail but larger file sizes and slower processing."
                }),
                "semantic_strength": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Strength of semantic segmentation for object separation. 0.0=geometry only, 1.0=strong semantic separation. 0.5 balances both."
                }),
                "enable_super_resolution": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply super-resolution to enhance detail during reconstruction. Improves quality but increases processing time."
                }),
                "filter_mask": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply mask filtering to clean up segmentation boundaries. Recommended for cleaner layer separation."
                }),
            }
        }

    def prepare_panorama_file(self, panorama_tensor, output_dir):
        """Save panorama tensor to temporary file for processing"""
        pano_pil = tensor_to_pil(panorama_tensor)
        pano_path = os.path.join(output_dir, "input_panorama.png")
        pano_pil.save(pano_path)
        return pano_path

    def reconstruct_world(self, hyw_runtime, panorama, classes, seed, target_size,
                         labels_fg1="", labels_fg2="", quality="standard", 
                         max_triangles=200000, semantic_strength=0.5,
                         enable_super_resolution=True, filter_mask=True):
        """Reconstruct 3D world from panorama"""
        
        try:
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save panorama to file
                pano_path = self.prepare_panorama_file(panorama, temp_dir)
                
                # Parse labels
                fg1_labels = [label.strip() for label in labels_fg1.split(',') if label.strip()]
                fg2_labels = [label.strip() for label in labels_fg2.split(',') if label.strip()]
                
                print(f"Reconstructing world with:")
                print(f"  - Classes: {classes}")
                print(f"  - FG1 Labels: {fg1_labels}")
                print(f"  - FG2 Labels: {fg2_labels}")
                print(f"  - Target size: {target_size}")
                print(f"  - Quality: {quality}")
                
                # Perform world reconstruction using runtime
                world_layers, metadata = hyw_runtime.reconstruct_world(
                    panorama_path=pano_path,
                    labels_fg1=fg1_labels,
                    labels_fg2=fg2_labels,
                    classes=classes,
                    seed=seed,
                    target_size=target_size,
                    output_dir=temp_dir
                )
                
                # Process mesh layers and add quality settings
                processed_layers = []
                for i, layer_info in enumerate(world_layers):
                    mesh = layer_info['mesh']
                    
                    # Apply quality-based mesh simplification
                    if quality == "preview":
                        target_triangles = min(max_triangles // 4, 50000)
                    elif quality == "standard":
                        target_triangles = max_triangles // 2
                    else:  # high quality
                        target_triangles = max_triangles
                    
                    # Simplify mesh if needed
                    current_triangles = len(mesh.triangles)
                    if current_triangles > target_triangles:
                        print(f"Simplifying layer {i} mesh from {current_triangles} to {target_triangles} triangles")
                        mesh = mesh.simplify_quadric_decimation(target_triangles)
                    
                    # Create processed layer info
                    processed_layer = {
                        'mesh': mesh,
                        'layer_id': i,
                        'triangle_count': len(mesh.triangles),
                        'vertex_count': len(mesh.vertices),
                        'original_info': layer_info
                    }
                    processed_layers.append(processed_layer)
                
                # Enhanced metadata
                enhanced_metadata = {
                    **metadata,
                    'quality': quality,
                    'max_triangles': max_triangles,
                    'semantic_strength': semantic_strength,
                    'enable_super_resolution': enable_super_resolution,
                    'filter_mask': filter_mask,
                    'layer_stats': [
                        {
                            'layer_id': layer['layer_id'],
                            'triangles': layer['triangle_count'],
                            'vertices': layer['vertex_count']
                        }
                        for layer in processed_layers
                    ],
                    'total_triangles': sum(layer['triangle_count'] for layer in processed_layers),
                    'total_vertices': sum(layer['vertex_count'] for layer in processed_layers)
                }
                
                print(f"Reconstruction complete:")
                print(f"  - Generated {len(processed_layers)} layers")
                print(f"  - Total triangles: {enhanced_metadata['total_triangles']}")
                print(f"  - Total vertices: {enhanced_metadata['total_vertices']}")
                
                return (processed_layers, enhanced_metadata)
                
        except Exception as e:
            print(f"Error in world reconstruction: {e}")
            import traceback
            traceback.print_exc()
            raise e


class HYW_MeshProcessor:
    """Process and refine reconstructed mesh layers"""
    
    CATEGORY = "HunyuanWorld/Reconstruct"
    RETURN_TYPES = ("HYW_MESH_LAYERS", "HYW_METADATA")
    RETURN_NAMES = ("processed_layers", "process_metadata")
    FUNCTION = "process_meshes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "world_layers": ("HYW_MESH_LAYERS", {
                    "tooltip": "Mesh layers from HYW_WorldReconstructor to process and refine. Each layer is processed independently."
                }),
                "operation": (["smooth", "decimate", "repair", "merge", "separate"], {
                    "default": "smooth",
                    "tooltip": "Processing operation: smooth=reduce noise, decimate=reduce triangles, repair=fix topology, merge=combine vertices, separate=clean components."
                }),
            },
            "optional": {
                "smoothing_iterations": ("INT", {
                    "default": 5, "min": 1, "max": 50,
                    "tooltip": "Number of Laplacian smoothing iterations for 'smooth' operation. More iterations = smoother but may lose detail. 3-10 typical."
                }),
                "decimation_ratio": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 0.9, "step": 0.1,
                    "tooltip": "Fraction of triangles to keep in 'decimate' operation. 0.5=half triangles, 0.1=very aggressive reduction, 0.9=mild reduction."
                }),
                "merge_distance": ("FLOAT", {
                    "default": 0.01, "min": 0.001, "max": 0.1, "step": 0.001,
                    "tooltip": "Distance threshold for merging nearby vertices in 'merge' operation. Smaller values=less merging, larger=more aggressive."
                }),
                "repair_holes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fill holes and fix degenerate triangles during 'repair' operation. Improves mesh manifold properties."
                }),
                "remove_duplicates": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove duplicate vertices and triangles during processing. Generally recommended for cleaner geometry."
                }),
                "filter_small_components": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove small disconnected mesh components. Helps clean up noise and floating geometry artifacts."
                }),
                "min_component_size": ("INT", {
                    "default": 100, "min": 10, "max": 10000,
                    "tooltip": "Minimum triangle count for mesh components to keep. Smaller components are removed. 100-1000 typical range."
                }),
            }
        }

    def process_meshes(self, world_layers, operation, smoothing_iterations=5,
                      decimation_ratio=0.5, merge_distance=0.01, repair_holes=True,
                      remove_duplicates=True, filter_small_components=True,
                      min_component_size=100):
        """Process mesh layers with various operations"""
        
        try:
            processed_layers = []
            
            for i, layer in enumerate(world_layers):
                mesh = layer['mesh'].copy()
                original_triangles = len(mesh.triangles)
                original_vertices = len(mesh.vertices)
                
                print(f"Processing layer {i} with operation: {operation}")
                
                if operation == "smooth":
                    # Apply Laplacian smoothing
                    mesh = mesh.filter_smooth_laplacian(
                        number_of_iterations=smoothing_iterations
                    )
                    
                elif operation == "decimate":
                    # Quadric decimation
                    target_triangles = int(original_triangles * decimation_ratio)
                    mesh = mesh.simplify_quadric_decimation(target_triangles)
                    
                elif operation == "repair":
                    # Repair mesh
                    if remove_duplicates:
                        mesh.remove_duplicated_vertices()
                        mesh.remove_duplicated_triangles()
                    
                    if repair_holes:
                        mesh.remove_degenerate_triangles()
                        mesh.remove_unreferenced_vertices()
                    
                    # Remove small components if requested
                    if filter_small_components:
                        triangle_clusters, cluster_n_triangles, cluster_area = (
                            mesh.cluster_connected_triangles()
                        )
                        triangles_to_remove = cluster_n_triangles < min_component_size
                        mesh.remove_triangles_by_mask(triangles_to_remove)
                        mesh.remove_unreferenced_vertices()
                    
                elif operation == "merge":
                    # Merge nearby vertices
                    mesh.merge_close_vertices(merge_distance)
                    mesh.remove_duplicated_triangles()
                    
                elif operation == "separate":
                    # This would separate into multiple meshes - for now just clean
                    mesh.remove_duplicated_vertices()
                    mesh.remove_duplicated_triangles()
                
                # Update layer info
                processed_layer = {
                    **layer,
                    'mesh': mesh,
                    'triangle_count': len(mesh.triangles),
                    'vertex_count': len(mesh.vertices),
                    'processing_applied': operation,
                    'original_triangle_count': original_triangles,
                    'original_vertex_count': original_vertices
                }
                
                processed_layers.append(processed_layer)
                
                print(f"Layer {i} processed: {original_triangles} -> {len(mesh.triangles)} triangles")
            
            # Create processing metadata
            process_metadata = {
                'operation': operation,
                'parameters': {
                    'smoothing_iterations': smoothing_iterations,
                    'decimation_ratio': decimation_ratio,
                    'merge_distance': merge_distance,
                    'repair_holes': repair_holes,
                    'remove_duplicates': remove_duplicates,
                    'filter_small_components': filter_small_components,
                    'min_component_size': min_component_size
                },
                'layer_processing_stats': [
                    {
                        'layer_id': layer['layer_id'],
                        'original_triangles': layer['original_triangle_count'],
                        'processed_triangles': layer['triangle_count'],
                        'triangle_reduction': (layer['original_triangle_count'] - layer['triangle_count']) / layer['original_triangle_count'] if layer['original_triangle_count'] > 0 else 0
                    }
                    for layer in processed_layers
                ],
                'total_original_triangles': sum(layer['original_triangle_count'] for layer in processed_layers),
                'total_processed_triangles': sum(layer['triangle_count'] for layer in processed_layers)
            }
            
            return (processed_layers, process_metadata)
            
        except Exception as e:
            print(f"Error in mesh processing: {e}")
            import traceback
            traceback.print_exc()
            raise e


class HYW_MeshAnalyzer:
    """Analyze mesh properties and quality"""
    
    CATEGORY = "HunyuanWorld/Utils"
    RETURN_TYPES = ("HYW_METADATA",)
    RETURN_NAMES = ("analysis_results",)
    FUNCTION = "analyze_meshes"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "world_layers": ("HYW_MESH_LAYERS", {
                    "tooltip": "Mesh layers to analyze for geometry statistics, quality metrics, and structural properties."
                }),
            },
            "optional": {
                "detailed_analysis": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Compute detailed geometry statistics including bounding boxes, surface areas, and spatial metrics."
                }),
                "check_manifold": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Check mesh manifold properties: edge/vertex manifold, watertight, orientable. Important for 3D printing/export."
                }),
                "compute_quality_metrics": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Calculate triangle quality metrics like area distribution and geometry health indicators."
                }),
            }
        }

    def analyze_meshes(self, world_layers, detailed_analysis=True, 
                      check_manifold=True, compute_quality_metrics=True):
        """Analyze mesh layers and return detailed statistics"""
        
        try:
            analysis_results = {
                'layer_count': len(world_layers),
                'layers': []
            }
            
            total_vertices = 0
            total_triangles = 0
            
            for i, layer in enumerate(world_layers):
                mesh = layer['mesh']
                layer_analysis = {
                    'layer_id': layer.get('layer_id', i),
                    'vertex_count': len(mesh.vertices),
                    'triangle_count': len(mesh.triangles),
                    'has_vertex_colors': mesh.has_vertex_colors(),
                    'has_vertex_normals': mesh.has_vertex_normals(),
                    'has_triangle_normals': mesh.has_triangle_normals(),
                }
                
                total_vertices += layer_analysis['vertex_count']
                total_triangles += layer_analysis['triangle_count']
                
                if detailed_analysis and len(mesh.vertices) > 0:
                    # Compute bounding box
                    bbox = mesh.get_axis_aligned_bounding_box()
                    layer_analysis['bounding_box'] = {
                        'min': bbox.min_bound.tolist(),
                        'max': bbox.max_bound.tolist(),
                        'extent': bbox.get_extent().tolist(),
                        'volume': bbox.volume()
                    }
                    
                    # Surface area
                    layer_analysis['surface_area'] = mesh.get_surface_area()
                
                if check_manifold:
                    # Check mesh manifold properties
                    layer_analysis['is_edge_manifold'] = mesh.is_edge_manifold()
                    layer_analysis['is_vertex_manifold'] = mesh.is_vertex_manifold()
                    layer_analysis['is_watertight'] = mesh.is_watertight()
                    layer_analysis['is_orientable'] = mesh.is_orientable()
                
                if compute_quality_metrics and len(mesh.triangles) > 0:
                    # Compute quality metrics
                    edge_lengths = np.asarray(mesh.compute_triangle_areas())
                    if len(edge_lengths) > 0:
                        layer_analysis['triangle_areas'] = {
                            'min': float(edge_lengths.min()),
                            'max': float(edge_lengths.max()),
                            'mean': float(edge_lengths.mean()),
                            'std': float(edge_lengths.std())
                        }
                
                analysis_results['layers'].append(layer_analysis)
            
            # Overall statistics
            analysis_results['totals'] = {
                'total_vertices': total_vertices,
                'total_triangles': total_triangles,
                'average_vertices_per_layer': total_vertices / len(world_layers) if world_layers else 0,
                'average_triangles_per_layer': total_triangles / len(world_layers) if world_layers else 0
            }
            
            # Quality assessment
            if world_layers:
                manifold_layers = sum(1 for layer in analysis_results['layers'] 
                                    if layer.get('is_edge_manifold', False) and layer.get('is_vertex_manifold', False))
                watertight_layers = sum(1 for layer in analysis_results['layers'] 
                                      if layer.get('is_watertight', False))
                
                analysis_results['quality_summary'] = {
                    'manifold_ratio': manifold_layers / len(world_layers),
                    'watertight_ratio': watertight_layers / len(world_layers),
                    'overall_quality': 'good' if manifold_layers > len(world_layers) * 0.8 else 'fair' if manifold_layers > len(world_layers) * 0.5 else 'poor'
                }
            
            return (analysis_results,)
            
        except Exception as e:
            print(f"Error in mesh analysis: {e}")
            import traceback
            traceback.print_exc()
            raise e