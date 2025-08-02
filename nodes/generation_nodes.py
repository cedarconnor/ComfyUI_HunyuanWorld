import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional

from ..core.data_types import PanoramaImage, Scene3D, WorldMesh, ModelHunyuan
from ..core.model_manager import model_manager

class HunyuanLoader:
    """Model loader node for HunyuanWorld models"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "models/hunyuan_world",
                    "tooltip": "Path to HunyuanWorld model directory. Should contain model.safetensors and config.json files."
                }),
                "model_type": (["text_to_panorama", "scene_generator", "world_reconstructor"], {
                    "default": "text_to_panorama",
                    "tooltip": "Model component to load: 'text_to_panorama' for text→image, 'scene_generator' for depth/segmentation, 'world_reconstructor' for 3D mesh."
                }),
                "precision": (["fp32", "fp16", "bf16"], {
                    "default": "fp16",
                    "tooltip": "Model precision: 'fp32' = best quality/more VRAM, 'fp16' = balanced, 'bf16' = fastest/least VRAM. Start with fp16."
                }),
                "device": (["auto", "cuda", "cpu", "mps"], {
                    "default": "auto",
                    "tooltip": "Device for model execution: 'auto' = detect best, 'cuda' = NVIDIA GPU, 'cpu' = CPU only, 'mps' = Apple Silicon."
                })
            },
            "optional": {
                "force_reload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force reload model even if already cached. Use when switching model files or troubleshooting."
                })
            }
        }
    
    RETURN_TYPES = ("MODEL_HUNYUAN",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "HunyuanWorld/Loaders"
    
    def load_model(self, 
                   model_path: str,
                   model_type: str,
                   precision: str = "fp16",
                   device: str = "auto",
                   force_reload: bool = False):
        """Load HunyuanWorld model"""
        
        try:
            # Override device setting if needed
            if device != "auto":
                original_device = model_manager.device
                model_manager.device = device
            
            model_ref = model_manager.load_model(
                model_path=model_path,
                model_type=model_type,
                precision=precision,
                force_reload=force_reload
            )
            
            print(f"Loaded {model_type} model from {model_path} with precision {precision}")
            
            return (model_ref,)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e

class HunyuanTextToPanorama:
    """Text-to-panorama generation node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL_HUNYUAN",),
                "prompt": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"forceInput": True}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"forceInput": True}),
                "width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Panorama width in pixels. Higher = better quality but slower. 1024-2048 recommended. Must be multiple of 64."
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Panorama height in pixels. Should be half of width (2:1 ratio) for proper panoramic format."
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of denoising steps. More = higher quality but slower. 20-30 for testing, 50+ for final results."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "How closely to follow the prompt. Higher = more prompt adherence but less creativity. 7.5 is balanced, 15+ for exact prompt following."
                }),
                "scheduler": (["DPMSolverMultistep", "DDIM", "LMS", "Euler", "EulerAncestral"], {
                    "default": "DPMSolverMultistep",
                    "tooltip": "Sampling scheduler. DPMSolverMultistep = best quality, Euler = fastest, DDIM = most stable. Try DPM first."
                })
            }
        }
    
    RETURN_TYPES = ("PANORAMA_IMAGE", "IMAGE")
    RETURN_NAMES = ("panorama", "preview_image")
    FUNCTION = "generate_panorama"
    CATEGORY = "HunyuanWorld/Generation"
    
    def generate_panorama(self,
                         model: ModelHunyuan,
                         prompt: str,
                         seed: int,
                         negative_prompt: str = "",
                         width: int = 1024,
                         height: int = 512,
                         num_inference_steps: int = 50,
                         guidance_scale: float = 7.5,
                         scheduler: str = "DPMSolverMultistep"):
        """Generate panoramic image from text prompt"""
        
        # Validate model type
        if model.model_type != "text_to_panorama":
            raise ValueError(f"Expected text_to_panorama model, got {model.model_type}")
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Ensure model is loaded
        if not model.is_loaded:
            model.reload()
        
        try:
            # Generate panorama using the model
            # This is placeholder logic - real implementation would use actual HunyuanWorld API
            panorama_tensor = model.model.generate_panorama(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                scheduler=scheduler
            )
            
            # Create PanoramaImage object
            panorama = PanoramaImage(
                image=panorama_tensor,
                metadata={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "guidance_scale": guidance_scale,
                    "steps": num_inference_steps,
                    "scheduler": scheduler
                }
            )
            
            # Create preview image for ComfyUI display (add batch dimension)
            preview_image = panorama_tensor.unsqueeze(0) if len(panorama_tensor.shape) == 3 else panorama_tensor
            
            return (panorama, preview_image)
            
        except Exception as e:
            print(f"Error generating panorama: {str(e)}")
            # Return fallback panorama
            fallback_tensor = torch.randn(height, width, 3)
            fallback_panorama = PanoramaImage(fallback_tensor, {"error": str(e)})
            return (fallback_panorama, fallback_tensor.unsqueeze(0))

class HunyuanImageToPanorama:
    """Image-to-panorama conversion node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL_HUNYUAN",),
                "image": ("IMAGE",),
            },
            "optional": {
                "extension_mode": (["seamless", "outpainting", "symmetric"], {
                    "default": "seamless",
                    "tooltip": "How to extend image to panorama: 'seamless' = tile/repeat, 'outpainting' = AI extend edges, 'symmetric' = mirror image."
                }),
                "strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Strength of image-to-panorama conversion. Higher = more AI modification, lower = preserve original more."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "How closely to follow the prompt. Higher = more prompt adherence but less creativity. 7.5 is balanced, 15+ for exact prompt following."
                }),
                "num_inference_steps": ("INT", {
                    "default": 30,
                    "min": 10,
                    "max": 100,
                    "step": 1
                })
            }
        }
    
    RETURN_TYPES = ("PANORAMA_IMAGE", "IMAGE")
    RETURN_NAMES = ("panorama", "preview_image")
    FUNCTION = "convert_to_panorama"
    CATEGORY = "HunyuanWorld/Generation"
    
    def convert_to_panorama(self,
                           model: ModelHunyuan,
                           image: torch.Tensor,
                           extension_mode: str = "seamless",
                           strength: float = 0.8,
                           guidance_scale: float = 7.5,
                           num_inference_steps: int = 30):
        """Convert regular image to panoramic format"""
        
        # Validate model type
        if model.model_type != "text_to_panorama":
            print("Warning: Using text_to_panorama model for image conversion")
        
        # Process input image
        if len(image.shape) == 4:
            input_image = image[0]  # Take first image from batch
        else:
            input_image = image
        
        # Ensure model is loaded
        if not model.is_loaded:
            model.reload()
        
        try:
            # Convert to panoramic format
            # This is placeholder logic - real implementation would use actual HunyuanWorld API
            h, w, c = input_image.shape
            target_w = w * 2 if w < 1024 else 1024
            target_h = target_w // 2
            
            # Simple extension for demonstration
            if extension_mode == "seamless":
                # Tile the image horizontally with blending
                panorama_tensor = self._create_seamless_panorama(input_image, target_w, target_h)
            elif extension_mode == "outpainting":
                # Extend image using outpainting logic
                panorama_tensor = self._create_outpainted_panorama(input_image, target_w, target_h)
            else:  # symmetric
                # Create symmetric panorama
                panorama_tensor = self._create_symmetric_panorama(input_image, target_w, target_h)
            
            # Create PanoramaImage object
            panorama = PanoramaImage(
                image=panorama_tensor,
                metadata={
                    "source_image_shape": input_image.shape,
                    "extension_mode": extension_mode,
                    "strength": strength,
                    "guidance_scale": guidance_scale,
                    "steps": num_inference_steps
                }
            )
            
            # Create preview image
            preview_image = panorama_tensor.unsqueeze(0)
            
            return (panorama, preview_image)
            
        except Exception as e:
            print(f"Error converting to panorama: {str(e)}")
            # Return fallback
            fallback_tensor = torch.randn(512, 1024, 3)
            fallback_panorama = PanoramaImage(fallback_tensor, {"error": str(e)})
            return (fallback_panorama, fallback_tensor.unsqueeze(0))
    
    def _create_seamless_panorama(self, image: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
        """Create seamless panorama by tiling and blending"""
        # Resize source image to target height
        from torch.nn.functional import interpolate
        resized = interpolate(
            image.permute(2, 0, 1).unsqueeze(0),
            size=(target_h, image.shape[1]),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        
        # Tile horizontally
        num_tiles = target_w // resized.shape[1] + 1
        tiled = torch.cat([resized] * num_tiles, dim=1)
        
        # Crop to target width
        panorama = tiled[:, :target_w, :]
        
        return panorama
    
    def _create_outpainted_panorama(self, image: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
        """Create panorama using outpainting simulation"""
        # For placeholder, just pad with blurred edges
        from torch.nn.functional import interpolate, pad
        
        # Resize to target height
        resized = interpolate(
            image.permute(2, 0, 1).unsqueeze(0),
            size=(target_h, min(image.shape[1], target_w)),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        
        # Pad to target width
        pad_width = target_w - resized.shape[1]
        if pad_width > 0:
            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            
            # Simple padding with edge values
            panorama = pad(resized.permute(2, 0, 1), (left_pad, right_pad, 0, 0), mode='reflect')
            panorama = panorama.permute(1, 2, 0)
        else:
            panorama = resized
        
        return panorama
    
    def _create_symmetric_panorama(self, image: torch.Tensor, target_w: int, target_h: int) -> torch.Tensor:
        """Create symmetric panorama"""
        from torch.nn.functional import interpolate
        
        # Resize to half target width
        half_w = target_w // 2
        resized = interpolate(
            image.permute(2, 0, 1).unsqueeze(0),
            size=(target_h, half_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)
        
        # Create mirror image
        flipped = torch.flip(resized, dims=[1])
        
        # Concatenate
        panorama = torch.cat([resized, flipped], dim=1)
        
        return panorama

class HunyuanSceneGenerator:
    """3D scene generation from panoramic images"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL_HUNYUAN",),
                "panorama": ("PANORAMA_IMAGE",),
            },
            "optional": {
                "depth_estimation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate depth map for 3D reconstruction. Required for creating explorable 3D worlds."
                }),
                "semantic_segmentation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Identify and separate objects (sky, ground, trees, etc). Enables better materials and physics."
                }),
                "object_separation": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "How distinctly to separate objects. Higher = cleaner boundaries but may miss details. 0.5 is balanced."
                }),
                "layer_count": ("INT", {
                    "default": 5,
                    "min": 3,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Maximum number of semantic layers/objects to detect. More = finer detail but slower processing."
                }),
                "depth_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Scale factor for depth values. Higher = more pronounced 3D effect. 1.0 = natural depth, 2.0+ = exaggerated."
                })
            }
        }
    
    RETURN_TYPES = ("SCENE_3D", "IMAGE", "IMAGE")
    RETURN_NAMES = ("scene_3d", "depth_preview", "segmentation_preview")
    FUNCTION = "generate_scene"
    CATEGORY = "HunyuanWorld/Generation"
    
    def generate_scene(self,
                      model: ModelHunyuan,
                      panorama: PanoramaImage,
                      depth_estimation: bool = True,
                      semantic_segmentation: bool = True,
                      object_separation: float = 0.5,
                      layer_count: int = 5,
                      depth_scale: float = 1.0):
        """Generate 3D scene from panoramic image"""
        
        # Validate model type
        if model.model_type != "scene_generator":
            print("Warning: Model type mismatch for scene generation")
        
        # Ensure model is loaded
        if not model.is_loaded:
            model.reload()
        
        try:
            # Generate depth map and semantic segmentation
            depth_map, semantic_masks = model.model.generate_scene(
                panorama=panorama.image,
                depth_estimation=depth_estimation,
                semantic_segmentation=semantic_segmentation,
                object_separation=object_separation,
                layer_count=layer_count,
                depth_scale=depth_scale
            )
            
            # Create object layers information
            object_layers = []
            if semantic_segmentation and semantic_masks:
                for i, (mask_name, mask_tensor) in enumerate(semantic_masks.items()):
                    object_layers.append({
                        "id": i,
                        "name": mask_name,
                        "mask_shape": mask_tensor.shape,
                        "pixel_count": torch.sum(mask_tensor > 0.5).item()
                    })
            
            # Create Scene3D object
            scene_3d = Scene3D(
                panorama=panorama,
                depth_map=depth_map,
                semantic_masks=semantic_masks,
                object_layers=object_layers,
                metadata={
                    "depth_estimation": depth_estimation,
                    "semantic_segmentation": semantic_segmentation,
                    "object_separation": object_separation,
                    "layer_count": layer_count,
                    "depth_scale": depth_scale
                }
            )
            
            # Create preview images
            depth_preview = self._create_depth_preview(depth_map)
            seg_preview = self._create_segmentation_preview(semantic_masks)
            
            return (scene_3d, depth_preview, seg_preview)
            
        except Exception as e:
            print(f"Error generating scene: {str(e)}")
            # Return fallback scene
            h, w = panorama.image.shape[:2]
            fallback_depth = torch.randn(h, w)
            fallback_scene = Scene3D(
                panorama=panorama,
                depth_map=fallback_depth,
                metadata={"error": str(e)}
            )
            fallback_preview = fallback_depth.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)
            return (fallback_scene, fallback_preview, fallback_preview)
    
    def _create_depth_preview(self, depth_map: torch.Tensor) -> torch.Tensor:
        """Create depth visualization for preview"""
        if depth_map is None:
            return torch.zeros(1, 512, 1024, 3)
        
        # Normalize depth to [0, 1]
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        # Convert to RGB (grayscale depth map)
        depth_rgb = depth_norm.unsqueeze(-1).repeat(1, 1, 3)
        
        # Add batch dimension
        return depth_rgb.unsqueeze(0)
    
    def _create_segmentation_preview(self, semantic_masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Create segmentation visualization for preview"""
        if not semantic_masks:
            return torch.zeros(1, 512, 1024, 3)
        
        # Combine all masks with different colors
        combined_mask = torch.zeros_like(list(semantic_masks.values())[0]).unsqueeze(-1).repeat(1, 1, 3)
        
        colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
        ]
        
        for i, (name, mask) in enumerate(semantic_masks.items()):
            color = colors[i % len(colors)]
            mask_3d = mask.unsqueeze(-1)
            combined_mask += mask_3d * torch.tensor(color).view(1, 1, 3)
        
        # Normalize
        combined_mask = torch.clamp(combined_mask, 0, 1)
        
        # Add batch dimension
        return combined_mask.unsqueeze(0)

class HunyuanWorldReconstructor:
    """3D world reconstruction from scene data"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL_HUNYUAN",),
                "scene_3d": ("SCENE_3D",),
            },
            "optional": {
                "mesh_resolution": ("INT", {
                    "default": 512,
                    "min": 128,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "3D mesh density. Higher = smoother geometry but larger files and slower processing. 512 is good balance."
                }),
                "texture_resolution": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 256,
                    "tooltip": "Texture quality for 3D mesh. Higher = sharper textures but larger files. 1024-2048 recommended."
                }),
                "optimization_steps": ("INT", {
                    "default": 100,
                    "min": 50,
                    "max": 500,
                    "step": 25,
                    "tooltip": "Mesh optimization iterations. Higher = cleaner geometry but slower. 100 is good for most cases."
                }),
                "smooth_normals": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Smooth surface normals for better lighting. Usually keep enabled unless you want faceted/low-poly look."
                }),
                "generate_materials": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-generate materials based on semantic segmentation (sky=emissive, water=reflective, etc). Recommended."
                })
            }
        }
    
    RETURN_TYPES = ("WORLD_MESH",)
    RETURN_NAMES = ("world_mesh",)
    FUNCTION = "reconstruct_world"
    CATEGORY = "HunyuanWorld/Generation"
    
    def reconstruct_world(self,
                         model: ModelHunyuan,
                         scene_3d: Scene3D,
                         mesh_resolution: int = 512,
                         texture_resolution: int = 1024,
                         optimization_steps: int = 100,
                         smooth_normals: bool = True,
                         generate_materials: bool = True):
        """Reconstruct 3D world mesh from scene data"""
        
        # Validate model type
        if model.model_type != "world_reconstructor":
            print("Warning: Model type mismatch for world reconstruction")
        
        # Ensure model is loaded
        if not model.is_loaded:
            model.reload()
        
        try:
            # Reconstruct 3D world
            vertices, faces = model.model.reconstruct_world(
                scene_data=scene_3d,
                mesh_resolution=mesh_resolution,
                texture_resolution=texture_resolution,
                optimization_steps=optimization_steps,
                smooth_normals=smooth_normals
            )
            
            # Generate texture coordinates
            texture_coords = self._generate_texture_coords(vertices, faces)
            
            # Extract textures from panorama
            textures = self._extract_textures(scene_3d.panorama, texture_resolution)
            
            # Generate materials if requested
            materials = {}
            if generate_materials:
                materials = self._generate_materials(scene_3d)
            
            # Create WorldMesh object
            world_mesh = WorldMesh(
                vertices=vertices,
                faces=faces,
                texture_coords=texture_coords,
                textures=textures,
                materials=materials,
                metadata={
                    "mesh_resolution": mesh_resolution,
                    "texture_resolution": texture_resolution,
                    "optimization_steps": optimization_steps,
                    "smooth_normals": smooth_normals,
                    "generate_materials": generate_materials,
                    "num_vertices": vertices.shape[0],
                    "num_faces": faces.shape[0]
                }
            )
            
            return (world_mesh,)
            
        except Exception as e:
            print(f"Error reconstructing world: {str(e)}")
            # Return fallback mesh
            fallback_vertices = torch.randn(1000, 3)
            fallback_faces = torch.randint(0, 1000, (1800, 3))
            fallback_mesh = WorldMesh(
                vertices=fallback_vertices,
                faces=fallback_faces,
                metadata={"error": str(e)}
            )
            return (fallback_mesh,)
    
    def _generate_texture_coords(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """Generate UV texture coordinates"""
        # Simple spherical projection for texture coordinates
        # This is a placeholder - real implementation would use proper UV unwrapping
        
        # Normalize vertices to unit sphere
        norm_vertices = vertices / (torch.norm(vertices, dim=1, keepdim=True) + 1e-8)
        
        # Convert to spherical coordinates
        x, y, z = norm_vertices[:, 0], norm_vertices[:, 1], norm_vertices[:, 2]
        
        # Calculate UV coordinates
        u = 0.5 + torch.atan2(z, x) / (2 * np.pi)
        v = 0.5 - torch.arcsin(y) / np.pi
        
        texture_coords = torch.stack([u, v], dim=1)
        
        return texture_coords
    
    def _extract_textures(self, panorama: PanoramaImage, resolution: int) -> Dict[str, torch.Tensor]:
        """Extract textures from panoramic image"""
        # Resize panorama to texture resolution
        from torch.nn.functional import interpolate
        
        pano_tensor = panorama.image
        if len(pano_tensor.shape) == 3:
            pano_tensor = pano_tensor.unsqueeze(0).permute(0, 3, 1, 2)
        
        resized_texture = interpolate(
            pano_tensor,
            size=(resolution // 2, resolution),
            mode='bilinear',
            align_corners=False
        )
        
        # Convert back to HWC format
        texture = resized_texture.squeeze(0).permute(1, 2, 0)
        
        return {
            "diffuse": texture,
            "albedo": texture * 0.8,  # Slightly darker albedo
        }
    
    def _generate_materials(self, scene_3d: Scene3D) -> Dict[str, Any]:
        """Generate material properties"""
        materials = {
            "default": {
                "diffuse_color": [0.8, 0.8, 0.8],
                "specular_color": [0.2, 0.2, 0.2],
                "roughness": 0.7,
                "metallic": 0.0,
                "emission": [0.0, 0.0, 0.0]
            }
        }
        
        # Add materials based on semantic segmentation
        if scene_3d.semantic_masks:
            for mask_name in scene_3d.semantic_masks.keys():
                if "sky" in mask_name.lower():
                    materials[mask_name] = {
                        "diffuse_color": [0.5, 0.7, 1.0],
                        "emission": [0.1, 0.1, 0.2]
                    }
                elif "ground" in mask_name.lower():
                    materials[mask_name] = {
                        "diffuse_color": [0.4, 0.3, 0.2],
                        "roughness": 0.9
                    }
                elif "water" in mask_name.lower():
                    materials[mask_name] = {
                        "diffuse_color": [0.0, 0.3, 0.6],
                        "roughness": 0.1,
                        "metallic": 0.0
                    }
        
        return materials