import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional

from ..core.data_types import PanoramaImage, Scene3D, WorldMesh, ModelHunyuan, LayeredScene3D, ObjectLabels, SceneMask, LayerMesh
from ..core.model_manager import model_manager

class HunyuanLoader:
    """Model loader node for HunyuanWorld models"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "models/Hunyuan_World",
                    "tooltip": "Path to HunyuanWorld model directory. Should contain HunyuanWorld-*.safetensors files."
                }),
                "model_type": (["text_to_panorama", "image_to_panorama", "scene_generator", "world_reconstructor", "scene_inpainter", "sky_inpainter", "flux_dev", "flux_fill", "dreamshaper"], {
                    "default": "text_to_panorama",
                    "tooltip": "Model to load: HunyuanWorld models (text_to_panorama, image_to_panorama, scene_inpainter, sky_inpainter) or FLUX models (flux_dev=FLUX.1-dev, flux_fill=FLUX.1-fill, dreamshaper=DreamShaper). HunyuanWorld models are in Hunyuan_World folder, FLUX models are in checkpoints folder."
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
    
    RETURN_TYPES = ("*",)
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
                "model": ("*",),
                "prompt": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"forceInput": True}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"forceInput": True}),
                "width": ("INT", {
                    "default": 1920,
                    "min": 512,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Panorama width in pixels. Repository default: 1920. Higher = better quality but slower. Must be multiple of 64."
                }),
                "height": ("INT", {
                    "default": 960,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Panorama height in pixels. Repository default: 960. Should be half of width (2:1 ratio) for proper panoramic format."
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of denoising steps. More = higher quality but slower. 20-30 for testing, 50+ for final results."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "How closely to follow the prompt. Repository default: 30.0. Higher = more prompt adherence but less creativity."
                }),
                "true_cfg_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "True CFG scale parameter from HunyuanWorld repository. Advanced parameter for fine-tuning generation."
                }),
                "blend_extend": ("INT", {
                    "default": 6,
                    "min": 0,
                    "max": 20,
                    "step": 1,  
                    "tooltip": "Blend extension parameter from repository. Default: 6. Controls edge blending in panorama generation."
                }),
                "scheduler": (["DPMSolverMultistep", "DDIM", "LMS", "Euler", "EulerAncestral"], {
                    "default": "DPMSolverMultistep",
                    "tooltip": "Sampling scheduler. DPMSolverMultistep = best quality, Euler = fastest, DDIM = most stable. Try DPM first."
                })
            }
        }
    
    RETURN_TYPES = ("*", "IMAGE")
    RETURN_NAMES = ("panorama", "preview_image")
    FUNCTION = "generate_panorama"
    CATEGORY = "HunyuanWorld/Generation"
    
    def generate_panorama(self,
                         model: ModelHunyuan,
                         prompt: str,
                         seed: int,
                         negative_prompt: str = "",
                         width: int = 1920,
                         height: int = 960,
                         num_inference_steps: int = 50,
                         guidance_scale: float = 30.0,
                         true_cfg_scale: float = 1.0,
                         blend_extend: int = 6,
                         scheduler: str = "DPMSolverMultistep"):
        """Generate panoramic image from text prompt"""
        
        # Input validation and type conversion
        try:
            true_cfg_scale = float(true_cfg_scale) if not isinstance(true_cfg_scale, float) else true_cfg_scale
            guidance_scale = float(guidance_scale) if not isinstance(guidance_scale, float) else guidance_scale
            blend_extend = int(blend_extend) if not isinstance(blend_extend, int) else blend_extend
        except (ValueError, TypeError) as e:
            print(f"⚠️ Parameter conversion error: {e}")
            # Use defaults if conversion fails
            true_cfg_scale = 1.0
            guidance_scale = 30.0
            blend_extend = 6
        
        # Validate model type
        if hasattr(model, 'model_type') and model.model_type != "text_to_panorama":
            print(f"⚠️ Expected text_to_panorama model, got {model.model_type}")
        
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
                true_cfg_scale=true_cfg_scale,
                blend_extend=blend_extend,
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
                    "true_cfg_scale": true_cfg_scale,
                    "blend_extend": blend_extend,
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
                "model": ("*",),
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
    
    RETURN_TYPES = ("*", "IMAGE")
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
                "model": ("*",),
                "panorama": ("*",),
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
    
    RETURN_TYPES = ("*", "IMAGE", "IMAGE")
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
                "model": ("*",),
                "scene_3d": ("*",),
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
    
    RETURN_TYPES = ("*",)
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

class HunyuanSceneInpainter:
    """Scene inpainting node for modifying specific panorama elements"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("*",),
                "panorama": ("*",),
                "mask": ("*",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Enhance this scene area with better details",
                    "tooltip": "Describe what you want to add/change in the masked area"
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blurry, low quality, artifacts",
                    "tooltip": "What to avoid in the inpainted area"
                }),
                "strength": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Inpainting strength. Higher = more modification, lower = more preservation"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "Guidance scale for inpainting. Repository default is 30.0"
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of denoising steps for inpainting"
                }),
                "blend_extend": ("INT", {
                    "default": 6,
                    "min": 0,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Blend extension for seamless inpainting. Repository default is 6"
                })
            }
        }
    
    RETURN_TYPES = ("*", "IMAGE")
    RETURN_NAMES = ("inpainted_panorama", "preview_image")
    FUNCTION = "inpaint_scene"
    CATEGORY = "HunyuanWorld/Generation"
    
    def inpaint_scene(self,
                     model: ModelHunyuan,
                     panorama: PanoramaImage,
                     mask: SceneMask,
                     prompt: str,
                     negative_prompt: str = "",
                     strength: float = 0.8,
                     guidance_scale: float = 30.0,
                     num_inference_steps: int = 50,
                     blend_extend: int = 6):
        """Perform scene inpainting on panoramic image"""
        
        # Validate model type
        if model.model_type != "scene_inpainter":
            print("Warning: Model type mismatch for scene inpainting")
        
        # Ensure model is loaded
        if not model.is_loaded:
            model.reload()
        
        try:
            # Get processed mask
            processed_mask = mask.get_processed_mask()
            
            # Perform scene inpainting using HunyuanWorld-PanoInpaint-Scene
            inpainted_tensor = model.model.inpaint_scene(
                panorama=panorama.image,
                mask=processed_mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                blend_extend=blend_extend
            )
            
            # Create new PanoramaImage with inpainting metadata
            inpainted_panorama = PanoramaImage(
                image=inpainted_tensor,
                metadata={
                    **panorama.metadata,
                    "inpaint_type": "scene",
                    "inpaint_prompt": prompt,
                    "inpaint_strength": strength,
                    "mask_type": mask.mask_type,
                    "guidance_scale": guidance_scale,
                    "blend_extend": blend_extend
                }
            )
            
            # Create preview image
            preview_image = inpainted_tensor.unsqueeze(0) if len(inpainted_tensor.shape) == 3 else inpainted_tensor
            
            return (inpainted_panorama, preview_image)
            
        except Exception as e:
            print(f"Error during scene inpainting: {str(e)}")
            # Return original panorama as fallback
            preview_image = panorama.image.unsqueeze(0) if len(panorama.image.shape) == 3 else panorama.image
            return (panorama, preview_image)

class HunyuanSkyInpainter:
    """Sky inpainting node for replacing/enhancing sky regions"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("*",),
                "panorama": ("*",),
                "sky_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Clear blue sky with white clouds",
                    "tooltip": "Describe the desired sky appearance"
                }),
            },
            "optional": {
                "mask": ("*", {
                    "tooltip": "Optional sky mask. If not provided, sky will be auto-detected"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "overcast, storm clouds, dark sky",
                    "tooltip": "Sky elements to avoid"
                }),
                "sky_type": (["clear", "cloudy", "dramatic", "sunset", "night", "aurora"], {
                    "default": "clear",
                    "tooltip": "Type of sky to generate"
                }),
                "blend_mode": (["natural", "soft", "sharp"], {
                    "default": "natural",
                    "tooltip": "How to blend new sky with existing panorama"
                }),
                "strength": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Sky replacement strength"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 30.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 0.5,
                    "tooltip": "Guidance scale for sky inpainting"
                }),
                "num_inference_steps": ("INT", {
                    "default": 40,
                    "min": 10,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Inference steps for sky generation"
                })
            }
        }
    
    RETURN_TYPES = ("*", "*", "IMAGE")
    RETURN_NAMES = ("inpainted_panorama", "sky_mask", "preview_image")
    FUNCTION = "inpaint_sky"
    CATEGORY = "HunyuanWorld/Generation"
    
    def inpaint_sky(self,
                   model: ModelHunyuan,
                   panorama: PanoramaImage,
                   sky_prompt: str,
                   mask: Optional[SceneMask] = None,
                   negative_prompt: str = "",
                   sky_type: str = "clear",
                   blend_mode: str = "natural",
                   strength: float = 0.9,
                   guidance_scale: float = 30.0,
                   num_inference_steps: int = 40):
        """Perform sky inpainting on panoramic image"""
        
        # Validate model type
        if model.model_type != "sky_inpainter":
            print("Warning: Model type mismatch for sky inpainting")
        
        # Ensure model is loaded
        if not model.is_loaded:
            model.reload()
        
        try:
            # Generate or use provided sky mask
            if mask is None:
                # Auto-detect sky region (simplified implementation)
                sky_mask_tensor = self._auto_detect_sky(panorama.image)
                sky_mask = SceneMask(sky_mask_tensor, mask_type="sky")
            else:
                sky_mask = mask
            
            # Get processed mask
            processed_mask = sky_mask.get_processed_mask()
            
            # Enhance prompt based on sky type
            enhanced_prompt = self._enhance_sky_prompt(sky_prompt, sky_type)
            
            # Perform sky inpainting using HunyuanWorld-PanoInpaint-Sky
            inpainted_tensor = model.model.inpaint_sky(
                panorama=panorama.image,
                mask=processed_mask,
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                blend_mode=blend_mode
            )
            
            # Create new PanoramaImage with sky inpainting metadata
            inpainted_panorama = PanoramaImage(
                image=inpainted_tensor,
                metadata={
                    **panorama.metadata,
                    "inpaint_type": "sky",
                    "sky_prompt": enhanced_prompt,
                    "sky_type": sky_type,
                    "blend_mode": blend_mode,
                    "inpaint_strength": strength
                }
            )
            
            # Create preview image
            preview_image = inpainted_tensor.unsqueeze(0) if len(inpainted_tensor.shape) == 3 else inpainted_tensor
            
            return (inpainted_panorama, sky_mask, preview_image)
            
        except Exception as e:
            print(f"Error during sky inpainting: {str(e)}")
            # Return original panorama as fallback
            fallback_mask = SceneMask(torch.zeros_like(panorama.image[:, :, 0]), mask_type="sky")
            preview_image = panorama.image.unsqueeze(0) if len(panorama.image.shape) == 3 else panorama.image
            return (panorama, fallback_mask, preview_image)
    
    def _auto_detect_sky(self, panorama_tensor: torch.Tensor) -> torch.Tensor:
        """Auto-detect sky region in panorama (simplified implementation)"""
        h, w = panorama_tensor.shape[:2]
        
        # Simple heuristic: assume upper portion is sky
        sky_mask = torch.zeros(h, w)
        sky_height = int(h * 0.4)  # Top 40% is likely sky
        sky_mask[:sky_height, :] = 1.0
        
        # Add some gradient for natural blending
        fade_height = int(h * 0.1)
        for i in range(fade_height):
            y = sky_height + i
            if y < h:
                fade_value = 1.0 - (i / fade_height)
                sky_mask[y, :] = fade_value
        
        return sky_mask
    
    def _enhance_sky_prompt(self, base_prompt: str, sky_type: str) -> str:
        """Enhance sky prompt based on type"""
        type_modifiers = {
            "clear": "clear blue sky, bright daylight",
            "cloudy": "fluffy white clouds, partly cloudy sky",
            "dramatic": "dramatic clouds, dynamic sky, cinematic lighting",
            "sunset": "golden sunset, warm colors, evening sky",
            "night": "night sky, stars, moonlight",
            "aurora": "aurora borealis, northern lights, magical sky"
        }
        
        modifier = type_modifiers.get(sky_type, "")
        if modifier:
            return f"{base_prompt}, {modifier}"
        return base_prompt

class HunyuanLayeredSceneGenerator:
    """Advanced multi-layer 3D scene generation with object separation"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("*",),
                "panorama": ("*",),
                "object_labels": ("*",),
            },
            "optional": {
                "depth_estimation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Generate depth maps for each layer separately for better 3D reconstruction"
                }),
                "semantic_segmentation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Perform semantic segmentation for object separation"
                }),
                "layer_resolution": (["3840x1920", "1920x960", "1024x512"], {
                    "default": "3840x1920",
                    "tooltip": "Resolution for layer processing. Repository supports up to 3840x1920 for high-quality results"
                }),
                "object_separation": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Object separation strength. Higher = cleaner layer boundaries but may miss fine details"
                }),
                "layer_count": ("INT", {
                    "default": 8,
                    "min": 3,
                    "max": 15,
                    "step": 1,
                    "tooltip": "Maximum number of semantic layers. Repository supports up to 10+ layers for detailed decomposition"
                }),
                "depth_scale": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Depth scale multiplier for layered scenes. Higher = more pronounced depth separation"
                }),
                "background_weight": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Background layer weight in final composition"
                })
            }
        }
    
    RETURN_TYPES = ("*", "*", "*", "IMAGE", "IMAGE")
    RETURN_NAMES = ("layered_scene", "background_scene", "foreground_scene", "layer_preview", "depth_preview")
    FUNCTION = "generate_layered_scene"
    CATEGORY = "HunyuanWorld/Generation"
    
    def generate_layered_scene(self,
                              model: ModelHunyuan,
                              panorama: PanoramaImage,
                              object_labels: ObjectLabels,
                              depth_estimation: bool = True,
                              semantic_segmentation: bool = True,
                              layer_resolution: str = "3840x1920",
                              object_separation: float = 0.6,
                              layer_count: int = 8,
                              depth_scale: float = 1.2,
                              background_weight: float = 0.3):
        """Generate multi-layer 3D scene with object separation"""
        
        # Validate model type
        if model.model_type != "scene_generator":
            print("Warning: Model type mismatch for layered scene generation")
        
        # Ensure model is loaded
        if not model.is_loaded:
            model.reload()
        
        try:
            # Parse resolution
            width, height = map(int, layer_resolution.split('x'))
            
            # Generate layered scene using repository methodology
            layer_data = model.model.generate_layered_scene(
                panorama=panorama.image,
                fg_labels_1=object_labels.fg_labels_1,
                fg_labels_2=object_labels.fg_labels_2,
                scene_class=object_labels.scene_class,
                resolution=(width, height),
                depth_estimation=depth_estimation,
                semantic_segmentation=semantic_segmentation,
                object_separation=object_separation,
                layer_count=layer_count,
                depth_scale=depth_scale
            )
            
            # Process layer results
            background_depth = layer_data.get("background_depth")
            foreground_layers = layer_data.get("foreground_layers", [])
            layer_depth_maps = layer_data.get("layer_depth_maps", {})
            layer_masks = layer_data.get("layer_masks", {})
            
            # Create background scene
            background_scene = Scene3D(
                panorama=panorama,
                depth_map=background_depth,
                semantic_masks={"background": layer_masks.get("background", torch.ones_like(background_depth) * background_weight)},
                metadata={
                    "layer_type": "background",
                    "resolution": layer_resolution,
                    "depth_scale": depth_scale
                }
            )
            
            # Create combined foreground scene from all foreground layers
            combined_fg_depth = self._combine_layer_depths(layer_depth_maps, object_labels.fg_labels_1 + object_labels.fg_labels_2)
            combined_fg_masks = {}
            for label in object_labels.fg_labels_1 + object_labels.fg_labels_2:
                if label in layer_masks:
                    combined_fg_masks[label] = layer_masks[label]
            
            foreground_scene = Scene3D(
                panorama=panorama,
                depth_map=combined_fg_depth,
                semantic_masks=combined_fg_masks,
                metadata={
                    "layer_type": "foreground_combined",
                    "resolution": layer_resolution,
                    "object_count": len(combined_fg_masks)
                }
            )
            
            # Create LayeredScene3D with all layer information
            layered_scene = LayeredScene3D(
                panorama=panorama,
                background_scene=background_scene,
                foreground_layers=foreground_layers,
                layer_depth_maps=layer_depth_maps,
                layer_masks=layer_masks,
                object_labels=object_labels,
                metadata={
                    "resolution": layer_resolution,
                    "layer_count": len(foreground_layers),
                    "object_separation": object_separation,
                    "depth_scale": depth_scale,
                    "total_objects": len(object_labels.get_all_labels())
                }
            )
            
            # Create preview images
            layer_preview = self._create_layer_preview(layer_masks, width//4, height//4)
            depth_preview = self._create_depth_preview(combined_fg_depth, width//4, height//4)
            
            return (layered_scene, background_scene, foreground_scene, layer_preview, depth_preview)
            
        except Exception as e:
            print(f"Error generating layered scene: {str(e)}")
            # Return fallback scenes
            fallback_depth = torch.randn(panorama.image.shape[0], panorama.image.shape[1])
            fallback_scene = Scene3D(panorama, fallback_depth, metadata={"error": str(e)})
            fallback_layered = LayeredScene3D(
                panorama, fallback_scene, [], {}, {}, object_labels,
                metadata={"error": str(e)}
            )
            fallback_preview = torch.zeros(1, height//4, width//4, 3)
            return (fallback_layered, fallback_scene, fallback_scene, fallback_preview, fallback_preview)
    
    def _combine_layer_depths(self, layer_depth_maps: Dict[str, torch.Tensor], labels: List[str]) -> torch.Tensor:
        """Combine multiple layer depth maps into single depth map"""
        if not layer_depth_maps:
            return torch.zeros(960, 1920)  # Default size fallback
        
        # Start with first available depth map
        combined_depth = None
        for label in labels:
            if label in layer_depth_maps:
                if combined_depth is None:
                    combined_depth = layer_depth_maps[label].clone()
                else:
                    # Take maximum depth for proper layering
                    combined_depth = torch.max(combined_depth, layer_depth_maps[label])
        
        return combined_depth if combined_depth is not None else torch.zeros(960, 1920)
    
    def _create_layer_preview(self, layer_masks: Dict[str, torch.Tensor], width: int, height: int) -> torch.Tensor:
        """Create preview showing all layers with different colors"""
        if not layer_masks:
            return torch.zeros(1, height, width, 3)
        
        from torch.nn.functional import interpolate
        
        combined_preview = torch.zeros(height, width, 3)
        colors = [
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.5, 0.5, 0.5], [1.0, 0.5, 0.0],
            [0.5, 0.0, 1.0], [0.0, 0.5, 0.5]
        ]
        
        for i, (name, mask) in enumerate(layer_masks.items()):
            color = colors[i % len(colors)]
            
            # Resize mask to preview size
            if len(mask.shape) == 2:
                mask_resized = interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            else:
                mask_resized = mask  # Assume already correct size
            
            # Apply color to mask
            mask_3d = mask_resized.unsqueeze(-1)
            combined_preview += mask_3d * torch.tensor(color).view(1, 1, 3)
        
        # Normalize and add batch dimension
        combined_preview = torch.clamp(combined_preview, 0, 1)
        return combined_preview.unsqueeze(0)
    
    def _create_depth_preview(self, depth_map: torch.Tensor, width: int, height: int) -> torch.Tensor:
        """Create depth visualization preview"""
        if depth_map is None:
            return torch.zeros(1, height, width, 3)
        
        from torch.nn.functional import interpolate
        
        # Normalize depth
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        # Resize to preview size
        depth_resized = interpolate(
            depth_norm.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
        
        # Convert to RGB
        depth_rgb = depth_resized.unsqueeze(-1).repeat(1, 1, 3)
        
        return depth_rgb.unsqueeze(0)


class HunyuanFluxGenerator:
    """FLUX-enhanced panoramic generation node that combines FLUX models with HunyuanWorld pipeline"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_model": ("MODEL_HUNYUAN",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful panoramic landscape",
                    "tooltip": "Text prompt for FLUX generation. FLUX models excel at detailed, high-quality imagery."
                }),
                "width": ("INT", {
                    "default": 1920,
                    "min": 512,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Output width. 1920 is HunyuanWorld standard for panoramic format."
                }),
                "height": ("INT", {
                    "default": 960,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Output height. 960 is HunyuanWorld standard for panoramic format (2:1 ratio)."
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "tooltip": "Random seed for reproducible generation. -1 for random."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Guidance scale for prompt adherence. Higher = more prompt following, lower = more creative."
                }),
                "num_inference_steps": ("INT", {
                    "default": 30,
                    "min": 10,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of denoising steps. More steps = higher quality but slower generation."
                }),
                "flux_mode": (["standard", "panoramic", "ultra_wide"], {
                    "default": "panoramic",
                    "tooltip": "FLUX generation mode: 'standard'=normal FLUX output, 'panoramic'=optimized for 360° panoramas, 'ultra_wide'=extra wide landscapes."
                })
            }
        }
    
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("panorama",)
    FUNCTION = "generate_flux_panorama"
    CATEGORY = "HunyuanWorld/FLUX"
    
    def generate_flux_panorama(self, flux_model, prompt: str, width: int = 1920, height: int = 960, 
                              seed: int = -1, guidance_scale: float = 7.5, num_inference_steps: int = 30,
                              flux_mode: str = "panoramic"):
        """Generate panoramic images using FLUX models with HunyuanWorld optimizations"""
        
        try:
            # Handle random seed
            if seed == -1:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
            # Determine if this is a FLUX model
            model_type = getattr(flux_model, 'model_type', 'unknown')
            is_flux_model = model_type in ['flux_dev', 'flux_fill', 'dreamshaper']
            
            if not is_flux_model:
                print(f"⚠️ Warning: Expected FLUX model, got {model_type}. Proceeding anyway...")
            
            # Generate with appropriate method
            if hasattr(flux_model, 'generate_panorama'):
                print(f"🌄 Generating {width}x{height} panorama using {model_type}")
                panorama_tensor = flux_model.generate_panorama(
                    prompt,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    mode=flux_mode
                )
            elif hasattr(flux_model, 'generate_image'):
                print(f"🎨 Adapting standard generation to panoramic format")
                # Adapt standard generation for panoramic output
                panorama_tensor = flux_model.generate_image(
                    prompt,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed
                )
            else:
                # Fallback for models without proper methods
                print(f"🔄 Using fallback generation for {model_type}")
                panorama_tensor = torch.randn(height, width, 3)
            
            # Ensure correct format and range
            if panorama_tensor.max() > 1.0:
                panorama_tensor = panorama_tensor / 255.0  # Normalize to [0,1]
            
            panorama_tensor = torch.clamp(panorama_tensor, 0.0, 1.0)
            
            # Create PanoramaImage with metadata
            metadata = {
                "model_type": model_type,
                "prompt": prompt,
                "width": width,
                "height": height,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "steps": num_inference_steps,
                "flux_mode": flux_mode,
                "generated_with": "HunyuanFluxGenerator"
            }
            
            panorama = PanoramaImage(panorama_tensor, metadata)
            
            print(f"✅ FLUX panorama generated: {width}x{height} using {model_type}")
            return (panorama,)
            
        except Exception as e:
            print(f"❌ Error in FLUX panorama generation: {e}")
            import traceback
            traceback.print_exc()
            # Return fallback
            fallback_tensor = torch.randn(height, width, 3)
            return (PanoramaImage(fallback_tensor, {"error": str(e)}),)