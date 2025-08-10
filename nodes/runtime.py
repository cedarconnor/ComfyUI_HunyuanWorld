import sys
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import json
import numpy as np
from PIL import Image

# Add HunyuanWorld-1.0 to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "HunyuanWorld-1.0"))

# Import HunyuanWorld modules
try:
    from hy3dworld import Text2PanoramaPipelines, Image2PanoramaPipelines
    from hy3dworld import LayerDecomposition, WorldComposer
    from hy3dworld.utils import Perspective, process_file
    HUNYUANWORLD_AVAILABLE = True
except ImportError as e:
    print(f"Warning: HunyuanWorld modules not available: {e}")
    HUNYUANWORLD_AVAILABLE = False

@dataclass
class RuntimeConfig:
    """Configuration for HunyuanWorld runtime"""
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    model_paths: Dict[str, str] = None
    optimization: Dict[str, bool] = None
    defaults: Dict[str, Any] = None

class HYWRuntime:
    """Runtime adapter for HunyuanWorld integration with ComfyUI"""
    
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self._text2pano_pipe = None
        self._image2pano_pipe = None
        self._layer_decomposer = None
        self._world_composer = None
        
        # Set torch dtype
        if cfg.dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif cfg.dtype == "float16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32
            
        # Verify HunyuanWorld is available
        if not HUNYUANWORLD_AVAILABLE:
            raise RuntimeError("HunyuanWorld modules are not available. Please check installation.")
    
    def _find_comfyui_root(self):
        """Find ComfyUI root directory by looking for characteristic files/folders"""
        # Start from current file location and search upward
        current_dir = os.path.dirname(__file__)  # Start from nodes/ directory
        
        # Look for ComfyUI markers (main.py, web/, custom_nodes/)
        for i in range(10):  # Limit search depth
            if (os.path.exists(os.path.join(current_dir, "main.py")) and 
                os.path.exists(os.path.join(current_dir, "web")) and
                os.path.exists(os.path.join(current_dir, "custom_nodes"))):
                return current_dir
            parent = os.path.dirname(current_dir)
            if parent == current_dir:  # Reached filesystem root
                break
            current_dir = parent
        
        # Improved fallback: try common ComfyUI paths
        possible_roots = [
            "C:\\ComfyUI",
            os.path.join(os.environ.get("USERPROFILE", ""), "ComfyUI"),
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Original fallback
        ]
        
        for root in possible_roots:
            if (os.path.exists(root) and 
                os.path.exists(os.path.join(root, "main.py")) and 
                os.path.exists(os.path.join(root, "models"))):
                return root
        
        # Final fallback
        return "C:\\ComfyUI"
    
    def load_text2pano_pipeline(self):
        """Load text-to-panorama pipeline"""
        if self._text2pano_pipe is None:
            print("Loading Text2Panorama pipeline...")
            
            flux_model_path = self.cfg.model_paths.get("flux_text", "models/unet/flux1-dev-fp8.safetensors")
            lora_path = self.cfg.model_paths.get("pano_text_lora", "models/Hunyuan_World/HunyuanWorld-PanoDiT-Text.safetensors")
            
            # Convert relative paths to absolute paths from ComfyUI root
            if not os.path.isabs(flux_model_path):
                comfyui_root = self._find_comfyui_root()
                flux_model_path = os.path.join(comfyui_root, flux_model_path)
                
            if not os.path.isabs(lora_path):
                comfyui_root = self._find_comfyui_root()
                lora_path = os.path.join(comfyui_root, lora_path)
            
            # Check if model files exist
            if not os.path.exists(flux_model_path):
                raise FileNotFoundError(f"FLUX text model not found at: {flux_model_path}")
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"HunyuanWorld text LoRA not found at: {lora_path}")
            
            # Load from local files
            self._text2pano_pipe = Text2PanoramaPipelines.from_single_file(
                flux_model_path,
                torch_dtype=self.torch_dtype
            ).to(self.cfg.device)
            
            # Load LoRA weights from local file
            self._text2pano_pipe.load_lora_weights(lora_path, adapter_name="hunyuanworld_text")
            
            # Apply optimizations
            if self.cfg.optimization.get("enable_model_cpu_offload", True):
                self._text2pano_pipe.enable_model_cpu_offload()
            if self.cfg.optimization.get("enable_vae_tiling", True):
                self._text2pano_pipe.enable_vae_tiling()
                
        return self._text2pano_pipe
    
    def load_image2pano_pipeline(self):
        """Load image-to-panorama pipeline"""
        if self._image2pano_pipe is None:
            print("Loading Image2Panorama pipeline...")
            
            flux_model_path = self.cfg.model_paths.get("flux_image", "models/unet/flux1-fill-dev.safetensors")
            lora_path = self.cfg.model_paths.get("pano_image_lora", "models/Hunyuan_World/HunyuanWorld-PanoDiT-Image.safetensors")
            
            # Convert relative paths to absolute paths from ComfyUI root
            if not os.path.isabs(flux_model_path):
                comfyui_root = self._find_comfyui_root()
                flux_model_path = os.path.join(comfyui_root, flux_model_path)
                
            if not os.path.isabs(lora_path):
                comfyui_root = self._find_comfyui_root()
                lora_path = os.path.join(comfyui_root, lora_path)
            
            # Check if model files exist
            if not os.path.exists(flux_model_path):
                raise FileNotFoundError(f"FLUX image model not found at: {flux_model_path}")
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"HunyuanWorld image LoRA not found at: {lora_path}")
            
            # Load from local files
            self._image2pano_pipe = Image2PanoramaPipelines.from_single_file(
                flux_model_path,
                torch_dtype=self.torch_dtype
            ).to(self.cfg.device)
            
            # Load LoRA weights from local file
            self._image2pano_pipe.load_lora_weights(lora_path, adapter_name="hunyuanworld_image")
            
            # Apply optimizations
            if self.cfg.optimization.get("enable_model_cpu_offload", True):
                self._image2pano_pipe.enable_model_cpu_offload()
            if self.cfg.optimization.get("enable_vae_tiling", True):
                self._image2pano_pipe.enable_vae_tiling()
                
        return self._image2pano_pipe
    
    def load_world_components(self, seed=42, target_size=3840):
        """Load world reconstruction components"""
        if self._layer_decomposer is None:
            print("Loading LayerDecomposition...")
            self._layer_decomposer = LayerDecomposition()
            
        if self._world_composer is None:
            print("Loading WorldComposer...")
            kernel_scale = max(1, int(target_size / 1920))
            device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
            
            self._world_composer = WorldComposer(
                device=device,
                resolution=(target_size, target_size // 2),
                seed=seed,
                filter_mask=True,
                kernel_scale=kernel_scale,
            )
            
        return self._layer_decomposer, self._world_composer
    
    def generate_text_panorama(self, prompt: str, negative_prompt: Optional[str] = None, 
                             height: int = 960, width: int = 1920, 
                             guidance_scale: float = 30.0, num_inference_steps: int = 50,
                             seed: int = 42, blend_extend: int = 6, 
                             true_cfg_scale: float = 0.0, shifting_extend: int = 0) -> Tuple[Image.Image, Dict]:
        """Generate panorama from text prompt"""
        pipe = self.load_text2pano_pipeline()
        
        # Generate image
        result = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            generator=torch.Generator("cpu").manual_seed(seed),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            blend_extend=blend_extend,
            true_cfg_scale=true_cfg_scale,
        )
        
        image = result.images[0]
        meta = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "blend_extend": blend_extend,
            "true_cfg_scale": true_cfg_scale,
            "shifting_extend": shifting_extend
        }
        
        return image, meta
    
    def generate_image_panorama(self, prompt: str, image: Image.Image, mask: Optional[Image.Image] = None,
                              negative_prompt: Optional[str] = None,
                              height: int = 960, width: int = 1920,
                              guidance_scale: float = 30.0, num_inference_steps: int = 50,
                              seed: int = 42, blend_extend: int = 6,
                              true_cfg_scale: float = 2.0, shifting_extend: int = 0,
                              fov: float = 80, theta: float = 0, phi: float = 0) -> Tuple[Image.Image, Dict]:
        """Generate panorama from input image"""
        pipe = self.load_image2pano_pipeline()
        
        # Process input image if needed
        processed_img = image
        processed_mask = mask
        
        if mask is None:
            # Convert perspective image to equirectangular
            import cv2
            perspective_img = np.array(image)
            if len(perspective_img.shape) == 3:
                perspective_img = cv2.cvtColor(perspective_img, cv2.COLOR_RGB2BGR)
            
            height_fov, width_fov = perspective_img.shape[:2]
            if width_fov > height_fov:
                ratio = width_fov / height_fov
                w = int((fov / 360) * width)
                h = int(w / ratio)
                perspective_img = cv2.resize(perspective_img, (w, h), interpolation=cv2.INTER_AREA)
            else:
                ratio = height_fov / width_fov
                h = int((fov / 180) * height)
                w = int(h / ratio)
                perspective_img = cv2.resize(perspective_img, (w, h), interpolation=cv2.INTER_AREA)
            
            equ = Perspective(perspective_img, fov, theta, phi, crop_bound=False)
            img_array, mask_array = equ.GetEquirec(height, width)
            
            # Erode mask
            mask_array = cv2.erode(mask_array.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=5)
            img_array = img_array * mask_array
            mask_array = mask_array.astype(np.uint8) * 255
            mask_array = 255 - mask_array
            
            processed_mask = Image.fromarray(mask_array[:, :, 0])
            processed_img = Image.fromarray(cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_BGR2RGB))
        
        # Generate panorama
        result = pipe(
            prompt=prompt,
            image=processed_img,
            mask_image=processed_mask,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed),
            blend_extend=blend_extend,
            shifting_extend=shifting_extend,
            true_cfg_scale=true_cfg_scale,
        )
        
        image = result.images[0]
        meta = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "blend_extend": blend_extend,
            "true_cfg_scale": true_cfg_scale,
            "shifting_extend": shifting_extend,
            "fov": fov,
            "theta": theta,
            "phi": phi
        }
        
        return image, meta
    
    def reconstruct_world(self, panorama_path: str, labels_fg1: List[str] = None, 
                         labels_fg2: List[str] = None, classes: str = "outdoor",
                         seed: int = 42, target_size: int = 3840,
                         output_dir: str = "temp_world_output") -> Tuple[List, Dict]:
        """Reconstruct 3D world from panorama"""
        layer_decomposer, world_composer = self.load_world_components(seed, target_size)
        
        labels_fg1 = labels_fg1 or []
        labels_fg2 = labels_fg2 or []
        
        # Create temporary output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Layer decomposition
        fg1_infos = [{
            "image_path": panorama_path,
            "output_path": output_dir,
            "labels": labels_fg1,
            "class": classes,
        }]
        
        fg2_infos = [{
            "image_path": os.path.join(output_dir, 'remove_fg1_image.png'),
            "output_path": output_dir,
            "labels": labels_fg2,
            "class": classes,
        }]
        
        # Perform layer decomposition
        layer_decomposer(fg1_infos, layer=0)
        if labels_fg1:  # Only decompose if we have labels
            layer_decomposer(fg2_infos, layer=1)
            layer_decomposer(fg2_infos, layer=2)
        
        # Load separate panorama layers
        separate_pano, fg_bboxes = world_composer._load_separate_pano_from_dir(output_dir, sr=True)
        
        # Generate layered world mesh
        layered_world_mesh = world_composer.generate_world(
            separate_pano=separate_pano, fg_bboxes=fg_bboxes, world_type='mesh'
        )
        
        meta = {
            "labels_fg1": labels_fg1,
            "labels_fg2": labels_fg2,
            "classes": classes,
            "seed": seed,
            "target_size": target_size,
            "num_layers": len(layered_world_mesh)
        }
        
        return layered_world_mesh, meta