import os
import json
import hashlib
from typing import Dict, Tuple

from .runtime import HYWRuntime, RuntimeConfig


class HYW_ModelLoader:
    """HunyuanWorld Model Loader node for ComfyUI"""
    
    CATEGORY = "HunyuanWorld/Loaders"
    RETURN_TYPES = ("HYW_RUNTIME",)
    RETURN_NAMES = ("hyw_runtime",)
    FUNCTION = "load_models"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_text_model": ("STRING", {
                    "default": "models/unet/flux1-dev-fp8.safetensors",
                    "tooltip": "Path to the FLUX text-to-image model file (.safetensors). Used as base model for text-to-panorama generation."
                }),
                "flux_image_model": ("STRING", {
                    "default": "models/unet/flux1-fill-dev.safetensors", 
                    "tooltip": "Path to the FLUX image-to-image model file (.safetensors). Used as base model for image-to-panorama generation."
                }),
                "text_lora_path": ("STRING", {
                    "default": "models/Hunyuan_World/HunyuanWorld-PanoDiT-Text.safetensors",
                    "tooltip": "Path to the HunyuanWorld text-to-panorama LoRA weights. Fine-tunes FLUX for 360° panorama generation from text prompts."
                }),
                "image_lora_path": ("STRING", {
                    "default": "models/Hunyuan_World/HunyuanWorld-PanoDiT-Image.safetensors",
                    "tooltip": "Path to the HunyuanWorld image-to-panorama LoRA weights. Fine-tunes FLUX for 360° panorama generation from input images."
                }),
                "device": ("STRING", {
                    "default": "cuda:0",
                    "tooltip": "Device to load models on. Use 'cuda:0' for GPU acceleration, 'cpu' for CPU-only inference. GPU recommended for reasonable speed."
                }),
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Data type precision for model weights. bfloat16: Best speed/quality balance. float16: Faster but may have precision issues. float32: Highest quality but slower."
                }),
            },
            "optional": {
                "enable_cpu_offload": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable model CPU offloading to save GPU memory. Models are moved to CPU when not in use. Recommended for limited VRAM."
                }),
                "enable_vae_tiling": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable VAE tiling to reduce memory usage during image encoding/decoding. Processes image in tiles instead of full resolution."
                }),
                "enable_xformers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable xFormers memory-efficient attention. Reduces memory usage and may improve speed. Requires xFormers to be installed."
                }),
            }
        }

    def load_models(self, flux_text_model, flux_image_model, text_lora_path, image_lora_path,
                   device, dtype, enable_cpu_offload=True, enable_vae_tiling=True, 
                   enable_xformers=True):
        """Load HunyuanWorld models and return runtime instance"""
        
        # Create configuration
        model_paths = {
            "flux_text": flux_text_model,
            "flux_image": flux_image_model,
            "pano_text_lora": text_lora_path,
            "pano_image_lora": image_lora_path
        }
        
        optimization = {
            "enable_model_cpu_offload": enable_cpu_offload,
            "enable_vae_tiling": enable_vae_tiling,
            "enable_xformers": enable_xformers
        }
        
        defaults = {
            "pano_size": [1920, 960],
            "guidance_scale": 30.0,
            "num_inference_steps": 50,
            "blend_extend": 6,
            "true_cfg_scale": 0.0,
            "shifting_extend": 0
        }
        
        cfg = RuntimeConfig(
            device=device,
            dtype=dtype,
            model_paths=model_paths,
            optimization=optimization,
            defaults=defaults
        )
        
        # Create runtime instance
        runtime = HYWRuntime(cfg)
        
        return (runtime,)


class HYW_Config:
    """HunyuanWorld Configuration node for ComfyUI"""
    
    CATEGORY = "HunyuanWorld/Loaders"
    RETURN_TYPES = ("HYW_CONFIG",)
    RETURN_NAMES = ("hyw_config",)
    FUNCTION = "create_config"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pano_width": ("INT", {
                    "default": 1920, "min": 256, "max": 16384, "step": 256,
                    "tooltip": "Width of the generated panorama in pixels. Standard: 1920px. Higher values need more VRAM but produce higher quality."
                }),
                "pano_height": ("INT", {
                    "default": 960, "min": 128, "max": 8192, "step": 128,
                    "tooltip": "Height of the generated panorama in pixels. Standard: 960px. Should be half of width for proper 360° aspect ratio."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 30.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "How strongly the model follows the prompt. Higher values = more adherence to prompt but less creativity. Range: 1-50 typical."
                }),
                "num_inference_steps": ("INT", {
                    "default": 50, "min": 1, "max": 200,
                    "tooltip": "Number of denoising steps. More steps = higher quality but slower generation. 20-50 is usually sufficient."
                }),
                "blend_extend": ("INT", {
                    "default": 6, "min": 0, "max": 20,
                    "tooltip": "Panorama seam blending pixels. Higher values create smoother 360° transitions but may blur details at seam boundaries."
                }),
                "true_cfg_scale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Advanced CFG scaling for better prompt adherence. 0.0 = disabled. 1.0-5.0 may improve quality for complex prompts."
                }),
                "shifting_extend": ("INT", {
                    "default": 0, "min": 0, "max": 10,
                    "tooltip": "Panorama shifting extension for better continuity. 0 = disabled. Higher values may improve 360° wrap-around quality."
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 2**31-1,
                    "tooltip": "Random seed for reproducible generation. Same seed + settings = same result. Use -1 for random seed."
                }),
                "target_size": ("INT", {
                    "default": 3840, "min": 1024, "max": 8192, "step": 256,
                    "tooltip": "Target resolution for 3D world reconstruction. Higher values = more detail but slower processing and more memory usage."
                }),
                "fov": ("FLOAT", {
                    "default": 80.0, "min": 10.0, "max": 180.0,
                    "tooltip": "Field of view in degrees for input image perspective correction. 80° is typical for photos. Wider for fisheye/action cameras."
                }),
                "theta": ("FLOAT", {
                    "default": 0.0, "min": -180.0, "max": 180.0,
                    "tooltip": "Horizontal rotation angle in degrees for input image placement in panorama. 0° = center, ±180° = opposite side."
                }),
                "phi": ("FLOAT", {
                    "default": 0.0, "min": -90.0, "max": 90.0,
                    "tooltip": "Vertical rotation angle in degrees for input image placement. 0° = horizon level, +90° = looking up, -90° = looking down."
                }),
            }
        }

    def create_config(self, pano_width, pano_height, guidance_scale, num_inference_steps,
                     blend_extend, true_cfg_scale, shifting_extend, seed=42, 
                     target_size=3840, fov=80.0, theta=0.0, phi=0.0):
        """Create configuration dictionary for HunyuanWorld operations"""
        
        config = {
            "pano_size": [int(pano_width), int(pano_height)],
            "guidance_scale": float(guidance_scale),
            "num_inference_steps": int(num_inference_steps),
            "blend_extend": int(blend_extend),
            "true_cfg_scale": float(true_cfg_scale),
            "shifting_extend": int(shifting_extend),
            "seed": int(seed),
            "target_size": int(target_size),
            "fov": float(fov),
            "theta": float(theta),
            "phi": float(phi)
        }
        
        return (config,)


class HYW_SettingsLoader:
    """Load HunyuanWorld settings from JSON file"""
    
    CATEGORY = "HunyuanWorld/Loaders" 
    RETURN_TYPES = ("HYW_RUNTIME",)
    RETURN_NAMES = ("hyw_runtime",)
    FUNCTION = "load_from_settings"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "settings_path": ("STRING", {
                    "default": "settings.json",
                    "tooltip": "Path to JSON settings file containing model paths, device config, and default parameters. Relative to node pack directory."
                }),
            },
            "optional": {
                "override_device": ("STRING", {
                    "default": "",
                    "tooltip": "Override device from settings file. Leave empty to use settings file value. Example: 'cuda:0', 'cuda:1', 'cpu'."
                }),
                "override_dtype": (["", "bfloat16", "float16", "float32"], {
                    "default": "",
                    "tooltip": "Override data type from settings file. Leave empty to use settings file value. bfloat16 recommended for best balance."
                }),
            }
        }

    def load_from_settings(self, settings_path, override_device="", override_dtype=""):
        """Load runtime from settings file"""
        
        # Get absolute path relative to the node pack directory
        node_pack_dir = os.path.dirname(os.path.dirname(__file__))
        if not os.path.isabs(settings_path):
            settings_path = os.path.join(node_pack_dir, settings_path)
        
        if not os.path.exists(settings_path):
            raise FileNotFoundError(f"Settings file not found: {settings_path}")
        
        # Load settings
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        
        # Apply overrides
        device = override_device if override_device else settings.get("device", "cuda:0")
        dtype = override_dtype if override_dtype else settings.get("dtype", "bfloat16")
        
        # Create configuration
        cfg = RuntimeConfig(
            device=device,
            dtype=dtype,
            model_paths=settings.get("model_paths", {}),
            optimization=settings.get("optimization", {}),
            defaults=settings.get("defaults", {})
        )
        
        # Create runtime instance
        runtime = HYWRuntime(cfg)
        
        return (runtime,)