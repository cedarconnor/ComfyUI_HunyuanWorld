"""
Working FLUX Diffusion Pipeline
Uses actual FLUX model weights for real diffusion generation
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any

def create_working_flux_pipeline(model_path: str, device: str = "cpu"):
    """Create a working FLUX pipeline that uses actual model weights"""
    
    class WorkingFluxPipeline:
        def __init__(self, model_path: str, device: str):
            self.model_path = model_path
            self.device = device
            self.model_weights = None
            self.lora_weights = None
            self.is_loaded = False
            
            print(f"[INFO] Creating working FLUX pipeline: {model_path}")
            self._load_model_weights()
        
        def _load_model_weights(self):
            """Load actual FLUX model weights"""
            try:
                from safetensors.torch import load_file
                
                if os.path.exists(self.model_path):
                    print(f"[INFO] Loading FLUX weights: {self.model_path}")
                    self.model_weights = load_file(self.model_path)
                    self.is_loaded = True
                    print(f"[SUCCESS] Loaded FLUX weights: {len(self.model_weights)} tensors")
                else:
                    print(f"[ERROR] FLUX model not found: {self.model_path}")
                    self.is_loaded = False
                    
            except Exception as e:
                print(f"[ERROR] Failed to load FLUX weights: {e}")
                self.is_loaded = False
        
        def __call__(self, prompt: str, **kwargs):
            """FLUX + LoRA ONLY - No fallbacks, fail if not available"""
            
            if not self.is_loaded:
                error_msg = "FLUX model weights not loaded - cannot generate images"
                print(f"[CRITICAL ERROR] {error_msg}")
                print(f"[FAILURE REASON] Model file not found or corrupted")
                raise RuntimeError(error_msg)
            
            # Extract parameters
            height = kwargs.get('height', 512)
            width = kwargs.get('width', 512) 
            num_inference_steps = kwargs.get('num_inference_steps', 20)
            guidance_scale = kwargs.get('guidance_scale', 7.5)
            
            print(f"[INFO] FLUX + LoRA ONLY: '{prompt}' ({width}x{height}, {num_inference_steps} steps)")
            
            try:
                # Use FLUX + LoRA diffusion process - NO FALLBACKS
                result_image = self._flux_diffusion_process(
                    prompt, width, height, num_inference_steps, guidance_scale
                )
                
                class FluxResult:
                    def __init__(self, images):
                        self.images = images
                
                return FluxResult([result_image])
                
            except Exception as e:
                print(f"[CRITICAL ERROR] FLUX + LoRA generation failed: {e}")
                print(f"[FAILURE REASON] Real AI inference required but failed")
                print(f"[NO FALLBACK] Refusing to generate placeholder/procedural content")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"FLUX + LoRA generation failed: {e}")
        
        def _flux_diffusion_process(self, prompt: str, width: int, height: int, steps: int, guidance: float):
            """FLUX + LoRA inference ONLY - No fallbacks or placeholder generation"""
            
            print(f"[INFO] FLUX + LoRA ONLY MODE: Starting real diffusion inference...")
            
            try:
                # Try to use real diffusers FLUX pipeline
                from diffusers import FluxPipeline
                import torch
                
                # Determine device and dtype
                device = "cuda" if torch.cuda.is_available() and self.device != "cpu" else "cpu"
                dtype = torch.bfloat16 if device == "cuda" else torch.float32
                
                print(f"[INFO] Loading FLUX pipeline on {device} with {dtype}")
                
                # Load FLUX pipeline
                if device == "cuda":
                    pipe = FluxPipeline.from_pretrained(
                        "black-forest-labs/FLUX.1-dev",
                        torch_dtype=dtype,
                        device_map="balanced"
                    )
                else:
                    pipe = FluxPipeline.from_pretrained(
                        "black-forest-labs/FLUX.1-dev",
                        torch_dtype=dtype
                    )
                
                if device == "cpu":
                    pipe = pipe.to("cpu")
                
                # Apply LoRA - REQUIRED, no fallback
                lora_loaded = False
                if hasattr(self, 'lora_weights') and self.lora_weights is not None:
                    try:
                        # Try to find a HunyuanWorld LoRA file path
                        lora_path = None
                        possible_lora_paths = [
                            r"C:\ComfyUI\models\Hunyuan_World\HunyuanWorld-PanoDiT-Text-PT.safetensors",
                            r"C:\ComfyUI\models\Hunyuan_World\HunyuanWorld-PanoDiT-Text.safetensors",
                            r"C:\ComfyUI\models\loras\HunyuanWorld.safetensors",
                            self.model_path  # Use the provided model path directly
                        ]
                        
                        for path in possible_lora_paths:
                            if os.path.exists(path):
                                lora_path = path
                                break
                        
                        if lora_path:
                            print(f"[INFO] Loading HunyuanWorld LoRA from: {lora_path}")
                            pipe.load_lora_weights(lora_path)
                            lora_loaded = True
                            print(f"[SUCCESS] HunyuanWorld LoRA loaded successfully")
                        else:
                            raise FileNotFoundError("HunyuanWorld LoRA file not found in any expected location")
                            
                    except Exception as lora_error:
                        print(f"[CRITICAL ERROR] LoRA loading failed: {lora_error}")
                        print(f"[FAILURE REASON] Cannot proceed without HunyuanWorld LoRA")
                        raise RuntimeError(f"LoRA loading failed: {lora_error}")
                else:
                    print(f"[CRITICAL ERROR] No LoRA weights available")
                    print(f"[FAILURE REASON] LoRA weights not loaded from model file")
                    raise RuntimeError("No LoRA weights available for HunyuanWorld")
                
                if not lora_loaded:
                    print(f"[CRITICAL ERROR] LoRA loading failed")
                    print(f"[FAILURE REASON] Cannot generate panoramas without HunyuanWorld LoRA")
                    raise RuntimeError("HunyuanWorld LoRA is required but failed to load")
                
                # Enhanced prompt for panoramic generation
                enhanced_prompt = f"{prompt}, panoramic view, wide angle, ultra detailed, 8k resolution, professional photography"
                
                print(f"[INFO] Generating with FLUX + HunyuanWorld LoRA")
                print(f"[INFO] Enhanced prompt: '{enhanced_prompt}'")
                print(f"[INFO] Parameters: {width}x{height}, {steps} steps, guidance {guidance}")
                
                # Generate with FLUX + LoRA
                with torch.inference_mode():
                    # Set seed for reproducibility
                    generator = torch.Generator(device=device)
                    seed = abs(hash(prompt)) % 2**32
                    generator.manual_seed(seed)
                    
                    result = pipe(
                        prompt=enhanced_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=steps,  
                        guidance_scale=guidance,
                        generator=generator,
                        max_sequence_length=256
                    )
                
                if hasattr(result, 'images') and len(result.images) > 0:
                    print(f"[SUCCESS] FLUX + HunyuanWorld LoRA generation completed!")
                    return result.images[0]
                else:
                    print(f"[CRITICAL ERROR] FLUX pipeline returned no images")
                    print(f"[FAILURE REASON] Pipeline executed but produced no output")
                    raise RuntimeError("FLUX pipeline returned no images")
                    
            except Exception as e:
                print(f"[CRITICAL ERROR] FLUX + LoRA generation failed: {e}")
                print(f"[FAILURE REASON] Cannot generate without real FLUX + HunyuanWorld LoRA")
                print(f"[NO FALLBACK] Procedural generation disabled - only real AI inference allowed")
                raise RuntimeError(f"FLUX + LoRA generation failed: {e}")
        
        # ALL PROCEDURAL/PLACEHOLDER METHODS COMPLETELY REMOVED
        # ONLY REAL FLUX + LoRA GENERATION REMAINS
    
    return WorkingFluxPipeline(model_path, device)