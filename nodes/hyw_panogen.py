import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple


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


class HYW_PanoGen:
    """HunyuanWorld Panorama Generation node for ComfyUI"""
    
    CATEGORY = "HunyuanWorld/Generate"
    RETURN_TYPES = ("IMAGE", "HYW_METADATA")
    RETURN_NAMES = ("panorama", "metadata")
    FUNCTION = "generate_panorama"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hyw_runtime": ("HYW_RUNTIME",),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "A beautiful landscape panorama"
                }),
                "height": ("INT", {"default": 960, "min": 128, "max": 8192, "step": 128}),
                "width": ("INT", {"default": 1920, "min": 256, "max": 16384, "step": 256}),
                "guidance_scale": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31-1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "low quality, blurry, distorted"
                }),
                "input_image": ("IMAGE",),
                "mask": ("IMAGE",),
                "blend_extend": ("INT", {"default": 6, "min": 0, "max": 20}),
                "true_cfg_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "shifting_extend": ("INT", {"default": 0, "min": 0, "max": 10}),
                "fov": ("FLOAT", {"default": 80.0, "min": 10.0, "max": 180.0}),
                "theta": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
                "phi": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0}),
                "hyw_config": ("HYW_CONFIG",),
            }
        }

    def generate_panorama(self, hyw_runtime, prompt, height, width, guidance_scale, 
                         num_inference_steps, seed, negative_prompt="", 
                         input_image=None, mask=None, blend_extend=6, 
                         true_cfg_scale=0.0, shifting_extend=0, fov=80.0, 
                         theta=0.0, phi=0.0, hyw_config=None):
        """Generate panorama using HunyuanWorld"""
        
        # Use config values if provided
        if hyw_config is not None:
            width = hyw_config.get("pano_size", [width, height])[0]
            height = hyw_config.get("pano_size", [width, height])[1] 
            guidance_scale = hyw_config.get("guidance_scale", guidance_scale)
            num_inference_steps = hyw_config.get("num_inference_steps", num_inference_steps)
            blend_extend = hyw_config.get("blend_extend", blend_extend)
            true_cfg_scale = hyw_config.get("true_cfg_scale", true_cfg_scale)
            shifting_extend = hyw_config.get("shifting_extend", shifting_extend)
            seed = hyw_config.get("seed", seed)
            fov = hyw_config.get("fov", fov)
            theta = hyw_config.get("theta", theta)
            phi = hyw_config.get("phi", phi)
        
        try:
            if input_image is not None:
                # Image-to-panorama mode
                input_pil = tensor_to_pil(input_image)
                mask_pil = tensor_to_pil(mask) if mask is not None else None
                
                pano_pil, metadata = hyw_runtime.generate_image_panorama(
                    prompt=prompt,
                    image=input_pil,
                    mask=mask_pil,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    blend_extend=blend_extend,
                    true_cfg_scale=true_cfg_scale,
                    shifting_extend=shifting_extend,
                    fov=fov,
                    theta=theta,
                    phi=phi
                )
            else:
                # Text-to-panorama mode
                pano_pil, metadata = hyw_runtime.generate_text_panorama(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    blend_extend=blend_extend,
                    true_cfg_scale=true_cfg_scale,
                    shifting_extend=shifting_extend
                )
            
            # Convert PIL to ComfyUI tensor
            pano_tensor = pil_to_tensor(pano_pil)
            
            return (pano_tensor, metadata)
            
        except Exception as e:
            print(f"Error in panorama generation: {e}")
            import traceback
            traceback.print_exc()
            raise e


class HYW_PanoGenBatch:
    """HunyuanWorld Batch Panorama Generation node for ComfyUI"""
    
    CATEGORY = "HunyuanWorld/Generate"
    RETURN_TYPES = ("IMAGE", "HYW_METADATA")
    RETURN_NAMES = ("panoramas", "metadata_batch")
    FUNCTION = "generate_batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hyw_runtime": ("HYW_RUNTIME",),
                "prompts": ("STRING", {
                    "multiline": True, 
                    "default": "A beautiful landscape panorama\nA city skyline at sunset\nA forest with mountains"
                }),
                "height": ("INT", {"default": 960, "min": 128, "max": 8192, "step": 128}),
                "width": ("INT", {"default": 1920, "min": 256, "max": 16384, "step": 256}),
                "guidance_scale": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "base_seed": ("INT", {"default": 42, "min": 0, "max": 2**31-1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "low quality, blurry, distorted"
                }),
                "blend_extend": ("INT", {"default": 6, "min": 0, "max": 20}),
                "true_cfg_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "shifting_extend": ("INT", {"default": 0, "min": 0, "max": 10}),
            }
        }

    def generate_batch(self, hyw_runtime, prompts, height, width, guidance_scale,
                      num_inference_steps, base_seed, negative_prompt="",
                      blend_extend=6, true_cfg_scale=0.0, shifting_extend=0):
        """Generate multiple panoramas from a list of prompts"""
        
        prompt_list = [p.strip() for p in prompts.split('\n') if p.strip()]
        
        if not prompt_list:
            raise ValueError("No valid prompts provided")
        
        batch_tensors = []
        batch_metadata = []
        
        for i, prompt in enumerate(prompt_list):
            try:
                # Use different seed for each generation
                seed = base_seed + i
                
                pano_pil, metadata = hyw_runtime.generate_text_panorama(
                    prompt=prompt,
                    negative_prompt=negative_prompt if negative_prompt else None,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    blend_extend=blend_extend,
                    true_cfg_scale=true_cfg_scale,
                    shifting_extend=shifting_extend
                )
                
                pano_tensor = pil_to_tensor(pano_pil)
                batch_tensors.append(pano_tensor)
                batch_metadata.append(metadata)
                
                print(f"Generated panorama {i+1}/{len(prompt_list)}: {prompt}")
                
            except Exception as e:
                print(f"Error generating panorama {i+1} '{prompt}': {e}")
                continue
        
        if not batch_tensors:
            raise RuntimeError("Failed to generate any panoramas")
        
        # Concatenate all tensors along batch dimension
        combined_tensor = torch.cat(batch_tensors, dim=0)
        
        combined_metadata = {
            "batch_size": len(batch_tensors),
            "prompts": [meta["prompt"] for meta in batch_metadata],
            "seeds": [meta["seed"] for meta in batch_metadata],
            "common_params": {
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "negative_prompt": negative_prompt,
                "blend_extend": blend_extend,
                "true_cfg_scale": true_cfg_scale,
                "shifting_extend": shifting_extend
            }
        }
        
        return (combined_tensor, combined_metadata)