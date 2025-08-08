import torch
import numpy as np
from PIL import Image


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


class HYW_PanoInpaint_Scene:
    """HunyuanWorld Scene Inpainting node for ComfyUI"""
    
    CATEGORY = "HunyuanWorld/Generate"
    RETURN_TYPES = ("IMAGE", "HYW_METADATA")
    RETURN_NAMES = ("inpainted_panorama", "metadata")
    FUNCTION = "inpaint_scene"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hyw_runtime": ("HYW_RUNTIME",),
                "panorama": ("IMAGE",),
                "mask": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "detailed scene elements, high quality"
                }),
                "strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "guidance_scale": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31-1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "low quality, blurry, artifacts"
                }),
                "blend_extend": ("INT", {"default": 6, "min": 0, "max": 20}),
                "true_cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "shifting_extend": ("INT", {"default": 0, "min": 0, "max": 10}),
            }
        }

    def inpaint_scene(self, hyw_runtime, panorama, mask, prompt, strength, 
                     guidance_scale, num_inference_steps, seed, 
                     negative_prompt="", blend_extend=6, true_cfg_scale=2.0, 
                     shifting_extend=0):
        """Inpaint scene elements in panorama"""
        
        try:
            # Convert tensors to PIL
            pano_pil = tensor_to_pil(panorama)
            mask_pil = tensor_to_pil(mask)
            
            # Get panorama dimensions 
            height, width = pano_pil.size[1], pano_pil.size[0]
            
            # Use image-to-panorama pipeline for inpainting
            result_pil, metadata = hyw_runtime.generate_image_panorama(
                prompt=prompt,
                image=pano_pil,
                mask=mask_pil,
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
            
            # Add inpainting-specific metadata
            metadata.update({
                "inpaint_type": "scene",
                "strength": strength,
                "mask_shape": mask_pil.size
            })
            
            # Convert back to tensor
            result_tensor = pil_to_tensor(result_pil)
            
            return (result_tensor, metadata)
            
        except Exception as e:
            print(f"Error in scene inpainting: {e}")
            import traceback
            traceback.print_exc()
            raise e


class HYW_PanoInpaint_Advanced:
    """Advanced Panorama Inpainting with multiple regions"""
    
    CATEGORY = "HunyuanWorld/Generate"
    RETURN_TYPES = ("IMAGE", "HYW_METADATA")
    RETURN_NAMES = ("inpainted_panorama", "metadata")
    FUNCTION = "inpaint_advanced"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hyw_runtime": ("HYW_RUNTIME",),
                "panorama": ("IMAGE",),
                "mask": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "high quality inpainting"
                }),
                "inpaint_type": (["scene", "sky", "foreground", "background"], {"default": "scene"}),
                "strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "guidance_scale": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31-1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "low quality, blurry, artifacts"
                }),
                "region_prompts": ("STRING", {
                    "multiline": True,
                    "default": "region1: trees and forest\nregion2: sky and clouds"
                }),
                "blend_extend": ("INT", {"default": 6, "min": 0, "max": 20}),
                "true_cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "feather_mask": ("BOOLEAN", {"default": True}),
                "preserve_edges": ("BOOLEAN", {"default": True}),
            }
        }

    def inpaint_advanced(self, hyw_runtime, panorama, mask, prompt, inpaint_type,
                        strength, guidance_scale, num_inference_steps, seed,
                        negative_prompt="", region_prompts="", blend_extend=6,
                        true_cfg_scale=2.0, feather_mask=True, preserve_edges=True):
        """Advanced inpainting with region-specific prompts"""
        
        try:
            # Convert tensors to PIL
            pano_pil = tensor_to_pil(panorama)
            mask_pil = tensor_to_pil(mask)
            
            # Process region prompts if provided
            regions = {}
            if region_prompts:
                for line in region_prompts.split('\n'):
                    if ':' in line:
                        region_name, region_prompt = line.split(':', 1)
                        regions[region_name.strip()] = region_prompt.strip()
            
            # Apply mask feathering if requested
            if feather_mask:
                import cv2
                mask_array = np.array(mask_pil)
                if len(mask_array.shape) == 3:
                    mask_array = mask_array[:, :, 0]
                
                # Apply Gaussian blur for feathering
                blurred_mask = cv2.GaussianBlur(mask_array.astype(np.float32), (15, 15), 5)
                mask_pil = Image.fromarray(blurred_mask.astype(np.uint8))
            
            # Adjust prompt based on inpainting type
            type_prompts = {
                "scene": "detailed scene elements, natural lighting",
                "sky": "beautiful sky, clouds, atmospheric lighting",
                "foreground": "detailed foreground objects, sharp focus",
                "background": "background scenery, depth, atmospheric perspective"
            }
            
            enhanced_prompt = f"{prompt}, {type_prompts.get(inpaint_type, '')}"
            
            # Get panorama dimensions
            height, width = pano_pil.size[1], pano_pil.size[0]
            
            # Perform inpainting
            result_pil, metadata = hyw_runtime.generate_image_panorama(
                prompt=enhanced_prompt,
                image=pano_pil,
                mask=mask_pil,
                negative_prompt=negative_prompt if negative_prompt else None,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                blend_extend=blend_extend,
                true_cfg_scale=true_cfg_scale,
                shifting_extend=0
            )
            
            # Add advanced inpainting metadata
            metadata.update({
                "inpaint_type": inpaint_type,
                "strength": strength,
                "mask_shape": mask_pil.size,
                "regions": regions,
                "feather_mask": feather_mask,
                "preserve_edges": preserve_edges,
                "enhanced_prompt": enhanced_prompt
            })
            
            # Convert back to tensor
            result_tensor = pil_to_tensor(result_pil)
            
            return (result_tensor, metadata)
            
        except Exception as e:
            print(f"Error in advanced inpainting: {e}")
            import traceback
            traceback.print_exc()
            raise e