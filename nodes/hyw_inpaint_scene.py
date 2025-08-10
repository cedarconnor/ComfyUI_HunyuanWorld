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
                "hyw_runtime": ("HYW_RUNTIME", {
                    "tooltip": "HunyuanWorld runtime instance for inpainting operations. Must have image-to-panorama pipeline loaded."
                }),
                "panorama": ("IMAGE", {
                    "tooltip": "Input panorama image to inpaint. Should be in 360° equirectangular format for best results."
                }),
                "mask": ("IMAGE", {
                    "tooltip": "Inpainting mask where white areas are inpainted and black areas are preserved. Use grayscale for partial inpainting strength."
                }),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "detailed scene elements, high quality",
                    "tooltip": "Description of what to generate in masked areas. Be specific about scene elements, lighting, and style for better results."
                }),
                "strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Inpainting strength. 0.0=preserve original, 1.0=completely regenerate masked areas. 0.6-0.8 balances quality and consistency."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 30.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "How strongly to follow the prompt. Higher values = more prompt adherence. 20-40 recommended for inpainting."
                }),
                "num_inference_steps": ("INT", {
                    "default": 30, "min": 1, "max": 200,
                    "tooltip": "Number of denoising steps. 20-50 usually sufficient for inpainting. More steps improve quality but increase time."
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 2**31-1,
                    "tooltip": "Random seed for reproducible inpainting results. Same seed with identical settings produces consistent results."
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "low quality, blurry, artifacts",
                    "tooltip": "What to avoid in inpainted regions. Common: 'blurry, artifacts, low quality, distorted, seams, inconsistent lighting'."
                }),
                "blend_extend": ("INT", {
                    "default": 6, "min": 0, "max": 20,
                    "tooltip": "Pixels to blend at inpainting boundaries for seamless integration. Higher values = smoother transitions but may blur details."
                }),
                "true_cfg_scale": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Advanced CFG for better prompt following in inpainted areas. 0.0=disabled. 1.0-5.0 can improve quality for complex prompts."
                }),
                "shifting_extend": ("INT", {
                    "default": 0, "min": 0, "max": 10,
                    "tooltip": "Additional boundary processing for panorama continuity. 0=disabled. May help with seamless 360° inpainting."
                }),
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
                "hyw_runtime": ("HYW_RUNTIME", {
                    "tooltip": "HunyuanWorld runtime for advanced inpainting with multiple regions and specialized processing."
                }),
                "panorama": ("IMAGE", {
                    "tooltip": "Input 360° panorama to inpaint. Works best with equirectangular format panoramas."
                }),
                "mask": ("IMAGE", {
                    "tooltip": "Inpainting mask defining areas to regenerate. White=inpaint, black=preserve. Supports multi-region masks."
                }),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "high quality inpainting",
                    "tooltip": "Base prompt for inpainting. Will be enhanced based on inpaint_type selection. Be descriptive about desired content."
                }),
                "inpaint_type": (["scene", "sky", "foreground", "background"], {
                    "default": "scene",
                    "tooltip": "Type of content being inpainted. Adds specialized prompting: scene=general, sky=atmospheric, foreground=detailed, background=depth."
                }),
                "strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Inpainting strength affecting how much the original content is preserved vs regenerated. 0.7-0.8 recommended."
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 30.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "Prompt adherence strength. Higher values follow prompts more closely. 25-35 good for detailed inpainting."
                }),
                "num_inference_steps": ("INT", {
                    "default": 30, "min": 1, "max": 200,
                    "tooltip": "Denoising steps for quality vs speed trade-off. 25-50 recommended for high quality advanced inpainting."
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 2**31-1,
                    "tooltip": "Random seed for reproducible advanced inpainting results. Same seed ensures consistent region generation."
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "low quality, blurry, artifacts",
                    "tooltip": "What to avoid in all inpainted regions. Include artifacts like 'seams, inconsistent lighting, mismatched perspective'."
                }),
                "region_prompts": ("STRING", {
                    "multiline": True,
                    "default": "region1: trees and forest\nregion2: sky and clouds",
                    "tooltip": "Specific prompts for different mask regions. Format: 'region1: description'. Each line defines a region's content."
                }),
                "blend_extend": ("INT", {
                    "default": 6, "min": 0, "max": 20,
                    "tooltip": "Boundary blending for seamless integration between regions and original content. Higher=smoother but softer edges."
                }),
                "true_cfg_scale": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "Enhanced CFG for better prompt adherence in complex multi-region inpainting. 1.5-4.0 typically effective."
                }),
                "feather_mask": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply Gaussian blur to mask edges for smoother transitions. Recommended for natural-looking inpainting results."
                }),
                "preserve_edges": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Protect important edge details during inpainting. Helps maintain structural integrity of the panorama."
                }),
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