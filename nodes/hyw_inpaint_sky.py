import torch
import numpy as np
from PIL import Image, ImageFilter
import cv2


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


class HYW_PanoInpaint_Sky:
    """HunyuanWorld Sky Inpainting node for ComfyUI"""
    
    CATEGORY = "HunyuanWorld/Generate"
    RETURN_TYPES = ("IMAGE", "HYW_METADATA")
    RETURN_NAMES = ("inpainted_panorama", "metadata")
    FUNCTION = "inpaint_sky"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hyw_runtime": ("HYW_RUNTIME",),
                "panorama": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": "beautiful sky with clouds, natural lighting"
                }),
                "guidance_scale": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31-1}),
                "sky_region_height": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 0.8, "step": 0.05}),
            },
            "optional": {
                "sky_mask": ("IMAGE",),
                "negative_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "ground, buildings, objects, low quality"
                }),
                "blend_extend": ("INT", {"default": 6, "min": 0, "max": 20}),
                "true_cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "feather_amount": ("INT", {"default": 20, "min": 0, "max": 50}),
                "sky_type": (["clear", "cloudy", "dramatic", "sunset", "sunrise", "stormy"], {"default": "cloudy"}),
            }
        }

    def create_sky_mask(self, panorama_pil, sky_region_height, feather_amount=20):
        """Create automatic sky mask for the upper portion of panorama"""
        width, height = panorama_pil.size
        
        # Create mask for upper portion
        mask = Image.new('L', (width, height), 0)
        sky_height = int(height * sky_region_height)
        
        # Fill upper region
        mask_array = np.array(mask)
        mask_array[:sky_height] = 255
        
        # Apply feathering to transition area
        if feather_amount > 0:
            transition_start = max(0, sky_height - feather_amount)
            transition_end = min(height, sky_height + feather_amount)
            
            for y in range(transition_start, transition_end):
                alpha = 1.0 - abs(y - sky_height) / feather_amount
                mask_array[y] = int(255 * alpha)
        
        return Image.fromarray(mask_array)

    def inpaint_sky(self, hyw_runtime, panorama, prompt, guidance_scale, 
                   num_inference_steps, seed, sky_region_height, sky_mask=None,
                   negative_prompt="", blend_extend=6, true_cfg_scale=2.0,
                   feather_amount=20, sky_type="cloudy"):
        """Inpaint sky region in panorama"""
        
        try:
            # Convert tensor to PIL
            pano_pil = tensor_to_pil(panorama)
            
            # Create or use provided sky mask
            if sky_mask is not None:
                mask_pil = tensor_to_pil(sky_mask)
            else:
                mask_pil = self.create_sky_mask(pano_pil, sky_region_height, feather_amount)
            
            # Enhance prompt based on sky type
            sky_prompts = {
                "clear": "clear blue sky, few white clouds, bright daylight",
                "cloudy": "cloudy sky, white and gray clouds, natural lighting",
                "dramatic": "dramatic sky, dark clouds, dynamic lighting",
                "sunset": "sunset sky, orange and pink clouds, golden hour lighting", 
                "sunrise": "sunrise sky, soft pastel colors, morning light",
                "stormy": "stormy sky, dark dramatic clouds, atmospheric"
            }
            
            enhanced_prompt = f"{prompt}, {sky_prompts.get(sky_type, '')}, panoramic sky"
            
            # Get panorama dimensions
            height, width = pano_pil.size[1], pano_pil.size[0]
            
            # Perform sky inpainting
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
            
            # Add sky-specific metadata
            metadata.update({
                "inpaint_type": "sky",
                "sky_type": sky_type,
                "sky_region_height": sky_region_height,
                "feather_amount": feather_amount,
                "mask_shape": mask_pil.size,
                "enhanced_prompt": enhanced_prompt,
                "auto_mask_used": sky_mask is None
            })
            
            # Convert back to tensor
            result_tensor = pil_to_tensor(result_pil)
            
            return (result_tensor, metadata)
            
        except Exception as e:
            print(f"Error in sky inpainting: {e}")
            import traceback
            traceback.print_exc()
            raise e


class HYW_SkyMaskGenerator:
    """Generate sky masks for panorama inpainting"""
    
    CATEGORY = "HunyuanWorld/Utils"
    RETURN_TYPES = ("IMAGE", "HYW_METADATA")
    RETURN_NAMES = ("sky_mask", "metadata")
    FUNCTION = "generate_sky_mask"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
                "sky_region_height": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 0.8, "step": 0.05}),
                "feather_amount": ("INT", {"default": 20, "min": 0, "max": 100}),
                "mask_type": (["simple", "gradient", "horizon_detection"], {"default": "gradient"}),
            },
            "optional": {
                "horizon_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blur_radius": ("INT", {"default": 5, "min": 0, "max": 20}),
            }
        }

    def detect_horizon(self, panorama_pil, threshold=0.5):
        """Detect horizon line in panorama using edge detection"""
        try:
            # Convert to grayscale
            gray = panorama_pil.convert('L')
            gray_array = np.array(gray)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray_array, 50, 150)
            
            # Find horizontal lines (potential horizon)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(gray_array.shape[1] * threshold))
            
            if lines is not None:
                # Find the most horizontal line
                best_line = None
                min_angle = float('inf')
                
                for line in lines:
                    rho, theta = line[0]
                    angle = abs(theta - np.pi/2)  # Distance from horizontal
                    
                    if angle < min_angle:
                        min_angle = angle
                        best_line = line[0]
                
                if best_line is not None:
                    rho, theta = best_line
                    # Convert to y-coordinate
                    horizon_y = int(rho / np.sin(theta)) if np.sin(theta) != 0 else gray_array.shape[0] // 2
                    return max(0, min(gray_array.shape[0], horizon_y))
            
        except Exception as e:
            print(f"Horizon detection failed: {e}")
        
        # Fallback to middle of image
        return panorama_pil.size[1] // 2

    def generate_sky_mask(self, panorama, sky_region_height, feather_amount, 
                         mask_type, horizon_threshold=0.5, blur_radius=5):
        """Generate sky mask based on specified method"""
        
        try:
            pano_pil = tensor_to_pil(panorama)
            width, height = pano_pil.size
            
            if mask_type == "simple":
                # Simple rectangular mask
                mask = Image.new('L', (width, height), 0)
                sky_height = int(height * sky_region_height)
                mask_array = np.array(mask)
                mask_array[:sky_height] = 255
                mask_pil = Image.fromarray(mask_array)
                
            elif mask_type == "gradient":
                # Gradient mask with feathering
                mask = Image.new('L', (width, height), 0)
                sky_height = int(height * sky_region_height)
                mask_array = np.array(mask)
                
                # Fill upper region
                mask_array[:sky_height] = 255
                
                # Apply gradient feathering
                if feather_amount > 0:
                    transition_start = max(0, sky_height - feather_amount)
                    transition_end = min(height, sky_height + feather_amount)
                    
                    for y in range(transition_start, transition_end):
                        alpha = 1.0 - abs(y - sky_height) / feather_amount
                        mask_array[y] = int(255 * alpha)
                
                mask_pil = Image.fromarray(mask_array)
                
            elif mask_type == "horizon_detection":
                # Automatic horizon detection
                horizon_y = self.detect_horizon(pano_pil, horizon_threshold)
                
                mask = Image.new('L', (width, height), 0)
                mask_array = np.array(mask)
                mask_array[:horizon_y] = 255
                
                # Apply feathering around detected horizon
                if feather_amount > 0:
                    transition_start = max(0, horizon_y - feather_amount)
                    transition_end = min(height, horizon_y + feather_amount)
                    
                    for y in range(transition_start, transition_end):
                        alpha = 1.0 - abs(y - horizon_y) / feather_amount
                        mask_array[y] = int(255 * alpha)
                
                mask_pil = Image.fromarray(mask_array)
            
            # Apply blur if requested
            if blur_radius > 0:
                mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # Convert to tensor
            mask_tensor = pil_to_tensor(mask_pil)
            
            metadata = {
                "mask_type": mask_type,
                "sky_region_height": sky_region_height,
                "feather_amount": feather_amount,
                "blur_radius": blur_radius,
                "horizon_threshold": horizon_threshold if mask_type == "horizon_detection" else None,
                "mask_size": mask_pil.size
            }
            
            return (mask_tensor, metadata)
            
        except Exception as e:
            print(f"Error generating sky mask: {e}")
            import traceback
            traceback.print_exc()
            raise e