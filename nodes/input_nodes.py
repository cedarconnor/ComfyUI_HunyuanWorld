import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple

from ..core.data_types import PanoramaImage, ObjectLabels, SceneMask

class HunyuanTextInput:
    """Text input node for world generation prompts"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful natural landscape with mountains and forests",
                    "tooltip": "Text description of the world you want to generate. Be specific about environment, lighting, objects, and atmosphere for best results."
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**32 - 1,
                    "step": 1,
                    "tooltip": "Random seed for reproducible generation. Use -1 for random seed, or specific number to recreate exact results."
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional: Describe what you DON'T want in the generated world (e.g., 'blurry, low quality, distorted')."
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("prompt", "seed", "negative_prompt")
    FUNCTION = "process_text_input"
    CATEGORY = "HunyuanWorld/Input"
    
    def process_text_input(self, prompt: str, seed: int, negative_prompt: str = ""):
        """Process and validate text input"""
        
        # Handle random seed
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        # Clean and validate prompt
        prompt = prompt.strip()
        if not prompt:
            prompt = "A beautiful natural landscape"
        
        negative_prompt = negative_prompt.strip()
        
        return (prompt, seed, negative_prompt)

class HunyuanImageInput:
    """Image input node for image-to-panorama conversion"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "resize_mode": (["stretch", "crop", "pad"], {
                    "default": "stretch",
                    "tooltip": "How to resize input image: 'stretch' changes aspect ratio, 'crop' maintains ratio by cutting, 'pad' maintains ratio by adding borders."
                }),
                "target_width": ("INT", {
                    "default": 1024,
                    "min": 512,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Target width for panoramic image. Higher values = better quality but more VRAM. 1024-2048 recommended."
                }),
                "target_height": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Target height for panoramic image. Should be half of width for proper 2:1 panoramic ratio."
                }),
                "preprocessing": (["none", "enhance", "denoise"], {
                    "default": "none",
                    "tooltip": "Image preprocessing: 'none' = no changes, 'enhance' = improve contrast/brightness, 'denoise' = reduce noise."
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("processed_image", "width", "height")
    FUNCTION = "process_image_input"
    CATEGORY = "HunyuanWorld/Input"
    
    def process_image_input(self, 
                          image: torch.Tensor,
                          resize_mode: str = "stretch",
                          target_width: int = 1024,
                          target_height: int = 512,
                          preprocessing: str = "none"):
        """Process and prepare image for panoramic conversion"""
        
        # ComfyUI images are typically in shape (B, H, W, C)
        if len(image.shape) == 4:
            batch_size, height, width, channels = image.shape
            # Use first image if batch
            if batch_size > 1:
                image = image[0:1]
        else:
            height, width, channels = image.shape[-3:]
        
        # Resize image based on mode
        processed_image = self._resize_image(image, target_width, target_height, resize_mode)
        
        # Apply preprocessing if requested
        if preprocessing == "enhance":
            processed_image = self._enhance_image(processed_image)
        elif preprocessing == "denoise":
            processed_image = self._denoise_image(processed_image)
        
        return (processed_image, target_width, target_height)
    
    def _resize_image(self, image: torch.Tensor, target_w: int, target_h: int, mode: str) -> torch.Tensor:
        """Resize image according to specified mode"""
        
        # Convert to PIL for easier manipulation
        if len(image.shape) == 4:  # Batch dimension
            pil_image = self._tensor_to_pil(image[0])
        else:
            pil_image = self._tensor_to_pil(image)
        
        original_w, original_h = pil_image.size
        
        if mode == "stretch":
            # Simply stretch to target size
            resized = pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
        elif mode == "crop":
            # Crop to target aspect ratio, then resize
            target_aspect = target_w / target_h
            current_aspect = original_w / original_h
            
            if current_aspect > target_aspect:
                # Image is wider, crop width
                new_w = int(original_h * target_aspect)
                left = (original_w - new_w) // 2
                pil_image = pil_image.crop((left, 0, left + new_w, original_h))
            else:
                # Image is taller, crop height
                new_h = int(original_w / target_aspect)
                top = (original_h - new_h) // 2
                pil_image = pil_image.crop((0, top, original_w, top + new_h))
            
            resized = pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
        elif mode == "pad":
            # Pad to target aspect ratio, then resize
            target_aspect = target_w / target_h
            current_aspect = original_w / original_h
            
            if current_aspect > target_aspect:
                # Image is wider, pad height
                new_h = int(original_w / target_aspect)
                new_image = Image.new("RGB", (original_w, new_h), (0, 0, 0))
                paste_y = (new_h - original_h) // 2
                new_image.paste(pil_image, (0, paste_y))
            else:
                # Image is taller, pad width
                new_w = int(original_h * target_aspect)
                new_image = Image.new("RGB", (new_w, original_h), (0, 0, 0))
                paste_x = (new_w - original_w) // 2
                new_image.paste(pil_image, (paste_x, 0))
            
            resized = new_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # Convert back to tensor
        result_tensor = self._pil_to_tensor(resized)
        
        # Ensure batch dimension
        if len(result_tensor.shape) == 3:
            result_tensor = result_tensor.unsqueeze(0)
        
        return result_tensor
    
    def _enhance_image(self, image: torch.Tensor) -> torch.Tensor:
        """Apply basic image enhancement"""
        # Simple enhancement: adjust contrast and brightness
        enhanced = torch.clamp(image * 1.1 + 0.05, 0.0, 1.0)
        return enhanced
    
    def _denoise_image(self, image: torch.Tensor) -> torch.Tensor:
        """Apply basic denoising"""
        # Simple denoising: slight gaussian blur
        # This is a placeholder - in real implementation you'd use proper denoising
        return image
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image"""
        # Assume tensor is in range [0, 1] and format (H, W, C)
        if len(tensor.shape) == 4:
            tensor = tensor[0]  # Remove batch dim
        
        # Convert to numpy and scale to [0, 255]
        numpy_array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # Ensure RGB format
        if numpy_array.shape[-1] == 3:
            return Image.fromarray(numpy_array, 'RGB')
        elif numpy_array.shape[-1] == 4:
            return Image.fromarray(numpy_array, 'RGBA')
        else:
            # Grayscale
            return Image.fromarray(numpy_array.squeeze(), 'L')
    
    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor"""
        # Convert to numpy array
        numpy_array = np.array(pil_image)
        
        # Normalize to [0, 1]
        tensor = torch.from_numpy(numpy_array).float() / 255.0
        
        # Ensure 3D (H, W, C)
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        
        return tensor

class HunyuanPromptProcessor:
    """Advanced prompt processing and enhancement node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "style": (["realistic", "artistic", "fantasy", "sci-fi", "minimalist"], {
                    "default": "realistic",
                    "tooltip": "Visual style for the generated world. 'realistic' for photorealistic, 'artistic' for painterly look."
                }),
                "lighting": (["natural", "dramatic", "soft", "golden_hour", "night"], {
                    "default": "natural",
                    "tooltip": "Lighting conditions: 'natural' = balanced daylight, 'golden_hour' = warm sunset/sunrise, 'dramatic' = strong shadows."
                }),
                "atmosphere": (["clear", "misty", "stormy", "serene", "mysterious"], {
                    "default": "clear",
                    "tooltip": "Atmospheric mood: 'clear' = no special effects, 'misty' = fog/haze, 'stormy' = dramatic weather."
                }),
                "quality_boost": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically add quality-enhancing terms to prompt (4k, high detail, sharp). Recommended for better results."
                }),
                "custom_suffix": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Additional prompt text to append. Use for specific details, camera settings, or artistic directions."
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "process_prompt"
    CATEGORY = "HunyuanWorld/Input"
    
    def process_prompt(self, 
                      prompt: str,
                      style: str = "realistic",
                      lighting: str = "natural",
                      atmosphere: str = "clear",
                      quality_boost: bool = True,
                      custom_suffix: str = ""):
        """Process and enhance prompts for better generation"""
        
        enhanced_parts = [prompt.strip()]
        
        # Add style modifiers
        style_modifiers = {
            "realistic": "photorealistic, highly detailed",
            "artistic": "artistic, painterly, creative composition",
            "fantasy": "fantasy style, magical atmosphere, otherworldly",
            "sci-fi": "futuristic, science fiction, high-tech",
            "minimalist": "minimalist, clean, simple composition"
        }
        
        lighting_modifiers = {
            "natural": "natural lighting",
            "dramatic": "dramatic lighting, strong shadows",
            "soft": "soft lighting, gentle shadows",
            "golden_hour": "golden hour lighting, warm tones",
            "night": "night scene, moonlight, artificial lighting"
        }
        
        atmosphere_modifiers = {
            "clear": "clear atmosphere",
            "misty": "misty, fog, atmospheric haze",
            "stormy": "stormy weather, dramatic clouds",
            "serene": "peaceful, calm, serene atmosphere",
            "mysterious": "mysterious, enigmatic mood"
        }
        
        # Add modifiers
        if style in style_modifiers:
            enhanced_parts.append(style_modifiers[style])
        
        if lighting in lighting_modifiers:
            enhanced_parts.append(lighting_modifiers[lighting])
        
        if atmosphere in atmosphere_modifiers:
            enhanced_parts.append(atmosphere_modifiers[atmosphere])
        
        # Add quality boost
        if quality_boost:
            enhanced_parts.append("high quality, 4k resolution, sharp details")
        
        # Add custom suffix
        if custom_suffix.strip():
            enhanced_parts.append(custom_suffix.strip())
        
        # Join all parts
        enhanced_prompt = ", ".join(enhanced_parts)
        
        return (enhanced_prompt,)

class HunyuanObjectLabeler:
    """Object labeling node for multi-layer scene generation"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fg_labels_1": ("STRING", {
                    "multiline": True,
                    "default": "trees, mountains, buildings",
                    "tooltip": "Primary foreground object labels (layer 1), separated by commas. These will be processed first with highest priority."
                }),
                "fg_labels_2": ("STRING", {
                    "multiline": True, 
                    "default": "rocks, grass, bushes",
                    "tooltip": "Secondary foreground object labels (layer 2), separated by commas. These will be processed second with lower priority."
                }),
                "scene_class": (["outdoor", "indoor", "landscape", "urban", "natural", "architectural"], {
                    "default": "outdoor",
                    "tooltip": "Scene classification to guide object detection and layer separation. Repository default is 'outdoor'."
                }),
            },
            "optional": {
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Minimum confidence threshold for object detection. Higher = more selective detection."
                }),
                "label_weights": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional label weights in format 'label1:weight1, label2:weight2'. E.g., 'trees:1.0, buildings:0.8'"
                })
            }
        }
    
    RETURN_TYPES = ("OBJECT_LABELS",)
    RETURN_NAMES = ("object_labels",)
    FUNCTION = "create_object_labels"
    CATEGORY = "HunyuanWorld/Input"
    
    def create_object_labels(self,
                           fg_labels_1: str,
                           fg_labels_2: str,
                           scene_class: str = "outdoor",
                           confidence_threshold: float = 0.5,
                           label_weights: str = ""):
        """Create object labels for layered scene generation"""
        
        # Parse label lists
        labels_1 = [label.strip() for label in fg_labels_1.split(",") if label.strip()]
        labels_2 = [label.strip() for label in fg_labels_2.split(",") if label.strip()]
        
        # Parse label weights
        weights = {}
        if label_weights.strip():
            for weight_pair in label_weights.split(","):
                if ":" in weight_pair:
                    label, weight = weight_pair.split(":", 1)
                    try:
                        weights[label.strip()] = float(weight.strip())
                    except ValueError:
                        print(f"Invalid weight format: {weight_pair}")
        
        # Create ObjectLabels instance
        object_labels = ObjectLabels(
            fg_labels_1=labels_1,
            fg_labels_2=labels_2,
            scene_class=scene_class,
            confidence_threshold=confidence_threshold,
            label_weights=weights,
            metadata={
                "total_labels": len(labels_1) + len(labels_2),
                "primary_labels": len(labels_1),
                "secondary_labels": len(labels_2)
            }
        )
        
        return (object_labels,)

class HunyuanMaskCreator:
    """Create scene masks for inpainting and layered processing"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_input": ("IMAGE", {
                    "tooltip": "Input mask image (black/white or grayscale). White areas will be processed."
                }),
                "mask_type": (["scene", "sky", "object", "custom"], {
                    "default": "scene",
                    "tooltip": "Type of mask: 'scene' for scene inpainting, 'sky' for sky replacement, 'object' for object-specific, 'custom' for general use."
                }),
            },
            "optional": {
                "invert": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Invert the mask (make black areas white and vice versa)"
                }),
                "feather": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Feathering amount for soft mask edges. Higher = softer transitions."
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Threshold for converting grayscale to binary mask. Values above threshold become white."
                }),
                "target_regions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional comma-separated list of target region names (e.g., 'foreground, background, sky')"
                })
            }
        }
    
    RETURN_TYPES = ("SCENE_MASK", "IMAGE")
    RETURN_NAMES = ("scene_mask", "preview_mask")
    FUNCTION = "create_mask"
    CATEGORY = "HunyuanWorld/Input"
    
    def create_mask(self,
                   mask_input: torch.Tensor,
                   mask_type: str = "scene",
                   invert: bool = False,
                   feather: float = 0.0,
                   threshold: float = 0.5,
                   target_regions: str = ""):
        """Create scene mask from input image"""
        
        # Process input mask
        if len(mask_input.shape) == 4:
            # Remove batch dimension
            mask_tensor = mask_input[0]
        else:
            mask_tensor = mask_input
        
        # Convert to grayscale if needed
        if len(mask_tensor.shape) == 3 and mask_tensor.shape[-1] > 1:
            # Convert RGB to grayscale
            mask_tensor = torch.mean(mask_tensor, dim=-1)
        elif len(mask_tensor.shape) == 3:
            mask_tensor = mask_tensor[:, :, 0]
        
        # Apply threshold to create binary mask
        binary_mask = (mask_tensor > threshold).float()
        
        # Parse target regions
        regions = []
        if target_regions.strip():
            regions = [region.strip() for region in target_regions.split(",") if region.strip()]
        
        # Create SceneMask instance
        scene_mask = SceneMask(
            mask=binary_mask,
            mask_type=mask_type,
            invert=invert,
            feather=feather,
            target_regions=regions,
            metadata={
                "threshold": threshold,
                "original_shape": mask_input.shape,
                "mask_coverage": torch.sum(binary_mask).item() / (binary_mask.shape[0] * binary_mask.shape[1])
            }
        )
        
        # Create preview image (convert mask back to 3-channel for display)
        processed_mask = scene_mask.get_processed_mask()
        preview_mask = processed_mask.unsqueeze(-1).repeat(1, 1, 3).unsqueeze(0)
        
        return (scene_mask, preview_mask)