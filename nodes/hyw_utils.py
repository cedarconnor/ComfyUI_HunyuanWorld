import hashlib
import json
import numpy as np
import torch
from PIL import Image, ImageFilter
import cv2
from datetime import datetime
from typing import Dict, List, Tuple, Optional


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


class HYW_SeamlessWrap360:
    """Make panorama seamless by blending left and right edges"""
    
    CATEGORY = "HunyuanWorld/Utils"
    RETURN_TYPES = ("IMAGE", "HYW_METADATA")
    RETURN_NAMES = ("seamless_panorama", "wrap_metadata")
    FUNCTION = "make_seamless"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
                "blend_mode": (["linear", "cosine", "gaussian", "feather"], {"default": "cosine"}),
                "blend_width": ("INT", {"default": 32, "min": 8, "max": 256, "step": 8}),
            },
            "optional": {
                "enable_vertical_wrap": ("BOOLEAN", {"default": False}),
                "vertical_blend_height": ("INT", {"default": 16, "min": 4, "max": 128, "step": 4}),
                "preserve_center": ("BOOLEAN", {"default": True}),
                "edge_detection": ("BOOLEAN", {"default": True}),
            }
        }

    def create_blend_mask(self, width, height, blend_width, mode="cosine"):
        """Create blending mask for seamless wrapping"""
        mask = np.ones((height, width), dtype=np.float32)
        
        if mode == "linear":
            # Linear blend
            for x in range(blend_width):
                alpha = x / blend_width
                mask[:, x] = alpha
                mask[:, width - blend_width + x] = 1 - alpha
                
        elif mode == "cosine":
            # Cosine blend (smoother)
            for x in range(blend_width):
                alpha = 0.5 * (1 - np.cos(np.pi * x / blend_width))
                mask[:, x] = alpha
                mask[:, width - blend_width + x] = 1 - alpha
                
        elif mode == "gaussian":
            # Gaussian blend
            sigma = blend_width / 3.0
            for x in range(blend_width):
                alpha = np.exp(-0.5 * ((x - blend_width/2) / sigma) ** 2)
                alpha = np.clip(alpha, 0, 1)
                mask[:, x] = alpha
                mask[:, width - blend_width + x] = 1 - alpha
                
        elif mode == "feather":
            # Feathered blend with soft edges
            for x in range(blend_width):
                t = x / blend_width
                alpha = 3 * t**2 - 2 * t**3  # Smoothstep function
                mask[:, x] = alpha
                mask[:, width - blend_width + x] = 1 - alpha
        
        return mask

    def detect_edge_content(self, image_array, blend_width):
        """Detect if edges have significant content differences"""
        try:
            height, width = image_array.shape[:2]
            
            # Extract left and right edge regions
            left_edge = image_array[:, :blend_width]
            right_edge = image_array[:, -blend_width:]
            
            # Flip right edge to match left edge orientation
            right_edge_flipped = np.fliplr(right_edge)
            
            # Calculate difference
            diff = np.abs(left_edge.astype(float) - right_edge_flipped.astype(float))
            mean_diff = np.mean(diff)
            
            # Return True if edges are significantly different
            return mean_diff > 30  # Threshold for difference
            
        except Exception as e:
            print(f"Edge detection failed: {e}")
            return True  # Default to blending if detection fails

    def make_seamless(self, panorama, blend_mode, blend_width, 
                     enable_vertical_wrap=False, vertical_blend_height=16,
                     preserve_center=True, edge_detection=True):
        """Make panorama seamless for 360-degree viewing"""
        
        try:
            pano_pil = tensor_to_pil(panorama)
            pano_array = np.array(pano_pil)
            
            height, width = pano_array.shape[:2]
            
            # Check if blending is needed
            needs_blending = True
            if edge_detection:
                needs_blending = self.detect_edge_content(pano_array, blend_width)
                print(f"Edge content analysis: {'Blending needed' if needs_blending else 'Edges already similar'}")
            
            result_array = pano_array.copy()
            
            if needs_blending:
                # Horizontal seamless wrapping
                blend_mask = self.create_blend_mask(width, height, blend_width, blend_mode)
                
                # Extract edge regions
                left_edge = pano_array[:, :blend_width].copy()
                right_edge = pano_array[:, -blend_width:].copy()
                
                # Create blended edge
                if len(pano_array.shape) == 3:  # Color image
                    for c in range(pano_array.shape[2]):
                        # Blend left edge with right edge
                        blended_left = (left_edge[:, :, c] * blend_mask[:, :blend_width] + 
                                      right_edge[:, :, c] * (1 - blend_mask[:, :blend_width]))
                        
                        # Blend right edge with left edge
                        blended_right = (right_edge[:, :, c] * blend_mask[:, -blend_width:] + 
                                       left_edge[:, :, c] * (1 - blend_mask[:, -blend_width:]))
                        
                        result_array[:, :blend_width, c] = blended_left
                        result_array[:, -blend_width:, c] = blended_right
                else:  # Grayscale
                    blended_left = (left_edge * blend_mask[:, :blend_width] + 
                                  right_edge * (1 - blend_mask[:, :blend_width]))
                    
                    blended_right = (right_edge * blend_mask[:, -blend_width:] + 
                                   left_edge * (1 - blend_mask[:, -blend_width:]))
                    
                    result_array[:, :blend_width] = blended_left
                    result_array[:, -blend_width:] = blended_right
                
                # Vertical seamless wrapping (polar caps)
                if enable_vertical_wrap:
                    v_mask = np.ones((height, width), dtype=np.float32)
                    
                    # Create vertical blend mask
                    for y in range(vertical_blend_height):
                        alpha = 0.5 * (1 - np.cos(np.pi * y / vertical_blend_height))
                        v_mask[y, :] = alpha
                        v_mask[height - vertical_blend_height + y, :] = alpha
                    
                    # Apply vertical blending (simplified - mirror top/bottom)
                    top_region = result_array[:vertical_blend_height]
                    bottom_region = result_array[-vertical_blend_height:]
                    
                    # Blend top with flipped bottom
                    flipped_bottom = np.flipud(bottom_region)
                    flipped_top = np.flipud(top_region)
                    
                    if len(result_array.shape) == 3:
                        for c in range(result_array.shape[2]):
                            result_array[:vertical_blend_height, :, c] = (
                                top_region[:, :, c] * v_mask[:vertical_blend_height, :] + 
                                flipped_bottom[:, :, c] * (1 - v_mask[:vertical_blend_height, :])
                            )
                            result_array[-vertical_blend_height:, :, c] = (
                                bottom_region[:, :, c] * v_mask[-vertical_blend_height:, :] + 
                                flipped_top[:, :, c] * (1 - v_mask[-vertical_blend_height:, :])
                            )
                    else:
                        result_array[:vertical_blend_height, :] = (
                            top_region * v_mask[:vertical_blend_height, :] + 
                            flipped_bottom * (1 - v_mask[:vertical_blend_height, :])
                        )
                        result_array[-vertical_blend_height:, :] = (
                            bottom_region * v_mask[-vertical_blend_height:, :] + 
                            flipped_top * (1 - v_mask[-vertical_blend_height:, :])
                        )
            
            # Convert back to PIL and then tensor
            result_pil = Image.fromarray(result_array.astype(np.uint8))
            result_tensor = pil_to_tensor(result_pil)
            
            # Create metadata
            wrap_metadata = {
                'blend_mode': blend_mode,
                'blend_width': blend_width,
                'enable_vertical_wrap': enable_vertical_wrap,
                'vertical_blend_height': vertical_blend_height,
                'preserve_center': preserve_center,
                'edge_detection': edge_detection,
                'blending_applied': needs_blending,
                'image_size': [width, height]
            }
            
            return (result_tensor, wrap_metadata)
            
        except Exception as e:
            print(f"Error in seamless wrapping: {e}")
            import traceback
            traceback.print_exc()
            raise e


class HYW_PanoramaValidator:
    """Validate panorama quality and properties"""
    
    CATEGORY = "HunyuanWorld/Utils"
    RETURN_TYPES = ("HYW_METADATA", "BOOLEAN")
    RETURN_NAMES = ("validation_results", "is_valid")
    FUNCTION = "validate_panorama"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panorama": ("IMAGE",),
            },
            "optional": {
                "expected_aspect_ratio": ("FLOAT", {"default": 2.0, "min": 1.5, "max": 4.0}),
                "min_resolution": ("INT", {"default": 1024, "min": 256, "max": 8192}),
                "check_seams": ("BOOLEAN", {"default": True}),
                "check_distortion": ("BOOLEAN", {"default": True}),
                "check_exposure": ("BOOLEAN", {"default": True}),
                "quality_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
            }
        }

    def check_aspect_ratio(self, image, expected_ratio=2.0, tolerance=0.1):
        """Check if image has correct panorama aspect ratio"""
        width, height = image.size
        actual_ratio = width / height
        return abs(actual_ratio - expected_ratio) <= tolerance

    def check_seam_quality(self, image_array, check_width=32):
        """Check quality of left-right seam"""
        try:
            height, width = image_array.shape[:2]
            
            left_edge = image_array[:, :check_width]
            right_edge = image_array[:, -check_width:]
            
            # Compare edges
            diff = np.abs(left_edge.astype(float) - right_edge.astype(float))
            mean_diff = np.mean(diff)
            
            # Good seam should have low difference
            seam_quality = max(0, 1 - mean_diff / 128.0)
            return seam_quality, mean_diff
            
        except Exception as e:
            print(f"Seam check failed: {e}")
            return 0.5, 64.0

    def check_distortion_artifacts(self, image_array):
        """Check for common panorama distortion artifacts"""
        try:
            # Check for extreme stretching at poles (top/bottom edges)
            height, width = image_array.shape[:2]
            
            # Sample top and bottom strips
            top_strip = image_array[:height//10, :]
            bottom_strip = image_array[-height//10:, :]
            
            # Calculate variance - high variance indicates stretching artifacts
            top_variance = np.var(top_strip)
            bottom_variance = np.var(bottom_strip)
            center_variance = np.var(image_array[height//3:2*height//3, width//4:3*width//4])
            
            # Compare edge variance to center variance
            if center_variance > 0:
                top_ratio = top_variance / center_variance
                bottom_ratio = bottom_variance / center_variance
                
                # Good panorama should have reasonable variance ratios
                distortion_score = 1.0 - min(1.0, max(top_ratio, bottom_ratio) / 2.0)
            else:
                distortion_score = 0.5
            
            return distortion_score
            
        except Exception as e:
            print(f"Distortion check failed: {e}")
            return 0.5

    def check_exposure_balance(self, image_array):
        """Check exposure and brightness balance"""
        try:
            # Convert to grayscale for analysis
            if len(image_array.shape) == 3:
                gray = np.mean(image_array, axis=2)
            else:
                gray = image_array
            
            # Calculate overall brightness statistics
            mean_brightness = np.mean(gray)
            brightness_std = np.std(gray)
            
            # Check for over/underexposure
            overexposed_pixels = np.sum(gray > 240) / gray.size
            underexposed_pixels = np.sum(gray < 15) / gray.size
            
            # Good exposure should have reasonable brightness and low clipping
            exposure_score = 1.0
            
            if overexposed_pixels > 0.05:  # More than 5% overexposed
                exposure_score -= overexposed_pixels * 2
            
            if underexposed_pixels > 0.05:  # More than 5% underexposed
                exposure_score -= underexposed_pixels * 2
            
            if brightness_std < 10:  # Too uniform (flat)
                exposure_score -= 0.3
            
            exposure_score = max(0.0, min(1.0, exposure_score))
            
            return exposure_score, {
                'mean_brightness': float(mean_brightness),
                'brightness_std': float(brightness_std),
                'overexposed_ratio': float(overexposed_pixels),
                'underexposed_ratio': float(underexposed_pixels)
            }
            
        except Exception as e:
            print(f"Exposure check failed: {e}")
            return 0.5, {}

    def validate_panorama(self, panorama, expected_aspect_ratio=2.0, min_resolution=1024,
                         check_seams=True, check_distortion=True, check_exposure=True,
                         quality_threshold=0.7):
        """Validate panorama quality and properties"""
        
        try:
            pano_pil = tensor_to_pil(panorama)
            pano_array = np.array(pano_pil)
            
            validation_results = {
                'image_size': list(pano_pil.size),
                'aspect_ratio': pano_pil.size[0] / pano_pil.size[1],
                'checks_performed': []
            }
            
            scores = []
            
            # Check aspect ratio
            aspect_valid = self.check_aspect_ratio(pano_pil, expected_aspect_ratio)
            validation_results['aspect_ratio_valid'] = aspect_valid
            validation_results['checks_performed'].append('aspect_ratio')
            scores.append(1.0 if aspect_valid else 0.0)
            
            # Check resolution
            width, height = pano_pil.size
            resolution_valid = width >= min_resolution and height >= min_resolution // 2
            validation_results['resolution_valid'] = resolution_valid
            validation_results['checks_performed'].append('resolution')
            scores.append(1.0 if resolution_valid else 0.0)
            
            # Check seams
            if check_seams:
                seam_quality, seam_diff = self.check_seam_quality(pano_array)
                validation_results['seam_quality'] = seam_quality
                validation_results['seam_difference'] = seam_diff
                validation_results['checks_performed'].append('seams')
                scores.append(seam_quality)
            
            # Check distortion
            if check_distortion:
                distortion_score = self.check_distortion_artifacts(pano_array)
                validation_results['distortion_score'] = distortion_score
                validation_results['checks_performed'].append('distortion')
                scores.append(distortion_score)
            
            # Check exposure
            if check_exposure:
                exposure_score, exposure_details = self.check_exposure_balance(pano_array)
                validation_results['exposure_score'] = exposure_score
                validation_results['exposure_details'] = exposure_details
                validation_results['checks_performed'].append('exposure')
                scores.append(exposure_score)
            
            # Calculate overall quality score
            overall_score = np.mean(scores) if scores else 0.0
            validation_results['overall_quality_score'] = overall_score
            validation_results['quality_threshold'] = quality_threshold
            
            # Determine if panorama is valid
            is_valid = overall_score >= quality_threshold
            validation_results['is_valid'] = is_valid
            
            # Add quality rating
            if overall_score >= 0.9:
                quality_rating = "excellent"
            elif overall_score >= 0.8:
                quality_rating = "good"
            elif overall_score >= 0.6:
                quality_rating = "fair"
            else:
                quality_rating = "poor"
            
            validation_results['quality_rating'] = quality_rating
            
            print(f"Panorama validation results:")
            print(f"  - Overall quality: {quality_rating} ({overall_score:.2f})")
            print(f"  - Valid: {is_valid}")
            print(f"  - Checks performed: {', '.join(validation_results['checks_performed'])}")
            
            return (validation_results, is_valid)
            
        except Exception as e:
            print(f"Error in panorama validation: {e}")
            import traceback
            traceback.print_exc()
            raise e


class HYW_MetadataManager:
    """Manage and track metadata throughout the workflow"""
    
    CATEGORY = "HunyuanWorld/Utils"
    RETURN_TYPES = ("HYW_METADATA",)
    RETURN_NAMES = ("combined_metadata",)
    FUNCTION = "combine_metadata"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "primary_metadata": ("HYW_METADATA",),
            },
            "optional": {
                "secondary_metadata": ("HYW_METADATA",),
                "additional_metadata": ("HYW_METADATA",),
                "user_notes": ("STRING", {"multiline": True, "default": ""}),
                "workflow_stage": ("STRING", {"default": "final"}),
                "generate_hash": ("BOOLEAN", {"default": True}),
            }
        }

    def graph_hash(self, metadata: dict) -> str:
        """Generate hash from metadata for tracking"""
        try:
            # Create a stable string representation
            metadata_str = json.dumps(metadata, sort_keys=True, default=str)
            return hashlib.sha256(metadata_str.encode("utf-8")).hexdigest()[:16]
        except Exception as e:
            print(f"Hash generation failed: {e}")
            return "unknown"

    def combine_metadata(self, primary_metadata, secondary_metadata=None, 
                        additional_metadata=None, user_notes="", 
                        workflow_stage="final", generate_hash=True):
        """Combine multiple metadata sources"""
        
        try:
            combined = {
                'workflow_stage': workflow_stage,
                'primary_data': primary_metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add secondary metadata if provided
            if secondary_metadata:
                combined['secondary_data'] = secondary_metadata
            
            # Add additional metadata if provided  
            if additional_metadata:
                combined['additional_data'] = additional_metadata
            
            # Add user notes
            if user_notes.strip():
                combined['user_notes'] = user_notes.strip()
            
            # Generate workflow hash
            if generate_hash:
                combined['workflow_hash'] = self.graph_hash(combined)
            
            # Extract key metrics for quick access
            metrics = {}
            
            # Try to extract common metrics from all metadata sources
            for source_name, source_data in [
                ('primary', primary_metadata),
                ('secondary', secondary_metadata),
                ('additional', additional_metadata)
            ]:
                if source_data:
                    # Extract generation parameters
                    if 'prompt' in source_data:
                        metrics['prompt'] = source_data['prompt']
                    if 'seed' in source_data:
                        metrics['seed'] = source_data['seed']
                    if 'guidance_scale' in source_data:
                        metrics['guidance_scale'] = source_data['guidance_scale']
                    
                    # Extract quality metrics
                    if 'overall_quality_score' in source_data:
                        metrics['quality_score'] = source_data['overall_quality_score']
                    if 'total_triangles' in source_data:
                        metrics['mesh_complexity'] = source_data['total_triangles']
            
            combined['key_metrics'] = metrics
            
            return (combined,)
            
        except Exception as e:
            print(f"Error in metadata combination: {e}")
            import traceback
            traceback.print_exc()
            raise e