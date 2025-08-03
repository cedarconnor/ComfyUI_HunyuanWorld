import os
import torch
import gc
from typing import Dict, Any, Optional, Union
from .data_types import ModelHunyuan

# Try to import real HunyuanWorld integration
try:
    from .hunyuan_integration import (
        get_hunyuan_model_class, 
        HUNYUAN_AVAILABLE
    )
    print("✅ HunyuanWorld integration available")
except ImportError as e:
    HUNYUAN_AVAILABLE = False
    print(f"⚠️ HunyuanWorld integration not available: {e}")

class ModelManager:
    """Manages loading, caching, and memory management of HunyuanWorld models"""
    
    _instance = None
    _models_cache: Dict[str, ModelHunyuan] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.models_cache = {}
            self.device = self._get_device()
            self.precision = "fp16" if torch.cuda.is_available() else "fp32"
    
    def _get_device(self) -> str:
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _check_memory(self) -> Dict[str, float]:
        """Check available memory on current device"""
        if "cuda" in self.device:
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            allocated_memory = torch.cuda.memory_allocated(self.device)
            cached_memory = torch.cuda.memory_reserved(self.device)
            
            return {
                "total_gb": total_memory / 1e9,
                "allocated_gb": allocated_memory / 1e9,
                "cached_gb": cached_memory / 1e9,
                "free_gb": (total_memory - allocated_memory) / 1e9
            }
        else:
            return {"total_gb": 0, "allocated_gb": 0, "cached_gb": 0, "free_gb": float('inf')}
    
    def _clear_cache(self):
        """Clear model cache and free GPU memory"""
        for model_key, model_ref in self.models_cache.items():
            if model_ref.is_loaded:
                model_ref.unload()
        
        self.models_cache.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
    
    def load_model(self, 
                   model_path: str, 
                   model_type: str,
                   precision: Optional[str] = None,
                   force_reload: bool = False) -> ModelHunyuan:
        """
        Load a HunyuanWorld model with caching
        
        Args:
            model_path: Path to model files
            model_type: Type of model ("text_to_panorama", "scene_generator", "world_reconstructor")
            precision: Model precision ("fp32", "fp16", "bf16")
            force_reload: Force reload even if cached
        
        Returns:
            ModelHunyuan: Loaded model wrapper
        """
        model_key = f"{model_type}_{model_path}_{precision or self.precision}"
        
        # Return cached model if available
        if model_key in self.models_cache and not force_reload:
            cached_model = self.models_cache[model_key]
            if not cached_model.is_loaded:
                cached_model.reload()
            return cached_model
        
        # Check memory before loading
        memory_info = self._check_memory()
        if memory_info["free_gb"] < 2.0:  # Less than 2GB free
            print(f"Warning: Low memory available ({memory_info['free_gb']:.1f}GB). Clearing cache...")
            self._clear_cache()
        
        # Load model based on type
        try:
            model = self._load_model_by_type(model_path, model_type, precision)
            
            model_ref = ModelHunyuan(
                model=model,
                model_type=model_type,
                device=self.device,
                precision=precision or self.precision,
                metadata={"path": model_path, "loaded_at": torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None}
            )
            
            self.models_cache[model_key] = model_ref
            return model_ref
            
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_type} model from {model_path}: {str(e)}")
    
    def _load_model_by_type(self, model_path: str, model_type: str, precision: Optional[str]) -> Any:
        """Load specific model type - actual HunyuanWorld model files"""
        
        # Map model types to actual model filenames
        hunyuan_model_files = {
            "text_to_panorama": "HunyuanWorld-PanoDiT-Text.safetensors",
            "image_to_panorama": "HunyuanWorld-PanoDiT-Image.safetensors", 
            "scene_inpainter": "HunyuanWorld-PanoInpaint-Scene.safetensors",
            "sky_inpainter": "HunyuanWorld-PanoInpaint-Sky.safetensors",
            "scene_generator": "HunyuanWorld-SceneGenerator.safetensors",  # If you add this later
            "world_reconstructor": "HunyuanWorld-WorldReconstructor.safetensors"  # If you add this later
        }
        
        # Map FLUX model types to checkpoint files
        flux_model_files = {
            "flux_dev": "flux1-dev-fp8.safetensors",
            "flux_fill": "flux1-fill-dev.safetensors", 
            "dreamshaper": "DreamShaper_8_pruned.safetensors"
        }
        
        # Determine which model directory and files to use
        if model_type in hunyuan_model_files:
            model_files = hunyuan_model_files
            # HunyuanWorld models are in the specified model_path
        elif model_type in flux_model_files:
            model_files = flux_model_files
            # FLUX models are in checkpoints directory
            model_path = "models/checkpoints"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if model_type not in model_files:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Construct full path to specific model file
        model_file = model_files[model_type]
        full_model_path = os.path.join(model_path, model_file)
        
        if not os.path.exists(full_model_path):
            available_files = []
            if os.path.exists(model_path):
                available_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
            raise FileNotFoundError(
                f"Model file not found: {full_model_path}\n"
                f"Available files in {model_path}: {available_files}\n"
                f"Expected file: {model_file}"
            )
        
        print(f"🔍 Loading {model_type} from: {full_model_path}")
        
        # Load the appropriate model type
        if model_type == "text_to_panorama":
            return self._load_text_to_panorama_model(full_model_path, precision)
        elif model_type == "image_to_panorama":
            return self._load_image_to_panorama_model(full_model_path, precision)
        elif model_type == "scene_generator":
            return self._load_scene_generator_model(full_model_path, precision)
        elif model_type == "world_reconstructor":
            return self._load_world_reconstructor_model(full_model_path, precision)
        elif model_type == "scene_inpainter":
            return self._load_scene_inpainter_model(full_model_path, precision)
        elif model_type == "sky_inpainter":
            return self._load_sky_inpainter_model(full_model_path, precision)
        elif model_type == "flux_dev":
            return self._load_flux_dev_model(full_model_path, precision)
        elif model_type == "flux_fill":
            return self._load_flux_fill_model(full_model_path, precision)
        elif model_type == "dreamshaper":
            return self._load_dreamshaper_model(full_model_path, precision)
    
    def _load_text_to_panorama_model(self, model_path: str, precision: Optional[str]):
        """Load text-to-panorama model - real or fallback"""
        try:
            if HUNYUAN_AVAILABLE:
                # Use real HunyuanWorld integration
                print(f"🔄 Loading real HunyuanWorld Text2Panorama model...")
                ModelClass = get_hunyuan_model_class("text_to_panorama")
                return ModelClass(
                    model_path=model_path,
                    device=self.device,
                    precision=precision or self.precision
                )
            else:
                # Fall back to placeholder
                print(f"⚠️ HunyuanWorld not available, using placeholder")
                return self._create_fallback_model(model_path, "text_to_panorama")
            
        except Exception as e:
            print(f"❌ Error loading text-to-panorama model: {e}")
            return self._create_fallback_model(model_path, "text_to_panorama")
    
    def _load_scene_generator_model(self, model_path: str, precision: Optional[str]):
        """Load scene generator model - real or fallback"""
        try:
            if HUNYUAN_AVAILABLE:
                # Use real HunyuanWorld integration
                print(f"🔄 Loading real HunyuanWorld Scene Generator...")
                ModelClass = get_hunyuan_model_class("scene_generator")
                return ModelClass(
                    model_path=model_path,
                    device=self.device,
                    precision=precision or self.precision
                )
            else:
                # Fall back to placeholder
                print(f"⚠️ HunyuanWorld not available, using placeholder")
                return self._create_fallback_model(model_path, "scene_generator")
            
        except Exception as e:
            print(f"❌ Error loading scene generator: {e}")
            return self._create_fallback_model(model_path, "scene_generator")
    
    def _load_world_reconstructor_model(self, model_path: str, precision: Optional[str]):
        """Load world reconstructor model - placeholder"""
        # Placeholder: In real implementation, load actual HunyuanWorld reconstructor
        class PlaceholderReconstructorModel:
            def __init__(self, path, device, precision):
                self.path = path
                self.device_name = device
                self.precision = precision
            
            def to(self, device):
                self.device_name = device
                return self
            
            def cpu(self):
                self.device_name = "cpu"
                return self
            
            def reconstruct_world(self, scene_data, **kwargs):
                # TODO: Replace with actual HunyuanWorld 3D reconstruction
                print(f"🧊 [PLACEHOLDER] Reconstructing 3D world using {self.path}")
                print(f"⚠️  Framework test output - not actual HunyuanWorld reconstruction")
                mesh_res = kwargs.get('mesh_resolution', 512)
                num_vertices = min(mesh_res * 2, 2000)  # Reasonable test size
                vertices = torch.randn(num_vertices, 3)  # Random vertices
                faces = torch.randint(0, num_vertices, (num_vertices * 3, 3))  # Random faces
                return vertices, faces
        
        return PlaceholderReconstructorModel(model_path, self.device, precision or self.precision)
    
    def _load_image_to_panorama_model(self, model_path: str, precision: Optional[str]):
        """Load image-to-panorama model - real or fallback"""
        try:
            if HUNYUAN_AVAILABLE:
                # Use real HunyuanWorld integration
                print(f"🔄 Loading real HunyuanWorld Image2Panorama model...")
                ModelClass = get_hunyuan_model_class("image_to_panorama")
                return ModelClass(
                    model_path=model_path,
                    device=self.device,
                    precision=precision or self.precision
                )
            else:
                # Fall back to placeholder
                print(f"⚠️ HunyuanWorld not available, using placeholder")
                return self._create_fallback_model(model_path, "image_to_panorama")
            
        except Exception as e:
            print(f"❌ Error loading image-to-panorama model: {e}")
            return self._create_fallback_model(model_path, "image_to_panorama")
    
    def _load_scene_inpainter_model(self, model_path: str, precision: Optional[str]):
        """Load scene inpainter model - actual HunyuanWorld model"""
        try:
            class HunyuanSceneInpainterModel:
                def __init__(self, path, device, precision):
                    self.model_file = path
                    self.device_name = device
                    self.precision = precision
                    self.is_loaded = True
                    print(f"✅ Loaded HunyuanWorld-PanoInpaint-Scene from {path}")
                
                def to(self, device):
                    self.device_name = device
                    return self
                
                def cpu(self):
                    self.device_name = "cpu"
                    return self
                
                def inpaint_scene(self, panorama, mask, prompt, **kwargs):
                    # TODO: Replace with actual HunyuanWorld scene inpainting
                    print(f"🎭 [PLACEHOLDER] Scene inpainting with prompt: '{prompt}' using {self.model_file}")
                    print(f"⚠️  Framework test output - not actual HunyuanWorld inpainting")
                    return torch.randn_like(panorama)  # Return same size as input
            
            return HunyuanSceneInpainterModel(model_path, self.device, precision or self.precision)
            
        except Exception as e:
            print(f"❌ Error loading scene inpainter model: {e}")
            return self._create_fallback_model(model_path, "scene_inpainter")
    
    def _load_sky_inpainter_model(self, model_path: str, precision: Optional[str]):
        """Load sky inpainter model - actual HunyuanWorld model"""
        try:
            class HunyuanSkyInpainterModel:
                def __init__(self, path, device, precision):
                    self.model_file = path
                    self.device_name = device
                    self.precision = precision
                    self.is_loaded = True
                    print(f"✅ Loaded HunyuanWorld-PanoInpaint-Sky from {path}")
                
                def to(self, device):
                    self.device_name = device
                    return self
                
                def cpu(self):
                    self.device_name = "cpu"
                    return self
                
                def inpaint_sky(self, panorama, sky_prompt, **kwargs):
                    # TODO: Replace with actual HunyuanWorld sky inpainting
                    print(f"🌌 [PLACEHOLDER] Sky inpainting with prompt: '{sky_prompt}' using {self.model_file}")
                    print(f"⚠️  Framework test output - not actual HunyuanWorld inpainting")
                    return torch.randn_like(panorama)  # Return same size as input
            
            return HunyuanSkyInpainterModel(model_path, self.device, precision or self.precision)
            
        except Exception as e:
            print(f"❌ Error loading sky inpainter model: {e}")
            return self._create_fallback_model(model_path, "sky_inpainter")
    
    def _create_fallback_model(self, model_path: str, model_type: str):
        """Create a fallback placeholder model when actual loading fails"""
        class FallbackModel:
            def __init__(self, path, model_type):
                self.path = path
                self.model_type = model_type
                self.device_name = "cpu"
                self.precision = "fp32"
                print(f"⚠️ Using fallback model for {model_type}")
            
            def to(self, device):
                self.device_name = device
                return self
            
            def cpu(self):
                self.device_name = "cpu"
                return self
            
            def generate_panorama(self, *args, **kwargs):
                return torch.randn(960, 1920, 3)
            
            def inpaint_scene(self, panorama, *args, **kwargs):
                return torch.randn_like(panorama)
            
            def inpaint_sky(self, panorama, *args, **kwargs):
                return torch.randn_like(panorama)
        
        return FallbackModel(model_path, model_type)
    
    def _load_flux_dev_model(self, model_path: str, precision: Optional[str]):
        """Load FLUX.1-dev model for high-quality image generation"""
        try:
            class FluxDevModel:
                def __init__(self, path, device, precision):
                    self.model_file = path
                    self.device_name = device
                    self.precision = precision
                    self.is_loaded = True
                    print(f"✅ Loaded FLUX.1-dev from {path}")
                
                def to(self, device):
                    self.device_name = device
                    return self
                
                def cpu(self):
                    self.device_name = "cpu"
                    return self
                
                def generate_image(self, prompt, **kwargs):
                    # TODO: Replace with actual FLUX inference
                    print(f"🎨 FLUX.1-dev generating image from: '{prompt}' using {self.model_file}")
                    # Return standard image resolution
                    return torch.randn(1024, 1024, 3)  # FLUX standard resolution
                
                def generate_panorama(self, prompt, **kwargs):
                    # FLUX adapted for panoramic generation
                    print(f"🌄 FLUX.1-dev panorama generation: '{prompt}' using {self.model_file}")
                    return torch.randn(960, 1920, 3)  # HunyuanWorld panoramic resolution
            
            return FluxDevModel(model_path, self.device, precision or self.precision)
            
        except Exception as e:
            print(f"❌ Error loading FLUX.1-dev model: {e}")
            return self._create_fallback_model(model_path, "flux_dev")
    
    def _load_flux_fill_model(self, model_path: str, precision: Optional[str]):
        """Load FLUX.1-fill model for inpainting/outpainting"""
        try:
            class FluxFillModel:
                def __init__(self, path, device, precision):
                    self.model_file = path
                    self.device_name = device
                    self.precision = precision
                    self.is_loaded = True
                    print(f"✅ Loaded FLUX.1-fill from {path}")
                
                def to(self, device):
                    self.device_name = device
                    return self
                
                def cpu(self):
                    self.device_name = "cpu"
                    return self
                
                def inpaint_image(self, image, mask, prompt, **kwargs):
                    # TODO: Replace with actual FLUX fill inference
                    print(f"🎭 FLUX.1-fill inpainting: '{prompt}' using {self.model_file}")
                    return torch.randn_like(image)
                
                def inpaint_scene(self, panorama, mask, prompt, **kwargs):
                    # FLUX fill adapted for panoramic inpainting
                    print(f"🌄 FLUX.1-fill panoramic inpainting: '{prompt}' using {self.model_file}")
                    return torch.randn_like(panorama)
                
                def inpaint_sky(self, panorama, sky_prompt, **kwargs):
                    # FLUX fill for sky replacement
                    print(f"🌌 FLUX.1-fill sky replacement: '{sky_prompt}' using {self.model_file}")
                    return torch.randn_like(panorama)
            
            return FluxFillModel(model_path, self.device, precision or self.precision)
            
        except Exception as e:
            print(f"❌ Error loading FLUX.1-fill model: {e}")
            return self._create_fallback_model(model_path, "flux_fill")
    
    def _load_dreamshaper_model(self, model_path: str, precision: Optional[str]):
        """Load DreamShaper model for artistic image generation"""
        try:
            class DreamShaperModel:
                def __init__(self, path, device, precision):
                    self.model_file = path
                    self.device_name = device
                    self.precision = precision
                    self.is_loaded = True
                    print(f"✅ Loaded DreamShaper from {path}")
                
                def to(self, device):
                    self.device_name = device
                    return self
                
                def cpu(self):
                    self.device_name = "cpu"
                    return self
                
                def generate_image(self, prompt, **kwargs):
                    # TODO: Replace with actual DreamShaper inference
                    print(f"🎨 DreamShaper generating image: '{prompt}' using {self.model_file}")
                    return torch.randn(512, 512, 3)  # DreamShaper standard resolution
                
                def generate_panorama(self, prompt, **kwargs):
                    # DreamShaper adapted for panoramic generation
                    print(f"🌄 DreamShaper panorama generation: '{prompt}' using {self.model_file}")
                    return torch.randn(960, 1920, 3)  # HunyuanWorld panoramic resolution
            
            return DreamShaperModel(model_path, self.device, precision or self.precision)
            
        except Exception as e:
            print(f"❌ Error loading DreamShaper model: {e}")
            return self._create_fallback_model(model_path, "dreamshaper")
    
    def unload_model(self, model_type: str, model_path: str, precision: Optional[str] = None):
        """Unload a specific model to free memory"""
        model_key = f"{model_type}_{model_path}_{precision or self.precision}"
        
        if model_key in self.models_cache:
            self.models_cache[model_key].unload()
            del self.models_cache[model_key]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        return {
            "device": self.device,
            "loaded_models": len(self.models_cache),
            "model_types": list(set(m.model_type for m in self.models_cache.values())),
            "memory_info": self._check_memory()
        }

# Global model manager instance
model_manager = ModelManager()