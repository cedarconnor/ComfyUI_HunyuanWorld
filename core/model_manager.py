import os
import torch
import gc
from typing import Dict, Any, Optional, Union
from .data_types import ModelHunyuan

# Import HunyuanWorld integration
from .hunyuan_integration import (
    get_hunyuan_model_class, 
    HUNYUAN_AVAILABLE
)

if HUNYUAN_AVAILABLE:
    print("[SUCCESS] HunyuanWorld integration available")
else:
    raise ImportError("HunyuanWorld integration required. Please follow setup instructions in README.md")

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
        
        # Map model types to actual model filenames based on user's setup
        hunyuan_model_files = {
            "text_to_panorama": "HunyuanWorld-PanoDiT-Text.safetensors",
            "image_to_panorama": "HunyuanWorld-PanoDiT-Image.safetensors", 
            "scene_inpainter": "HunyuanWorld-PanoInpaint-Scene.safetensors",
            "sky_inpainter": "HunyuanWorld-PanoInpaint-Sky.safetensors",
            "scene_generator": "HunyuanWorld-SceneGenerator.safetensors",  # If you add this later
            "world_reconstructor": "HunyuanWorld-WorldReconstructor.safetensors"  # If you add this later
        }
        
        # Map FLUX model types to checkpoint files - updated paths per user setup
        flux_model_files = {
            "flux_dev": "flux1-dev.safetensors",  # In models/unet per user setup
            "flux_fill": "flux1-fill-dev.safetensors", 
            "dreamshaper": "DreamShaper_8_pruned.safetensors"
        }
        
        # Determine which model directory and files to use
        if model_type in hunyuan_model_files:
            model_files = hunyuan_model_files
            # HunyuanWorld models are in the specified model_path
        elif model_type in flux_model_files:
            model_files = flux_model_files
            # FLUX models are in unet directory per user setup
            model_path = "models/unet"
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
        """Load text-to-panorama model - HunyuanWorld AI"""
        if not HUNYUAN_AVAILABLE:
            raise RuntimeError("HunyuanWorld integration required for text-to-panorama model. Please follow setup instructions.")
        
        try:
            print(f"🔄 Loading HunyuanWorld Text2Panorama model...")
            ModelClass = get_hunyuan_model_class("text_to_panorama")
            return ModelClass(
                model_path=model_path,
                device=self.device,
                precision=precision or self.precision
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load text-to-panorama model: {e}")
    
    def _load_scene_generator_model(self, model_path: str, precision: Optional[str]):
        """Load scene generator model - HunyuanWorld AI"""
        if not HUNYUAN_AVAILABLE:
            raise RuntimeError("HunyuanWorld integration required for scene generator model. Please follow setup instructions.")
        
        try:
            print(f"🔄 Loading HunyuanWorld Scene Generator...")
            ModelClass = get_hunyuan_model_class("scene_generator")
            return ModelClass(
                model_path=model_path,
                device=self.device,
                precision=precision or self.precision
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load scene generator model: {e}")
    
    def _load_world_reconstructor_model(self, model_path: str, precision: Optional[str]):
        """Load world reconstructor model - create placeholder since this model doesn't exist yet"""
        try:
            print(f"🔄 Creating placeholder WorldReconstructor...")
            
            class PlaceholderWorldReconstructor:
                def __init__(self, path, device, precision):
                    self.model_file = path
                    self.device_name = device
                    self.precision = precision
                    self.is_loaded = True
                    print(f"ℹ️ WorldReconstructor not available - using Scene Generator fallback")
                
                def to(self, device):
                    self.device_name = device
                    return self
                
                def cpu(self):
                    self.device_name = "cpu"
                    return self
                
                def reconstruct_world(self, scene_data, **kwargs):
                    # Fallback to scene generation for now
                    print("⚠️ WorldReconstructor not implemented - using Scene Generator fallback")
                    raise NotImplementedError("WorldReconstructor model not available. Use HunyuanSceneGenerator instead.")
            
            return PlaceholderWorldReconstructor(model_path, self.device, precision or self.precision)
            
        except Exception as e:
            raise RuntimeError(f"Failed to create world reconstructor placeholder: {e}")
    
    def _load_image_to_panorama_model(self, model_path: str, precision: Optional[str]):
        """Load image-to-panorama model - HunyuanWorld AI"""
        if not HUNYUAN_AVAILABLE:
            raise RuntimeError("HunyuanWorld integration required for image-to-panorama model. Please follow setup instructions.")
        
        try:
            print(f"🔄 Loading HunyuanWorld Image2Panorama model...")
            ModelClass = get_hunyuan_model_class("image_to_panorama")
            return ModelClass(
                model_path=model_path,
                device=self.device,
                precision=precision or self.precision
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load image-to-panorama model: {e}")
    
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
                    # TODO: Implement actual HunyuanWorld scene inpainting
                    raise NotImplementedError("HunyuanWorld-PanoInpaint-Scene integration coming soon")
            
            return HunyuanSceneInpainterModel(model_path, self.device, precision or self.precision)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load scene inpainter model: {e}")
    
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
                    # TODO: Implement actual HunyuanWorld sky inpainting
                    raise NotImplementedError("HunyuanWorld-PanoInpaint-Sky integration coming soon")
            
            return HunyuanSkyInpainterModel(model_path, self.device, precision or self.precision)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load sky inpainter model: {e}")
    
    
    def _load_flux_dev_model(self, model_path: str, precision: Optional[str]):
        """Load FLUX.1-dev model - handled by HunyuanWorld integration"""
        if not HUNYUAN_AVAILABLE:
            raise RuntimeError("HunyuanWorld integration required for FLUX models. Please follow setup instructions.")
        
        ModelClass = get_hunyuan_model_class("text_to_panorama")
        return ModelClass(model_path, self.device, precision or self.precision)
    
    def _load_flux_fill_model(self, model_path: str, precision: Optional[str]):
        """Load FLUX.1-fill model - handled by HunyuanWorld integration"""
        if not HUNYUAN_AVAILABLE:
            raise RuntimeError("HunyuanWorld integration required for FLUX models. Please follow setup instructions.")
        
        ModelClass = get_hunyuan_model_class("image_to_panorama")
        return ModelClass(model_path, self.device, precision or self.precision)
    
    def _load_dreamshaper_model(self, model_path: str, precision: Optional[str]):
        """DreamShaper support removed - use HunyuanWorld models only"""
        raise RuntimeError("DreamShaper models not supported. Use HunyuanWorld text_to_panorama or image_to_panorama models.")
    
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