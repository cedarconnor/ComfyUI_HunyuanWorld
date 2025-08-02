import os
import torch
import gc
from typing import Dict, Any, Optional, Union
from .data_types import ModelHunyuan

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
        """Load specific model type - placeholder for actual HunyuanWorld integration"""
        
        # This is a placeholder implementation
        # In real implementation, this would load actual HunyuanWorld models
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Placeholder model loading logic
        if model_type == "text_to_panorama":
            return self._load_text_to_panorama_model(model_path, precision)
        elif model_type == "scene_generator":
            return self._load_scene_generator_model(model_path, precision)
        elif model_type == "world_reconstructor":
            return self._load_world_reconstructor_model(model_path, precision)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _load_text_to_panorama_model(self, model_path: str, precision: Optional[str]):
        """Load text-to-panorama model - placeholder"""
        # Placeholder: In real implementation, load actual HunyuanWorld text-to-panorama model
        class PlaceholderModel:
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
            
            def generate_panorama(self, prompt, **kwargs):
                # Placeholder generation
                return torch.randn(512, 1024, 3)  # Dummy panorama
        
        return PlaceholderModel(model_path, self.device, precision or self.precision)
    
    def _load_scene_generator_model(self, model_path: str, precision: Optional[str]):
        """Load scene generator model - placeholder"""
        # Placeholder: In real implementation, load actual HunyuanWorld scene generator
        class PlaceholderSceneModel:
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
            
            def generate_scene(self, panorama, **kwargs):
                # Placeholder scene generation
                h, w = panorama.shape[-2:]
                depth = torch.randn(h, w)
                return depth, {}
        
        return PlaceholderSceneModel(model_path, self.device, precision or self.precision)
    
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
                # Placeholder world reconstruction
                vertices = torch.randn(1000, 3)  # 1000 vertices
                faces = torch.randint(0, 1000, (1800, 3))  # 1800 triangular faces
                return vertices, faces
        
        return PlaceholderReconstructorModel(model_path, self.device, precision or self.precision)
    
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