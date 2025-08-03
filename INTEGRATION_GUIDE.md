# HunyuanWorld-1.0 Integration Guide

This guide shows how to integrate actual HunyuanWorld-1.0 model inference with the existing ComfyUI framework.

## 🔧 **Installation Steps**

### **Step 1: Install Dependencies**

```bash
# Navigate to the ComfyUI HunyuanWorld directory
cd ComfyUI/custom_nodes/ComfyUI_HunyuanWorld

# Install HunyuanWorld dependencies
pip install -r requirements_hunyuan.txt

# Optional: Install performance optimizations
pip install xformers  # For memory-efficient attention
```

### **Step 2: Clone HunyuanWorld-1.0 Repository**

The integration expects the official HunyuanWorld-1.0 repository to be present:

```bash
# Clone into the node directory (if not already present)
cd ComfyUI/custom_nodes/ComfyUI_HunyuanWorld
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git

# Your directory should now look like:
# ComfyUI_HunyuanWorld/
#   ├── HunyuanWorld-1.0/          ← Official repository
#   ├── core/
#   ├── nodes/
#   └── workflows/
```

### **Step 3: Download Models**

The integration uses FLUX.1-dev as the base model with HunyuanWorld LoRA adapters:

```bash
# Install huggingface-cli if not already installed
pip install huggingface-hub

# Download base FLUX model (will be downloaded automatically on first use)
# Base model: black-forest-labs/FLUX.1-dev
# LoRA adapter: tencent/HunyuanWorld-1

# For image-to-panorama, also need:
# Base model: black-forest-labs/FLUX.1-fill-dev
```

**Note**: Models will be automatically downloaded from Hugging Face on first use. Ensure you have:
- Sufficient disk space (10GB+ for FLUX models)
- Stable internet connection for initial download
- Hugging Face account (for gated models, if any)

### **Step 4: Verify Installation**

```bash
# Test the integration
python -c "
import sys
sys.path.append('ComfyUI/custom_nodes/ComfyUI_HunyuanWorld')
from core.hunyuan_integration import HUNYUAN_AVAILABLE
print(f'HunyuanWorld Integration: {\"✅ Available\" if HUNYUAN_AVAILABLE else \"❌ Not Available\"}')
"
```

## 🚀 **How It Works**

### **Model Loading Architecture**

1. **Automatic Detection**: The integration automatically detects if HunyuanWorld-1.0 is available
2. **Fallback System**: If not available, falls back to placeholder behavior
3. **Real Inference**: When available, uses actual HunyuanWorld pipelines

### **Console Output**

**With Integration**:
```
✅ Added HunyuanWorld path: /path/to/HunyuanWorld-1.0
✅ HunyuanWorld imports successful
✅ HunyuanWorld integration available
🔄 Loading real HunyuanWorld Text2Panorama model...
✅ HunyuanWorld Text2Panorama loaded successfully
```

**Without Integration**:
```
⚠️ HunyuanWorld import failed: No module named 'hy3dworld'
⚠️ HunyuanWorld integration not available
⚠️ HunyuanWorld not available, using placeholder
```

## 📋 **Supported Features**

### ✅ **Currently Integrated**

| Feature | Status | Model Used |
|---------|--------|------------|
| **Text-to-Panorama** | ✅ Full Integration | FLUX.1-dev + HunyuanWorld LoRA |
| **Image-to-Panorama** | ✅ Full Integration | FLUX.1-fill-dev + HunyuanWorld LoRA |
| **Basic Scene Generation** | ⚠️ Partial | Custom components |

### 🔄 **Integration Parameters**

The integration supports all original HunyuanWorld parameters:

#### Text-to-Panorama:
- `prompt`: Text description
- `height`: Default 960 (HunyuanWorld standard)
- `width`: Default 1920 (HunyuanWorld standard)
- `num_inference_steps`: Default 50
- `guidance_scale`: Default 30.0 (HunyuanWorld optimized)
- `true_cfg_scale`: Default 0.0
- `blend_extend`: Default 6
- `shifting_extend`: Default 0

#### Image-to-Panorama:
- `image`: Input image tensor
- `strength`: Modification strength (0.1-1.0)
- `num_inference_steps`: Default 30
- `guidance_scale`: Default 7.5

## 🔍 **Technical Details**

### **Model Integration Pattern**

```python
# The integration follows this pattern:
class HunyuanTextToPanoramaModel:
    def __init__(self, model_path, device, precision):
        # Load FLUX.1-dev as base model
        self.pipeline = Text2PanoramaPipelines.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=dtype,
            device_map=device
        )
        
        # Load HunyuanWorld LoRA adaptation
        self.pipeline.load_lora_weights("tencent/HunyuanWorld-1")
        
    def generate_panorama(self, prompt, **kwargs):
        # Use real HunyuanWorld inference
        result = self.pipeline(prompt=prompt, **kwargs)
        return tensor_result
```

### **Memory Requirements**

| Component | VRAM Usage | Precision |
|-----------|------------|-----------|
| FLUX.1-dev | ~12GB | fp16 |
| FLUX.1-dev | ~24GB | fp32 |
| HunyuanWorld LoRA | ~500MB | Additional |

**Recommended**: 16GB+ VRAM for comfortable usage

### **Performance Optimization**

```python
# Use these settings for different scenarios:

# High Quality (16GB+ VRAM):
precision = "fp16"
num_inference_steps = 50
guidance_scale = 30.0

# Balanced (12GB VRAM):
precision = "fp16"
num_inference_steps = 30
guidance_scale = 25.0

# Fast/Low Memory (8GB VRAM):
precision = "fp16"
num_inference_steps = 20
guidance_scale = 15.0
```

## 🛠️ **Troubleshooting**

### **Common Issues**

#### 1. Import Errors
```
❌ HunyuanWorld import failed: No module named 'hy3dworld'
```
**Solution**: Ensure HunyuanWorld-1.0 repository is cloned in the correct location

#### 2. Model Download Failures
```
❌ Failed to load HunyuanWorld pipeline: HTTP Error 403
```
**Solution**: 
- Login to Hugging Face: `huggingface-cli login`
- Check internet connection
- Ensure sufficient disk space

#### 3. CUDA Out of Memory
```
❌ RuntimeError: CUDA out of memory
```
**Solution**:
- Use fp16 precision
- Reduce inference steps
- Close other GPU applications
- Use smaller image dimensions

#### 4. Model Loading Slow
```
🔄 Loading HunyuanWorld Text2Panorama model... (taking >5 minutes)
```
**Solution**:
- First load downloads models (~10GB) - this is normal
- Subsequent loads should be much faster
- Check disk I/O performance

### **Verification Steps**

```bash
# 1. Check Python environment
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 2. Check HunyuanWorld presence
ls ComfyUI/custom_nodes/ComfyUI_HunyuanWorld/HunyuanWorld-1.0/

# 3. Check dependencies
pip list | grep -E "(diffusers|transformers|torch)"

# 4. Test integration
python ComfyUI/custom_nodes/ComfyUI_HunyuanWorld/core/hunyuan_integration.py
```

## 📈 **Performance Comparison**

| Method | Speed | Quality | VRAM | Status |
|--------|-------|---------|------|--------|
| Placeholder | ⚡ Instant | ❌ Random | 0GB | Always available |
| HunyuanWorld | 🐌 30-60s | ✅ High | 12GB+ | Requires setup |

## 🔄 **Advanced Features (Future)**

### **Scene Inpainting Integration**
To add scene and sky inpainting:

1. Implement `HunyuanSceneInpainterModel` using HunyuanWorld-PanoInpaint models
2. Add mask processing pipeline
3. Integrate with ComfyUI mask workflows

### **Layer Decomposition Integration**
To add multi-layer scene generation:

1. Implement `LayerDecomposition` integration
2. Add `WorldComposer` for 3D reconstruction
3. Create layered mesh export

### **Additional Models**
- ZIM integration for advanced inpainting
- Real-ESRGAN for upscaling
- Custom depth estimation models

## 🎯 **Usage Examples**

### **Basic Text-to-Panorama**
```python
# In ComfyUI workflow:
# 1. HunyuanTextInput: "Mountain landscape at sunset"
# 2. HunyuanLoader: model_type="text_to_panorama"
# 3. HunyuanTextToPanorama: default settings
# 4. Result: Real AI-generated panorama (not placeholder)
```

### **Image Extension**
```python
# In ComfyUI workflow:
# 1. LoadImage: Load your photo
# 2. HunyuanImageInput: Process image
# 3. HunyuanLoader: model_type="image_to_panorama" 
# 4. HunyuanImageToPanorama: strength=0.8
# 5. Result: AI-extended panoramic version
```

## 🔗 **Resources**

- **Official Repository**: https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
- **FLUX Models**: https://huggingface.co/black-forest-labs
- **HunyuanWorld LoRA**: https://huggingface.co/tencent/HunyuanWorld-1
- **ComfyUI Documentation**: https://github.com/comfyanonymous/ComfyUI

---

**✅ Integration Complete**: After following this guide, your ComfyUI HunyuanWorld nodes will use real AI inference instead of placeholder data!