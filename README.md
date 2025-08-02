# HunyuanWorld ComfyUI Integration

A comprehensive ComfyUI custom node package for [Tencent's HunyuanWorld-1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) that enables generating immersive, explorable, and interactive 3D worlds from text prompts or images.

## 🌟 Features

- **Text-to-World Generation**: Create complete 360° 3D environments from text descriptions
- **Image-to-Panorama**: Convert regular images to panoramic format
- **3D Scene Reconstruction**: Generate depth maps and semantic segmentation
- **Mesh Export**: Export 3D worlds in multiple formats (OBJ, PLY, GLB, FBX)
- **Interactive Viewer**: Built-in 3D visualization with multiple display modes
- **Memory Management**: Intelligent model loading and GPU memory optimization
- **Flexible Workflows**: Support for complex multi-step generation pipelines

## 📋 Requirements

### System Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended (4GB minimum)
- **RAM**: 16GB+ system RAM recommended
- **Storage**: 10GB+ free space for models
- **OS**: Windows 10/11, Linux, or macOS

### Software Requirements
- **ComfyUI**: Latest version
- **Python**: 3.10 or newer
- **PyTorch**: 2.0.0+ with CUDA support
- **CUDA**: 11.8 or newer (for NVIDIA GPUs)

## 🔧 Installation

### 1. Clone the Repository

Navigate to your ComfyUI custom nodes directory and clone:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/your-repo/HunyuanWorld.git
# OR download and extract the ZIP file
```

### 2. Install Dependencies

```bash
cd HunyuanWorld
pip install -r requirements.txt
```

**For CUDA users**, ensure you have the correct PyTorch version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Download Models

#### Standard ComfyUI Model Directory Structure

**⚠️ IMPORTANT: This repository contains only the ComfyUI node code. All model files must be downloaded separately.**

Create the following directory structure and download the required models to these locations:

```
ComfyUI/
├── models/                                    # ← YOU NEED TO DOWNLOAD ALL FILES BELOW
│   ├── hunyuan_world/                         # ⬇️ Download: HunyuanWorld models (1.5GB total)
│   │   ├── HunyuanWorld-PanoDiT-Text.safetensors     # ⬇️ 478MB: Text to panorama
│   │   ├── HunyuanWorld-PanoDiT-Image.safetensors    # ⬇️ 478MB: Image to panorama  
│   │   ├── HunyuanWorld-PanoInpaint-Scene.safetensors # ⬇️ 478MB: Scene inpainting
│   │   └── HunyuanWorld-PanoInpaint-Sky.safetensors   # ⬇️ 120MB: Sky inpainting
│   ├── flux/                                  # ⬇️ Download: FLUX models (optional, for enhanced quality)
│   │   ├── flux1-dev.safetensors              # ⬇️ ~12GB: FLUX.1 [dev] model
│   │   └── flux1-schnell.safetensors          # ⬇️ ~12GB: FLUX.1 [schnell] model  
│   └── clip/                                  # ⬇️ Download: Text encoders (required for FLUX)
│       ├── clip_l.safetensors                 # ⬇️ ~1GB: CLIP text encoder
│       └── t5xxl_fp16.safetensors             # ⬇️ ~5GB: T5 text encoder
└── custom_nodes/
    └── HunyuanWorld/                          # ✅ Included: This package (node code only)
```

#### HunyuanWorld Model Downloads

**Required HunyuanWorld Models** (1.5GB total):

| Model | Description | Size | Download |
|-------|-------------|------|----------|
| **HunyuanWorld-PanoDiT-Text** | Text to Panorama Model | 478MB | [Download](https://huggingface.co/Tencent-Hunyuan/HunyuanWorld) |
| **HunyuanWorld-PanoDiT-Image** | Image to Panorama Model | 478MB | [Download](https://huggingface.co/Tencent-Hunyuan/HunyuanWorld) |
| **HunyuanWorld-PanoInpaint-Scene** | PanoInpaint Model for scene | 478MB | [Download](https://huggingface.co/Tencent-Hunyuan/HunyuanWorld) |
| **HunyuanWorld-PanoInpaint-Sky** | PanoInpaint Model for sky | 120MB | [Download](https://huggingface.co/Tencent-Hunyuan/HunyuanWorld) |

**Download Commands:**
```bash
# Create directory
mkdir -p ComfyUI/models/hunyuan_world

# Download HunyuanWorld models
huggingface-cli download Tencent-Hunyuan/HunyuanWorld HunyuanWorld-PanoDiT-Text.safetensors --local-dir ComfyUI/models/hunyuan_world/
huggingface-cli download Tencent-Hunyuan/HunyuanWorld HunyuanWorld-PanoDiT-Image.safetensors --local-dir ComfyUI/models/hunyuan_world/
huggingface-cli download Tencent-Hunyuan/HunyuanWorld HunyuanWorld-PanoInpaint-Scene.safetensors --local-dir ComfyUI/models/hunyuan_world/
huggingface-cli download Tencent-Hunyuan/HunyuanWorld HunyuanWorld-PanoInpaint-Sky.safetensors --local-dir ComfyUI/models/hunyuan_world/
```

#### FLUX Models (Optional Enhancement)

**FLUX.1 Models** (Optional, for enhanced generation quality):
- **FLUX.1 [dev]**: Best quality, slower generation (~12GB VRAM)
- **FLUX.1 [schnell]**: Faster generation, good quality (~8GB VRAM)

```bash
# Download FLUX models (optional)
wget -O ComfyUI/models/flux/flux1-dev.safetensors "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors"
wget -O ComfyUI/models/flux/flux1-schnell.safetensors "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors"

# CLIP and T5 encoders (required if using FLUX)
wget -O ComfyUI/models/clip/clip_l.safetensors "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
wget -O ComfyUI/models/clip/t5xxl_fp16.safetensors "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
```

#### Quick Setup Commands

**All-in-one setup** (install dependencies + download HunyuanWorld models):
```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Create directories
mkdir -p ComfyUI/models/hunyuan_world
mkdir -p ComfyUI/models/flux
mkdir -p ComfyUI/models/clip

# Download all HunyuanWorld models (1.5GB total)
huggingface-cli download Tencent-Hunyuan/HunyuanWorld --local-dir ComfyUI/models/hunyuan_world/ --include "*.safetensors"
```

**Alternative: Manual Download**
1. Visit [HunyuanWorld on Hugging Face](https://huggingface.co/Tencent-Hunyuan/HunyuanWorld)
2. Download the 4 `.safetensors` files listed above
3. Place them in `ComfyUI/models/hunyuan_world/`

#### Storage Requirements

| Component | Size | Required |
|-----------|------|----------|
| **HunyuanWorld Models** | 1.5GB | ✅ Required |
| **FLUX Models** | ~24GB | ⚠️ Optional (for enhanced quality) |
| **Text Encoders** | ~6GB | ⚠️ Only if using FLUX |

**Minimum setup**: 1.5GB (HunyuanWorld models only)  
**Full setup with FLUX**: ~31GB total

### 4. Verify Installation

**Before testing, ensure you have downloaded the required model files (see section 3 above).**

1. **Restart ComfyUI** completely
2. **Check for nodes**: Look for "HunyuanWorld" category in the node browser
3. **Test basic workflow**: Create a simple Text Input → Model Loader → Text to Panorama chain
4. **Model loading**: The first time you load a model, it may take several minutes to initialize

## 🚀 Quick Start

### Basic Text-to-World Workflow

1. **Add Nodes**:
   - `HunyuanTextInput` - Enter your prompt
   - `HunyuanLoader` - Load the text-to-panorama model
   - `HunyuanTextToPanorama` - Generate panoramic image
   - `HunyuanViewer` - Preview the result

2. **Connect the Pipeline**:
   ```
   HunyuanTextInput → HunyuanTextToPanorama ← HunyuanLoader
                    ↓
                 HunyuanViewer
   ```

3. **Configure Settings**:
   - Model path: `models/hunyuan_world`
   - Model type: `text_to_panorama`
   - Prompt: "A beautiful mountain landscape with forests"

4. **Generate**: Click "Queue Prompt" and wait for generation

### Advanced 3D World Pipeline

For full 3D world generation:
```
HunyuanTextInput → HunyuanTextToPanorama ← HunyuanLoader
                ↓
            HunyuanSceneGenerator ← HunyuanLoader (scene_generator)
                ↓
        HunyuanWorldReconstructor ← HunyuanLoader (world_reconstructor)
                ↓
           HunyuanMeshExporter
```

## ⚙️ Configuration

### Model Configuration

Edit `configs/default_config.yaml` to customize:

```yaml
models:
  default_precision: "fp16"  # or "fp32" for better quality
  max_memory_usage: 0.8      # GPU memory limit
  
generation:
  text_to_panorama:
    default_width: 1024      # Panorama width
    default_height: 512      # Panorama height
    default_steps: 50        # Inference steps
```

### Memory Optimization

For **Low VRAM systems (4-6GB)**:
```yaml
models:
  default_precision: "fp16"
  max_memory_usage: 0.7
  
performance:
  memory:
    auto_clear_cache: true
    model_unload_timeout: 60
```

For **High VRAM systems (12GB+)**:
```yaml
models:
  default_precision: "fp32"  # Better quality
  max_memory_usage: 0.9
  
generation:
  text_to_panorama:
    default_width: 2048      # Higher resolution
    default_height: 1024
```

## 📝 Node Reference

### Input Nodes

| Node | Purpose | Key Parameters |
|------|---------|----------------|
| **HunyuanTextInput** | Text prompt input | `prompt`, `seed`, `negative_prompt` |
| **HunyuanImageInput** | Image preprocessing | `resize_mode`, `target_width`, `preprocessing` |
| **HunyuanPromptProcessor** | Prompt enhancement | `style`, `lighting`, `atmosphere` |

### Generation Nodes

| Node | Purpose | Key Parameters |
|------|---------|----------------|
| **HunyuanLoader** | Model loading | `model_path`, `model_type`, `precision` |
| **HunyuanTextToPanorama** | Text to 360° image | `width`, `height`, `guidance_scale` |
| **HunyuanImageToPanorama** | Image to panorama | `extension_mode`, `strength` |
| **HunyuanSceneGenerator** | 3D scene creation | `depth_estimation`, `semantic_segmentation` |
| **HunyuanWorldReconstructor** | 3D mesh generation | `mesh_resolution`, `texture_resolution` |

### Output Nodes

| Node | Purpose | Key Parameters |
|------|---------|----------------|
| **HunyuanViewer** | 3D preview | `display_mode`, `output_size` |
| **HunyuanMeshExporter** | Export 3D models | `format`, `compression`, `include_materials` |
| **HunyuanDataInfo** | Data information | Shows detailed statistics |

## 🔍 Troubleshooting

### Common Issues

**"Model not found" Error**
- ✅ Check model path in `HunyuanLoader` node
- ✅ Verify files exist in `ComfyUI/models/hunyuan_world/`
- ✅ Ensure correct model type is selected

**"Out of Memory" Error**
- ✅ Reduce image resolution (width/height)
- ✅ Change precision to `fp16` in model loader
- ✅ Close other applications using GPU memory
- ✅ Enable `auto_clear_cache` in config

**"CUDA not available" Warning**
- ✅ Install CUDA-enabled PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- ✅ Verify NVIDIA drivers are up to date
- ✅ Check CUDA installation: `nvidia-smi`

**Nodes not appearing**
- ✅ Restart ComfyUI completely
- ✅ Check for Python errors in console
- ✅ Verify all dependencies are installed: `pip install -r requirements.txt`

**Slow generation**
- ✅ Reduce inference steps (try 20-30 instead of 50)
- ✅ Use smaller resolutions for testing
- ✅ Enable mixed precision (`fp16`)
- ✅ Close unnecessary browser tabs/applications

### Performance Tips

1. **First Run**: Models take time to load initially
2. **Memory**: Monitor GPU memory with `nvidia-smi`
3. **Quality vs Speed**: Higher steps = better quality but slower
4. **Batch Processing**: Process multiple prompts together when possible

### Debug Mode

Enable detailed logging:
```yaml
logging:
  level: "DEBUG"
  log_model_loading: true
  log_generation_steps: true
  log_memory_usage: true
```

## 📚 Example Workflows

### Text-to-Panorama
```json
{
  "prompt": "A serene Japanese garden with cherry blossoms, koi pond, and traditional architecture",
  "style": "realistic",
  "lighting": "golden_hour",
  "width": 1024,
  "height": 512,
  "steps": 50
}
```

### Image-to-World
```json
{
  "extension_mode": "outpainting",
  "depth_estimation": true,
  "semantic_segmentation": true,
  "mesh_resolution": 512,
  "export_format": "OBJ"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License. See LICENSE file for details.

The HunyuanWorld-1.0 models are subject to their own licensing terms from Tencent.

## 🔗 Resources

- **HunyuanWorld Repository**: https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI
- **Documentation**: Check `CLAUDE.md` for detailed technical implementation notes

## 🆘 Support

- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Community help and feature requests
- **Discord**: ComfyUI community Discord server

---

**⚠️ Note**: This is a community integration package. For official support and the latest model versions, please refer to the original HunyuanWorld-1.0 repository by Tencent.