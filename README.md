# ComfyUI HunyuanWorld - Professional 3D World Generation

A comprehensive ComfyUI custom node package for [Tencent's HunyuanWorld-1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) that enables professional-grade generation of immersive, explorable 3D worlds from text prompts or images with advanced features including panorama inpainting, multi-layer scene decomposition, and high-quality export capabilities.

## 🌟 Key Features

### Core Generation Capabilities
- **🎨 Text-to-World Generation**: Create complete 360° 3D environments from detailed text descriptions
- **🖼️ Image-to-Panorama**: Convert regular images to high-resolution panoramic format with intelligent extension
- **🎯 Advanced Panorama Inpainting**: Professional scene and sky inpainting with mask-based control
- **🏗️ Multi-Layer 3D Scene Decomposition**: Separate foreground/background layers with object-specific processing
- **📐 High-Quality 3D Reconstruction**: Generate detailed depth maps, semantic segmentation, and explorable meshes

### Professional Workflow Features
- **🔧 Complete Node Ecosystem**: 15+ specialized nodes covering the entire pipeline from input to export
- **🎛️ Advanced Parameter Control**: Repository-accurate settings with professional-grade fine-tuning
- **🚀 Batch Processing Support**: Optimized workflows for production environments
- **💾 Multiple Export Formats**: OBJ, PLY, GLB, FBX with Draco compression support
- **👁️ Interactive 3D Viewer**: Built-in Three.js-based viewer with layer controls and real-time preview

### Advanced Technical Features
- **🧠 Intelligent Model Loading**: Support for 6 model types (text_to_panorama, image_to_panorama, scene_inpainter, sky_inpainter, scene_generator, world_reconstructor)
- **🎭 Object Labeling System**: Automated foreground object detection and classification
- **🗜️ Draco Mesh Compression**: Professional-grade 3D asset optimization for production pipelines
- **📊 Real-time Analytics**: Performance monitoring, memory usage tracking, and detailed export statistics
- **🌐 Web Integration**: Modern browser-based 3D visualization with full ComfyUI integration

## 📋 System Requirements

### Minimum Requirements
- **GPU**: NVIDIA RTX 3060 (8GB VRAM) or equivalent
- **RAM**: 16GB system memory
- **Storage**: 15GB free space (5GB for models + 10GB working space)
- **OS**: Windows 10/11, Linux Ubuntu 20.04+, or macOS 12+

### Recommended Configuration
- **GPU**: NVIDIA RTX 4080/4090 (16GB+ VRAM) for ultra-high quality
- **RAM**: 32GB+ for large batch processing
- **Storage**: 50GB+ SSD for optimal performance
- **CUDA**: 11.8 or newer

### Software Dependencies
- **ComfyUI**: Latest stable version
- **Python**: 3.10-3.11
- **PyTorch**: 2.0.0+ with CUDA support
- **Three.js**: Automatically loaded for web viewer

## 🔧 Installation

### 1. Install ComfyUI Node Package

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/your-username/ComfyUI_HunyuanWorld.git
cd ComfyUI_HunyuanWorld
pip install -r requirements.txt
```

### 2. Download Required Models

**⚠️ CRITICAL**: This package contains only node code. You must download model files separately.

Create the model directory structure:
```bash
mkdir -p ComfyUI/models/hunyuan_world
```

**Required HunyuanWorld Models** (download all 6 models):

| Model File | Purpose | Size | Required |
|------------|---------|------|----------|
| `HunyuanWorld-PanoDiT-Text.safetensors` | Text → Panorama | 478MB | ✅ Essential |
| `HunyuanWorld-PanoDiT-Image.safetensors` | Image → Panorama | 478MB | ✅ Essential |
| `HunyuanWorld-PanoInpaint-Scene.safetensors` | Scene Inpainting | 478MB | ✅ Essential |
| `HunyuanWorld-PanoInpaint-Sky.safetensors` | Sky Inpainting | 120MB | ✅ Essential |
| `HunyuanWorld-SceneGenerator.safetensors` | 3D Scene Generation | 1.2GB | ⚠️ Advanced Features |
| `HunyuanWorld-WorldReconstructor.safetensors` | 3D Reconstruction | 1.5GB | ⚠️ Advanced Features |

**Quick Download** (requires huggingface-cli):
```bash
# Install huggingface CLI
pip install huggingface-hub

# Download essential models (1.5GB total)
huggingface-cli download Tencent-Hunyuan/HunyuanWorld \
  HunyuanWorld-PanoDiT-Text.safetensors \
  HunyuanWorld-PanoDiT-Image.safetensors \
  HunyuanWorld-PanoInpaint-Scene.safetensors \
  HunyuanWorld-PanoInpaint-Sky.safetensors \
  --local-dir ComfyUI/models/hunyuan_world/

# Download advanced models (2.7GB additional)
huggingface-cli download Tencent-Hunyuan/HunyuanWorld \
  HunyuanWorld-SceneGenerator.safetensors \
  HunyuanWorld-WorldReconstructor.safetensors \
  --local-dir ComfyUI/models/hunyuan_world/
```

### 3. Verify Installation

1. **Restart ComfyUI** completely
2. **Check Node Categories**: Look for "HunyuanWorld" in node browser
3. **Test Basic Workflow**: Load workflow from `workflows/` folder
4. **Verify Models**: Check console for successful model loading

## 🚀 Quick Start Guide

### Basic Text-to-Panorama Workflow

1. **Load Workflow**: Import `workflows/text_to_world_basic.json`
2. **Configure Prompt**: Use HunyuanTextInput node
   ```
   Example: "A majestic mountain landscape with snow-capped peaks, alpine lakes, and evergreen forests, captured during golden hour with dramatic lighting"
   ```
3. **Set Model Path**: Point HunyuanLoader to `models/hunyuan_world`
4. **Select Model Type**: Choose `text_to_panorama`
5. **Generate**: Click "Queue Prompt"

### Professional Panorama Inpainting

1. **Load Workflow**: Import `workflows/professional_panorama_inpainting_workflow.json`
2. **Prepare Assets**:
   - Base panorama image
   - Scene mask (black/white image for areas to modify)
   - Sky mask (for sky replacement)
3. **Configure Inpainting**:
   - Scene inpainting prompt: "Add a wooden dock with boats"
   - Sky inpainting prompt: "Dramatic sunset with golden clouds"
4. **Process**: Run the complete pipeline

### Production Batch Processing

1. **Load Workflow**: Import `workflows/production_batch_processing_workflow.json`
2. **Configure Template**: Set up prompt template with variables
3. **Batch Settings**: Configure parallel processing nodes
4. **Export Pipeline**: Set up automated Draco compression export
5. **Monitor**: Track progress with real-time analytics

## 📚 Complete Node Reference

### Input Processing Nodes
| Node | Function | Key Parameters |
|------|----------|----------------|
| **HunyuanTextInput** | Text prompt processing | `prompt`, `seed`, `negative_prompt` |
| **HunyuanImageInput** | Image preprocessing & enhancement | `resize_mode`, `target_resolution`, `preprocessing` |
| **HunyuanPromptProcessor** | Advanced prompt enhancement | `style`, `lighting`, `atmosphere`, `quality_boost` |
| **HunyuanObjectLabeler** | Object detection & labeling | `fg_labels_1`, `fg_labels_2`, `scene_class` |
| **HunyuanMaskCreator** | Mask creation for inpainting | `mask_type`, `feather`, `target_regions` |

### Core Generation Nodes
| Node | Function | Key Parameters |
|------|----------|----------------|
| **HunyuanLoader** | Model loading & management | `model_path`, `model_type`, `precision`, `device` |
| **HunyuanTextToPanorama** | Text → 360° panorama | `width=1920`, `height=960`, `guidance_scale=30.0` |
| **HunyuanImageToPanorama** | Image → panorama extension | `extension_mode`, `strength`, `blend_extend=6` |
| **HunyuanSceneInpainter** | Professional scene editing | `guidance_scale`, `strength`, `blend_mode` |
| **HunyuanSkyInpainter** | Sky replacement & enhancement | `sky_prompt`, `atmospheric_control` |
| **HunyuanLayeredSceneGenerator** | Multi-layer 3D decomposition | `layer_count`, `object_separation`, `depth_accuracy` |
| **HunyuanWorldReconstructor** | 3D mesh generation | `quality_level`, `mesh_resolution`, `texture_resolution` |

### Advanced Export & Viewing Nodes
| Node | Function | Key Parameters |
|------|----------|----------------|
| **HunyuanViewer** | Interactive 3D visualization | `display_mode`, `layer_controls`, `output_size` |
| **HunyuanMeshExporter** | Standard 3D export | `format`, `texture_resolution`, `compression` |
| **HunyuanDracoExporter** | Professional compressed export | `compression_level`, `quantization_bits`, `optimize_size` |
| **HunyuanLayeredMeshExporter** | Multi-layer export pipeline | `export_background`, `layer_naming`, `format` |
| **HunyuanDataInfo** | Analytics & statistics | Real-time performance monitoring |

## 🎯 Professional Workflow Examples

### 1. Architectural Visualization Pipeline
```
HunyuanTextInput → HunyuanPromptProcessor → HunyuanTextToPanorama
                                                    ↓
HunyuanObjectLabeler → HunyuanLayeredSceneGenerator → HunyuanDracoExporter
```

### 2. Panorama Enhancement & Inpainting
```
LoadImage → HunyuanImageInput → HunyuanImageToPanorama
    ↓               ↓
HunyuanMaskCreator → HunyuanSceneInpainter → HunyuanSkyInpainter → Export
```

### 3. Production Batch Processing
```
Template Input → Multiple HunyuanTextToPanorama (Parallel)
                            ↓
                 Batch HunyuanLayeredSceneGenerator
                            ↓
                 Automated HunyuanDracoExporter Pipeline
```

## ⚙️ Advanced Configuration

### High-Quality Settings (16GB+ VRAM)
```python
# In HunyuanTextToPanorama
width = 3840
height = 1920
guidance_scale = 30.0
num_inference_steps = 50
true_cfg_scale = 1.0
blend_extend = 6
```

### Production Optimization (8-12GB VRAM)
```python
# Optimized for speed/memory balance
width = 1920
height = 960
guidance_scale = 25.0
num_inference_steps = 30
precision = "fp16"
```

### Batch Processing Configuration
```python
# For production environments
batch_size = 4
parallel_workers = 2
auto_clear_cache = True
compression_level = 7
```

## 🔍 Troubleshooting

### Model Loading Issues
```bash
# Check model files exist
ls -la ComfyUI/models/hunyuan_world/
# Should show all 4-6 .safetensors files

# Verify file integrity
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Optimization
- **Reduce Resolution**: Start with 1024x512 for testing
- **Enable FP16**: Use `precision="fp16"` in HunyuanLoader
- **Clear Cache**: Enable `auto_clear_cache` between generations
- **Model Unloading**: Use `unload_after_generation=True`

### Performance Tuning
- **Parallel Processing**: Use multiple generation nodes for batch work
- **Draco Compression**: Optimize export file sizes for production
- **Layer Management**: Selectively enable/disable layers for faster preview

## 🚀 Production Features

### Web-Based 3D Viewer
- **Real-time Rendering**: Three.js-based interactive viewer
- **Layer Controls**: Toggle visibility and opacity per layer
- **Export Integration**: Direct export from viewer interface
- **Performance Monitoring**: FPS tracking and triangle count display

### Professional Export Pipeline
- **Multiple Formats**: OBJ, PLY, GLB, FBX with full material support
- **Draco Compression**: Industry-standard mesh optimization
- **Batch Export**: Automated processing for production workflows
- **Quality Analytics**: Compression ratios and optimization statistics

### Enterprise Integration
- **API Compatibility**: RESTful endpoints for external integration
- **Workflow Templates**: Pre-configured professional workflows
- **Asset Management**: Organized output with naming conventions
- **Performance Metrics**: Detailed analytics for production monitoring

## 📄 License & Credits

**License**: Apache 2.0 License - see LICENSE file for details

**Credits**:
- **HunyuanWorld-1.0**: Tencent Hunyuan Team
- **ComfyUI Integration**: Community development
- **3D Viewer**: Three.js library
- **Mesh Compression**: Google Draco

## 🔗 Resources & Support

- **🏠 HunyuanWorld Official**: https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
- **💬 ComfyUI Community**: https://github.com/comfyanonymous/ComfyUI
- **📖 Technical Documentation**: See `CLAUDE.md` for implementation details
- **🐛 Bug Reports**: GitHub Issues
- **💡 Feature Requests**: GitHub Discussions

## 🆘 Getting Help

1. **Check Documentation**: Review node tooltips and workflow examples
2. **Community Discord**: Join ComfyUI Discord for real-time help
3. **GitHub Issues**: Report bugs with detailed error logs
4. **Professional Support**: Contact for enterprise deployment assistance

---

**⚠️ Important**: This is a community-developed integration package. Model files must be downloaded separately from the official HunyuanWorld repository. For the latest model updates and official support, please refer to Tencent's HunyuanWorld-1.0 repository.

**🎯 Production Ready**: This package is designed for professional workflows and production environments, with comprehensive testing and optimization for real-world use cases.