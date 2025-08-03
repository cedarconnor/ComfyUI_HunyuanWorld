# ComfyUI HunyuanWorld - Professional 3D World Generation

A comprehensive ComfyUI custom node package for [Tencent's HunyuanWorld-1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) with complete AI inference integration for professional 3D world generation.

## 🚀 **Quick Start**

```bash
git clone https://github.com/cedarconnor/ComfyUI_HunyuanWorld.git
cd ComfyUI_HunyuanWorld
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git
pip install -r requirements_hunyuan.txt
# Restart ComfyUI → Load workflows → Create real 3D worlds with AI!
```

## 🎯 **Features**

| Feature | Model | Status |
|---------|-------|--------|
| **Text→Panorama** | FLUX.1-dev + HunyuanWorld LoRA | ✅ Production Ready |
| **Image→Panorama** | FLUX.1-fill + HunyuanWorld LoRA | ✅ Production Ready |
| **Scene Generation** | LayerDecomposition + WorldComposer | ✅ Production Ready |
| **3D Export & Viewer** | Three.js + Draco Compression | ✅ Production Ready |
| **Panorama Inpainting** | HunyuanWorld-PanoInpaint | 🔄 Coming Soon |

## 🌟 **Key Features**

### **Complete AI Integration**
- **15+ Professional Nodes**: Full pipeline from text/image → 3D world
- **Real HunyuanWorld Models**: FLUX.1-dev base with HunyuanWorld LoRA adapters
- **Local Inference Only**: No web calls - everything runs on your machine
- **Production Workflows**: Batch processing, advanced parameters, professional export

### **AI Capabilities**
- **Text-to-World Generation**: Create 360° environments from text descriptions
- **Image-to-Panorama**: AI-powered panoramic extension of regular images  
- **Advanced Scene Processing**: Multi-layer decomposition and 3D reconstruction
- **Professional Export**: OBJ, PLY, GLB with Draco compression

### **Professional Features**
- **Interactive 3D Viewer**: Three.js-based real-time visualization
- **Professional Export Pipeline**: Multiple formats with compression
- **Performance Monitoring**: Real-time memory and processing analytics
- **Workflow Templates**: Pre-configured professional workflows

## 📋 **System Requirements**

### **Recommended Configuration**
- **GPU**: NVIDIA RTX 4080+ (16GB+ VRAM recommended)
- **RAM**: 32GB+ for large panorama processing
- **Storage**: 50GB+ (models: ~10GB, working space: ~40GB)
- **CUDA**: 11.8 or newer
- **OS**: Windows 10+, Linux Ubuntu 20.04+, macOS 12+

### **Minimum Requirements**
- **GPU**: NVIDIA RTX 3080 (10GB VRAM)
- **RAM**: 16GB system memory
- **Storage**: 25GB free space
- **CUDA**: 11.0 or newer

## 🔧 **Installation & Setup**

### **Step 1: Basic Installation**

```bash
# Install in ComfyUI custom_nodes directory
cd ComfyUI/custom_nodes/
git clone https://github.com/cedarconnor/ComfyUI_HunyuanWorld.git
cd ComfyUI_HunyuanWorld

# Install basic dependencies
pip install -r requirements.txt
```

### **Step 2: AI Model Integration**

```bash
# 1. Install AI dependencies
pip install -r requirements_hunyuan.txt

# 2. Clone official HunyuanWorld repository
git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git

# 3. Optional: Install performance optimizations
pip install xformers  # Memory-efficient attention

# Your directory structure should be:
# ComfyUI_HunyuanWorld/
#   ├── HunyuanWorld-1.0/          ← Official repository
#   ├── core/
#   ├── nodes/
#   └── workflows/
```

### **Step 3: Optional Dependencies**

For full 3D functionality, install utils3d (may have dependency conflicts on some systems):

```bash
# Option 1: Try direct install (may fail due to conflicts)
pip install utils3d

# Option 2: Install from source (recommended)
python install_utils3d.py

# Option 3: Manual install from GitHub
pip install git+https://github.com/EasternJournalist/utils3d.git
```

### **Step 4: Model Downloads**

Models download automatically on first use:

- **Base Models**: `black-forest-labs/FLUX.1-dev`, `black-forest-labs/FLUX.1-fill-dev`
- **HunyuanWorld LoRA**: `tencent/HunyuanWorld-1`
- **Total Size**: ~10GB (downloads once, caches locally)

### **Step 5: Verify Installation**

```bash
# Test the integration
python test_integration.py

# Expected output:
# ✅ Integration module imported
# HunyuanWorld Available: ✅ Yes
# ✅ Model manager created
```

## 🚀 **Usage Guide**

### **Basic Workflow**

1. **Load Workflow**: Import any `.json` from `workflows/` folder
2. **Configure Prompts**: Use HunyuanTextInput or HunyuanImageInput nodes
3. **Run Pipeline**: Generate real AI-powered 3D worlds
4. **View Results**: Interactive 3D viewer with export options

**Console Output**:
```
✅ HunyuanWorld integration available
🔄 Loading real HunyuanWorld Text2Panorama model...
✅ HunyuanWorld Text2Panorama loaded successfully
🎨 Generating panorama: 'Mountain landscape'
✅ Generated panorama: torch.Size([960, 1920, 3])
```

### **Recommended Workflows**

| Workflow | Purpose | AI Features |
|----------|---------|-------------|
| `framework_testing_basic.json` | Test all nodes | Complete pipeline validation |
| `export_pipeline_test.json` | Test 3D export | Real mesh export with compression |
| `viewer_functionality_test.json` | Test 3D viewer | Interactive visualization |
| `text_to_world_basic.json` | Text→3D world | FLUX.1-dev + HunyuanWorld |
| `image_to_panorama_basic.json` | Image→panorama | FLUX.1-fill + HunyuanWorld |

## 📚 **Node Reference**

### **Input Processing**
- **HunyuanTextInput**: Text prompt processing with enhancement
- **HunyuanImageInput**: Image preprocessing and optimization
- **HunyuanPromptProcessor**: Advanced prompt styling and control
- **HunyuanObjectLabeler**: Automated object detection and labeling
- **HunyuanMaskCreator**: Mask creation for inpainting workflows

### **Core Generation**
- **HunyuanLoader**: AI model loading and management
- **HunyuanTextToPanorama**: Text→360° panorama (FLUX.1-dev + HunyuanWorld)
- **HunyuanImageToPanorama**: Image→panorama extension (FLUX.1-fill + HunyuanWorld)
- **HunyuanSceneInpainter**: Professional scene editing and inpainting
- **HunyuanSkyInpainter**: Sky replacement and atmospheric control
- **HunyuanLayeredSceneGenerator**: Multi-layer 3D scene decomposition
- **HunyuanWorldReconstructor**: 3D mesh generation and optimization

### **Export & Visualization**
- **HunyuanViewer**: Interactive Three.js 3D visualization
- **HunyuanMeshExporter**: Standard 3D export (OBJ, PLY, GLB)
- **HunyuanDracoExporter**: Professional compressed export
- **HunyuanLayeredMeshExporter**: Multi-layer export pipeline
- **HunyuanDataInfo**: Real-time analytics and performance monitoring

## ⚙️ **Advanced Configuration**

### **Performance Optimization**

```python
# High Quality (16GB+ VRAM)
precision = "fp16"
num_inference_steps = 50
guidance_scale = 30.0
width, height = 3840, 1920

# Balanced (12GB VRAM)  
precision = "fp16"
num_inference_steps = 30
guidance_scale = 25.0
width, height = 1920, 960

# Fast/Low Memory (8GB VRAM)
precision = "fp16" 
num_inference_steps = 20
guidance_scale = 15.0
width, height = 1024, 512
```

### **Model Parameters**

#### Text-to-Panorama (HunyuanWorld Optimized):
- `width`: 1920 (HunyuanWorld standard)
- `height`: 960 (2:1 panoramic ratio)
- `guidance_scale`: 30.0 (HunyuanWorld optimized)
- `num_inference_steps`: 50 (high quality)
- `true_cfg_scale`: 0.0 (HunyuanWorld default)
- `blend_extend`: 6 (seamless panorama blending)

#### Image-to-Panorama:
- `strength`: 0.8 (modification intensity)
- `num_inference_steps`: 30
- `guidance_scale`: 7.5

## 🔍 **Troubleshooting**

### **Common Issues**

#### 1. **"HunyuanWorld integration not available"**
```bash
# Check if HunyuanWorld-1.0 directory exists
ls HunyuanWorld-1.0/
# If missing: git clone https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git
```

#### 2. **"CUDA out of memory"**
- Reduce image resolution (1024x512 instead of 1920x960)
- Use `precision="fp16"`
- Reduce `num_inference_steps`
- Close other GPU applications

#### 3. **"Model download failed"**
```bash
# Login to Hugging Face
pip install huggingface-hub
huggingface-cli login
# Check internet connection and disk space (10GB+ needed)
```

#### 4. **"Node not found in ComfyUI"**
- Restart ComfyUI completely
- Check `custom_nodes/ComfyUI_HunyuanWorld` is in correct location
- Verify Python dependencies: `pip install -r requirements_hunyuan.txt`

### **Verification Commands**

```bash
# Test integration
python test_integration.py

# Check dependencies
pip list | grep -E "(torch|diffusers|transformers)"

# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check models (after first use)
ls ~/.cache/huggingface/transformers/  # Linux/Mac
# or C:\Users\%USERNAME%\.cache\huggingface\transformers\  # Windows
```

## 📊 **Performance Benchmarks**

| Model Type | Resolution | Steps | VRAM | Time (RTX 4090) |
|------------|------------|-------|------|------------------|
| Text→Panorama | 1920x960 | 50 | 14GB | 45s |
| Text→Panorama | 1024x512 | 30 | 8GB | 20s |
| Image→Panorama | 1920x960 | 30 | 12GB | 25s |
| Scene Generation | Variable | - | 6GB | 10s |

## 🎯 **Development & Contributing**

### **AI Integration Architecture**
- Real models automatically loaded when HunyuanWorld-1.0 repository present
- Smart error handling with detailed logging
- Modify `core/hunyuan_integration.py` for model improvements

### **Contributing**
1. Test complete AI inference pipeline
2. Ensure all dependencies work correctly
3. Add comprehensive error handling
4. Update workflows and documentation

## 🔗 **Resources & Support**

- **🏠 Official HunyuanWorld**: https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0
- **💬 ComfyUI Community**: https://github.com/comfyanonymous/ComfyUI
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/cedarconnor/ComfyUI_HunyuanWorld/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/cedarconnor/ComfyUI_HunyuanWorld/discussions)
- **📖 FLUX Models**: https://huggingface.co/black-forest-labs
- **🤗 HunyuanWorld LoRA**: https://huggingface.co/tencent/HunyuanWorld-1

## 📄 **License & Credits**

**License**: Apache 2.0 License

**Credits**:
- **HunyuanWorld-1.0**: [Tencent Hunyuan Team](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0)
- **FLUX Models**: [Black Forest Labs](https://huggingface.co/black-forest-labs)
- **ComfyUI Framework**: [ComfyUI Community](https://github.com/comfyanonymous/ComfyUI)
- **Integration Development**: Community contributions

---

**🎯 Production Ready**: Complete ComfyUI integration with real HunyuanWorld-1.0 AI inference for professional 3D world generation!