# HunyuanWorld ComfyUI Workflows

Professional workflow templates for HunyuanWorld-1.0 AI inference with complete 3D world generation pipeline.

## Core AI Workflows

### 1. `framework_testing_basic.json` ✅
**Purpose**: Complete AI pipeline validation
- Tests all major nodes: Input → AI Generation → 3D Reconstruction → Export
- Real HunyuanWorld AI inference with FLUX.1-dev + HunyuanWorld LoRA
- **Features**: Text-to-panorama, scene generation, 3D mesh creation
- **Output**: High-quality 3D worlds with interactive viewer

### 2. `text_to_world_basic.json` ✅
**Purpose**: Text-to-3D world generation
- Complete pipeline from text prompt to explorable 3D environment
- Uses FLUX.1-dev + HunyuanWorld LoRA for panorama generation
- **Features**: Advanced prompt processing, scene decomposition, mesh export
- **Output**: Professional-quality 3D worlds from text descriptions

### 3. `image_to_panorama_basic.json` ✅
**Purpose**: Image-to-panorama AI extension
- Converts regular images to 360° panoramic format using AI
- Uses FLUX.1-fill + HunyuanWorld LoRA for intelligent extension
- **Features**: Smart image preprocessing, AI-powered panorama generation
- **Output**: High-resolution 360° panoramas from input images

## Professional Workflows

### 4. `professional_text_to_world_advanced.json` ✅
**Purpose**: Advanced text-to-world with professional controls
- Multi-layer scene generation with object labeling
- Advanced prompt processing and style controls
- **Features**: High-resolution output, layered 3D decomposition
- **Output**: Production-quality 3D environments

### 5. `professional_panorama_inpainting_workflow.json` 🔄
**Purpose**: Professional panorama editing and inpainting
- Scene and sky inpainting with mask-based control
- **Features**: Advanced editing capabilities, atmospheric control
- **Status**: Coming soon with HunyuanWorld-PanoInpaint integration

### 6. `professional_image_to_world_enhanced.json` ✅
**Purpose**: Enhanced image-to-world generation
- Complete pipeline from input image to 3D world
- Advanced scene processing and reconstruction
- **Features**: Multi-resolution processing, enhanced 3D generation

## Export & Testing Workflows

### 7. `export_pipeline_test.json` ✅
**Purpose**: 3D export functionality validation
- Tests OBJ, PLY, GLB export with Draco compression
- **Features**: Multiple export formats, compression testing
- **Output**: Optimized 3D files for production use

### 8. `viewer_functionality_test.json` ✅
**Purpose**: Interactive 3D viewer testing
- Tests Three.js-based interactive visualization
- Performance monitoring and analytics
- **Features**: Real-time controls, layer management, export integration

## Batch Processing Workflows

### 9. `batch_processing.json` ✅
**Purpose**: Efficient batch 3D world generation
- Parallel processing of multiple prompts
- Optimized for production environments
- **Features**: Shared model loading, parallel AI inference

### 10. `production_batch_processing_workflow.json` ✅
**Purpose**: Production-scale batch processing
- Advanced batch management with quality controls
- **Features**: Automated workflows, quality assurance, batch export

### 11. `advanced_multi_input.json` ✅
**Purpose**: Complex multi-input scene generation
- Combines multiple text prompts and reference images
- **Features**: Multi-modal input processing, scene blending

## Usage Instructions

### **Basic Setup**
1. **Install HunyuanWorld**: Follow main README setup instructions
2. **Load Workflow**: Import any `.json` file into ComfyUI
3. **Configure Inputs**: Set prompts, images, and parameters
4. **Generate**: Run workflow for real AI inference

### **Expected Performance**
All workflows use real HunyuanWorld AI models:

| Workflow | Model Used | Inference Time | VRAM Required |
|----------|------------|----------------|---------------|
| Text→World | FLUX.1-dev + HunyuanWorld | 30-60s | 12GB+ |
| Image→Panorama | FLUX.1-fill + HunyuanWorld | 20-40s | 10GB+ |
| Scene Generation | LayerDecomposition | 10-20s | 8GB+ |
| 3D Reconstruction | WorldComposer | 15-30s | 6GB+ |

### **Console Output Examples**

**Successful AI Inference**:
```
✅ HunyuanWorld integration available
🔄 Loading real HunyuanWorld Text2Panorama model...
✅ HunyuanWorld Text2Panorama loaded successfully
🎨 Generating panorama: 'Mountain landscape'
✅ Generated panorama: torch.Size([960, 1920, 3])
🏗️ Generating 3D scene with LayerDecomposition
✅ Exporting 2847 vertices, 5694 faces to mountain_world.obj
```

## Workflow Customization

### **Parameter Optimization**
```python
# High Quality (16GB+ VRAM)
width, height = 1920, 960
num_inference_steps = 50
guidance_scale = 30.0

# Balanced (12GB VRAM)
width, height = 1024, 512
num_inference_steps = 30
guidance_scale = 25.0
```

### **Model Selection**
- **Text-to-Panorama**: FLUX.1-dev + HunyuanWorld LoRA
- **Image-to-Panorama**: FLUX.1-fill + HunyuanWorld LoRA
- **Scene Generation**: HunyuanWorld LayerDecomposition
- **3D Reconstruction**: HunyuanWorld WorldComposer

## Troubleshooting

### **"No AI inference detected"**
- Ensure HunyuanWorld-1.0 repository is cloned
- Check `pip install -r requirements_hunyuan.txt`
- Verify CUDA and sufficient VRAM

### **"Model loading failed"**
- Check internet connection for initial model download
- Ensure 10GB+ free disk space
- Login to Hugging Face: `huggingface-cli login`

### **Performance Issues**
- Reduce resolution for lower VRAM setups
- Use `precision="fp16"` for memory optimization
- Close other GPU applications

## Development Notes

All workflows are designed for real AI inference with HunyuanWorld-1.0 models. The integration automatically handles:
- Model loading and caching
- Memory optimization
- Error handling and validation
- Performance monitoring

## Contributing

When creating new workflows:
1. Test with real HunyuanWorld AI models
2. Include proper error handling
3. Document expected performance and requirements
4. Follow naming conventions for parameters