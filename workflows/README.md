# HunyuanWorld ComfyUI Workflow Templates

This directory contains sample workflow templates for the HunyuanWorld ComfyUI custom nodes. Each workflow demonstrates different use cases and capabilities of the integration.

## Available Workflows

### 1. Text-to-World Basic (`text_to_world_basic.json`)
**Purpose**: Complete pipeline from text prompt to 3D world mesh
**Use Case**: Generate explorable 3D worlds from natural language descriptions

**Key Features**:
- Text input with prompt enhancement
- Panoramic image generation
- 3D scene reconstruction
- Mesh export in OBJ format
- Real-time preview at each stage

**Nodes Used**:
- `HunyuanTextInput` - Text prompt input
- `HunyuanPromptProcessor` - Enhance prompts with style/lighting
- `HunyuanLoader` - Load different model components
- `HunyuanTextToPanorama` - Generate 360° images
- `HunyuanSceneGenerator` - Create depth maps and segmentation
- `HunyuanWorldReconstructor` - Build 3D mesh
- `HunyuanMeshExporter` - Export to file formats
- `HunyuanViewer` - Preview results

**Recommended Settings**:
- Resolution: 1024x512 for panorama
- Inference steps: 50 for high quality
- Mesh resolution: 512 for balanced performance
- Export format: OBJ with materials

### 2. Image-to-Panorama Basic (`image_to_panorama_basic.json`)
**Purpose**: Convert regular images to 360° panoramic format only
**Use Case**: Quick panorama creation for VR, 360° viewers, or further processing

**Key Features**:
- Simple image-to-panorama conversion
- Real-time preview of results
- Multiple extension algorithms
- High-resolution output support (up to 2048x1024)
- Detailed panorama information display

**Nodes Used**:
- `LoadImage` - Load source image
- `HunyuanImageInput` - Process and resize image  
- `HunyuanImageToPanorama` - Convert to 360° format
- `HunyuanViewer` - Display panorama info
- `PreviewImage` - Visual preview

**Extension Modes**:
- **Outpainting**: AI-extend image edges (recommended)
- **Seamless**: Tile/repeat image content
- **Symmetric**: Mirror image for panorama

**Recommended Settings**:
- Resolution: 2048x1024 for high quality
- Extension mode: "outpainting" for natural results
- Strength: 0.8 for balanced AI modification
- Steps: 35 for good quality/speed balance

### 3. Image-to-World Basic (`image_to_world_basic.json`)
**Purpose**: Convert existing images into 3D explorable worlds
**Use Case**: Transform photographs or artwork into immersive environments

**Key Features**:
- Image preprocessing and resizing
- Multiple panorama extension modes
- Complete 3D reconstruction pipeline
- GLB export format support

**Nodes Used**:
- `LoadImage` - Load source image
- `HunyuanImageInput` - Process and resize image
- `HunyuanImageToPanorama` - Convert to 360° format
- Complete 3D reconstruction pipeline

**Extension Modes**:
- **Seamless**: Tile/repeat image content
- **Outpainting**: AI-extend image edges  
- **Symmetric**: Mirror image for panorama

**Recommended Settings**:
- Resize mode: "crop" to maintain aspect ratio
- Extension mode: "outpainting" for natural results
- Lower inference steps (30) for faster processing
- Export format: GLB for modern 3D applications

### 4. Advanced Multi-Input (`advanced_multi_input.json`)
**Purpose**: Combine multiple text prompts and images for complex scenes
**Use Case**: Create diverse worlds with multiple themes or reference materials

**Key Features**:
- Multiple independent text inputs
- Different prompt processing styles
- Parallel panorama generation
- Combined 3D reconstruction
- High-resolution output (2048x1024)

**Workflow Structure**:
- **Input Layer**: 2 text prompts + 1 reference image
- **Processing Layer**: Individual prompt enhancement
- **Generation Layer**: Parallel panorama creation
- **Combination Layer**: Single high-quality 3D mesh
- **Export Layer**: High-resolution GLB output

**Use Cases**:
- Fantasy/realistic hybrid environments
- Multi-biome worlds (forest + ocean + desert)
- Architectural visualization with natural surroundings
- Game environment concept development

### 5. Batch Processing (`batch_processing.json`)
**Purpose**: Generate multiple 3D worlds efficiently in parallel
**Use Case**: Content creation pipelines, style exploration, asset generation

**Key Features**:
- 4 parallel text-to-world pipelines
- Shared model loading for efficiency
- Individual export paths and settings
- Optimized for production workflows

**Batch Configuration**:
- **Input**: 4 different environment prompts
- **Processing**: Parallel generation with shared models
- **Output**: 4 separate OBJ mesh files
- **Settings**: Balanced quality/speed (256 mesh resolution)

**Sample Prompts Included**:
1. Mountain forest with lake
2. Desert canyon landscape  
3. Tropical beach paradise
4. Snowy arctic wilderness

**Production Benefits**:
- Model loading overhead shared across batches
- Consistent processing parameters
- Organized output file naming
- Scalable to more parallel streams

## Getting Started

### Prerequisites
1. ComfyUI installed and running
2. HunyuanWorld models downloaded to `models/hunyuan_world/`
3. Sufficient VRAM (8GB+ recommended for high-quality generation)

### Quick Start
1. Open ComfyUI in your browser
2. Load any workflow template using "Load" button
3. Adjust prompts and settings as needed
4. Click "Queue Prompt" to start generation
5. Monitor progress in the UI
6. Find exported files in the `output/` directory

### Model Requirements
Each workflow requires specific HunyuanWorld model components:
- **text_to_panorama**: For generating panoramic images from text
- **scene_generator**: For depth estimation and semantic segmentation  
- **world_reconstructor**: For 3D mesh generation

Make sure all required models are available before running workflows.

### Performance Tips
- Start with lower resolutions (512x256) for testing
- Use fp16 precision to save VRAM
- Enable model caching for batch processing
- Monitor GPU memory usage during generation

## Customization Guide

### Modifying Prompts
- Edit text in `HunyuanTextInput` nodes
- Adjust `HunyuanPromptProcessor` settings for different styles
- Use negative prompts to avoid unwanted elements

### Quality vs Speed Trade-offs
- **Higher Quality**: More inference steps, higher resolutions, fp32 precision
- **Faster Generation**: Fewer steps, lower resolutions, fp16 precision
- **Balanced**: 30-50 steps, 1024x512 resolution, fp16 precision

### Export Formats
- **OBJ**: Widely supported, good for editing
- **PLY**: Simple geometry, smaller files
- **GLB**: Modern standard, includes materials
- **FBX**: Animation support, professional tools

### Memory Optimization
- Use smaller batch sizes if running out of VRAM
- Unload unused models between different workflow stages
- Consider using CPU for less critical operations

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce resolution, batch size, or use fp16 precision
2. **Model Not Found**: Check model paths in `HunyuanLoader` nodes
3. **Slow Generation**: Reduce inference steps or resolution
4. **Export Failures**: Check output directory permissions

### Performance Monitoring
- Use Task Manager/Activity Monitor to watch VRAM usage
- Monitor ComfyUI console for error messages
- Check generated files for quality issues

## Advanced Usage

### Custom Node Combinations
You can mix and match nodes from different workflows:
- Use image input with text enhancement
- Combine multiple panoramas before 3D generation
- Add custom post-processing nodes

### Integration with Other Nodes
HunyuanWorld nodes work well with standard ComfyUI nodes:
- Image preprocessing nodes
- Upscaling nodes for higher quality
- Video generation for animated worlds

### Workflow Optimization
- Cache loaded models across different generations
- Use workflow groups to organize complex pipelines
- Save frequently used configurations as presets

## Support and Resources

- Check the main README for installation instructions
- Review node tooltips for parameter explanations
- Monitor ComfyUI logs for debugging information
- Experiment with different parameter combinations

Happy world building! 🌍✨