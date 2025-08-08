# ComfyUI HunyuanWorld Node Pack

A comprehensive ComfyUI node pack for HunyuanWorld-1.0, enabling text-to-panorama generation and 3D world reconstruction directly within ComfyUI.

## Features

- **Text-to-Panorama**: Generate high-quality 360° panoramas from text prompts
- **Image-to-Panorama**: Convert perspective images to panoramic views  
- **3D World Reconstruction**: Transform panoramas into layered 3D meshes
- **Advanced Inpainting**: Scene and sky inpainting for panorama refinement
- **Texture Baking**: Generate PBR textures from panoramas
- **Multiple Export Formats**: GLB, GLTF, OBJ, PLY, STL support
- **Quality Tools**: Validation, seamless wrapping, and mesh analysis

## Installation

1. Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/your-repo/ComfyUI_HunyuanWorld.git
```

2. Install required dependencies:
```bash
cd ComfyUI_HunyuanWorld
pip install -r requirements.txt
```

3. Download HunyuanWorld models:
   - The nodes will automatically download models from Hugging Face on first use
   - Models: `tencent/HunyuanWorld-1` (LoRA weights)
   - Base models: `black-forest-labs/FLUX.1-dev`, `black-forest-labs/FLUX.1-Fill-dev`

4. Restart ComfyUI to load the new nodes.

## Node Categories

### 🔧 **Loaders**
- **HYW_ModelLoader**: Load and configure HunyuanWorld models
- **HYW_Config**: Create reusable parameter configurations  
- **HYW_SettingsLoader**: Load settings from JSON file

### 🎨 **Generation** 
- **HYW_PanoGen**: Text/Image-to-panorama generation
- **HYW_PanoGenBatch**: Batch panorama generation
- **HYW_PanoInpaint_Scene**: Scene element inpainting
- **HYW_PanoInpaint_Advanced**: Advanced multi-region inpainting
- **HYW_PanoInpaint_Sky**: Specialized sky inpainting

### 🏗️ **Reconstruction**
- **HYW_WorldReconstructor**: Convert panorama to layered 3D world
- **HYW_MeshProcessor**: Process and refine mesh geometry
- **HYW_MeshAnalyzer**: Analyze mesh quality and properties

### 📦 **Export**
- **HYW_TextureBaker**: Bake PBR textures from panoramas  
- **HYW_MeshExport**: Export meshes to various 3D formats
- **HYW_Thumbnailer**: Generate 3D world previews

### 🛠️ **Utils**
- **HYW_SeamlessWrap360**: Make panoramas seamless
- **HYW_SkyMaskGenerator**: Generate sky masks automatically
- **HYW_PanoramaValidator**: Validate panorama quality
- **HYW_MetadataManager**: Track workflow metadata

## Quick Start

### Basic Text-to-Panorama
1. Add **HYW_ModelLoader** node
2. Add **HYW_PanoGen** node  
3. Connect ModelLoader output to PanoGen input
4. Set your text prompt in PanoGen
5. Add **SaveImage** node to save result

### Full 3D Pipeline
1. **HYW_ModelLoader** → **HYW_PanoGen** (generate panorama)
2. **HYW_WorldReconstructor** (create 3D world from panorama)
3. **HYW_TextureBaker** (bake textures)
4. **HYW_MeshExport** (export to GLB/OBJ)

## Configuration

### Model Settings
Edit `settings.json` to configure default model paths and parameters:

```json
{
  "model_paths": {
    "pano_text": "tencent/HunyuanWorld-1",
    "pano_image": "tencent/HunyuanWorld-1",
    "flux_text": "black-forest-labs/FLUX.1-dev", 
    "flux_image": "black-forest-labs/FLUX.1-Fill-dev"
  },
  "device": "cuda:0",
  "dtype": "bfloat16",
  "defaults": {
    "pano_size": [1920, 960],
    "guidance_scale": 30.0,
    "steps": 50
  }
}
```

### Performance Optimization
- Enable CPU offloading to save VRAM
- Use VAE tiling for large images
- Adjust target mesh complexity based on hardware

## Supported Formats

### Input
- Text prompts (for generation)
- Images: PNG, JPG, WebP
- Panoramas: Equirectangular format (2:1 aspect ratio recommended)

### Output  
- **Images**: PNG, JPG  
- **3D Meshes**: GLB, GLTF, OBJ, PLY, STL
- **Textures**: PNG, JPG with PBR maps (albedo, normal, AO, roughness)

## Example Workflows

See the `examples/` directory for detailed workflow guides:
- `text_to_panorama.md` - Basic panorama generation
- `panorama_to_3d_world.md` - Full 3D reconstruction pipeline

## Hardware Requirements

### Minimum
- GPU: 8GB VRAM (RTX 3070 / RTX 4060)
- RAM: 16GB system memory
- Storage: 10GB for models

### Recommended  
- GPU: 12GB+ VRAM (RTX 4070 Ti / RTX 4080)
- RAM: 32GB system memory
- Storage: 20GB+ for models and outputs

## Troubleshooting

### Common Issues

**"HunyuanWorld modules not available"**
- Ensure HunyuanWorld-1.0 is cloned in the correct directory
- Check that all dependencies are installed

**CUDA out of memory**
- Enable model CPU offloading
- Reduce panorama resolution
- Use lower mesh target sizes

**Poor panorama quality**
- Increase guidance scale (25-35)
- Use more inference steps (50-100)
- Adjust prompt for better descriptions

**Mesh export fails**
- Check output directory permissions
- Reduce mesh complexity
- Try different export formats

### Performance Tips
- Use bfloat16 dtype for best VRAM efficiency  
- Enable VAE tiling for large panoramas
- Process in batches for multiple outputs
- Use preview quality for testing workflows

## Data Types

The node pack uses custom data types for efficient processing:
- **HYW_RUNTIME**: Model runtime instance
- **HYW_CONFIG**: Configuration parameters
- **HYW_METADATA**: Processing metadata
- **HYW_MESH_LAYERS**: Layered 3D mesh data
- **HYW_BAKED_TEXTURES**: Texture atlas data

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request

## License

This project follows the HunyuanWorld-1.0 Community License. See the original repository for details.

## Credits

- **HunyuanWorld-1.0**: Tencent AI Lab
- **FLUX Models**: Black Forest Labs  
- **ComfyUI Integration**: Community contribution

## Support

- Documentation: See `examples/` directory
- Issues: GitHub issue tracker
- Community: ComfyUI Discord server

---

*Built with ❤️ for the ComfyUI community*