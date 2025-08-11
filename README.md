NOT WORKING YET!!!!!
# ComfyUI HunyuanWorld Node Pack

A comprehensive ComfyUI node pack for HunyuanWorld offering text-to-panorama generation, 3D world reconstruction, and related utilities directly within ComfyUI.

**🔧 Development Status: Offline Mode In Progress**

This node pack aims for complete offline operation but currently requires additional configuration development. For immediate offline panorama generation, use the recommended workaround below.

## Requirements

- **ComfyUI** (latest)
- **Python** 3.x
- **CUDA-enabled GPU** (for accelerated models)
- **Dependencies**: see [requirements.txt](requirements.txt)

## Installation

### Automatic Installation (Recommended)

1. Clone this repository into your ComfyUI custom_nodes directory:
   ```bash
   cd C:\ComfyUI\custom_nodes
   git clone https://github.com/cedarconnor/ComfyUI_HunyuanWorld.git
   ```

2. Install all dependencies with a single command:
   ```bash
   cd C:\ComfyUI\custom_nodes\ComfyUI_HunyuanWorld
   C:\ComfyUI\.venv\Scripts\python.exe -m pip install -r requirements.txt
   C:\ComfyUI\.venv\Scripts\python.exe install.py
   ```

3. Place model files:
   - **FLUX Base Models** → `C:\ComfyUI\models\unet\`
     - flux1-dev-fp8.safetensors
     - flux1-fill-dev.safetensors
   - **HunyuanWorld Models** → `C:\ComfyUI\models\Hunyuan_World\`
     - HunyuanWorld-PanoDiT-Text.safetensors
     - HunyuanWorld-PanoDiT-Image.safetensors
     - HunyuanWorld-PanoInpaint-Scene.safetensors
     - HunyuanWorld-PanoInpaint-Sky.safetensors

4. Restart ComfyUI to load the new nodes.

### Manual Installation (If automatic fails)

If the automatic installation encounters issues, install dependencies manually:

```bash
# Core requirements
C:\ComfyUI\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Git-based dependencies
C:\ComfyUI\.venv\Scripts\python.exe -m pip install "git+https://github.com/EasternJournalist/utils3d.git"
C:\ComfyUI\.venv\Scripts\python.exe -m pip install "git+https://github.com/microsoft/MoGe.git"

# Additional HunyuanWorld dependencies
C:\ComfyUI\.venv\Scripts\python.exe -m pip install basicsr realesrgan zim-anything easydict
```

## Troubleshooting

**Import Errors**: If you see "HunyuanWorld modules are not available", ensure all dependencies are installed. Run the installation verification:
```bash
C:\ComfyUI\.venv\Scripts\python.exe install.py
```

**Model Requirements**: The node pack automatically detects your ComfyUI installation and loads components from standard directories:
- FLUX models: `C:\ComfyUI\models\unet\` (flux1-dev-fp8.safetensors, flux1-fill-dev.safetensors)
- HunyuanWorld LoRAs: `C:\ComfyUI\models\Hunyuan_World\` (HunyuanWorld-PanoDiT-Text.safetensors, HunyuanWorld-PanoDiT-Image.safetensors)
- CLIP encoder: `C:\ComfyUI\models\clip\clip_l.safetensors`
- T5 encoder: `C:\ComfyUI\models\text_encoders\t5xxl_fp16.safetensors`
- FLUX VAE: `C:\ComfyUI\models\vae\ae.safetensors`

**Recommended Offline Workaround**: 
1. Use ComfyUI's native **FLUX Dev** node to load `flux1-dev-fp8.safetensors`
2. Add **LoRA Loader** node with `HunyuanWorld-PanoDiT-Text.safetensors`
3. Generate panoramas using the combined FLUX + HunyuanWorld LoRA workflow
4. This provides the same panorama generation capability using ComfyUI's proven model loading system

**VRAM Issues**: Use CPU offloading and VAE tiling in the model loader settings for systems with limited VRAM.

**Performance**: For faster generation, ensure you have sufficient VRAM and use bfloat16 precision.

## Node Categories

### Loaders
- **HYW_ModelLoader**: Load HunyuanWorld models and create runtime instance
- **HYW_SettingsLoader**: Load settings from `settings.json`
- **HYW_Config**: Create or override inference configuration

### Generation
- **HYW_PanoGen**: Generate panorama (text or image conditioned)
- **HYW_PanoGenBatch**: Batch panorama generation
- **HYW_PanoInpaint_Scene**: Inpaint scene regions in panorama
- **HYW_PanoInpaint_Advanced**: Advanced multi-region inpainting
- **HYW_PanoInpaint_Sky**: Inpaint sky regions
- **HYW_SkyMaskGenerator**: Generate sky mask for panorama

### Reconstruction
- **HYW_WorldReconstructor**: Reconstruct 3D world mesh from panorama
- **HYW_MeshProcessor**: Process and clean mesh geometry
- **HYW_MeshAnalyzer**: Analyze mesh quality and statistics

### Export
- **HYW_TextureBaker**: Bake PBR textures from panorama and mesh data
- **HYW_MeshExport**: Export mesh and textures to GLB/OBJ/etc.
- **HYW_Thumbnailer**: Generate 3D world preview thumbnails

### Utils
- **HYW_SeamlessWrap360**: Seamlessly wrap panoramas horizontally
- **HYW_PanoramaValidator**: Validate panorama images
- **HYW_MetadataManager**: Manage workflow metadata and hashing

## Examples

See the `examples/` directory for detailed step-by-step guides:
- [Text-to-Panorama](examples/text_to_panorama.md)
- [Panorama-to-3D World](examples/panorama_to_3d_world.md)

## Configuration

Edit `settings.json` to configure default model paths, runtime parameters, and optimizations. For example:

```json
{
  "model_paths": {
    "flux_text": "models/unet/flux1-dev-fp8.safetensors",
    "flux_image": "models/unet/flux1-fill-dev.safetensors",
    "pano_text_lora": "models/Hunyuan_World/HunyuanWorld-PanoDiT-Text.safetensors",
    "pano_image_lora": "models/Hunyuan_World/HunyuanWorld-PanoDiT-Image.safetensors"
  },
  "device": "cuda:0",
  "dtype": "bfloat16",
  "defaults": {
    "pano_size": [1920, 960],
    "guidance_scale": 30.0,
    "num_inference_steps": 50,
    "blend_extend": 6,
    "true_cfg_scale": 0.0,
    "shifting_extend": 0
  },
  "optimization": {
    "enable_model_cpu_offload": true,
    "enable_vae_tiling": true,
    "enable_xformers": true
  }
}
```

## Contributing

Contributions welcome! Please open issues or PRs to add new features, improve documentation, or fix bugs.

## License

This project follows the HunyuanWorld-1.0 Community License. See the original repository for details.
