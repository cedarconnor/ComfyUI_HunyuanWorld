# Text-to-Panorama Workflow

This workflow demonstrates how to generate a panorama from a text prompt using HunyuanWorld.

## Nodes Required:
1. **HYW_ModelLoader** - Load the HunyuanWorld models
2. **HYW_PanoGen** - Generate panorama from text
3. **HYW_SeamlessWrap360** (optional) - Make panorama seamless 
4. **SaveImage** - Save the output

## Connection Flow:
```
HYW_ModelLoader -> HYW_PanoGen -> HYW_SeamlessWrap360 -> SaveImage
```

## Configuration:
- **Model Loader Settings**:
  - FLUX Text Model: "black-forest-labs/FLUX.1-dev"  
  - HunyuanWorld LoRA: "tencent/HunyuanWorld-1"
  - Device: "cuda:0" or "cpu"
  - Dtype: "bfloat16" 

- **Panorama Generation**:
  - Prompt: "A beautiful mountain landscape with forests and lakes"
  - Width: 1920, Height: 960 (2:1 aspect ratio)
  - Guidance Scale: 30.0
  - Steps: 50
  - Seed: 42

## Expected Output:
High-quality 360-degree panorama image suitable for VR/AR applications.