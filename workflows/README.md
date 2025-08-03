# HunyuanWorld ComfyUI Workflows

⚠️ **IMPORTANT**: These workflows test the ComfyUI node framework with **placeholder data**. Actual HunyuanWorld model inference is not yet implemented.

## Framework Testing Workflows

### 1. `framework_testing_basic.json` ✅
**Purpose**: Complete node architecture test
- Tests all major nodes: Input → Generation → Reconstruction → Export
- Generates placeholder panorama and 3D data
- **Functional**: 3D viewer, mesh export, data pipeline
- **Placeholder**: Model inference, AI generation

### 2. `export_pipeline_test.json` ✅
**Purpose**: Export functionality validation
- Tests OBJ, Draco compression, and viewer export
- **All export features are fully functional**
- Uses placeholder mesh data for testing

### 3. `viewer_functionality_test.json` ✅
**Purpose**: 3D viewer and analytics testing
- Tests interactive Three.js viewer
- Performance monitoring and statistics
- **Fully functional viewer with real-time controls**

## Legacy Workflows (Framework Compatible)

### 4. `text_to_world_basic.json` ⚠️
**Status**: Framework test with placeholder output
- Original workflow structure maintained
- Added framework testing metadata
- **Use for node architecture validation**

### 5. `image_to_panorama_basic.json` ⚠️
**Status**: Framework ready, placeholder panorama extension
- Image loading and processing works
- Extension algorithms use simple tiling (not AI)
- **Use for testing image-to-panorama node flow**

### 6. `image_to_world_basic.json` ⚠️
**Status**: Framework ready, placeholder 3D generation
- Complete pipeline from image to 3D mesh
- Uses placeholder depth estimation and reconstruction
- **Export and viewer functionality work perfectly**

### 7. `advanced_multi_input.json` ⚠️
**Status**: Complex workflow testing
- Multiple input processing
- Parallel placeholder generation
- **Tests advanced node combinations**

### 8. `batch_processing.json` ⚠️
**Status**: Batch framework testing
- Parallel placeholder generation
- Model loading efficiency testing
- **Tests production workflow architecture**

### 9. Professional Workflows ⚠️
- `professional_text_to_world_advanced.json`
- `professional_panorama_inpainting_workflow.json`
- `professional_image_to_world_enhanced.json`
- `production_batch_processing_workflow.json`

**Status**: Advanced framework testing with professional parameters

## Workflow Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully functional feature |
| ⚠️ | Framework ready, placeholder output |
| ❌ | Not implemented |

## Testing Instructions

### For Framework Validation:
1. Load any workflow in ComfyUI
2. Check that all nodes appear in correct categories
3. Verify data flows between nodes
4. Confirm functional components work (viewer, export)
5. Observe placeholder data generation messages

### For Export Testing:
1. Use `export_pipeline_test.json`
2. Run workflow to generate test mesh
3. Check output files are created
4. Test 3D viewer functionality
5. Verify Draco compression works

### For 3D Viewer Testing:
1. Use `viewer_functionality_test.json`
2. Generate placeholder mesh data
3. Click on HunyuanViewer output to open 3D viewer
4. Test mouse controls (rotate, zoom, pan)
5. Check layer visibility toggles
6. Verify performance stats display

### For Development:
1. Use workflows to test new node implementations
2. Replace placeholder functions with real model inference
3. Maintain compatibility with existing data types
4. Test export pipeline after model integration

## Current Framework Status

### ✅ **Fully Functional Components**
- **ComfyUI Integration**: All nodes load properly in correct categories
- **Data Pipeline**: PanoramaImage → Scene3D → WorldMesh → Export
- **3D Viewer**: Interactive Three.js viewer with real-time controls
- **Export System**: OBJ, PLY, GLB export with Draco compression
- **Performance Monitoring**: Real memory usage and statistics tracking
- **Workflow Management**: All templates load and execute properly

### ⚠️ **Placeholder Components** 
- **Text-to-Panorama**: Generates random data instead of AI inference
- **Image-to-Panorama**: Uses simple tiling instead of AI extension
- **Panorama Inpainting**: Framework ready but outputs placeholder data
- **3D Scene Generation**: Placeholder depth maps and segmentation
- **3D Reconstruction**: Random vertices/faces instead of real geometry

### ❌ **Missing Integration**
- Actual HunyuanWorld-1.0 Python API integration
- Real .safetensors model weight loading and inference
- Genuine AI-powered panorama generation and processing
- Production-quality 3D reconstruction algorithms

## Expected Console Output

When running workflows, you'll see messages like:
```
🎨 [PLACEHOLDER] Generating panorama from prompt: 'Mountain landscape' using HunyuanWorld-PanoDiT-Text.safetensors
⚠️  Framework test output - not actual HunyuanWorld inference

🏗️ [PLACEHOLDER] Generating 3D scene using models/hunyuan_world
⚠️  Framework test output - not actual HunyuanWorld scene generation

✅ Exporting 1000 vertices, 1800 faces to test_export.obj
✅ 3D Viewer loaded successfully with interactive controls
```

## Development Integration Path

### Phase 1: Model Loading (Current)
- ✅ Framework recognizes .safetensors files
- ✅ Model loading architecture complete
- ❌ Actual weight loading and GPU allocation

### Phase 2: Inference Integration (Next)
1. Replace `torch.randn()` with real HunyuanWorld API calls
2. Implement proper `.safetensors` loading in model classes
3. Add real panorama generation algorithms
4. Integrate scene inpainting and sky replacement

### Phase 3: Production Features (Future)
1. Add layered scene decomposition (HunyuanWorld's key feature)
2. Implement high-resolution pipeline (3840x1920)
3. Add advanced object labeling and semantic segmentation
4. Optimize for production batch processing

## Current Limitations

- **No real AI inference** - all generation uses placeholder data
- **Model files recognized but not used** for actual processing
- **Export and viewer work perfectly** with any mesh data
- **Node architecture is production-ready** for integration

## Future Integration

When HunyuanWorld models are integrated:
1. Replace placeholder functions in `model_manager.py`
2. Add real inference code to model classes
3. Test workflows will become production workflows
4. All export and viewer functionality will work unchanged

## Performance Testing

Use workflows to test:
- **Memory Usage**: Monitor VRAM consumption with placeholder data
- **Node Execution Speed**: Time framework overhead vs future AI inference
- **Export Performance**: Test real mesh export speed and file sizes
- **Viewer Responsiveness**: Validate Three.js performance with test data

## Support

- **Framework Issues**: Check node loading and data flow
- **Export Problems**: Verify file permissions and output directories
- **Viewer Issues**: Test browser compatibility and WebGL support
- **Integration Questions**: Refer to HunyuanWorld-1.0 official repository