# Panorama-to-3D World Workflow

This workflow demonstrates the full pipeline from panorama to exportable 3D world.

## Nodes Required:
1. **LoadImage** - Load input panorama
2. **HYW_ModelLoader** - Load HunyuanWorld models 
3. **HYW_WorldReconstructor** - Generate 3D world from panorama
4. **HYW_MeshProcessor** (optional) - Process and clean meshes
5. **HYW_TextureBaker** - Bake textures from panorama
6. **HYW_MeshExport** - Export to GLB/OBJ format

## Connection Flow:
```
LoadImage -> HYW_WorldReconstructor -> HYW_MeshProcessor -> HYW_TextureBaker -> HYW_MeshExport
     ^                ^                       ^                     ^              ^
     |                |                       |                     |              |
HYW_ModelLoader ------+                       |                     |              |
                                              |                     |              |
LoadImage (panorama) -------------------------+---------------------+              |
                                                                                   |
HYW_TextureBaker (baked_textures) ---------------------------------------------+
```

## Configuration:
- **World Reconstruction**:
  - Classes: "outdoor" 
  - Labels FG1: "tree, building, rock"
  - Labels FG2: "person, car, object"
  - Quality: "standard" or "high"
  - Target Size: 3840

- **Texture Baking**:
  - Resolution: 1024
  - Map Types: "albedo\nnormal\nao\nroughness" 
  - Enable AO: true
  
- **Export**:
  - Format: "glb" 
  - Include Materials: true
  - Export Individual Layers: true

## Expected Output:
- 3D mesh files (GLB/OBJ)
- Texture files (PNG)
- Metadata manifest