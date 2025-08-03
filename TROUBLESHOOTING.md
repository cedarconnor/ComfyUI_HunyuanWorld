# HunyuanWorld ComfyUI Import Troubleshooting Guide

If you're getting "node pack failed to import into comfyui", follow these steps to diagnose and fix the issue.

## Quick Diagnosis

### Step 1: Check ComfyUI Console Output

1. **Start ComfyUI** from command line to see detailed error messages
2. **Look for HunyuanWorld messages** in the console output
3. **Check for specific error details** - the new import system provides detailed reporting

Expected console output:
```
Loading HunyuanWorld data types...
Loading input nodes...
Loading generation nodes...
Loading output nodes...
✅ HunyuanWorld ComfyUI: Successfully loaded 18 nodes
Final node count: 18
Available nodes: ['HunyuanTextInput', 'HunyuanImageInput', ...]
```

### Step 2: Run Diagnostic Scripts

**Option A: Basic Syntax Check**
```bash
cd ComfyUI/custom_nodes/ComfyUI_HunyuanWorld
python syntax_check.py
```

**Option B: Detailed Import Testing**
```bash
cd ComfyUI/custom_nodes/ComfyUI_HunyuanWorld
python debug_imports.py
```

## Common Issues and Solutions

### Issue 1: Missing Dependencies

**Symptoms:**
- Import errors mentioning specific packages
- "ModuleNotFoundError" in console

**Solution:**
```bash
cd ComfyUI/custom_nodes/ComfyUI_HunyuanWorld
pip install -r requirements.txt
```

**Alternative for different Python environments:**
```bash
# If using conda
conda install torch torchvision numpy pillow

# If using specific Python version
python3.10 -m pip install -r requirements.txt
```

### Issue 2: Python Path Issues

**Symptoms:**
- "No module named 'core'" or similar
- Relative import failures

**Solution:**
The new `__init__.py` automatically handles path issues, but if problems persist:

1. **Restart ComfyUI completely** (stop and start, don't just refresh)
2. **Check ComfyUI's Python environment** is correct
3. **Verify file permissions** on the node directory

### Issue 3: Partial Node Loading

**Symptoms:**
- Only some nodes appear in ComfyUI
- Console shows "⚠️ Partial load" message

**What to do:**
1. **Check the specific failed imports** in console output
2. **Look for missing files** - all required files should exist:
   - `core/data_types.py`
   - `core/model_manager.py`
   - `nodes/input_nodes.py`
   - `nodes/generation_nodes.py`  
   - `nodes/output_nodes.py`

### Issue 4: ComfyUI Cache Issues

**Symptoms:**
- Old error messages persist
- Nodes don't appear after fixing imports

**Solution:**
1. **Fully restart ComfyUI** (command line: Ctrl+C, then restart)
2. **Clear browser cache** and refresh ComfyUI webpage
3. **Check for multiple ComfyUI instances** running

## Advanced Diagnostics

### Check File Integrity

Run this in the HunyuanWorld directory:
```bash
# Check all required files exist
ls -la core/
ls -la nodes/
ls -la utils/

# Verify file sizes (shouldn't be 0 bytes)
du -h core/*.py nodes/*.py
```

### Check Python Environment

```bash
# Verify Python can find torch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check if PIL/Pillow is available
python -c "from PIL import Image; print('PIL available')"

# Test numpy
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
```

### Manual Import Test

Create a test file `test_manual.py`:
```python
import sys
import os
sys.path.insert(0, os.getcwd())

# Test each module individually
try:
    from core.data_types import PanoramaImage
    print("✅ Core data types OK")
except Exception as e:
    print(f"❌ Core data types failed: {e}")

try:
    from nodes.input_nodes import HunyuanTextInput
    print("✅ Input nodes OK")
except Exception as e:
    print(f"❌ Input nodes failed: {e}")

try:
    from nodes.generation_nodes import HunyuanLoader
    print("✅ Generation nodes OK")
except Exception as e:
    print(f"❌ Generation nodes failed: {e}")

try:
    from nodes.output_nodes import HunyuanViewer
    print("✅ Output nodes OK")
except Exception as e:
    print(f"❌ Output nodes failed: {e}")
```

## Getting Help

### 1. Collect Information

Before reporting issues, gather:
- **ComfyUI version**
- **Python version** (`python --version`)
- **Operating system**
- **Full console output** from ComfyUI startup
- **Output from diagnostic scripts**

### 2. Check Console Output Carefully

The new import system provides detailed error reporting:
```
⚠️ HunyuanWorld ComfyUI: Partial load - 15 nodes loaded with some errors
   - HunyuanSceneInpainter not found in nodes.generation_nodes
   - Failed to import nodes.output_nodes: No module named 'trimesh'
```

This tells you exactly what's missing or broken.

### 3. Common Error Messages

**"No module named 'torch'"**
- Install PyTorch: `pip install torch torchvision`

**"No module named 'trimesh'"**
- Install 3D processing libs: `pip install trimesh pymeshlab`

**"Permission denied"**
- Check file permissions
- Run ComfyUI as administrator (Windows) or with proper permissions

**"Syntax error"**
- File corruption during download
- Re-download the node pack

## Verified Working Setup

### Minimum Requirements
- **ComfyUI**: Latest version
- **Python**: 3.10 or 3.11
- **PyTorch**: 2.0.0+
- **RAM**: 8GB minimum
- **Storage**: 2GB free space

### Dependencies That Must Work
```bash
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install numpy>=1.21.0
pip install pillow>=8.0.0
pip install opencv-python>=4.5.0
```

### File Structure Check
```
ComfyUI_HunyuanWorld/
├── __init__.py                 ✅ (Updated with new import system)
├── core/
│   ├── __init__.py            ✅
│   ├── data_types.py          ✅
│   └── model_manager.py       ✅
├── nodes/
│   ├── __init__.py            ✅
│   ├── input_nodes.py         ✅ (5 classes)
│   ├── generation_nodes.py    ✅ (8 classes)
│   └── output_nodes.py        ✅ (5 classes)
├── utils/
│   ├── __init__.py            ✅
│   └── validation.py          ✅
├── web/                       ✅
├── workflows/                 ✅
└── requirements.txt           ✅
```

## Success Verification

When everything works correctly, you should see:

1. **Console Output:**
   ```
   ✅ HunyuanWorld ComfyUI: Successfully loaded 18 nodes
   ```

2. **In ComfyUI Interface:**
   - New category "HunyuanWorld" in node browser
   - 18 nodes available across Input, Loaders, and Viewers categories

3. **Node Categories:**
   - **HunyuanWorld/Input**: HunyuanTextInput, HunyuanImageInput, etc.
   - **HunyuanWorld/Loaders**: HunyuanLoader
   - **HunyuanWorld/Viewers**: HunyuanViewer, HunyuanMeshExporter, etc.

If you're still having issues after following this guide, please provide the specific console output and diagnostic results when asking for help.