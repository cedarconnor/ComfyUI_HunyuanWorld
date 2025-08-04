# ComfyUI Node Imports
# Ensures proper module loading for ComfyUI integration

import sys
import os
from pathlib import Path

# Add the parent directory to path for relative imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Now import the generation nodes
try:
    from .generation_nodes import *
    from .input_nodes import *
    from .output_nodes import *
    print("[INFO] ComfyUI HunyuanWorld nodes loaded successfully")
except ImportError as e:
    print(f"[WARNING] ComfyUI node import failed: {e}")