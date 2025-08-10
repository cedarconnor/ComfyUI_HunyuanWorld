#!/usr/bin/env python3
"""
Post-installation script for ComfyUI HunyuanWorld Node Pack
Handles git-based dependencies and compatibility fixes
"""

import subprocess
import sys
import os
import platform

def run_pip_install(packages, description=""):
    """Install packages via pip with error handling"""
    if isinstance(packages, str):
        packages = [packages]
    
    print(f"Installing {description}...")
    for package in packages:
        try:
            print(f"  Installing {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"  Warning: Failed to install {package}: {e}")
            return False
    return True

def install_git_dependencies():
    """Install git-based dependencies"""
    git_packages = [
        ("git+https://github.com/EasternJournalist/utils3d.git", "utils3d (3D utilities)"),
        ("git+https://github.com/microsoft/MoGe.git", "MoGe (monocular geometry)")
    ]
    
    for package, description in git_packages:
        print(f"Installing {description}...")
        if not run_pip_install(package):
            print(f"Failed to install {description}")

def create_compatibility_fixes():
    """Create compatibility fixes for torchvision"""
    print("Applying compatibility fixes...")
    
    try:
        import torchvision
        torchvision_path = os.path.join(os.path.dirname(torchvision.__file__), 'transforms')
        functional_tensor_path = os.path.join(torchvision_path, 'functional_tensor.py')
        
        if not os.path.exists(functional_tensor_path):
            print("  Creating torchvision functional_tensor compatibility module...")
            with open(functional_tensor_path, 'w') as f:
                f.write('''# Compatibility module for older torchvision versions
from torchvision.transforms.functional import rgb_to_grayscale
''')
            print("  ✓ Compatibility module created")
        else:
            print("  ✓ Compatibility module already exists")
            
    except Exception as e:
        print(f"  Warning: Could not create compatibility fixes: {e}")

def verify_installation():
    """Verify that HunyuanWorld modules can be imported"""
    print("Verifying installation...")
    
    # Add HunyuanWorld-1.0 to path
    current_dir = os.path.dirname(__file__)
    hunyuan_path = os.path.join(current_dir, "HunyuanWorld-1.0")
    sys.path.insert(0, hunyuan_path)
    
    try:
        from hy3dworld import Text2PanoramaPipelines, Image2PanoramaPipelines
        from hy3dworld import LayerDecomposition, WorldComposer
        from hy3dworld.utils import Perspective, process_file
        print("  ✓ All HunyuanWorld modules imported successfully!")
        return True
    except ImportError as e:
        print(f"  ✗ Import verification failed: {e}")
        return False

def main():
    """Main installation routine"""
    print("=" * 60)
    print("ComfyUI HunyuanWorld Node Pack - Post Installation")
    print("=" * 60)
    
    # Install git dependencies
    install_git_dependencies()
    
    # Apply compatibility fixes
    create_compatibility_fixes()
    
    # Verify installation
    if verify_installation():
        print("\n" + "=" * 60)
        print("✓ Installation completed successfully!")
        print("✓ HunyuanWorld nodes should now work in ComfyUI")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ Installation verification failed")
        print("Please check the error messages above and try manual installation")
        print("=" * 60)

if __name__ == "__main__":
    main()