#!/usr/bin/env python3
"""
Install utils3d from source to avoid dependency conflicts
Run this if you get utils3d import errors
"""

import subprocess
import sys
import os

def install_utils3d():
    """Install utils3d from GitHub source"""
    print("🔧 Installing utils3d from source...")
    print("This avoids PyPI dependency conflicts")
    
    try:
        # Try installing from git
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/EasternJournalist/utils3d.git"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ utils3d installed successfully!")
            print("🔄 Please restart ComfyUI to apply changes")
            return True
        else:
            print(f"❌ Installation failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during installation: {e}")
        return False

def check_utils3d():
    """Check if utils3d is available"""
    try:
        import utils3d
        print("✅ utils3d is already installed and working")
        return True
    except ImportError:
        print("⚠️ utils3d not found or not working")
        return False

if __name__ == "__main__":
    print("🔍 Checking utils3d installation...")
    
    if check_utils3d():
        print("✅ No action needed - utils3d is working")
    else:
        print("🔧 Installing utils3d...")
        success = install_utils3d()
        
        if success:
            print("\n✅ Installation complete!")
            print("🚀 You can now use HunyuanWorld 3D features")
        else:
            print("\n❌ Installation failed")
            print("💡 Try installing manually:")
            print("   pip install git+https://github.com/EasternJournalist/utils3d.git")