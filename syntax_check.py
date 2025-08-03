#!/usr/bin/env python3
"""
Syntax validation script for HunyuanWorld nodes
"""

import ast
import sys
import os

def check_file_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST to check for syntax errors
        ast.parse(source, filename=file_path)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def main():
    """Check syntax of all Python files in the package"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    files_to_check = [
        "__init__.py",
        "core/data_types.py",
        "core/model_manager.py", 
        "nodes/input_nodes.py",
        "nodes/generation_nodes.py",
        "nodes/output_nodes.py",
        "utils/validation.py"
    ]
    
    print("=== HunyuanWorld Syntax Check ===")
    
    all_good = True
    for file_path in files_to_check:
        full_path = os.path.join(base_dir, file_path)
        
        if os.path.exists(full_path):
            is_valid, error = check_file_syntax(full_path)
            if is_valid:
                print(f"✅ {file_path}: VALID")
            else:
                print(f"❌ {file_path}: {error}")
                all_good = False
        else:
            print(f"⚠️ {file_path}: FILE NOT FOUND")
            all_good = False
    
    print(f"\n=== SUMMARY ===")
    if all_good:
        print("✅ All files have valid syntax!")
    else:
        print("❌ Some files have syntax errors or are missing")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)