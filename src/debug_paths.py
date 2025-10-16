#!/usr/bin/env python3
"""
Debug script to understand the directory structure
"""
import os

def check_directory_structure():
    print("=" * 60)
    print("🔍 DEBUGGING DIRECTORY STRUCTURE")
    print("=" * 60)
    
    # Current working directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Check parent directory
    parent_dir = os.path.dirname(current_dir)
    print(f"Parent directory: {parent_dir}")
    
    # List contents of parent directory
    print(f"\nContents of parent directory ({parent_dir}):")
    try:
        for item in sorted(os.listdir(parent_dir)):
            item_path = os.path.join(parent_dir, item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/")
            else:
                print(f"  📄 {item}")
    except Exception as e:
        print(f"  ❌ Error reading parent directory: {e}")
    
    # Check for possible FAISS directories
    possible_faiss_dirs = [
        "../faiss_misc_copy",
        "../faiss_misc",
        "../faiss_misc copy",
        "./faiss_misc_copy",
        "faiss_misc_copy",
        "../misc_faiss",
        "../faiss"
    ]
    
    print(f"\n🔍 Checking for FAISS directories:")
    for faiss_dir in possible_faiss_dirs:
        abs_path = os.path.abspath(faiss_dir)
        if os.path.exists(faiss_dir):
            print(f"  ✅ FOUND: {faiss_dir} -> {abs_path}")
            try:
                contents = os.listdir(faiss_dir)
                print(f"     Contents: {contents}")
                # Check for index files specifically
                if "index.faiss" in contents:
                    print(f"     ✅ index.faiss found!")
                if "index.pkl" in contents:
                    print(f"     ✅ index.pkl found!")
            except Exception as e:
                print(f"     ❌ Error reading directory: {e}")
        else:
            print(f"  ❌ NOT FOUND: {faiss_dir} -> {abs_path}")
    
    # Check for any directories containing "faiss" in parent directory
    print(f"\n🔍 Looking for any directories containing 'faiss' in parent directory:")
    try:
        for item in os.listdir(parent_dir):
            if "faiss" in item.lower() and os.path.isdir(os.path.join(parent_dir, item)):
                item_path = os.path.join(parent_dir, item)
                print(f"  📁 Found: {item}")
                try:
                    contents = os.listdir(item_path)
                    print(f"     Contents: {contents}")
                except Exception as e:
                    print(f"     ❌ Error reading: {e}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    check_directory_structure()
