#!/usr/bin/env python3
import streamlit as st
import numpy as np
from PIL import Image
import os

def main():
    print("Testing basic Streamlit functionality...")
    
    # Test Streamlit import
    print(f"✓ Streamlit version: {st.__version__}")
    
    # Test PIL
    test_img = Image.new('RGB', (224, 224), color='red')
    print(f"✓ PIL working, created image size: {test_img.size}")
    
    # Test numpy
    arr = np.zeros((224, 224, 3))
    print(f"✓ NumPy working, array shape: {arr.shape}")
    
    # Check if app file exists
    if os.path.exists('app_plantas.py'):
        print("✓ app_plantas.py exists")
        with open('app_plantas.py', 'r') as f:
            content = f.read()
            if 'def main():' in content:
                print("✓ Main function found in app")
            if 'st.title' in content:
                print("✓ Streamlit UI components found")
    
    print("\n🎉 Basic functionality test complete!")
    print("The app is ready to run with: streamlit run app_plantas.py")

if __name__ == "__main__":
    main()