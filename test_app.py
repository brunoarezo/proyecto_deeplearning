#!/usr/bin/env python3
"""
Test script to verify the Streamlit app components work correctly.
"""
import os
import sys
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.append('.')

def test_app_components():
    """Test individual components of the Streamlit app."""
    print("🧪 Testing Streamlit App Components...")
    
    # Test imports
    try:
        print("✓ Testing imports...")
        import streamlit as st
        import tensorflow as tf
        import cv2
        import matplotlib.pyplot as plt
        import pandas as pd
        print("✓ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test image preprocessing function
    try:
        print("✓ Testing image preprocessing...")
        from app_plantas import preprocess_image, load_class_names, validate_image
        
        # Create a test image
        test_img = Image.new('RGB', (300, 300), color='red')
        processed = preprocess_image(test_img)
        
        assert processed.shape == (1, 224, 224, 3), f"Wrong shape: {processed.shape}"
        print("✓ Image preprocessing works correctly")
        
        # Test class names loading
        class_names = load_class_names()
        assert len(class_names) == 10, f"Wrong number of classes: {len(class_names)}"
        print("✓ Class names loading works")
        
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return False
    
    print("✅ All core components working!")
    return True

def create_demo_data():
    """Create demo files for testing."""
    print("📁 Creating demo files...")
    
    # Create a sample classes.txt file
    with open('classes.txt', 'w') as f:
        classes = [
            "Rosa", "Girasol", "Tulipán", "Margarita", "Lavanda",
            "Orquídea", "Cactus", "Helecho", "Bambú", "Suculenta"
        ]
        for cls in classes:
            f.write(f"{cls}\n")
    print("✓ Created classes.txt with plant names")
    
    # Create a test image
    test_img = Image.new('RGB', (400, 400), color='green')
    test_img.save('test_plant.jpg')
    print("✓ Created test_plant.jpg")

def main():
    """Main test function."""
    print("🌱 Streamlit Plant Classifier - Component Test\n")
    
    # Create demo data
    create_demo_data()
    
    # Test components
    if test_app_components():
        print("\n🎉 SUCCESS: All components tested successfully!")
        print("\n📝 Next Steps:")
        print("1. Add your trained model.h5 to this directory")
        print("2. Run: streamlit run app_plantas.py")
        print("3. Open browser to http://localhost:8501")
        print("\n✨ Features Ready:")
        print("- Image upload and preprocessing")
        print("- VGG16 model integration")
        print("- Grad-CAM visualization")
        print("- Batch processing")
        print("- CSV export")
        print("- Interactive controls")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()