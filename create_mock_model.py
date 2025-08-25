import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Create a simple mock model for testing
def create_mock_model():
    """Create a mock VGG16 model for testing the Streamlit app."""
    # Use VGG16 base
    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze some layers
    for layer in vgg16_base.layers[:10]:
        layer.trainable = False
    
    # Add custom top layers
    model = models.Sequential([
        vgg16_base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 classes for our mock model
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    print("Creating mock model...")
    model = create_mock_model()
    
    # Save the model
    model.save("model.h5")
    print("Mock model saved as model.h5")
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()