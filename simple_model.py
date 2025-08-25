import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

print("Creating simple mock model...")
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.save('model.h5')
print("Mock model saved successfully!")