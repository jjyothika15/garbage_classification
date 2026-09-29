# garbage_classification.py
"""
Week 1: Transfer Learning for Garbage Classification using EfficientNetV2B0
Author: [Your Name]
Internship: AICTE – E-Waste Image Classification
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B0

# Step 1: Image Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Step 2: Placeholder for Dataset Paths
# Replace these paths with your own dataset folders later
train_dir = 'path_to_train_data'
val_dir = 'path_to_val_data'

# (Optional) Step 3: Data Loading (commented out since data not added yet)
# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     train_dir,
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE
# )

# val_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     val_dir,
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE
# )

# Step 4: Load Pretrained EfficientNetV2B0
base_model = EfficientNetV2B0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

# Step 5: Add Custom Layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Assuming 10 classes
])

# Step 6: Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 7: Print Summary
model.summary()
print("\n✅ Model created and compiled successfully!")
