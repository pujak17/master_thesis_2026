# training/train_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import os

# Paths
dataset_path = "./data/train"
model_output_path = "./models/charamel_emotion_model.h5"
img_size = (224, 224)
batch_size = 32

# Data augmentation + split
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False

# Top model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("🔧 Initial training (frozen MobileNetV2)...")
model.fit(train_data, validation_data=val_data, epochs=5)

# ✅ Fine-tuning: Unfreeze base model
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
print("🔧 Fine-tuning MobileNetV2 on your data...")
model.fit(train_data, validation_data=val_data, epochs=5)

# Save the final model
os.makedirs("./models", exist_ok=True)
model.save(model_output_path)
print(f"✅ Model saved to {model_output_path}")
