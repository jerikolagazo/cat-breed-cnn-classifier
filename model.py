# Import basic data manipulation, analysis, and visualization libraries.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import various tensorflow libraries for machine learning.
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models, optimizers, mixed_precision

# Import various operating system/file interaction libraries.
from urllib.request import urlretrieve
from PIL import Image
import zipfile
import os

# Define the variables that will be used for downloading, unzipping, and managing the dataset.
URL = "https://www.kaggle.com/api/v1/datasets/download/ma7555/cat-breeds-dataset"
ZIP_PATH = "content/cat-breeds-dataset.zip"
EXTRACTION_PATH = 'content/cat-breeds-dataset'

# Verifies if the dataset exist, if not, it downloads it. 
if (os.path.exists(f"content/cat-breeds-dataset/images") == False) or (os.path.exists(f"content/cat-breeds-dataset/images") == False):
    # Download the dataset using the URL and PATH,
    urlretrieve(URL, ZIP_PATH)
    # Unzip the downloaded dataset so that we are able to work the dataset.
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACTION_PATH)
    # Delete cat-breeds-dataset.zip now that we have the dataset.
    os.remove(ZIP_PATH)

# Defines variables that will be used to verifying the data.
invalid_files_removed = 0
corrupted_files_removed = 0
verify_files = False

# Goes through every folder and every image in each folder to remove corrupted images or
# invald files.
if verify_files:
    for root, _, files in os.walk('content/cat-breeds-dataset/images'):
        for file in files:
            # Verifies if the file is an image.
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                filepath = os.path.join(root, file)
                # Attempts to load the image to verify validity.
                try:
                    with Image.open(filepath) as img:
                        img.load()
                # Removes the file if it is corrupted.
                except Exception as e:
                    os.remove(filepath)
                    corrupted_files_removed += 1
            # Removes the file if it is not a .jpg, .jpeg, or .png.    
            else:
                os.remove(filepath)
                invalid_files_removed += 1

    print(f"Invalid files removed: {invalid_files_removed}")
    print(f"Corrupted files removed: {corrupted_files_removed}")


# Since we are about to build the model and train it, we will enable the usage of the GPU to make the
# the process much more quicker.

# First we will check if there is any physical GPU detected.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Afterwards, attempt to force enable memory growth to prevent so that TensorFlow does not preallocate
    # all GPU memory. 
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"Could not set memory growth: {e}")
else:
    print("No GPUs detected, training will run on CPU.")

# Begin constructing the the training and validations generators using ImageDataGenerator.
# This is so that the data can be preprocessed and agumentated later on.

# Define all the paramters that will be used for creating the generators.
TRAINING_DIR = 'content/cat-breeds-dataset/images'
IMAGE_SIZE = (200, 200)
BATCH_SIZE = 32
SEED_SIZE = 42

# Create a data generator for training images.
train_datagen = ImageDataGenerator(
  rescale=1./255,
  validation_split=.2
)

# Create a data generator for validation images.
validation_datagen = ImageDataGenerator(
  rescale=1./255,
  validation_split=.2
)

# Load and prepare the training images from the folder.
train_generator = train_datagen.flow_from_directory(
  TRAINING_DIR,               # Directory containing image folders
  target_size=IMAGE_SIZE,     # Resize all images to 100x100
  subset="training",          # Use the training split
  batch_size=BATCH_SIZE,      # Number of images per training batch
  class_mode='sparse',        # Labels are integer class indices
  seed=SEED_SIZE,             # Determines the split and shuffle (same for every epoch).
  shuffle=True                # Shuffle training images
)

# Load and prepare the validation images from the folder.
validation_generator = validation_datagen.flow_from_directory(
  TRAINING_DIR,               # Directory containing image folders.
  target_size=IMAGE_SIZE,     # Resize all images to 100x100.
  subset="validation",        # Use the validation split.
  batch_size=BATCH_SIZE,      # Number of images per training batch.
  class_mode='sparse',        # Labels are integer class indices.
  seed=SEED_SIZE,             # Determines the split and shuffle (same for every epoch).
)

# Define all the paramters that will be used for building and training the model.
EPOCHS_NUM = 15
CLASSES_NUM = train_generator.num_classes
LEARNING_RATE = 1e-4
# Sets mixed_precision to mixed_float16 to improve training speed.
mixed_precision.set_global_policy('mixed_float16')

# Load the ResNet50 base model pretrained on ImageNet.
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

# Define the model input.
inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = base_model(inputs, training=True)
# Add global average pooling and dropout for regularization.
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.1)(x)

# Add the final classification layer with softmax activation.
outputs = layers.Dense(CLASSES_NUM, activation='softmax')(x)

# Construct the final model using the input and output layers.
model = models.Model(inputs, outputs)

# Compile the model with the Adam optimizer, a suitable loss for multi-class classification,
# and accuracy as the evaluation metric.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model using the training and validation datasets.
history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS_NUM
)

# Save the trained model for evaulation or further training.
model.save(f"model_learning_rate_{LEARNING_RATE}")