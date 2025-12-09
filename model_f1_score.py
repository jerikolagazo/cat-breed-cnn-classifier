# Import basic data manipulation, analysis, and visualization libraries.
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import numpy as np

# Import basic data manipulation, analysis, and visualization libraries.
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import numpy as np

# Import various tensorflow libraries for machine learning.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models, optimizers, mixed_precision
from sklearn.metrics import f1_score

# Import various operating system/file interaction libraries.
import os

# Define all the paramters that will be used for creating the generators.
TEST_DIR = 'content/catbreedsrefined-7k/test'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Create a data generator for testing images.
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load and prepare the testing images from the folder.
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

# Load the trained model.
model = keras.models.load_model("model_v1.1.0.1_va.h5")
# Generate class probabilities for all test images.
probabilities = model.predict(test_generator)
# Get the predicted class index for each image (highest probability).
predictions = np.argmax(probabilities, axis=1)
# Get the true labels
y_true = test_generator.classes

# Compute F1 scores
f1_macro = f1_score(y_true, predictions, average="macro")
f1_weighted = f1_score(y_true, predictions, average="weighted")

print("F1 Score (Macro):", f1_macro)
print("F1 Score (Weighted):", f1_weighted)