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

# Import various operating system/file interaction libraries.
import os

# Define all the paramters that will be used for creating the generators.
TRAINING_DIR = 'content/cat-breeds-dataset/images'
IMAGE_SIZE = (200, 200)
BATCH_SIZE = 32
SEED_SIZE = 42

# Create a data generator for testing images.
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare the testing images from the folder.
test_generator = test_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

# Load the trained model.
model = keras.models.load_model("model_v1.0.1")
# Generate class probabilities for all test images.
probabilities = model.predict(test_generator)
# Get the predicted class index for each image (highest probability).
predictions = np.argmax(probabilities, axis=1)
# Get the highest probability value for each prediction (confidence score).
max_probs = np.max(probabilities, axis=1)

# Extract only the file names from the test image paths.
filenames = [fname.split('/')[-1] for fname in test_generator.filenames]
# Get the true class labels for the test images
answers = test_generator.classes

# Create a DataFrame to store the test results
results_df = pd.DataFrame({
    'filename': filenames,
    'answer': answers,
    'prediction': predictions,
    'max_probability': max_probs
})

# Calculate a binary score for each prediction (1 = correct, 0 = incorrect).
results_df['score'] = (results_df['prediction'] == results_df['answer']).astype(int)

# Compute overall accuracy metrics.
total = len(results_df)                     # Total number of test samples.
total_correct = results_df['score'].sum()   # Number of correctly predicted samples.

# Display overall accuracy results
print(f"Total Correct: {total_correct} / {total}")
print(f"Accuracy: {total_correct / total:.4f}")

# Save detailed test results to a CSV file
results_df.to_csv('results.csv', index=False)

# Calculate per-class accuracy.
accuracy_by_class = results_df.groupby('answer')['score'].mean().reset_index()
accuracy_by_class.rename(columns={'answer': 'class_id'}, inplace=True)

# Create plot that displays the accuracy by class.
accuracy_by_class_plot = alt.Chart(accuracy_by_class).mark_bar().encode(
    x = alt.X('class_id:O', title='Class'),
    y = alt.Y('score:Q', title='Accuracy'),
    color = alt.Color('score:Q', title='Score', scale=alt.Scale(scheme='blues'))
).properties(
    title='Accuracy by Class',
    width=600,
    height=400
)

# Saves the plot.
accuracy_by_class_plot.save('accuracy_by_class_plot.png')