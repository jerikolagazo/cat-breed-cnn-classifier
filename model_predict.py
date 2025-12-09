# Import various tensorflow libraries for machine learning.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models, optimizers, mixed_precision
from sklearn.metrics import f1_score

# Import basic data manipulation, analysis, and visualization libraries.
import numpy as np

# Dictionary, with the cat breed as the key and the ID as the value.
BREED_ID = {'Abyssinian': 0, 'American Bobtail': 1, 'American Curl': 2, 'American Shorthair': 3, 'Bengal': 4, 'Birman': 5, 'Bombay': 6, 'British Shorthair': 7, 'Egyptian Mau': 8, 'Exotic Shorthair': 9, 'Maine Coon': 10, 'Manx': 11, 'Norwegian Forest': 12, 'Persian': 13, 'Ragdoll': 14, 'Russian Blue': 15, 'Scottish Fold': 16, 'Siamese': 17, 'Sphynx': 18, 'Turkish Angora': 19}
# Inverts the dictionary so that the ID is the key and the breed is the value. This makes the data easier to work with.
ID_BREED = {v: k for k, v in BREED_ID.items()}

# Load the refined/tuned model.
model = keras.models.load_model("model_v1.1.0.1_va.h5")

# Loads the image as an PIL (so that  general image processing tasks can be performed).
# Moreover, resizes it in order to be evaluated.
IMG_PATH = "sample.jpg"
IMG = image.load_img(IMG_PATH, target_size=(224, 224))

# Convert the PIL into a NumPy array (height x width x channels).
IMG_ARRAY = image.img_to_array(IMG)
# Add an extra dimension at axis 0 to represent the batch size (model expects a batch of images)
IMG_ARRAY = np.expand_dims(IMG_ARRAY, axis=0)

# Preprocess the image array to match the format expected by ResNet50.
# This includes scaling pixel values and other model-specific adjustments.
IMG_ARRAY = preprocess_input(IMG_ARRAY)

# Run the model to predict what class the image belongs to.
pred = model.predict(IMG_ARRAY)

# Retrieve the breed ID and breed name.
BREED_ID = np.argmax(pred)
BREED_NAME = ID_BREED[BREED_ID]

print(f"IMAGE NAME: {IMG_PATH}")
print(f"BREED ID: {BREED_ID}")
print(f"BREED NAME: {BREED_NAME}")