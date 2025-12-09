# Import various tensorflow libraries for machine learning.
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models, optimizers,regularizers, mixed_precision
from sklearn.utils.class_weight import compute_class_weight

"""Since we are about to build the model and train it, we will enable the usage of the GPU to make the
the process quicker."""

# First we will check if there is any physical GPU detected.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Afterwards, attempt to force enable memory growth so that TensorFlow does not preallocate
    # all GPU memory. 
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"Could not set memory growth: {e}")
else:
    print("No GPUs detected, training will run on CPU.")

"""Begin constructing the training and validations generators using ImageDataGenerator.
This is so that the data can be preprocessed and agumentated later on."""

# Define all the paramters that will be used for creating the generators.
TRAIN_DIR = 'content/catbreedsrefined-7k/train'
VAL_DIR = 'content/catbreedsrefined-7k/val'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED_SIZE = 42

# Create a data generator for training images.
train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input,  # Preprocess images to match ResNet50 training format.
    rotation_range = 10,                        # Small rotations.
    width_shift_range = 0.05,                   # Horizontal translation.
    height_shift_range = 0.05,                  # Vertical translation.
    shear_range= 0.06,                          # Shearing.
    zoom_range = 0.06,                          # Zoom in/out.
    horizontal_flip = True,                     # Flip horizontally.
    brightness_range = [0.8, 1.2],              # Random brightness.
    fill_mode = 'nearest'                       # Filling missing pixels after transforms.
)

# Create a data generator for validation images.
validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input     # Preprocess images to match ResNet50 training format.
)

# Load and prepare the training images from the folder.
train_generator = train_datagen.flow_from_directory(
  TRAIN_DIR,                  # Directory containing image folders.
  target_size=IMAGE_SIZE,     # Resize all images.
  batch_size=BATCH_SIZE,      # Number of images per training batch.
  class_mode='sparse',        # Labels are integer class indices.
  seed=SEED_SIZE,             # Determines the split and shuffle (same for every epoch).
  shuffle=True                # Shuffle training images
)

# Load and prepare the validation images from the folder.
validation_generator = validation_datagen.flow_from_directory(
  VAL_DIR,                    # Directory containing image folders.
  target_size=IMAGE_SIZE,     # Resize all images.
  batch_size=BATCH_SIZE,      # Number of images per training batch.
  class_mode='sparse',        # Labels are integer class indices.
  seed=SEED_SIZE,             # Determines the split and shuffle (same for every epoch).
  shuffle=False               # Does NOT shuffle training images for consistent evaluation/predictions.
)

# Define all the paramters that will be used for building and training the model.
CLASSES_NUM = train_generator.num_classes               # The number of classes retrieved from the generator.
EPOCHS_NUM = 25                                         # The number of iterations that the model will run.
LEARNING_RATE = 1e-4                                    # Determines how much the model adjust paramters/weights are updated each step.
mixed_precision.set_global_policy('mixed_float16')      # Improve training speed.

# Load the ResNet50 base model.
base_model = ResNet50(
    weights='imagenet',                                 # Load the pretrained imagenet weights.
    include_top=False,                                  # Excludes the final classification layer.
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)       # Tells the model the shape and size of the input.
)

# Freezes all layers.
for layer in base_model.layers:
    layer.trainable = False

# Define the model input.
inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = base_model(inputs, training=True)

x = layers.GlobalAveragePooling2D()(x)  # Convert feature maps to a single vector per feature (reduces spatial dimensions).
x = layers.BatchNormalization()(x)      # Normalize activations to improve training stability and speed.
x = layers.Dropout(0.3)(x)              # Randomly drop 30% of units to reduce overfitting.

# Add the final classification layer with softmax activation.
# This converts the raw output scores into probabilities that sum to 1.
# The class with the highest probability is chosen as the model's prediction.
outputs = layers.Dense(
    CLASSES_NUM,
    activation='softmax',
    kernel_regularizer=regularizers.l2(1e-4)
    )(x)

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
model.save(f"model_v1.1.0.0")