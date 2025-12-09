# Import various tensorflow libraries for machine learning.
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models, optimizers,regularizers, mixed_precision
from sklearn.utils.class_weight import compute_class_weight

"""Since we are about to refine/tune the model, we will enable the usage of the GPU to make the
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

# Define all the paramters that will be used for training the model.
EPOCHS_NUM_PREVIOUS = 25
EPOCHS_NUM_NEXT = 50
LEARNING_RATE = 1e-5  
mixed_precision.set_global_policy('mixed_float16')      # Improve training speed.

# Load the model that will be refined/tuned.
model_name = "model_v1.1.0.0"
model_refined_name = "model_v1.1.0.1"
model = tf.keras.models.load_model(model_name)
base_model = model.get_layer('resnet50')

# Ensures that all BatchNormalization layers are frozen so their moving statistics are not updated during fine-tuning.
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Unfreeze only the last convolutional layers so we can fine-tune them on our dataset.
# This allows the model to adapt high-level features while keeping earlier layers frozen.
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        layer.trainable = True

# Creates a checkpoint that will be used to save the epoch with the best val_accuracy.
checkpoint_val_accuracy = ModelCheckpoint(
    filepath = f"{model_refined_name}_va.h5",     # Path that the  model will be saved at.
    monitor = f"val_accuracy",                    # Uses val_accuracy to determine epoch is the "best" model.
    save_best_only = True,                        # Saves the model according to the monitor.
    mode = "max",                                 # Tells Keras whether a higher or lower value of the monitored metric is better.
                                                  # 'max' means higher is better (e.g., accuracy), 'min' means lower is better (e.g., loss),
                                                  # and 'auto' lets Keras decide automatically. Only saves the model if the metric improves.
    verbose = 1                                   # Turns on messages during training.
)

# Creates a checkpoint that will be used to save the epoch with the best val_loss.
checkpoint_val_loss = tf.keras.callbacks.ModelCheckpoint(
    filepath = f"{model_refined_name}_vl.h5",     # Path that the  model will be saved at.
    monitor = "val_loss",                         # Uses val_loss to determine epoch is the "best" model.
    save_best_only = True,                        # Saves the model according to the monitor.
    mode = "min",                                 # Tells Keras whether a higher or lower value of the monitored metric is better.
                                                  # 'max' means higher is better (e.g., accuracy), 'min' means lower is better (e.g., loss),
                                                  # and 'auto' lets Keras decide automatically. Only saves the model if the metric improves.
    verbose = 1                                   # Turns on messages during training.
)

# List that will be used to save the epochs with the best val_accuracy / val_loss.
callbacks = [checkpoint_val_accuracy, checkpoint_val_loss]

# Recompile the model the Adam optimizer, a suitable loss for multi-class classification,
# and accuracy as the evaluation metric.
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Continues training the model from the previous epoch using the training and validation datasets.
history_more = model.fit(
    train_generator,
    validation_data = validation_generator,
    epochs = EPOCHS_NUM_NEXT,
    initial_epoch = EPOCHS_NUM_PREVIOUS,
    callbacks=callbacks
)

# Save the refined/tuned model.
model.save(model_refined_name)