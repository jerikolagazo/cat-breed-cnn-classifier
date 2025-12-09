# Imports the necesary libraries/packages that will be used for splitting the datasaet. 
import splitfolders
import shutil
import os

# List that will be used to simplify checking if all three directories exist.
REQUIRED_DIRS = [
    'content/catbreedsrefined-7k/train',
    'content/catbreedsrefined-7k/val',
    'content/catbreedsrefined-7k/test',
]

INPUT_DIR = 'content/catbreedsrefined-7k/images' # Directory the will be split.
OUTPUT_DIR = 'content/catbreedsrefined-7k' # Directory that the new folders will created.
SEED = 1337 # Determines the randomness of the split and ensures the split can be reproduced.
RATIO = (.7, .15, .15) # Determines the ratio of the train, val, and test split (in that order).

# Checks if the train, val, and test folders all exist, if not, executes the split.
if not all(os.path.exists(d) for d in REQUIRED_DIRS):
    splitfolders.ratio(
        INPUT_DIR,
        OUTPUT_DIR,
        seed = SEED,
        ratio = RATIO
    )

# Removes the folder that was split if it still exist as it is no longer
if os.path.exists('content/catbreedsrefined-7k/images'):
    shutil.rmtree('content/catbreedsrefined-7k/images')