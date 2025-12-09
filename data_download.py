# Import various operating system/file interaction libraries.
from urllib.request import urlretrieve
from PIL import Image
import zipfile
import os

# Define the variables that will be used for downloading, unzipping, and managing the dataset.
URL = "https://www.kaggle.com/api/v1/datasets/download/doctrinek/catbreedsrefined-7k"
ZIP_PATH = "content/catbreedsrefined-7k.zip"
EXTRACTION_PATH = 'content/catbreedsrefined-7k'

# Verifies if the dataset exist, if not, it downloads it. 
if (os.path.exists(f"content/catbreedsrefined-7k")) == False:
    # Download the dataset using the URL and PATH.
    urlretrieve(URL, ZIP_PATH)
    # Unzip the downloaded dataset so that we are able to work the dataset.
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACTION_PATH)
    # Delete the .zip now that we have the dataset.
    os.remove(ZIP_PATH)
    # Renames the extracted folder so that it is easier to work with.
    os.rename(f"{EXTRACTION_PATH}/CatBreedsRefined-v2", f"{EXTRACTION_PATH}/images")

# Defines variables that will be used to verifying the data.
invalid_files_removed = 0
corrupted_files_removed = 0

# Goes through every folder and every image in each folder to remove corrupted images or
# invald files (files that are not images).
for root, _, files in os.walk('content/catbreedsrefined-7k/images'):
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
        # Removes the file if it is not an image.
        else:
            os.remove(filepath)
            invalid_files_removed += 1
print(f"Removed Invalid Files: {invalid_files_removed}")
print(f"Remove Corrupted Files: {corrupted_files_removed}")