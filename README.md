# CV Project: Leaf Image Preprocessing

This project focuses on the preprocessing of leaf images for computer vision tasks. The `preprocess.py` script is designed to clean, enhance, and augment a dataset of leaf images to prepare them for training a machine learning model.

## Preprocessing Pipeline

The script implements the following preprocessing steps:

1.  **Image Loading and Resizing**:
    *   Loads images from the specified dataset directory.
    *   Resizes all images to a uniform dimension of 128x128 pixels.
    *   Converts images from BGR to RGB color space.

2.  **Normalization and Blurring**:
    *   Normalizes pixel values to a range of [0, 1].
    *   Applies a Gaussian blur with a 5x5 kernel to reduce noise.

3.  **Contrast Enhancement**:
    *   Uses Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the local contrast of the images.

4.  **Background Removal**:
    *   Converts images to grayscale to create a binary mask.
    *   Applies Otsu's thresholding to segment the leaf from the background.
    *   Uses a median blur to clean up the mask.
    *   Removes the background by applying the mask.

5.  **Data Splitting**:
    *   Splits the dataset into training, validation, and testing sets with a 70/15/15 distribution.

6.  **Data Augmentation**:
    *   Utilizes `ImageDataGenerator` from Keras and `albumentations` to apply a variety of augmentations to the training data, including:
        *   Rotation
        *   Width and height shifts
        *   Zoom
        *   Horizontal and vertical flips
        *   Brightness/contrast adjustments
        *   Motion blur

7.  **Additional Techniques**:
    *   The script also includes functions for other image processing techniques like sharpening, edge enhancement, and morphological transformations, with examples plotted for visualization.

## Usage

1.  **Setup**:
    *   Install the required Python libraries:
        ```bash
        pip install opencv-python-headless matplotlib scikit-learn albumentations tqdm tensorflow
        ```

2.  **Configure Paths**:
    *   Modify the `DATASET_DIR` and `OUTPUT_DIR` variables in `preprocess.py` to point to your image dataset and desired output location.

3.  **Run the script**:
    ```bash
    python preprocess.py
    ```

The script will process the images and save the preprocessed versions to the specified output directory. It will also display a sample of the original image alongside several processed versions to visualize the effects of the different techniques.
