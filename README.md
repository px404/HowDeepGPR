# HowDeepGPR
Model to estimate the location of items using GPR scans.

This repository contains the implementation of an ML pipeline to estimate the location of buried objects from images. The model is trained to estimate a location value for each image, which is later scaled back to the original coordinate system.

---

## Pipeline Overview

The pipeline flows:

### 1. Data Preparation
- **Structure**:
  - The dataset, obtained from Kaggle, consists of images and their corresponding location values.
  - It is divided into training, validation, and test sets.
  
- **Preprocessing**:
  - The `location` values are scaled using a factor of `128 / 9.2` to work within a larger numerical range to have floating point precision and simplify training.
  - Paths to image files are added to the dataset for easy loading during training.

- **Splitting**:
  - The dataset is split into training and validation sets using an 80/20 ratio, with random shuffling.

---

### 2. Data Augmentation
- **Why though**: applied to increase the diversity of the training data, reducing overfitting.
- **Augmentations**:
  - Horizontal shifts and vertical shifts to simulate slight variations in object positions.
  - Horizontal flips (randomly) to add variability.
  - Normalization of pixel values to `[0, 1]` - better convergence during training.
  Converted images to PyTorch tensors for model input.

---

### 3. Neural Network Architecture
The model is designed to learn from the augmented images and predict the location of the object.

#### Backbone Network
- **Convolutional Layers**:
  - Multiple `Conv2d` layers extract spatial features from the input.
  - Each layer is followed by Batch Normalization (`BatchNorm2d`) to stabilize training.
  - `ReLU` activations add non-linearity - for the model to learn complex patterns.
  
- **MaxPooling**:
  - Gradually reduces the spatial dimensions of the feature maps to focus on meaningful features and discard irrelevant details.

- **Output of Backbone**:
  - Image is compressed to a representation of size `(batch_size, 32, 5, 2)`.

#### Head
- **Fully Connected Layers**:
  - Extracted features flattened then passed through a fully connected (`Linear`) layer.
  - This reduces the dimensionality from 512 features to 128, and another layer to output a single normalized (acc to our coordinate normalization) value.

---

### 4. Loss Function and Training
- **Loss Function**: Mean Squared Error (MSE) used to compare predicted and ground truth locations.
- **Optimization**:
  - The model is trained with a batch size of 32 and uses random shuffling for the training data.
  - Validation set used (monitor performance, avoid overfitting).

---

## Repository Structure
```
.
├── data/                          # Dataset files
│   ├── location.csv               # Training locations
│   ├── sample_submission.csv      # Test file for predictions
│   ├── train_images/              # Training images
│   └── test_images/               # Test images
├── models/                        # Trained models
├── scripts/                       # Training and evaluation scripts
├── utils/                         # Utility functions
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

---

## How to Run the Code

### 1. Prerequisites
- Python 3.10
- Preferably make an environment.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Training the Model
- Run the training script in the notebook.

### 3. Evaluating with TTA
- Run the evaluation script.

---

## Considerations
- This project is essentially using CNNs(convolutional neural networks) to estimate locations of objects from images using a combination of preprocessing, augmentations, and TTA.
- Since we have labelled data training is easy, but this may not be the case with an unlabelled dataset. For this other implementations are necessary, one would be auto labelling of the dataset which I am currently taking a look into.
- We achieved a lower error inspite of using a similar pipeline. The original model was trained on an Nvidia P100 and completed training in 615.9s, while we used an A100.

---

#### Dataset
This project uses data from the Kaggle competition **[Find locations of buried objects from GPR data](https://kaggle.com/competitions/estimating-locations-of-buried-objects)**, provided by **Hoon**.  

If you use this dataset in your work, please cite it as follows:  
```
@misc{estimating-locations-of-buried-objects,
    author = {Hoon},
    title = {Find locations of buried objects from GPR data},
    year = {2023},
    howpublished = {\url{https://kaggle.com/competitions/estimating-locations-of-buried-objects}},
    note = {Kaggle}
}
```
