# Cat-Dog Classifier using Convolutional Neural Networks (CNN)

This repository contains a project for classifying images of cats and dogs using Convolutional Neural Networks (CNN). The model is implemented in Python using TensorFlow and Keras.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Cat-Dog Classifier is a deep learning project aimed at building a model that can distinguish between images of cats and dogs. This project uses a CNN to achieve high accuracy in classification tasks.

## Dataset

The dataset used for training and evaluation is the [Microsoft Cats and Dogs dataset](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765). It consists of 25,000 images of cats and dogs, with 12,500 images for each class. The dataset is available in the `Dataset/PetImages` directory.

## Model Architecture

The model is a Convolutional Neural Network with the following architecture:

- Input layer
- Convolutional layers with ReLU activation
- Max Pooling layers
- Fully connected (Dense) layers
- Dropout layers for regularization
- Output layer with Sigmoid activation for binary classification

## Training

The model is trained using the following parameters:

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Metrics: Accuracy
- Number of Epochs: 5
- Batch Size: 32

The dataset is split into training and validation sets with an 80-20 ratio. Data augmentation is applied to the training set to improve the model's generalization.

## Evaluation

The model's performance is evaluated using accuracy and loss metrics on the validation set. The evaluation results and training history are visualized using matplotlib.

## Usage

To use this project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/Mukund604/Cat_Dog_Classifier_CNN.git
    cd Cat_Dog_Classifier_CNN
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the dataset from [Microsoft](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765) and extract it into the `Dataset/PetImages` directory.

4. Train the model:
    ```sh
    python CNN.ipynb
    ```

5. Evaluate the model:
    ```sh
    python evaluate.py
    ```

6. Classify new images:
    ```sh
    python predict.py --image_path path_to_your_image.jpg
    ```

## Results

The model achieves an accuracy of approximately 84% on the validation set after training for 5 epochs. Example classification results and training history plots are provided in the `Sample-Prediction-Images` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
