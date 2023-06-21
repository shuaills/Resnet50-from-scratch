# MyResnet50

Welcome to MyResnet50! This project is a detailed, self-implemented version of the ResNet50 model for learning and inference acceleration. The goal of this project is to provide a clear understanding of the inner workings of the ResNet50 model by forgoing the use of high-level libraries. 

Two versions of the model are planned: a Python version for quick verification of the model's correctness, and a C++ version for computational performance.

## Components

The project consists of several key components:

1. **Inference Code (`main_inference.py`)**: This file demonstrates how to use various layers (convolution, batch normalization, ReLU, max pooling, bottleneck, average pooling, and fully connected) for inference. It also shows how to compute the timing for each step and how to find the predicted category from the model's final output.

2. **Data Preparation (`data_preparation.py`)**: This file contains functions for loading and handling model weights, as well as getting a list of images to predict and preprocessing those images.

## Usage

To use the model, you'll need to first preprocess your images with the `preprocess` function in `data_preparation.py`. Then, you can pass your preprocessed images through the various layers as demonstrated in `main_inference.py`.

## Examples

Examples of usage will be added as the project progresses.

## Contributing

Contributions to this project are welcome! Please fork the project and create a pull request with your changes.

## License

This project is licensed under the MIT License.

