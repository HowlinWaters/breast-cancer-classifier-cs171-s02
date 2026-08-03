# Binary Breast Tumor Image Classification

A PyTorch project comparing a custom convolutional neural network with a pretrained ResNeXt-50 model for classifying breast tumor images as benign or malignant.
This project was created for educational and research purposes and is not intended for clinical use.

Developed in Google Colab using PyTorch and an NVIDIA A100 GPU. The notebooks install `torchmetrics` directly.
Other dependencies may need to be pip installed if not already: numpy, matplotlib, Pillow, tqdm, scikit-learn, torch, torchvision.

**Breast Cancer Multispectral Image Dataset**: [Link to Dataset](https://www.kaggle.com/datasets/programmer3/breast-cancer-multispectral-image-dataset)

**Final Report**: [Link to Report](https://docs.google.com/document/d/1qbyoqJ2r4uVs0BYRoeMLRv9mYceR3-Bzp6oLKGl4fH4/edit?usp=sharing)

**Presentation Slides**: [Link to Slides](https://docs.google.com/presentation/d/14Z5-3gbHL0PMpIfpeh_W-vtfkQ39ZeFfjLEICDjvL0s/edit?usp=sharing)

## Overview

The project evaluates the tradeoff between a simple CNN designed from scratch and a deeper pretrained model.

The original dataset contains 10,000 balanced breast tumor images:

- 5,000 benign
- 5,000 malignant

To reduce training time, 3,000 images were randomly sampled without replacement:

- 1,500 benign
- 1,500 malignant

The sample was randomly split into:

- 70% training
- 15% validation
- 15% testing

## Preprocessing

Images were prepared using a PyTorch pipeline that:

- Resized images to 224 × 224
- Converted images to tensors
- Normalized RGB channels using ResNeXt-50-compatible values

## Models
Custom CNN
 - Four convolutional layers with 3x3 kernel, ReLU activation, 1 pixel border padding, and 2x2 max pooling
 - A fully connected classification head with 25% dropout to reduce overfitting

ResNeXt-50
 - A pretrained ResNeXt-50 model was used with its feature-extraction layers unchanged and a modified classification head.

## Training

Both models were trained using:

- Adam optimizer
- Cross-entropy loss
- Up to 10 epochs

The following hyperparameters were evaluated:

- Batch sizes: 16 and 32
- Learning rates: 0.1, 0.01, and 0.001

The best configuration for both models used:

- Batch size: 32
- Learning rate: 0.001
- Stopping epoch: 6 for custom model, 5 for ResNeXt-50

## Results

| Model | Test Accuracy | F1 Score |
|---|---:|---:|
| Custom CNN | 90.67% | 90.62% |
| ResNeXt-50 | 96.00% | 96.17% |

ResNeXt-50 achieved the strongest performance, while the simpler custom CNN still exceeded 90% test accuracy.

## Limitations

- The classifier only distinguishes benign from malignant tumors.
- It does not detect whether an image contains a tumor.
- It does not identify cancer subtype or stage.
- The original visible, infrared, and thermal modalities could not be recovered after dataset preprocessing.
