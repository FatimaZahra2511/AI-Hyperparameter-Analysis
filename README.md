# AI-Hyperparameter-Analysis

## Overview
This project involves building a Convolutional Neural Network (CNN) to classify images using the CIFAR-10 dataset. The model is trained with different hyperparameters and network architectures to evaluate their impact on classification performance.

## Project Objectives
1. Build a CNN model by defining various layers and connections.
2. Train the model using the CIFAR-10 dataset from Kaggle.
3. Compute accuracy during training and evaluate final performance.
4. Experiment with different hyperparameters and network architectures.
5. Plot the performance of the model based on the chosen hyperparameter.
6. Select the best-performing model and use it to classify 10 images.
7. Present results using a confusion matrix.

## Steps to Follow
### Step 1: Review Resources
- Download the source code from the video and test it on Google Colab or a local machine.

### Step 2: Set Up the Environment
Install the necessary dependencies:
```bash
pip install tensorflow keras numpy matplotlib seaborn scikit-learn
```

### Step 3: Download and Prepare CIFAR-10 Dataset
- Download CIFAR-10 from Kaggle: [CIFAR-10 Dataset](https://www.kaggle.com/c/cifar-10/)
- Load and preprocess the dataset in Python.

### Step 4: Build and Train a CNN Model
- Define a Convolutional Neural Network (CNN) architecture.
- Train the model using CIFAR-10 dataset.
- Compute accuracy during training and evaluate performance.

### Step 5: Experiment with Hyperparameters
- Try different values for:
  - Number of filters in convolutional layers
  - Batch size
  - Learning rate
  - Dropout rate
- Plot performance against hyperparameters.

### Step 6: Test Best Model on New Images
- Choose the model with the highest accuracy.
- Infer the class of 10 selected images.
- Present the results in a confusion matrix.

## Project Structure
```
AI-Hyperparameter-Analysis/
├── Accuracy1.ipynb        # Accuracy analysis 1
├── Accuracy2.ipynb        # Accuracy analysis 2
├── Accuracy3.ipynb        # Accuracy analysis 3
├── BatchSize.ipynb        # Batch size impact
├── DropoutRate.ipynb      # Dropout rate tuning
├── Filter.ipynb           # Filter optimization
├── LearningRate.ipynb     # Learning rate adjustment
├── Matrix.ipynb           # Matrix computations
├── LICENSE                # License file
└── README.md              # Documentation
```

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

