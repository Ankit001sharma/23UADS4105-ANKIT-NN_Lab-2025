Object 5

Program to Train and Evaluate a Convolutional Neural Network using Keras Library to Classify the MNIST Fashion Dataset

Description

This program trains a Convolutional Neural Network (CNN) using the Keras library to classify images from the Fashion MNIST dataset. The dataset consists of 70,000 grayscale images categorized into 10 different classes, each representing an article of clothing or footwear. The program demonstrates the impact of various hyperparameters, including:

Filter Size: The number of filters in the convolutional layers, impacting feature extraction.

Kernel Size: The size of the convolutional kernel used for feature detection.

Regularization: The use of kernel regularization (L2) to prevent overfitting.

Batch Size: The number of samples processed before updating model weights.

Optimization Algorithm: Different optimizers like Adam, SGD, and RMSprop affecting learning efficiency.

Features

Loads and preprocesses the Fashion MNIST dataset (normalization and reshaping).

Defines a CNN with adjustable filter size, kernel size, regularization, and optimizer.

Trains the model on the dataset with user-defined hyperparameters.

Evaluates model performance on test data.

Plots training loss and accuracy curves.

Generates a confusion matrix for classification performance visualization.

Model Architecture

The following table summarizes the layers used in the CNN model:

Layer

Type

Output Shape

Parameters

1

Conv2D

(26, 26, 32)

320

2

MaxPooling2D

(13, 13, 32)

0

3

Conv2D

(11, 11, 64)

18,496

4

MaxPooling2D

(5, 5, 64)

0

5

Flatten

(1600)

0

6

Dense

(128)

204,928

7

Dense

(10)

1,290

Requirements

Ensure you have the following dependencies installed:

pip install tensorflow keras matplotlib numpy seaborn scikit-learn

Usage

Run the script to train the CNN and analyze the performance:

python train_cnn.py

Modify the hyperparameters in the script to experiment with different configurations and observe their effect on accuracy.

Results

The script outputs training and validation accuracy/loss.

Plots training curves to analyze model performance over epochs.

Displays a confusion matrix to evaluate classification accuracy.

Conclusion

This program showcases the impact of CNN hyperparameters on classification accuracy. By tuning filter sizes, kernel size, regularization, batch size, and optimizers, users can optimize CNN performance for fashion item classification.

