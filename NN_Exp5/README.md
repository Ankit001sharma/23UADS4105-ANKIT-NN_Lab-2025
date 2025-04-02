# Object 5

## Program to Train and Evaluate a Convolutional Neural Network using Keras Library to Classify the MNIST Fashion Dataset

### Description
This program trains a Convolutional Neural Network (CNN) using the Keras library to classify images from the Fashion MNIST dataset. The dataset consists of 70,000 grayscale images categorized into 10 different classes, each representing an article of clothing or footwear. The program demonstrates the impact of various hyperparameters, including:

- **Filter Size**: The number of filters in the convolutional layers, impacting feature extraction.
- **Kernel Size**: The size of the convolutional kernel used for feature detection.
- **Regularization**: The use of kernel regularization (L2) to prevent overfitting.
- **Batch Size**: The number of samples processed before updating model weights.
- **Optimization Algorithm**: Different optimizers like Adam, SGD, and RMSprop affecting learning efficiency.

### Features
- Loads and preprocesses the Fashion MNIST dataset (normalization and reshaping).
- Defines a CNN with adjustable filter size, kernel size, regularization, and optimizer.
- Trains the model on the dataset with user-defined hyperparameters.
- Evaluates model performance on test data.
- Plots training loss and accuracy curves.
- Generates a confusion matrix for classification performance visualization.

### Requirements
Ensure you have the following dependencies installed:

```bash
pip install tensorflow keras matplotlib numpy seaborn scikit-learn
```

### Usage
Run the script to train the CNN and analyze the performance:

```bash
python train_cnn.py
```

Modify the hyperparameters in the script to experiment with different configurations and observe their effect on accuracy.

### Results
- The script outputs training and validation accuracy/loss.
- Plots training curves to analyze model performance over epochs.
- Displays a confusion matrix to evaluate classification accuracy.

### Conclusion
This program showcases the impact of CNN hyperparameters on classification accuracy. By tuning filter sizes, kernel size, regularization, batch size, and optimizers, users can optimize CNN performance for fashion item classification.

