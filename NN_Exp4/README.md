
# Object 4: This program evaluates the performance of a three-layer neural network with variations in activation functions, hidden layer sizes, learning rates, batch sizes, and the number of epochs. The goal is to analyze how these hyperparameters impact training efficiency and accuracy on the MNIST dataset.

## Description

The model consists of a single hidden layer with varying sizes (256, 128, or 64 neurons) and different activation functions (ReLU, Sigmoid, and Tanh). The output layer consists of 10 neurons representing digit classes (0-9). Gradient Descent Optimizer is used for training, and the loss is computed using softmax cross-entropy. Training time, accuracy, and loss curves are plotted for performance analysis.

## Code Explanation

### 1. Importing Dependencies

The required libraries (TensorFlow, NumPy, Matplotlib, Pandas, and Seaborn) are imported. `disable_eager_execution()` ensures TensorFlow runs in graph execution mode.

### 2. Loading and Preprocessing the MNIST Dataset

- The MNIST dataset is loaded and normalized.
- Images are reshaped into 1D arrays of size 784.
- Labels are converted to one-hot encoding.

### 3. Defining Model Parameters

- Weights are initialized using `tf.random.normal`.
- Different hidden layer sizes are explored: `[256, 128, 64]`.
- Three activation functions are tested: ReLU, Sigmoid, and Tanh.

### 4. Forward Propagation

A function `forward_propagation()` constructs a network with one hidden layer, allowing different configurations of activation functions and neuron counts.

### 5. Defining the Loss Function and Optimizer

- The loss function used is `softmax_cross_entropy_with_logits`.
- The gradient descent optimizer minimizes the loss with a learning rate of 0.1.

### 6. Defining Accuracy Metric

- Predictions are computed using `tf.argmax()`.
- Accuracy is calculated by comparing predictions with actual labels.

### 7. Training the Model

- The model is trained for 50 epochs using a batch size of 10.
- Loss and accuracy are tracked for each epoch.
- Different configurations are tested to compare performance.

### 8. Evaluation and Visualization

- Final test accuracy is computed for each model.
- Confusion matrices are plotted using Seaborn.
- Loss and accuracy curves are visualized to understand training trends.

## Comments and Observations

- ReLU generally performs better due to its ability to mitigate the vanishing gradient problem.
- In this model the choice of activation function affects the training speed & accuracy .
- Smaller batch sizes can improve generalization but slow down training.
- Increasing the number of neurons in hidden layers may enhance accuracy but increases computational cost.

## Requirements

- TensorFlow 2.x (running in compatibility mode for TF1.x syntax)
- NumPy, Matplotlib, Pandas, Seaborn

## Expected Output

- Training loss and accuracy for each configuration.
- Final test accuracy for each model.
- Confusion matrices for different configurations.

## Future Improvements

- Experiment with deeper architectures.
- Use dropout and batch normalization to improve generalization.
- Implement adaptive learning rate techniques.



