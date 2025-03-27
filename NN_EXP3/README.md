
## Object 3: Program to implement a three-layer neural network using TensorFlow library (only, no Keras) to classify MNIST handwritten digits dataset. Demonstrate the implementation of feed-forward and back-propagation approaches.

### Description

This project implements a simple feed-forward neural network using TensorFlow to classify handwritten digits from the MNIST dataset. The model consists of two hidden layers and uses the sigmoid activation function. The Adam optimizer is used for training, and accuracy is measured on both the training and test sets.

### Code Explanation

#### 1. Importing Dependencies

The required libraries, including TensorFlow and NumPy, are imported. `disable_eager_execution()` is used to ensure TensorFlow runs in graph execution mode.

#### 2. Loading and Preprocessing the MNIST Dataset

- The MNIST dataset is loaded and normalized by scaling pixel values between 0 and 1.
- The images are reshaped into 1D arrays of size 784.
- Labels are converted to one-hot encoding.

#### 3. Defining Model Hyperparameters

Key hyperparameters for the neural network include:

- `input_size = 784` (28x28 images flattened)
- `hidden1_size = 128`
- `hidden2_size = 64`
- `output_size = 10` (10 classes for digits 0-9)
- `learning_rate = 0.01`
- `batch_size = 100`
- `epochs = 20`

#### 4. Building the Neural Network

- Placeholders are defined for input (`X`) and output (`y`).
- Weights and biases are initialized using `tf.Variable`.
- The neural network consists of:
  - Layer 1: 128 neurons with sigmoid activation
  - Layer 2: 64 neurons with sigmoid activation
  - Output layer: 10 neurons (logits)

#### 5. Defining the Loss Function and Optimizer

- The loss function used is `softmax_cross_entropy_with_logits`.
- The Adam optimizer is used to minimize the loss.

#### 6. Defining Accuracy Metric

- Predictions are computed using `tf.argmax()`.
- Accuracy is calculated by comparing predictions with actual labels.

#### 7. Training the Model

- A TensorFlow session is created.
- The model is trained for 20 epochs using mini-batches of size 100.
- Loss and accuracy are printed at each epoch.
- Final training and test accuracy are displayed after training.

### Comments and Observations

- The model uses sigmoid activation, which can cause issues such as vanishing gradients. Using ReLU instead may improve performance.
- The dataset is small, so a more complex model is not required.
- Batch size is set to 100, which balances training stability and performance.
- The model does not use dropout or batch normalization, which could enhance generalization.

### Requirements

- TensorFlow 2.x (running in compatibility mode for TF1.x syntax)
- NumPy

### Running the Code

```bash
python mnist_nn.py
```

### Expected Output

- Training loss and accuracy at each epoch.
- Final train and test accuracy values.

### Future Improvements

- Replace sigmoid with ReLU for better training stability.
- Use dropout to prevent overfitting.
- Experiment with more advanced optimizers like RMSProp or learning rate scheduling.

