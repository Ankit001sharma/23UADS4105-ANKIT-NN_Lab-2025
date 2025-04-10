# Object 6: WAP to Train and Evaluate a Recurrent Neural Network using PyTorch Library to Predict the Next Value in a Sample Time Series Dataset

## Objective
### This project demonstrates how to train and evaluate a Recurrent Neural Network (RNN) using the PyTorch library. The model is trained on a real-world time series dataset of daily minimum temperatures in Melbourne to predict the next temperature value in the sequence.

## Technologies & Libraries Used
### PyTorch (torch): Used to create and train the RNN model.

## NumPy (numpy): For efficient numerical operations and array manipulations.

## Pandas (pandas): To read and handle the dataset (CSV format).

## Matplotlib (matplotlib.pyplot): For plotting the actual vs predicted values.

## Scikit-learn (sklearn.preprocessing.MinMaxScaler): For normalizing temperature values between 0 and 1, which helps in faster and more stable model training.

## Dataset Information
### Source: Public dataset hosted at JBrownlee’s GitHub

### Feature Used: Temp (daily minimum temperature in Celsius)

### Format: CSV with daily temperature readings

### Preprocessing: Converted to float32, reshaped to 2D, and normalized using MinMaxScaler

## Data Preparation Steps
### Loading: The dataset is read using Pandas from an online CSV URL.

### Normalization: The temperature values are scaled between 0 and 1 using MinMaxScaler.

## Sequence Creation:

### A helper function is used to create sequences of 30 consecutive temperature values (inputs) with the next value as the prediction target (output).

### This transforms the time series into a supervised learning problem.

### Conversion to Tensors: NumPy arrays are converted into PyTorch tensors of type float32.

## Train-Test Split:

### 80% of the sequences are used for training.

### 20% are reserved for testing.

## Model Architecture
### Model Type: Custom RNN built using PyTorch’s nn.Module

### Input Size: 1 (each time step has one temperature reading)

### Hidden Size: 64 (number of neurons in the RNN layer)

### Output Size: 1 (predicting one temperature value)

## Layers:
### RNN Layer: (nn.RNN) with batch_first=True (input shape: batch, sequence length, feature)

### Fully Connected Layer: (nn.Linear) that maps the final hidden state to the output prediction

## Training Configuration
### Loss Function: Mean Squared Error (nn.MSELoss)

## Optimizer: Adam Optimizer with a learning rate of 0.01

### Epochs: 100

## Training Loop:
### Model is set to training mode.

### Gradients are cleared using optimizer.zero_grad().

### The forward pass is computed.

### The loss is calculated and backpropagated.

### Optimizer updates the weights.

### Loss is printed every 10 epochs for tracking.

## Model Evaluation
### After training, the model is set to evaluation mode using model.eval().

### Predictions are made on the test set using torch.no_grad() to disable gradient calculations.

### Both predictions and actual values are inverse-transformed back to the original temperature scale for comparison.

## Visualization
### A plot is created comparing:

### Actual temperature values (from the test set)

### Predicted values (from the RNN)

### The x-axis represents the time steps, and the y-axis represents the temperature in Celsius.

### This visualization helps assess how accurately the RNN model forecasts temperature trends.

## Conclusion
### This project showcases:

### How to preprocess time series data for sequence learning

### How to design a basic RNN architecture in PyTorch

### How to train and evaluate an RNN model for regression tasks

### Visualization of actual vs predicted values for insight into model performance

