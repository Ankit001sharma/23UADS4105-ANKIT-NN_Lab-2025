# Time Series Prediction using RNN in PyTorch(Object 6:WAP to train and evaluate a Recurrent Neural Network using PyTorch Library to predict the next value in a sample time series dataset.)

This project demonstrates how to build and train a **Recurrent Neural Network (RNN)** using the **PyTorch** framework to predict the next value in a time series dataset. The model is trained to forecast the next day’s **minimum temperature** using historical daily data.

---

##  Dataset Description

###  Dataset: Daily Minimum Temperatures

- **Source:** [Jason Brownlee - UCI ML Repository](https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv)
- **Location:** Melbourne, Australia
- **Time Range:** January 1981 to December 1990
- **Columns:**
  - `Date`: Date in YYYY-MM-DD format
  - `Temp`: Daily minimum temperature in degrees Celsius

We only use the `Temp` column for time series forecasting.

---

##  Code Explanation

###  Step 1: Import Required Libraries

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
torch, torch.nn: Build and train the RNN model.

numpy, pandas: Data manipulation and loading.

matplotlib.pyplot: Visualize the predictions.

MinMaxScaler: Normalize the temperature values to a range [0, 1] for better training performance.

Step 2: Load and Preprocess the Dataset
python

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url)
temps = df['Temp'].values.astype('float32').reshape(-1, 1)

scaler = MinMaxScaler()
temps_scaled = scaler.fit_transform(temps)
Reads the dataset from the URL.

Extracts and reshapes the temperature values.

Normalizes the values using MinMaxScaler to ensure efficient learning.

Step 3: Create Input Sequences
python

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)
Generates sequences of length 30 from the time series.

Each sequence (X) is used to predict the next time step (y).

Step 4: Convert to Tensors and Split Dataset
python

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
Converts the input and output arrays into PyTorch tensors.

Splits the dataset into training (80%) and testing (20%) sets.

Step 5: Define the RNN Model
python

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # last time step
        return out
Defines a simple RNN model:

nn.RNN: A standard RNN layer.

nn.Linear: Maps the hidden state to the final output.

The last hidden state is used to predict the next value in the sequence.

Step 6: Train the Model
python

model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
Loss Function: Mean Squared Error (MSE)

Optimizer: Adam with learning rate 0.01

python

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
Trains the model for 100 epochs.

Prints the loss every 10 epochs.

Step 7: Evaluate the Model
python

model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.numpy())
Puts the model in evaluation mode.

Predicts the values on the test set.

Scales back the predictions and actual values to the original Celsius range.

Step 8: Plot the Results
python
plt.figure(figsize=(12, 6))
plt.plot(range(len(actual)), actual, label="Actual")
plt.plot(range(len(predictions)), predictions, label="Predicted", linestyle="--")
plt.title("RNN Prediction - Daily Minimum Temperatures")
plt.xlabel("Time Step")
plt.ylabel("Temperature (Celsius)")
plt.legend()
plt.show()
Plots the actual and predicted temperatures to visually compare the performance.

Requirements
Install the required libraries using pip:

pip install torch pandas numpy matplotlib scikit-learn
🔮 Possible Improvements
Replace nn.RNN with nn.LSTM or nn.GRU for better performance.

Introduce multi-step forecasting (predict the next 7 days instead of just one).

Add additional features like humidity, wind, etc., for multivariate time series.

Experiment with different optimizers and hyperparameters.







