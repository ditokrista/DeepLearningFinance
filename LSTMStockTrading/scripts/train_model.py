import numpy as np
import random
import pandas as pd
from pylab import mpl, plt
import seaborn as sns
mpl.rcParams['font.family'] = 'serif'
import joblib

from pandas import DatetimeIndex
import math, time
import itertools
import datetime as dt
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import torch as torch 
import torch.nn as nn
from torch.autograd import Variable

import requests
from pathlib import Path
import os

data_directory = Path(__file__).parent.parent
price_data_path = data_directory / "data" / "AAPL.csv" # Change file name to your desired stock data

price_data = pd.read_csv(price_data_path) 
price_data["date"] = pd.to_datetime(price_data["date"])
price_data = price_data.sort_values(by="date")

price_data = price_data.fillna(method="ffill")
values = price_data[["close"]].values.reshape(-1, 1)

look_back = 30


# Fit scaler on the first 80% of the raw series, then transform all values
'''
train_index = int(np.round(0.8 * len(values)))
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(values[:train_index])
values_scaled = scaler.transform(values)


'''
def create_sequence(values, look_back):
    x, y = [], []
    for index in range(len(values) - look_back):
        x.append(values[index: index + look_back])
        y.append(values[index + look_back])
    return np.array(x), np.array(y)


def load_data(stock, look_back, train_ratio=0.6, validation_ratio=0.2):
    train_size = int(len(stock) * train_ratio)
    validation_size = int(len(stock) * validation_ratio)
    test_size = len(stock) - train_size - validation_size

    train_values = stock[:train_size]
    validation_values = stock[train_size:train_size+validation_size]
    test_values = stock[train_size+validation_size:]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train_values)
    validation_scaled = scaler.transform(validation_values)
    test_scaled = scaler.transform(test_values)

    x_train, y_train = create_sequence(train_scaled, look_back)
    x_validation, y_validation = create_sequence(validation_scaled, look_back)
    x_test, y_test = create_sequence(test_scaled, look_back)

    print(f"Shapes - x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"Shapes - x_validation: {x_validation.shape}, y_validation: {y_validation.shape}")
    print(f"Shapes - x_test: {x_test.shape}, y_test: {y_test.shape}")

    return x_train, y_train, x_validation, y_validation, x_test, y_test, scaler
    
    
    '''
    data_raw = stock
    data = []

    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = len(data) - test_set_size

    x_train = data[:train_set_size, :-1,:]
    y_train = data[:train_set_size, -1, :]

    x_test = data[-test_set_size:, :-1, :]
    y_test = data[-test_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]
    '''
x_train, y_train, x_validation, y_validation, x_test, y_test, scaler = load_data(values, look_back, train_ratio=0.6, validation_ratio=0.2)

scaler_path = data_directory / "models" / "scaler.pkl"
joblib.dump(scaler, scaler_path)

x_train = torch.from_numpy(x_train).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
x_validation = torch.from_numpy(x_validation).type(torch.Tensor)
y_validation = torch.from_numpy(y_validation).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

print(y_train.size(), x_train.size(), y_validation.size(), x_validation.size(), y_test.size(), x_test.size())

input_dim = 1
hidden_dim = 128
num_layers = 2
dropout = 0.2
output_dim = 1

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, output_dim=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

    def forward(self, x):
        # Initialize hidden states without gradient tracking
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        
        # No need for .detach() since they don't have gradients anyway
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        
        out = self.fc1(out)
        out = self.elu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out

model = LSTM(input_dim, hidden_dim, num_layers, dropout, output_dim)

loss_fn = torch.nn.MSELoss(reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())

num_epochs = 200
hist = np.zeros(num_epochs)

seq_dim = look_back - 1

for t in range(num_epochs):
    # Remove this line - the LSTM model doesn't have init_hidden() method
    # The hidden states (h0, c0) are already initialized in the forward() method

    y_train_pred = model(x_train)
    y_validation_pred = model(x_validation)
    loss_prediction = loss_fn(y_train_pred, y_train)
    loss_validation = loss_fn(y_validation_pred, y_validation)
    if t % 10 == 0 and t != 0:
        print ("Epoch ", t, "MSE: ", loss_prediction.item(), "Validation MSE: ", loss_validation.item())
    
    hist[t] = loss_prediction.item()

    optimizer.zero_grad()
    loss_prediction.backward()
    optimizer.step()



print(np.shape(y_train_pred))
print(np.shape(y_validation_pred))

y_test_pred = model(x_test)

y_validation_pred = scaler.inverse_transform(y_validation_pred.detach().numpy())
y_validation = scaler.inverse_transform(y_validation.detach().numpy())

y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train.detach().numpy())

y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())

trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print(f"Train Score: {trainScore:.2f} RMSE")
validationScore = math.sqrt(mean_squared_error(y_validation[:,0], y_validation_pred[:,0]))
print(f"Validation Score: {validationScore:.2f} RMSE")
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print(f"Test Score: {testScore:.2f} RMSE")

import matplotlib.dates as mdates
# Create aligned date arrays for predictions

train_start_idx = look_back  # First look_back points lost to sequence creation
validation_start_idx = len(y_train_pred) + look_back  # Training size + look_back points lost
test_start_idx = len(y_train_pred) + len(y_validation_pred) + look_back  # Both previous sizes + look_back points lost

train_dates = price_data["date"].iloc[train_start_idx:train_start_idx + len(y_train_pred)]
validation_dates = price_data["date"].iloc[validation_start_idx:validation_start_idx + len(y_validation_pred)]
test_dates = price_data["date"].iloc[test_start_idx:test_start_idx + len(y_test_pred)]

figure, axes = plt.subplots(figsize=(15, 6))
axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
axes.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# Plot actual prices
axes.plot(train_dates, y_train, label="Train Price", color="blue")
axes.plot(train_dates, y_train_pred, label="Train Predicted", color="red")
axes.plot(validation_dates, y_validation, label="Validation Price", color="green")
axes.plot(validation_dates, y_validation_pred, label="Validation Predicted", color="orange")
axes.plot(test_dates, y_test, label="Actual Price", color="blue")
axes.plot(test_dates, y_test_pred, label="Predicted Price", color="red")
plt.title(f"LSTM {price_data_path.stem} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(data_directory / "models" / "training result" / f"{price_data_path.stem}_pred.png")
plt.show()

# Save complete model instead of just state_dict
model_path = data_directory / "models" / "complete_lstm_model.pth"
torch.save(model, model_path)
print(f"Model saved to: {model_path}")
