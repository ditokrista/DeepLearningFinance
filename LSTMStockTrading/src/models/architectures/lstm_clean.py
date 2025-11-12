"""
Clean LSTM Architecture Module
Pure model definition - no training, data loading, or feature engineering

This module contains only the neural network architecture definitions.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Enhanced LSTM Model for Stock Price Prediction

    Pure architecture definition with:
    - Multi-layer LSTM
    - Layer normalization for stable training
    - Fully connected layers with batch normalization
    - Dropout for regularization

    Args:
        input_dim (int): Number of input features
        hidden_dim (int): Size of hidden layers (default: 256)
        num_layers (int): Number of LSTM layers (default: 3)
        dropout (float): Dropout probability (default: 0.3)
        output_dim (int): Number of output features (default: 1)
    """

    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.3, output_dim=1):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, output_dim)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 4)

        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_dim)
        """
        # LSTM layers
        lstm_out, (hidden, cell) = self.lstm(x)

        # Take the last time step output
        out = lstm_out[:, -1, :]

        # Layer normalization
        out = self.layer_norm(out)

        # Fully connected layer 1
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Fully connected layer 2
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Output layer
        out = self.fc3(out)

        return out


class SimpleLSTM(nn.Module):
    """
    Simpler LSTM Model for baseline comparisons

    Args:
        input_dim (int): Number of input features
        hidden_dim (int): Size of hidden layers (default: 128)
        num_layers (int): Number of LSTM layers (default: 2)
        dropout (float): Dropout probability (default: 0.2)
        output_dim (int): Number of output features (default: 1)
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, output_dim=1):
        super(SimpleLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        # Activation and regularization
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the network

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_dim)
        """
        # Initialize hidden states
        device = x.device
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)

        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Take last time step
        out = out[:, -1, :]

        # Fully connected layers
        out = self.fc1(out)
        out = self.elu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


def get_model(model_type='enhanced', **kwargs):
    """
    Factory function to get model by type

    Args:
        model_type (str): Type of model ('enhanced' or 'simple')
        **kwargs: Model parameters (input_dim, hidden_dim, etc.)

    Returns:
        nn.Module: Initialized model
    """
    if model_type == 'enhanced':
        return LSTMModel(**kwargs)
    elif model_type == 'simple':
        return SimpleLSTM(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
