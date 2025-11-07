
"""
Optimized LSTM Model for Stock Price Prediction
Implements best practices in deep learning and financial time series forecasting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

class Config:
    # Use relative path that works in any environment
    data_directory = Path(__file__).resolve().parent.parent
    stock_symbol = "MSFT"  # Change as needed
    price_data_path = data_directory / "data" / f"{stock_symbol}.csv"
    
    # Data parameters
    look_back = 60  # Increased from 30 for better pattern recognition
    train_ratio = 0.7
    validation_ratio = 0.15
    test_ratio = 0.15
    
    # Model parameters
    input_dim = 1  # Will be updated based on features
    hidden_dim = 256  # Increased capacity
    num_layers = 3  # Deeper network
    dropout = 0.3
    output_dim = 1
    
    # Training parameters
    batch_size = 32
    num_epochs = 300
    learning_rate = 0.001
    patience = 30  # For early stopping
    gradient_clip = 1.0
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# ==================== Feature Engineering ====================

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for enhanced feature set
    """
    df = df.copy()
    
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Moving averages
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential moving averages
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Momentum
    df['momentum'] = df['close'] - df['close'].shift(4)
    
    # Volatility (Standard deviation of returns)
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Volume-based features (if available)
    if 'volume' in df.columns:
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price rate of change
    df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
    
    # Average True Range (ATR) - volatility measure
    if all(col in df.columns for col in ['high', 'low', 'close']):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
    
    return df

# ==================== Data Preparation ====================

def create_sequences(data, look_back):
    """
    Create sequences for time series prediction
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back, 0])  # Predict close price (first column)
    return np.array(X), np.array(y)

def prepare_data(config, use_technical_indicators=True):
    """
    Prepare data with proper scaling and feature engineering
    
    Key approach to prevent data leakage while maintaining continuity:
    1. Fit scaler ONLY on training data
    2. Transform entire dataset with training scaler
    3. Create sequences from entire scaled dataset (maintains smooth timeline)
    4. Then split sequences into train/val/test
    
    This ensures:
    - No data leakage (scaler never sees val/test statistics)
    - Continuous timeline (no jumps between train/val/test in plots)
    """
    # Load data
    df = pd.read_csv(config.price_data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Add technical indicators
    if use_technical_indicators:
        df = calculate_technical_indicators(df)
        
        # Select features (close price must be first for prediction target)
        feature_columns = ['close', 'returns', 'sma_5', 'sma_20', 'rsi', 
                          'macd', 'macd_signal', 'bb_position', 'bb_width',
                          'momentum', 'volatility', 'roc']
        
        # Filter to available columns
        feature_columns = [col for col in feature_columns if col in df.columns]
    else:
        feature_columns = ['close']
    
    # Remove NaN values created by technical indicators
    df = df.dropna().reset_index(drop=True)
    
    # Extract values
    values = df[feature_columns].values
    dates = df['date'].values
    
    # Split data BEFORE scaling to prevent data leakage
    train_size = int(len(values) * config.train_ratio)
    val_size = int(len(values) * config.validation_ratio)
    
    train_data = values[:train_size]
    
    # Fit scaler ONLY on training data (prevent data leakage)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_data)
    
    # Transform the ENTIRE dataset with training statistics
    values_scaled = scaler.transform(values)
    
    # Create sequences from the entire scaled dataset (maintains continuity)
    X_all, y_all = create_sequences(values_scaled, config.look_back)
    
    # Adjust dates to account for sequences
    dates_adjusted = dates[config.look_back:]
    
    # Now split the sequences (not the raw data)
    # Account for look_back period that was lost
    train_seq_size = train_size - config.look_back
    val_seq_size = val_size
    
    X_train = X_all[:train_seq_size]
    y_train = y_all[:train_seq_size]
    
    X_val = X_all[train_seq_size:train_seq_size + val_seq_size]
    y_val = y_all[train_seq_size:train_seq_size + val_seq_size]
    
    X_test = X_all[train_seq_size + val_seq_size:]
    y_test = y_all[train_seq_size + val_seq_size:]
    
    # Split dates accordingly
    train_dates = dates_adjusted[:train_seq_size]
    val_dates = dates_adjusted[train_seq_size:train_seq_size + val_seq_size]
    test_dates = dates_adjusted[train_seq_size + val_seq_size:]
    
    print(f"\n{'='*60}")
    print(f"Data Preparation Summary")
    print(f"{'='*60}")
    print(f"Features used: {len(feature_columns)}")
    print(f"Feature names: {feature_columns}")
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation shape: X={X_val.shape}, y={y_val.shape}")
    print(f"Test shape: X={X_test.shape}, y={y_test.shape}")
    print(f"{'='*60}\n")
    
    # Update config with actual input dimension
    config.input_dim = len(feature_columns)
    
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            scaler, train_dates, val_dates, test_dates, feature_columns)

# ==================== Model Architecture ====================

class ImprovedLSTM(nn.Module):
    """
    Enhanced LSTM with better architecture and regularization
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        super(ImprovedLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers with layer normalization
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
        
        # Fully connected layers with residual connections
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, output_dim)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 4)
        
    def forward(self, x):
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take the last time step output
        out = lstm_out[:, -1, :]
        
        # Layer normalization
        out = self.layer_norm(out)
        
        # Fully connected layers with batch norm and dropout
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out

# ==================== Training ====================

class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """
    def __init__(self, patience=20, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0

def train_model(model, train_loader, val_loader, config):
    """
    Train model with best practices
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n{'='*60}")
    print(f"Training on device: {config.device}")
    print(f"{'='*60}\n")
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(config.device), y_batch.to(config.device)
            
            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions.squeeze(), y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(config.device), y_batch.to(config.device)
                predictions = model(X_batch)
                loss = criterion(predictions.squeeze(), y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.num_epochs}] "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                      config.data_directory / "models" / f"{config.stock_symbol}_best_model.pth")
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    print(f"\nTraining completed. Best validation loss: {best_val_loss:.6f}")
    
    return train_losses, val_losses

# ==================== Evaluation ====================

def calculate_trading_metrics(y_true, y_pred):
    """
    Calculate trading-specific metrics
    """
    # Direction accuracy (most important for trading)
    y_true_diff = np.diff(y_true)
    y_pred_diff = np.diff(y_pred)
    direction_accuracy = np.mean((y_true_diff > 0) == (y_pred_diff > 0)) * 100
    
    # Standard regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calculate returns-based metrics
    returns_true = np.diff(y_true) / y_true[:-1]
    returns_pred = np.diff(y_pred) / y_pred[:-1]
    
    # Sharpe Ratio (assuming 252 trading days, 0% risk-free rate)
    if np.std(returns_pred) > 0:
        sharpe_ratio = np.mean(returns_pred) / np.std(returns_pred) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Max Drawdown
    cumulative_returns = np.cumprod(1 + returns_pred)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown) * 100
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Direction_Accuracy': direction_accuracy,
        'Sharpe_Ratio': sharpe_ratio,
        'Max_Drawdown': max_drawdown
    }
    
    return metrics

def evaluate_model(model, X_test, y_test, scaler, config, phase="Test"):
    """
    Evaluate model and return predictions
    """
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(config.device)
        predictions = model(X_test_tensor).cpu().numpy()
    
    # Inverse transform predictions and actual values
    # Create dummy array for inverse transform
    n_features = scaler.n_features_in_
    dummy_pred = np.zeros((len(predictions), n_features))
    dummy_pred[:, 0] = predictions.flatten()
    predictions_rescaled = scaler.inverse_transform(dummy_pred)[:, 0]
    
    dummy_true = np.zeros((len(y_test), n_features))
    dummy_true[:, 0] = y_test
    y_test_rescaled = scaler.inverse_transform(dummy_true)[:, 0]
    
    # Calculate metrics
    metrics = calculate_trading_metrics(y_test_rescaled, predictions_rescaled)
    
    print(f"\n{'='*60}")
    print(f"{phase} Set Metrics")
    print(f"{'='*60}")
    for metric, value in metrics.items():
        print(f"{metric:.<30} {value:.4f}")
    print(f"{'='*60}\n")
    
    return predictions_rescaled, y_test_rescaled, metrics

# ==================== Visualization ====================

def plot_results(train_pred, train_actual, val_pred, val_actual, 
                test_pred, test_actual, train_dates, val_dates, test_dates,
                train_losses, val_losses, config):
    """
    Create comprehensive visualization of results
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Full prediction plot
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(train_dates, train_actual, 'b-', label='Train Actual', alpha=0.7, linewidth=1)
    ax1.plot(train_dates, train_pred, 'r--', label='Train Predicted', alpha=0.7, linewidth=1)
    ax1.plot(val_dates, val_actual, 'g-', label='Val Actual', alpha=0.7, linewidth=1)
    ax1.plot(val_dates, val_pred, 'orange', linestyle='--', label='Val Predicted', alpha=0.7, linewidth=1)
    ax1.plot(test_dates, test_actual, 'b-', label='Test Actual', linewidth=2)
    ax1.plot(test_dates, test_pred, 'r--', label='Test Predicted', linewidth=2)
    ax1.set_title(f'{config.stock_symbol} Stock Price Prediction - Full Timeline', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Test set zoom
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(test_dates, test_actual, 'b-', label='Actual', linewidth=2)
    ax2.plot(test_dates, test_pred, 'r--', label='Predicted', linewidth=2)
    ax2.set_title('Test Set Predictions (Zoomed)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=10)
    ax2.set_ylabel('Price ($)', fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Training history
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(train_losses, label='Training Loss', linewidth=2)
    ax3.plot(val_losses, label='Validation Loss', linewidth=2)
    ax3.set_title('Training History', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('Loss (MSE)', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Prediction error distribution
    ax4 = fig.add_subplot(gs[2, 0])
    errors = test_actual - test_pred
    ax4.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax4.set_title('Test Set Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Prediction Error ($)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. Scatter plot: Actual vs Predicted
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.scatter(test_actual, test_pred, alpha=0.5)
    min_val = min(test_actual.min(), test_pred.min())
    max_val = max(test_actual.max(), test_pred.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax5.set_title('Actual vs Predicted Prices', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Actual Price ($)', fontsize=10)
    ax5.set_ylabel('Predicted Price ($)', fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = config.data_directory / "models" / "training result" / f"{config.stock_symbol}_optimized_results.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Results saved to: {save_path}")
    
    plt.show()

# ==================== Main Execution ====================

def main():
    """
    Main execution function
    """
    print("\n" + "="*60)
    print("OPTIMIZED LSTM STOCK PRICE PREDICTION MODEL")
    print("="*60)
    
    # Prepare data
    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler, train_dates, val_dates, test_dates, feature_columns) = prepare_data(
        config, use_technical_indicators=True
    )
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Initialize model
    model = ImprovedLSTM(
        config.input_dim,
        config.hidden_dim,
        config.num_layers,
        config.dropout,
        config.output_dim
    ).to(config.device)
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, config)
    
    # Evaluate on all sets
    train_pred, train_actual, train_metrics = evaluate_model(
        model, X_train, y_train, scaler, config, "Training"
    )
    val_pred, val_actual, val_metrics = evaluate_model(
        model, X_val, y_val, scaler, config, "Validation"
    )
    test_pred, test_actual, test_metrics = evaluate_model(
        model, X_test, y_test, scaler, config, "Test"
    )
    
    # Visualize results
    plot_results(
        train_pred, train_actual, val_pred, val_actual,
        test_pred, test_actual, train_dates, val_dates, test_dates,
        train_losses, val_losses, config
    )
    
    # Save model and scaler
    model_path = config.data_directory / "models" / f"{config.stock_symbol}_optimized_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    scaler_path = config.data_directory / "models" / f"{config.stock_symbol}_scaler_optimized.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save metrics
    metrics_df = pd.DataFrame({
        'Set': ['Train', 'Validation', 'Test'],
        'RMSE': [train_metrics['RMSE'], val_metrics['RMSE'], test_metrics['RMSE']],
        'MAE': [train_metrics['MAE'], val_metrics['MAE'], test_metrics['MAE']],
        'Direction_Accuracy': [train_metrics['Direction_Accuracy'], 
                               val_metrics['Direction_Accuracy'], 
                               test_metrics['Direction_Accuracy']],
        'R2': [train_metrics['R2'], val_metrics['R2'], test_metrics['R2']],
        'Sharpe_Ratio': [train_metrics['Sharpe_Ratio'], 
                        val_metrics['Sharpe_Ratio'], 
                        test_metrics['Sharpe_Ratio']]
    })
    
    metrics_path = config.data_directory / "models" / f"{config.stock_symbol}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
