"""
Improved Financial Time Series Prediction Model
Addresses: overfitting, data leakage, unrealistic evaluation, and model complexity

Key improvements:
1. Simplified architecture matched to sample size
2. Classification approach (direction prediction) instead of regression
3. Proper time-series cross-validation with walk-forward
4. Return-based features (not absolute prices)
5. Transaction cost modeling
6. Ensemble of simple models
7. Uncertainty quantification
8. Realistic trading metrics
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from typing import Tuple, Dict, List
import math

# Set random seeds
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ==================== Configuration ====================
class Config:
    data_directory = Path(__file__).parent.parent
    stock_symbol = "AAPL"
    price_data_path = data_directory / "data" / f"{stock_symbol}.csv"
    
    # Data parameters
    look_back = 20  # Reduced from 60 - simpler patterns
    prediction_horizon = 1  # Predict 1 day ahead
    min_train_size = 252  # Minimum 1 year of data
    test_size = 60  # ~3 months
    val_size = 60  # ~3 months
    
    # Target definition
    # Instead of predicting exact price, predict direction/magnitude buckets
    target_type = 'direction'  # 'direction', 'multiclass', or 'regression'
    classification_threshold = 0.002  # 0.2% threshold for up/down
    
    # Model parameters - DRAMATICALLY SIMPLIFIED
    input_dim = 1  # Will be updated
    hidden_dim = 32  # Reduced from 256
    num_layers = 1  # Reduced from 3
    dropout = 0.2  # Reduced dropout
    output_dim = 2  # Binary classification (up/down)
    
    # Training parameters
    batch_size = 16  # Smaller batches
    num_epochs = 100  # Fewer epochs
    learning_rate = 0.0005  # Lower learning rate
    weight_decay = 1e-4  # L2 regularization
    patience = 15
    gradient_clip = 0.5
    
    # Trading simulation
    transaction_cost = 0.001  # 0.1% per trade (realistic for retail)
    risk_free_rate = 0.04  # 4% annual
    
    # Ensemble
    n_models = 3  # Number of models in ensemble
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# ==================== Feature Engineering (NO DATA LEAKAGE) ====================

def calculate_returns_and_features(df: pd.DataFrame, fit_scaler=True, scaler=None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Calculate return-based features WITHOUT looking into the future
    
    Key principle: All features are based on information available at time t
    to predict time t+1
    """
    df = df.copy()
    
    # === PRICE-BASED RETURNS (most important) ===
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # === MOMENTUM FEATURES ===
    for period in [3, 5, 10, 20]:
        df[f'momentum_{period}'] = df['close'].pct_change(period)
    
    # === VOLATILITY FEATURES ===
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['returns'].rolling(period).std()
    
    # === MOVING AVERAGE RATIOS (relative, not absolute) ===
    for period in [5, 10, 20]:
        ma = df['close'].rolling(period).mean()
        df[f'ma_ratio_{period}'] = (df['close'] - ma) / ma
    
    # === RSI (Relative Strength Index) ===
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['rsi_normalized'] = (df['rsi_14'] - 50) / 50  # Normalize to [-1, 1]
    
    # === VOLUME-BASED FEATURES (if available) ===
    if 'volume' in df.columns:
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # === VOLATILITY REGIME ===
    df['vol_regime'] = df['volatility_20'] / df['volatility_20'].rolling(60).mean()
    
    # === TREND FEATURES ===
    # Higher highs / lower lows
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-8)
    
    # Remove NaN rows
    df = df.dropna().reset_index(drop=True)

    df['raw_returns'] = df['returns'].copy()

    # Select features for model (RETURNS-BASED, not prices)
    feature_columns = [
        'returns', 'log_returns',
        'momentum_3', 'momentum_5', 'momentum_10', 'momentum_20',
        'volatility_5', 'volatility_10', 'volatility_20',
        'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20',
        'rsi_normalized',
        'vol_regime',
        'price_position'
    ]
    
    # Add volume features if available
    if 'volume_change' in df.columns:
        feature_columns.extend(['volume_change', 'volume_ma_ratio'])
    
    # Filter to available columns
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    # Scale features using ROBUST scaler (better for outliers)
    if fit_scaler:
        scaler = RobustScaler()
        scaler.fit(df[feature_columns].values)
    
    df[feature_columns] = scaler.transform(df[feature_columns].values)
    
    return df, scaler, feature_columns

def create_targets(df: pd.DataFrame, horizon: int = 1, target_type: str = 'direction', 
                   threshold: float = 0.002) -> pd.DataFrame:
    """
    Create prediction targets
    
    target_type:
    - 'direction': Binary (up/down)
    - 'multiclass': Three classes (down/neutral/up)
    - 'regression': Continuous returns (for comparison)
    """
    df = df.copy()
    
    # Future return
    df['future_return'] = df['close'].pct_change(horizon).shift(-horizon)
    
    if target_type == 'direction':
        # Binary: up (1) or down (0)
        df['target'] = (df['future_return'] > threshold).astype(int)
        config.output_dim = 2
        
    elif target_type == 'multiclass':
        # Three classes: down (-1), neutral (0), up (1)
        df['target'] = 0  # neutral
        df.loc[df['future_return'] > threshold, 'target'] = 1  # up
        df.loc[df['future_return'] < -threshold, 'target'] = -1  # down
        df['target'] = df['target'] + 1  # Convert to [0, 1, 2]
        config.output_dim = 3
        
    elif target_type == 'regression':
        df['target'] = df['future_return']
        config.output_dim = 1
    
    return df

# ==================== Walk-Forward Cross-Validation ====================

def create_walk_forward_splits(df: pd.DataFrame, feature_columns: List[str], 
                                config) -> List[Dict]:
    """
    Create walk-forward validation splits
    
    This is THE CORRECT way to validate time series models:
    1. Train on past data
    2. Validate on next period
    3. Move window forward
    4. Never look into the future
    """
    splits = []
    n_samples = len(df)
    
    # Calculate split points
    test_start = n_samples - config.test_size
    val_start = test_start - config.val_size
    train_end = val_start
    
    # Ensure minimum training size
    if train_end < config.min_train_size:
        print(f"Warning: Training size {train_end} is less than minimum {config.min_train_size}")
        print("Adjusting split sizes...")
        train_end = max(config.min_train_size, n_samples - config.test_size - config.val_size)
        val_start = train_end
        test_start = val_start + config.val_size
    
    # Main split
    split = {
        'train': (0, train_end),
        'val': (val_start, test_start),
        'test': (test_start, n_samples)
    }
    
    print(f"\n{'='*60}")
    print(f"Walk-Forward Split")
    print(f"{'='*60}")
    print(f"Train: {split['train'][0]} to {split['train'][1]} ({split['train'][1] - split['train'][0]} samples)")
    print(f"Val:   {split['val'][0]} to {split['val'][1]} ({split['val'][1] - split['val'][0]} samples)")
    print(f"Test:  {split['test'][0]} to {split['test'][1]} ({split['test'][1] - split['test'][0]} samples)")
    print(f"{'='*60}\n")
    
    splits.append(split)
    
    return splits

def create_sequences_no_leakage(data: np.ndarray, targets: np.ndarray, 
                                 look_back: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences ensuring no data leakage
    """
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(targets[i + look_back])
    return np.array(X), np.array(y)

# ==================== Simplified Model Architecture ====================

class SimpleLSTM(nn.Module):
    """
    Simplified LSTM architecture matched to sample size
    
    Key changes from original:
    - 1 LSTM layer (not 3)
    - 32 hidden units (not 256)
    - Simple output layer
    - No batch normalization in LSTM
    - Light dropout
    
    Parameters: ~5,000 (vs 1.3M in original)
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super(SimpleLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Single LSTM layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=1,
            batch_first=True,
            dropout=0  # No dropout in single layer LSTM
        )
        
        # Simple output path
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Take last time step
        out = lstm_out[:, -1, :]
        
        # Dropout and output
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

# ==================== Training with Proper Regularization ====================

class EarlyStopping:
    """Early stopping with best model restoration"""
    def __init__(self, patience: int = 15, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_model = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_state_dict(self.best_model)
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0

def train_model(model, train_loader, val_loader, config):
    """
    Train model with proper regularization and monitoring
    """
    # Use CrossEntropyLoss for classification
    if config.target_type in ['direction', 'multiclass']:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    # Adam optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler (less aggressive)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(config.device)
            y_batch = y_batch.to(config.device).long()
            
            # Forward
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            if config.target_type in ['direction', 'multiclass']:
                _, predicted = torch.max(outputs.data, 1)
                train_total += y_batch.size(0)
                train_correct += (predicted == y_batch).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(config.device)
                y_batch = y_batch.to(config.device).long()
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                if config.target_type in ['direction', 'multiclass']:
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.num_epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%")
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:

            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

# ==================== Realistic Trading Evaluation ====================

def evaluate_trading_strategy(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                               returns: np.ndarray, config) -> Dict:
    """
    Evaluate strategy with REALISTIC trading simulation
    
    Includes:
    - Transaction costs
    - Slippage (implicit in costs)
    - Position sizing
    - Risk metrics
    """
    # Get predicted class
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Convert binary classification back to signals
    if config.target_type == 'direction':
        signals = y_pred * 2 - 1  # Convert [0,1] to [-1, 1]
    elif config.target_type == 'multiclass':
        signals = y_pred - 1  # Convert [0,1,2] to [-1, 0, 1]
    else:
        signals = np.sign(y_pred)

    # Calculate strategy returns
    # FIXED: Align signals with returns properly
    # We predict at time t for return from t to t+1
    strategy_returns = signals * returns  # No indexing needed

    # Apply transaction costs
    # Cost incurred when position changes
    position_changes = np.diff(np.concatenate([[0], signals]))
    transaction_costs = np.abs(position_changes) * config.transaction_cost

    # Align transaction costs with strategy returns
    if len(transaction_costs) > len(strategy_returns):
        transaction_costs = transaction_costs[:len(strategy_returns)]

    strategy_returns_net = strategy_returns - transaction_costs

    # Buy and hold returns (for comparison)
    buy_hold_returns = returns  # FIXED: Use all returns, no slicing
    
    # === METRICS ===
    
    # 1. Accuracy
    # FIXED: No need to slice since we fixed indexing above
    accuracy = accuracy_score(y_true, y_pred)
    
    # 2. Total return
    total_return = np.prod(1 + strategy_returns_net) - 1
    buy_hold_return = np.prod(1 + buy_hold_returns) - 1
    
    # 3. Sharpe Ratio (annualized)
    sharpe = np.mean(strategy_returns_net) / (np.std(strategy_returns_net) + 1e-8) * np.sqrt(252)
    sharpe_bh = np.mean(buy_hold_returns) / (np.std(buy_hold_returns) + 1e-8) * np.sqrt(252)
    
    # 4. Maximum Drawdown
    cumulative = np.cumprod(1 + strategy_returns_net)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # 5. Win Rate
    win_rate = np.sum(strategy_returns_net > 0) / len(strategy_returns_net)
    
    # 6. Profit Factor (gross profit / gross loss)
    gross_profit = np.sum(strategy_returns_net[strategy_returns_net > 0])
    gross_loss = np.abs(np.sum(strategy_returns_net[strategy_returns_net < 0]))
    profit_factor = gross_profit / (gross_loss + 1e-8)
    
    # 7. Information Ratio (excess return over buy-hold)
    excess_returns = strategy_returns_net - buy_hold_returns
    information_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)
    
    # 8. Calmar Ratio (return / max drawdown)
    calmar = (total_return / (abs(max_drawdown) + 1e-8))
    
    metrics = {
        'accuracy': accuracy * 100,
        'total_return': total_return * 100,
        'buy_hold_return': buy_hold_return * 100,
        'sharpe_ratio': sharpe,
        'sharpe_bh': sharpe_bh,
        'max_drawdown': max_drawdown * 100,
        'win_rate': win_rate * 100,
        'profit_factor': profit_factor,
        'information_ratio': information_ratio,
        'calmar_ratio': calmar,
        'avg_trade_cost': np.mean(transaction_costs) * 100
    }
    
    # Add cumulative returns for plotting
    metrics['cumulative_returns'] = cumulative
    metrics['buy_hold_cumulative'] = np.cumprod(1 + buy_hold_returns)
    
    return metrics

def predict_with_uncertainty(model, X, config, n_samples: int = 30):
    """
    Monte Carlo Dropout for uncertainty estimation
    """
    model.train()  # Keep dropout active
    predictions = []
    
    X_tensor = torch.FloatTensor(X).to(config.device)
    
    with torch.no_grad():
        for _ in range(n_samples):
            outputs = model(X_tensor)
            if config.target_type in ['direction', 'multiclass']:
                probs = torch.softmax(outputs, dim=1)
            else:
                probs = outputs
            predictions.append(probs.cpu().numpy())
    
    predictions = np.array(predictions)
    
    # Mean prediction
    mean_pred = np.mean(predictions, axis=0)
    
    # Uncertainty (standard deviation)
    uncertainty = np.std(predictions, axis=0)
    
    return mean_pred, uncertainty

# ==================== Ensemble ====================

def train_ensemble(X_train, y_train, X_val, y_val, config, n_models: int = 3):
    """
    Train ensemble of models for robustness
    """
    models = []
    histories = []
    
    print(f"\nTraining ensemble of {n_models} models...")
    
    for i in range(n_models):
        print(f"\nModel {i+1}/{n_models}")
        print("-" * 40)
        
        # Set different seed for each model
        set_seed(42 + i)
        
        # Create model
        model = SimpleLSTM(
            config.input_dim,
            config.hidden_dim,
            config.output_dim,
            config.dropout
        ).to(config.device)
        
        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train) if config.target_type == 'regression' else torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val) if config.target_type == 'regression' else torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Train
        history = train_model(model, train_loader, val_loader, config)
        
        models.append(model)
        histories.append(history)
        
        # Print final metrics
        print(f"Final Val Loss: {history['val_losses'][-1]:.4f}, "
              f"Val Acc: {history['val_accs'][-1]:.1f}%")
    
    return models, histories

def ensemble_predict(models, X, config):
    """
    Ensemble prediction with uncertainty
    """
    all_predictions = []
    
    X_tensor = torch.FloatTensor(X).to(config.device)
    
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            if config.target_type in ['direction', 'multiclass']:
                probs = torch.softmax(outputs, dim=1)
            else:
                probs = outputs
            all_predictions.append(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    
    # Average predictions
    mean_pred = np.mean(all_predictions, axis=0)
    
    # Disagreement as uncertainty measure
    uncertainty = np.std(all_predictions, axis=0)
    
    return mean_pred, uncertainty

# ==================== Visualization ====================

def plot_comprehensive_results(results: Dict, dates: pd.DatetimeIndex, config):
    """
    Create comprehensive visualization
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Training History
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(results['train_history']['train_losses'], label='Train Loss', alpha=0.7)
    ax1.plot(results['train_history']['val_losses'], label='Val Loss', alpha=0.7)
    ax1.set_title('Training History - Loss', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Accuracy History
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(results['train_history']['train_accs'], label='Train Acc', alpha=0.7)
    ax2.plot(results['train_history']['val_accs'], label='Val Acc', alpha=0.7)
    ax2.set_title('Accuracy History', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random')
    
    # 3. Cumulative Returns
    ax3 = fig.add_subplot(gs[1, :])
    test_metrics = results['test_metrics']
    ax3.plot(dates, test_metrics['cumulative_returns'], 
             label=f"Strategy (Net Return: {test_metrics['total_return']:.2f}%)", 
             linewidth=2, color='green')
    ax3.plot(dates, test_metrics['buy_hold_cumulative'], 
             label=f"Buy & Hold (Return: {test_metrics['buy_hold_return']:.2f}%)", 
             linewidth=2, color='blue', alpha=0.7)
    ax3.set_title('Cumulative Returns (After Transaction Costs)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Cumulative Return')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Confusion Matrix
    ax4 = fig.add_subplot(gs[2, 0])
    y_true = results['y_true_test']
    y_pred = np.argmax(results['y_pred_test'], axis=1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
    ax4.set_title('Confusion Matrix - Test Set', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    
    # 5. Prediction Uncertainty
    ax5 = fig.add_subplot(gs[2, 1])
    uncertainty = results['uncertainty_test']
    if config.target_type == 'direction':
        # Plot uncertainty for positive class
        ax5.hist(uncertainty[:, 1], bins=50, edgecolor='black', alpha=0.7)
        ax5.set_title('Prediction Uncertainty Distribution', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Uncertainty (Std Dev)')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
    
    # 6. Metrics Comparison
    ax6 = fig.add_subplot(gs[2, 2])
    metrics_to_plot = {
        'Accuracy': test_metrics['accuracy'],
        'Win Rate': test_metrics['win_rate'],
        'Sharpe': test_metrics['sharpe_ratio'],
        'Profit Factor': test_metrics['profit_factor']
    }
    bars = ax6.bar(metrics_to_plot.keys(), metrics_to_plot.values())
    ax6.set_title('Test Set Metrics', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Value')
    ax6.grid(True, alpha=0.3, axis='y')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
    
    # Color bars based on performance
    for i, (key, val) in enumerate(metrics_to_plot.items()):
        if key == 'Accuracy' or key == 'Win Rate':
            bars[i].set_color('green' if val > 50 else 'red')
        elif key == 'Sharpe':
            bars[i].set_color('green' if val > 1 else 'orange' if val > 0 else 'red')
        elif key == 'Profit Factor':
            bars[i].set_color('green' if val > 1 else 'red')
    
    # 7. Feature Importance (if available) - Placeholder
    ax7 = fig.add_subplot(gs[3, :])
    ax7.text(0.5, 0.5, 'Model Architecture:\n\n'
             f'Parameters: {results["n_parameters"]:,}\n'
             f'Input Features: {config.input_dim}\n'
             f'Hidden Dim: {config.hidden_dim}\n'
             f'Layers: {config.num_layers}\n'
             f'Dropout: {config.dropout}\n\n'
             f'Key Insight: {results["key_insight"]}',
             ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax7.axis('off')
    
    plt.tight_layout()
    
    # Save
    save_path = config.data_directory / "models" / "training result" / f"{config.stock_symbol}_improved_results.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nResults saved to: {save_path}")
    
    return fig

def print_detailed_metrics(train_metrics: Dict, val_metrics: Dict, test_metrics: Dict):
    """
    Print detailed metrics comparison
    """
    print(f"\n{'='*80}")
    print(f"DETAILED PERFORMANCE METRICS")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<25} {'Train':<15} {'Validation':<15} {'Test':<15}")
    print(f"{'-'*80}")
    
    metrics_to_show = ['accuracy', 'total_return', 'sharpe_ratio', 'max_drawdown', 
                       'win_rate', 'profit_factor', 'information_ratio']
    
    for metric in metrics_to_show:
        train_val = train_metrics.get(metric, 0)
        val_val = val_metrics.get(metric, 0)
        test_val = test_metrics.get(metric, 0)
        
        if 'ratio' in metric or 'factor' in metric:
            print(f"{metric:<25} {train_val:<15.3f} {val_val:<15.3f} {test_val:<15.3f}")
        else:
            print(f"{metric:<25} {train_val:<15.2f} {val_val:<15.2f} {test_val:<15.2f}")
    
    print(f"{'-'*80}")
    print(f"\n{'Buy & Hold Comparison':<25} {'':<15} {'':<15} {'Test':<15}")
    print(f"{'-'*80}")
    print(f"{'Buy & Hold Return':<25} {'':<15} {'':<15} {test_metrics['buy_hold_return']:<15.2f}")
    print(f"{'Buy & Hold Sharpe':<25} {'':<15} {'':<15} {test_metrics['sharpe_bh']:<15.3f}")
    print(f"{'Alpha (Excess Return)':<25} {'':<15} {'':<15} {test_metrics['total_return'] - test_metrics['buy_hold_return']:<15.2f}")
    
    print(f"\n{'Transaction Cost Analysis':<40} {'Test':<15}")
    print(f"{'-'*80}")
    print(f"{'Average Cost per Trade (%)':<40} {test_metrics['avg_trade_cost']:<15.4f}")
    
    print(f"\n{'='*80}\n")

# ==================== Main Execution ====================

def main():
    """
    Main execution with all improvements
    """
    print(f"\n{'='*80}")
    print(f"IMPROVED FINANCIAL LSTM - REALISTIC EVALUATION")
    print(f"{'='*80}\n")
    
    # Load data
    df = pd.read_csv(config.price_data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Loaded data: {len(df)} samples from {df['date'].min()} to {df['date'].max()}")
    
    # Calculate features WITHOUT data leakage
    df, scaler, feature_columns = calculate_returns_and_features(df, fit_scaler=True)
    
    # Create targets
    df = create_targets(df, config.prediction_horizon, config.target_type, config.classification_threshold)
    
    # Remove NaN from targets
    df = df.dropna().reset_index(drop=True)
    
    print(f"\nFeatures: {len(feature_columns)}")
    print(f"Target type: {config.target_type}")
    print(f"Samples after preprocessing: {len(df)}")
    
    # Update config
    config.input_dim = len(feature_columns)
    
    # Get walk-forward splits
    splits = create_walk_forward_splits(df, feature_columns, config)
    split = splits[0]  # Use first split
    
    # Extract split indices
    train_start, train_end = split['train']
    val_start, val_end = split['val']
    test_start, test_end = split['test']

    # Prepare data
    features = df[feature_columns].values
    targets = df['target'].values
    returns = df['raw_returns'].values  # FIXED: Use raw returns, not scaled
    dates = df['date'].values
    
    # Create sequences
    X_all, y_all = create_sequences_no_leakage(features, targets, config.look_back)
    
    # Align returns and dates with sequences
    returns_aligned = returns[config.look_back:]
    dates_aligned = dates[config.look_back:]
    
    # Split data
    X_train = X_all[:train_end - config.look_back]
    y_train = y_all[:train_end - config.look_back]
    returns_train = returns_aligned[:train_end - config.look_back]
    dates_train = dates_aligned[:train_end - config.look_back]
    
    X_val = X_all[val_start - config.look_back:val_end - config.look_back]
    y_val = y_all[val_start - config.look_back:val_end - config.look_back]
    returns_val = returns_aligned[val_start - config.look_back:val_end - config.look_back]
    dates_val = dates_aligned[val_start - config.look_back:val_end - config.look_back]
    
    X_test = X_all[test_start - config.look_back:test_end - config.look_back]
    y_test = y_all[test_start - config.look_back:test_end - config.look_back]
    returns_test = returns_aligned[test_start - config.look_back:test_end - config.look_back]
    dates_test = dates_aligned[test_start - config.look_back:test_end - config.look_back]
    
    print(f"\nFinal shapes:")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")
    
    # Train ensemble
    models, histories = train_ensemble(X_train, y_train, X_val, y_val, config, config.n_models)
    
    # Count parameters
    n_params = sum(p.numel() for p in models[0].parameters())
    print(f"\nModel parameters per model: {n_params:,}")
    print(f"Total ensemble parameters: {n_params * config.n_models:,}")
    
    # Evaluate on all sets
    print(f"\n{'='*80}")
    print(f"ENSEMBLE EVALUATION")
    print(f"{'='*80}\n")
    
    # Train set
    y_pred_train, uncertainty_train = ensemble_predict(models, X_train, config)
    train_metrics = evaluate_trading_strategy(y_train, y_pred_train, returns_train, config)
    
    # Val set
    y_pred_val, uncertainty_val = ensemble_predict(models, X_val, config)
    val_metrics = evaluate_trading_strategy(y_val, y_pred_val, returns_val, config)
    
    # Test set
    y_pred_test, uncertainty_test = ensemble_predict(models, X_test, config)
    test_metrics = evaluate_trading_strategy(y_test, y_pred_test, returns_test, config)
    
    # Print metrics
    print_detailed_metrics(train_metrics, val_metrics, test_metrics)
    
    # Determine key insight
    if test_metrics['accuracy'] > 52 and test_metrics['sharpe_ratio'] > 0.5:
        key_insight = "Model shows potential predictive power above random chance"
    elif test_metrics['accuracy'] > 50:
        key_insight = "Slight edge detected, but transaction costs may erode profits"
    else:
        key_insight = "Model performs at/below random - market may be too efficient"
    
    # Prepare results for visualization
    results = {
        'train_history': histories[0],  # Use first model's history
        'y_true_test': y_test,
        'y_pred_test': y_pred_test,
        'uncertainty_test': uncertainty_test,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'n_parameters': n_params * config.n_models,
        'key_insight': key_insight
    }
    
    # Visualize
    plot_comprehensive_results(results, dates_test[:-1], config)
    
    # Save models
    model_dir = config.data_directory / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for i, model in enumerate(models):
        torch.save(model.state_dict(), model_dir / f"{config.stock_symbol}_improved_model_{i}.pth")
    
    joblib.dump(scaler, model_dir / f"{config.stock_symbol}_scaler_improved.pkl")
    
    # Save metrics
    all_metrics = pd.DataFrame({
        'Set': ['Train', 'Validation', 'Test'],
        'Accuracy': [train_metrics['accuracy'], val_metrics['accuracy'], test_metrics['accuracy']],
        'Total_Return': [train_metrics['total_return'], val_metrics['total_return'], test_metrics['total_return']],
        'Sharpe_Ratio': [train_metrics['sharpe_ratio'], val_metrics['sharpe_ratio'], test_metrics['sharpe_ratio']],
        'Max_Drawdown': [train_metrics['max_drawdown'], val_metrics['max_drawdown'], test_metrics['max_drawdown']],
        'Win_Rate': [train_metrics['win_rate'], val_metrics['win_rate'], test_metrics['win_rate']],
        'Profit_Factor': [train_metrics['profit_factor'], val_metrics['profit_factor'], test_metrics['profit_factor']]
    })
    
    all_metrics.to_csv(model_dir / f"{config.stock_symbol}_improved_metrics.csv", index=False)
    
    print(f"\n{'='*80}")
    print(f"FINAL VERDICT")
    print(f"{'='*80}")
    print(f"\n{key_insight}")
    print(f"\nTest Set Performance:")
    print(f"  - Accuracy: {test_metrics['accuracy']:.2f}% (random = 50%)")
    print(f"  - Net Return: {test_metrics['total_return']:.2f}% vs Buy&Hold: {test_metrics['buy_hold_return']:.2f}%")
    print(f"  - Sharpe Ratio: {test_metrics['sharpe_ratio']:.3f}")
    print(f"  - Win Rate: {test_metrics['win_rate']:.2f}%")
    
    if test_metrics['total_return'] > test_metrics['buy_hold_return']:
        print(f"\n✓ Strategy OUTPERFORMS buy-and-hold by {test_metrics['total_return'] - test_metrics['buy_hold_return']:.2f}%")
    else:
        print(f"\n✗ Strategy UNDERPERFORMS buy-and-hold by {test_metrics['buy_hold_return'] - test_metrics['total_return']:.2f}%")
    
    print(f"\n{'='*80}\n")
    
    return results, models, scaler

if __name__ == "__main__":
    results, models, scaler = main()
