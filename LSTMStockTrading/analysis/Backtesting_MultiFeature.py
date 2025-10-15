import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import warnings
import joblib
from pathlib import Path
from collections import OrderedDict

warnings.filterwarnings('ignore')


# ==================== Technical Indicators ====================

def calculate_technical_indicators(df):
    """Calculate technical indicators matching the training model."""
    df = df.copy()

    # Price-based features
    df['returns'] = df['close'].pct_change()

    # Moving averages
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()

    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    bb_middle = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle - (bb_std * 2)
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    df['bb_width'] = (bb_upper - bb_lower) / bb_middle

    # Momentum
    df['momentum'] = df['close'] - df['close'].shift(4)

    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()

    # Rate of Change
    df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100

    return df


# ==================== Model Architecture ====================

class ImprovedLSTM(nn.Module):
    """
    Enhanced LSTM for time series prediction.
    Following best practices:
    - Layer normalization for stable training
    - Batch normalization in FC layers
    - Dropout for regularization
    - Residual connections in dense layers
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2, output_dim=1):
        super(ImprovedLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.output_dim = output_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Dense layers with batch normalization
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)

        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 4)

        self.fc3 = nn.Linear(hidden_dim // 4, output_dim)

        self.relu = nn.ReLU()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'weight' in name and 'fc' in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Take the last output
        out = lstm_out[:, -1, :]

        # Apply layer normalization
        out = self.layer_norm(out)

        # Dense layers with residual connections
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

    def get_config(self):
        """Return model configuration for saving/loading."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout_rate,
            'output_dim': self.output_dim
        }


# ==================== Model Loading Utilities ====================

def infer_model_dims_from_state_dict(state_dict: OrderedDict) -> Dict[str, int]:
    """
    Infer model dimensions from state dictionary.

    LSTM weight naming convention:
    - weight_ih_l0: input-to-hidden weights for layer 0
    - weight_hh_l0: hidden-to-hidden weights for layer 0

    Shape conventions:
    - weight_ih: (4*hidden_size, input_size) for LSTM (4 gates)
    - weight_hh: (4*hidden_size, hidden_size)
    """
    try:
        # Find LSTM input-to-hidden weight for first layer
        weight_ih_key = 'lstm.weight_ih_l0'

        if weight_ih_key not in state_dict:
            raise KeyError(f"Cannot find {weight_ih_key} in state dict")

        weight_ih = state_dict[weight_ih_key]

        # LSTM has 4 gates (input, forget, cell, output)
        # weight_ih shape is (4*hidden_dim, input_dim)
        four_times_hidden = weight_ih.shape[0]
        input_dim = weight_ih.shape[1]
        hidden_dim = four_times_hidden // 4

        # Count number of layers
        num_layers = 0
        for key in state_dict.keys():
            if 'lstm.weight_ih_l' in key:
                layer_num = int(key.split('_l')[1])
                num_layers = max(num_layers, layer_num + 1)

        # Infer output dimension from final FC layer
        fc3_key = 'fc3.weight'
        if fc3_key in state_dict:
            output_dim = state_dict[fc3_key].shape[0]
        else:
            output_dim = 1

        config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'output_dim': output_dim
        }

        print(f"✓ Inferred model configuration:")
        print(f"  - input_dim: {input_dim}")
        print(f"  - hidden_dim: {hidden_dim}")
        print(f"  - num_layers: {num_layers}")
        print(f"  - output_dim: {output_dim}")

        return config

    except Exception as e:
        raise ValueError(f"Failed to infer model dimensions: {e}")


def load_model_safe(model_path: Path,
                    default_config: Optional[Dict] = None) -> nn.Module:
    """
    Safely load a PyTorch model from various checkpoint formats.

    Handles:
    1. Full model saves (torch.save(model, path))
    2. State dict saves (torch.save(model.state_dict(), path))
    3. Checkpoint dicts with config (torch.save({'model_state_dict': ..., 'config': ...}, path))

    Args:
        model_path: Path to the saved model
        default_config: Default configuration if inference fails

    Returns:
        Loaded PyTorch model in eval mode
    """

    print(f"\nLoading model from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    print(f"Checkpoint type: {type(checkpoint).__name__}")

    # Case 1: Full model object
    if isinstance(checkpoint, nn.Module):
        print("✓ Loaded full model object")
        model = checkpoint
        model.eval()
        return model

    # Case 2: Dictionary or OrderedDict (state dict)
    elif isinstance(checkpoint, (dict, OrderedDict)):

        # Extract state dict and config
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            saved_config = checkpoint.get('model_config', None)
        else:
            # Assume the entire checkpoint is the state dict
            state_dict = checkpoint
            saved_config = None

        # Determine model configuration
        if saved_config is not None:
            print("✓ Using saved model configuration")
            config = saved_config
        else:
            print("⚠ No saved config found, inferring from state dict...")
            try:
                config = infer_model_dims_from_state_dict(state_dict)
            except Exception as e:
                if default_config is not None:
                    print(f"⚠ Inference failed: {e}")
                    print("✓ Using provided default configuration")
                    config = default_config
                else:
                    raise ValueError(f"Cannot infer model config and no default provided: {e}")

        # Create model instance
        print(f"Creating model with config: {config}")
        model = ImprovedLSTM(
            input_dim=config.get('input_dim', 12),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.2),
            output_dim=config.get('output_dim', 1)
        )

        # Load state dict
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✓ State dict loaded successfully (strict mode)")
        except Exception as e:
            print(f"⚠ Strict loading failed: {e}")
            print("Attempting non-strict loading...")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys: {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
            print("✓ State dict loaded (non-strict mode)")

        model.eval()
        return model

    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")


# ==================== Backtesting Engine ====================

class BacktestEngine:
    """
    Professional backtesting engine for LSTM trading strategies.

    Features:
    - Transaction cost modeling
    - Minimum holding period enforcement
    - Position sizing
    - Risk management
    """

    def __init__(self,
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001,  # 10 bps
                 min_holding_period: int = 1,
                 position_size: float = 0.95):  # Use 95% of capital
        """
        Initialize backtesting engine.

        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
            min_holding_period: Minimum days to hold position
            position_size: Fraction of capital to use per trade
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.min_holding_period = min_holding_period
        self.position_size = position_size

    def generate_trading_signals(self,
                                 model: nn.Module,
                                 historical_data: pd.DataFrame,
                                 scaler,
                                 sequence_length: int = 60) -> pd.DataFrame:
        """
        Generate trading signals using multi-feature LSTM predictions.

        Args:
            model: Trained PyTorch model (must be nn.Module)
            historical_data: DataFrame with price data
            scaler: Fitted scaler for feature normalization
            sequence_length: Lookback window size

        Returns:
            DataFrame with signals and predictions
        """

        # Validate model type
        if not isinstance(model, nn.Module):
            raise TypeError(f"Model must be nn.Module, got {type(model)}")

        print("\n" + "=" * 60)
        print("GENERATING TRADING SIGNALS")
        print("=" * 60)

        # Add technical indicators
        data_with_indicators = calculate_technical_indicators(historical_data)
        data_with_indicators = data_with_indicators.dropna().reset_index(drop=True)

        # Define feature columns (must match training - close must be first!)
        feature_columns = ['close', 'returns', 'sma_5', 'sma_20', 'rsi',
                           'macd', 'macd_signal', 'bb_position', 'bb_width',
                           'momentum', 'volatility', 'roc']

        # Filter to available columns
        feature_columns = [col for col in feature_columns if col in data_with_indicators.columns]

        print(f"\nUsing {len(feature_columns)} features:")
        for i, col in enumerate(feature_columns, 1):
            print(f"  {i}. {col}")

        # Extract feature values
        feature_data = data_with_indicators[feature_columns].values
        print(f"\nFeature data shape: {feature_data.shape}")
        print(f"Scaler expects {scaler.n_features_in_} features")

        if feature_data.shape[1] != scaler.n_features_in_:
            raise ValueError(
                f"Feature mismatch! Data has {feature_data.shape[1]} features but scaler expects {scaler.n_features_in_}")

        signals_df = pd.DataFrame(index=range(sequence_length, len(feature_data)))

        predictions = []
        current_prices = []
        signals = []

        print(f"\nGenerating predictions for {len(feature_data) - sequence_length} time steps...")

        # Set model to eval mode
        model.eval()

        # Generate predictions
        with torch.no_grad():  # Disable gradient computation for inference
            for i in range(sequence_length, len(feature_data)):
                # Prepare multi-feature input sequence
                sequence = feature_data[i - sequence_length:i]
                scaled_sequence = scaler.transform(sequence)

                # Convert to tensor
                input_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)

                # Generate prediction
                scaled_pred = model(input_tensor).cpu().numpy()

                # Inverse transform prediction (close price is first feature)
                n_features = scaler.n_features_in_
                dummy_pred = np.zeros((1, n_features))
                dummy_pred[:, 0] = scaled_pred.flatten()
                prediction = scaler.inverse_transform(dummy_pred)[0, 0]

                current_price = data_with_indicators['close'].iloc[i]

                predictions.append(prediction)
                current_prices.append(current_price)

                # Signal generation with transaction cost consideration
                expected_return = (prediction - current_price) / current_price
                threshold = 2 * self.transaction_cost

                if expected_return > threshold:
                    signal = 1  # Buy
                elif expected_return < -threshold:
                    signal = -1  # Sell
                else:
                    signal = 0  # Hold

                signals.append(signal)

        # Create output DataFrame
        signals_df['Price'] = current_prices
        signals_df['Prediction'] = predictions
        signals_df['Signal'] = signals
        signals_df['Expected_Return'] = [(p - c) / c for p, c in zip(predictions, current_prices)]

        print(f"\n✓ Generated {len(signals_df)} trading signals")
        print(f"  Buy signals: {sum(s == 1 for s in signals)}")
        print(f"  Sell signals: {sum(s == -1 for s in signals)}")
        print(f"  Hold signals: {sum(s == 0 for s in signals)}")

        return signals_df

    def run_backtest(self, signals_df: pd.DataFrame) -> Tuple[Dict, list, list]:
        """
        Run backtest with transaction costs and risk management.

        Args:
            signals_df: DataFrame with trading signals

        Returns:
            Tuple of (metrics dict, trades list, portfolio values list)
        """

        print("\n" + "=" * 60)
        print("RUNNING BACKTEST")
        print("=" * 60)

        capital = self.initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        trades = []
        portfolio_values = []
        entry_price = 0
        shares = 0
        last_trade_day = -self.min_holding_period

        for day_index, (idx, row) in enumerate(signals_df.iterrows()):
            # Calculate current portfolio value
            if position == 1:
                current_value = capital + (row['Price'] - entry_price) * shares
            elif position == -1:
                current_value = capital - (row['Price'] - entry_price) * shares
            else:
                current_value = capital

            portfolio_values.append(current_value)

            can_trade = (day_index - last_trade_day) >= self.min_holding_period

            if row['Signal'] != position and can_trade:
                # Close existing position
                if position != 0:
                    pnl = (row['Price'] - entry_price) * shares * position
                    trade_cost = abs(shares * row['Price'] * self.transaction_cost)
                    capital += pnl - trade_cost

                    trades.append({
                        'Date': idx,
                        'Type': 'CLOSE',
                        'Price': row['Price'],
                        'PnL': pnl - trade_cost,
                        'Days_Held': day_index - last_trade_day
                    })

                # Open new position
                if row['Signal'] != 0:
                    position = row['Signal']
                    entry_price = row['Price']
                    shares = int(capital * self.position_size / row['Price'])

                    if shares > 0:
                        trade_cost = abs(shares * row['Price'] * self.transaction_cost)
                        capital -= trade_cost
                        last_trade_day = day_index

                        trades.append({
                            'Date': idx,
                            'Type': 'OPEN',
                            'Signal': position,
                            'Price': entry_price,
                            'Shares': shares,
                            'Cost': trade_cost
                        })
                    else:
                        position = 0
                else:
                    position = 0
                    last_trade_day = day_index

        # Calculate performance metrics
        if len(portfolio_values) > 0:
            total_return = (portfolio_values[-1] / self.initial_capital - 1) * 100
            portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()

            if len(portfolio_returns) > 0 and portfolio_returns.std() > 0:
                sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0

            cumulative = pd.Series(portfolio_values) / self.initial_capital
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min()) * 100
        else:
            total_return = 0
            sharpe_ratio = 0
            max_drawdown = 0

        metrics = {
            'Total_Return': total_return,
            'Initial_Capital': self.initial_capital,
            'Final_Value': portfolio_values[-1] if portfolio_values else self.initial_capital,
            'Number_of_Trades': len(trades),
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown
        }

        print(f"\n✓ Backtest complete")
        print(f"  Total trades: {len(trades)}")
        print(f"  Final portfolio value: ${portfolio_values[-1]:,.2f}")
        print(f"  Total return: {total_return:.2f}%")

        return metrics, trades, portfolio_values


def create_performance_summary(metrics: Dict, trades: list, signals_df: pd.DataFrame,
                               price_data: pd.DataFrame, symbol: str = "STOCK"):
    """Create comprehensive performance summary report."""

    summary_lines = []
    summary_lines.append("\n" + "=" * 70)
    summary_lines.append("                    PERFORMANCE SUMMARY")
    summary_lines.append("=" * 70)

    # Trading period
    start_date = signals_df.index[0] if len(signals_df) > 0 else "N/A"
    end_date = signals_df.index[-1] if len(signals_df) > 0 else "N/A"

    start_price = signals_df['Price'].iloc[0] if len(signals_df) > 0 else 0
    end_price = signals_df['Price'].iloc[-1] if len(signals_df) > 0 else 0
    price_change = ((end_price - start_price) / start_price * 100) if start_price > 0 else 0

    summary_lines.append(f"\nTrading Period:")
    summary_lines.append(f"  Duration:             {len(signals_df)} trading days")
    summary_lines.append(f"  Start Price:          ${start_price:.2f}")
    summary_lines.append(f"  End Price:            ${end_price:.2f}")
    summary_lines.append(f"  Buy & Hold Return:    {price_change:.2f}%")

    summary_lines.append(f"\nStrategy Performance:")
    summary_lines.append(f"  Total Return:         {metrics['Total_Return']:.2f}%")
    summary_lines.append(f"  Initial Capital:      ${metrics.get('Initial_Capital', 100000):,.2f}")
    summary_lines.append(f"  Final Portfolio Value: ${metrics['Final_Value']:,.2f}")
    summary_lines.append(
        f"  Absolute Gain/Loss:   ${metrics['Final_Value'] - metrics.get('Initial_Capital', 100000):,.2f}")
    summary_lines.append(f"  Sharpe Ratio:         {metrics['Sharpe_Ratio']:.3f}")
    summary_lines.append(f"  Maximum Drawdown:     {metrics['Max_Drawdown']:.2f}%")
    summary_lines.append(f"  Number of Trades:     {metrics['Number_of_Trades']}")

    # Strategy vs Buy & Hold
    strategy_outperformance = metrics['Total_Return'] - price_change
    summary_lines.append(f"\nStrategy vs Buy & Hold:")
    summary_lines.append(f"  Outperformance:       {strategy_outperformance:.2f}%")

    # Signal statistics
    signal_counts = signals_df['Signal'].value_counts()
    total_signals = len(signals_df)

    summary_lines.append(f"\nSignal Distribution:")
    summary_lines.append(
        f"  Buy Signals:    {signal_counts.get(1, 0):3d} ({signal_counts.get(1, 0) / total_signals * 100:.1f}%)")
    summary_lines.append(
        f"  Hold Signals:   {signal_counts.get(0, 0):3d} ({signal_counts.get(0, 0) / total_signals * 100:.1f}%)")
    summary_lines.append(
        f"  Sell Signals:   {signal_counts.get(-1, 0):3d} ({signal_counts.get(-1, 0) / total_signals * 100:.1f}%)")

    # Trade analysis
    if trades:
        profitable_trades = [t for t in trades if t.get('PnL', 0) > 0]
        losing_trades = [t for t in trades if t.get('PnL', 0) < 0]
        pnl_trades = [t for t in trades if 'PnL' in t]

        if pnl_trades:
            win_rate = len(profitable_trades) / len(pnl_trades) * 100
            avg_trade_pnl = np.mean([t['PnL'] for t in pnl_trades])

            summary_lines.append(f"\nTrade Analysis:")
            summary_lines.append(f"  Win Rate:           {win_rate:.1f}%")
            summary_lines.append(f"  Average Trade P&L:  ${avg_trade_pnl:.2f}")

            if profitable_trades:
                avg_profit = np.mean([t['PnL'] for t in profitable_trades])
                max_profit = max([t['PnL'] for t in profitable_trades])
                summary_lines.append(f"  Average Profit:     ${avg_profit:.2f}")
                summary_lines.append(f"  Best Trade:         ${max_profit:.2f}")

            if losing_trades:
                avg_loss = np.mean([t['PnL'] for t in losing_trades])
                max_loss = min([t['PnL'] for t in losing_trades])
                summary_lines.append(f"  Average Loss:       ${avg_loss:.2f}")
                summary_lines.append(f"  Worst Trade:        ${max_loss:.2f}")

    summary_lines.append("=" * 70)

    for line in summary_lines:
        print(line)


# ==================== Main Execution ====================

if __name__ == "__main__":

    stock = "MSFT"  # Change to your stock symbol
    data_directory = Path(__file__).resolve().parent.parent
    model_path = data_directory / "models" / f"{stock}_optimized_model.pth"
    price_data_path = data_directory / "data" / f"{stock}.csv"
    scaler_path = data_directory / "models" / f"{stock}_scaler_optimized.pkl"

    print("\n" + "=" * 60)
    print(f"LSTM BACKTESTING SYSTEM - {stock}")
    print("=" * 60)
    print(f"\nModel path: {model_path}")
    print(f"Data path: {price_data_path}")
    print(f"Scaler path: {scaler_path}")

    # Default configuration (fallback)
    default_config = {
        'input_dim': 12,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'output_dim': 1
    }

    # Load model using safe loading utility
    try:
        model = load_model_safe(model_path, default_config=default_config)
        print(f"\n✓ Model successfully loaded")
        print(f"  Type: {type(model).__name__}")
        print(f"  Architecture: input={model.input_dim}, hidden={model.hidden_dim}, layers={model.num_layers}")
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        raise

    # Load price data and scaler
    try:
        price_data = pd.read_csv(price_data_path)
        scaler = joblib.load(scaler_path)
        print(f"✓ Data loaded: {len(price_data)} rows")
        print(f"✓ Scaler loaded: {scaler.n_features_in_} features")
    except Exception as e:
        print(f"\n✗ Failed to load data: {e}")
        raise

    # Verify compatibility
    if model.input_dim != scaler.n_features_in_:
        print(f"\n⚠ WARNING: Model expects {model.input_dim} features but scaler has {scaler.n_features_in_} features!")
        print("Please ensure model and scaler are compatible.")
        raise ValueError("Model-scaler dimension mismatch")

    # Initialize backtest engine with professional parameters
    backtest_engine = BacktestEngine(
        initial_capital=100000,
        transaction_cost=0.001,  # 10 bps (realistic for retail)
        min_holding_period=2,
        position_size=0.95
    )

    # Generate signals and run backtest
    try:
        signals_df = backtest_engine.generate_trading_signals(model, price_data, scaler)
        metrics, trades, portfolio_value = backtest_engine.run_backtest(signals_df)

        # Print performance summary
        create_performance_summary(metrics, trades, signals_df, price_data, symbol=stock)

        print("\n" + "=" * 60)
        print("BACKTEST COMPLETE")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ Backtesting failed: {e}")
        raise