import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import warnings
import joblib
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')


# ==================== Configuration ====================

@dataclass
class BacktestConfig:
    """Configuration for improved backtesting with risk management."""

    # Capital and position sizing
    initial_capital: float = 100000
    base_position_size: float = 0.65  # Reduced from 0.95 to 0.65 for safety
    max_position_size: float = 0.70   # Maximum position size cap
    min_position_size: float = 0.30   # Minimum position size floor

    # Transaction costs
    transaction_cost: float = 0.001  # 10 bps (0.1%)

    # Trading thresholds (asymmetric for bias reduction)
    buy_threshold: float = 0.02      # 2% predicted gain to buy (lower than sell)
    sell_threshold: float = 0.025    # 2.5% predicted loss to sell (higher threshold)
    hold_threshold: float = 0.015    # 1.5% neutral zone for holding

    # Holding periods
    min_holding_period: int = 5      # Increased from 2 to 5 days
    max_holding_period: int = 60     # Force exit after 60 days

    # Risk management - Stop loss & Take profit
    stop_loss_pct: float = 0.08      # 8% stop loss
    trailing_stop_pct: float = 0.10  # 10% trailing stop from peak
    take_profit_pct: float = 0.15    # 15% take profit target
    partial_profit_pct: float = 0.10 # 10% partial profit taking

    # Drawdown controls
    max_drawdown_limit: float = 0.20 # 20% max drawdown circuit breaker
    drawdown_scale_factor: float = 0.5  # Reduce position size by 50% in drawdown

    # Volatility-based adjustments
    volatility_lookback: int = 20    # Days to calculate volatility
    high_volatility_threshold: float = 0.03  # 3% daily volatility = high
    low_volatility_threshold: float = 0.01   # 1% daily volatility = low

    # Trend following
    use_trend_filter: bool = True
    trend_ma_short: int = 50         # Short-term trend (50-day MA)
    trend_ma_long: int = 200         # Long-term trend (200-day MA)

    # Regime detection
    use_regime_detection: bool = True
    regime_lookback: int = 60        # Days for regime classification


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


# ==================== Technical Indicators ====================

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators matching the training model."""
    df = df.copy()

    # Price-based features
    df['returns'] = df['close'].pct_change()

    # Moving averages
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()

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

    # Volatility (20-day rolling)
    df['volatility'] = df['returns'].rolling(window=20).std()

    # Rate of Change
    df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100

    return df


def detect_market_regime(df: pd.DataFrame, lookback: int = 60) -> MarketRegime:
    """
    Detect current market regime based on recent price action.

    Returns:
        MarketRegime: Current market classification
    """
    if len(df) < lookback:
        return MarketRegime.SIDEWAYS

    recent_data = df.iloc[-lookback:]
    returns = recent_data['returns'].dropna()

    if len(returns) < 2:
        return MarketRegime.SIDEWAYS

    # Calculate metrics
    mean_return = returns.mean()
    volatility = returns.std()

    # Check for high volatility
    if volatility > 0.03:  # 3% daily volatility
        return MarketRegime.VOLATILE

    # Check for trend
    if mean_return > 0.001:  # Positive drift
        return MarketRegime.BULL
    elif mean_return < -0.001:  # Negative drift
        return MarketRegime.BEAR
    else:
        return MarketRegime.SIDEWAYS


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
    """Infer model dimensions from state dictionary."""
    try:
        weight_ih_key = 'lstm.weight_ih_l0'

        if weight_ih_key not in state_dict:
            raise KeyError(f"Cannot find {weight_ih_key} in state dict")

        weight_ih = state_dict[weight_ih_key]

        four_times_hidden = weight_ih.shape[0]
        input_dim = weight_ih.shape[1]
        hidden_dim = four_times_hidden // 4

        num_layers = 0
        for key in state_dict.keys():
            if 'lstm.weight_ih_l' in key:
                layer_num = int(key.split('_l')[1])
                num_layers = max(num_layers, layer_num + 1)

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


def load_model_safe(model_path: Path, default_config: Optional[Dict] = None) -> nn.Module:
    """Safely load a PyTorch model from various checkpoint formats."""

    print(f"\nLoading model from: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

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

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            saved_config = checkpoint.get('model_config', None)
        else:
            state_dict = checkpoint
            saved_config = None

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

        print(f"Creating model with config: {config}")
        model = ImprovedLSTM(
            input_dim=config.get('input_dim', 12),
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.2),
            output_dim=config.get('output_dim', 1)
        )

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


# ==================== Improved Backtesting Engine ====================

class ImprovedBacktestEngine:
    """
    Professional backtesting engine with advanced risk management.

    Features:
    - Asymmetric buy/sell thresholds (reduces bear bias)
    - Volatility-based position sizing
    - Stop-loss and take-profit mechanisms
    - Trailing stop-loss
    - Maximum drawdown circuit breaker
    - Trend filtering
    - Market regime detection
    - Enhanced performance tracking
    """

    def __init__(self, config: BacktestConfig = None):
        """Initialize improved backtesting engine."""
        self.config = config if config else BacktestConfig()

        # Track state
        self.peak_portfolio_value = self.config.initial_capital
        self.current_drawdown = 0.0
        self.in_drawdown_protection = False

        # Performance tracking
        self.trade_log = []
        self.daily_log = []

    def calculate_position_size(self, current_volatility: float, regime: MarketRegime) -> float:
        """
        Calculate dynamic position size based on volatility and market regime.

        Args:
            current_volatility: Current market volatility
            regime: Current market regime

        Returns:
            Position size as fraction of capital
        """
        base_size = self.config.base_position_size

        # Adjust for volatility
        if current_volatility > self.config.high_volatility_threshold:
            volatility_multiplier = 0.7  # Reduce size in high volatility
        elif current_volatility < self.config.low_volatility_threshold:
            volatility_multiplier = 1.1  # Increase size in low volatility
        else:
            volatility_multiplier = 1.0

        # Adjust for market regime
        if regime == MarketRegime.VOLATILE:
            regime_multiplier = 0.6  # Very conservative in volatile markets
        elif regime == MarketRegime.BEAR:
            regime_multiplier = 0.8  # Somewhat conservative in bear markets
        else:
            regime_multiplier = 1.0

        # Adjust for drawdown
        if self.in_drawdown_protection:
            drawdown_multiplier = self.config.drawdown_scale_factor
        else:
            drawdown_multiplier = 1.0

        # Calculate final position size
        position_size = base_size * volatility_multiplier * regime_multiplier * drawdown_multiplier

        # Apply bounds
        position_size = max(self.config.min_position_size,
                          min(self.config.max_position_size, position_size))

        return position_size

    def check_trend_alignment(self, current_price: float, sma_50: float,
                             sma_200: float, signal: int) -> bool:
        """
        Check if trade signal aligns with overall trend.

        Args:
            current_price: Current stock price
            sma_50: 50-day simple moving average
            sma_200: 200-day simple moving average
            signal: Trading signal (1=buy, -1=sell, 0=hold)

        Returns:
            True if signal aligns with trend, False otherwise
        """
        if not self.config.use_trend_filter:
            return True

        # Determine trend
        if current_price > sma_50 > sma_200:
            trend = "uptrend"
        elif current_price < sma_50 < sma_200:
            trend = "downtrend"
        else:
            trend = "neutral"

        # In uptrend: prefer buys, discourage sells
        if trend == "uptrend":
            return signal >= 0  # Allow buy and hold, block sell

        # In downtrend: allow sells, be cautious with buys
        elif trend == "downtrend":
            return signal <= 0  # Allow sell and hold, block buy

        # Neutral: allow all signals
        else:
            return True

    def generate_trading_signals(self,
                                 model: nn.Module,
                                 historical_data: pd.DataFrame,
                                 scaler,
                                 sequence_length: int = 60) -> pd.DataFrame:
        """
        Generate trading signals with improved thresholds and bias reduction.

        Args:
            model: Trained PyTorch model
            historical_data: DataFrame with price data
            scaler: Fitted scaler for feature normalization
            sequence_length: Lookback window size

        Returns:
            DataFrame with signals, predictions, and metadata
        """

        if not isinstance(model, nn.Module):
            raise TypeError(f"Model must be nn.Module, got {type(model)}")

        print("\n" + "=" * 70)
        print("GENERATING TRADING SIGNALS (IMPROVED)")
        print("=" * 70)

        # Add technical indicators
        data_with_indicators = calculate_technical_indicators(historical_data)
        data_with_indicators = data_with_indicators.dropna().reset_index(drop=True)

        # Define feature columns (must match training)
        feature_columns = ['close', 'returns', 'sma_5', 'sma_20', 'rsi',
                          'macd', 'macd_signal', 'bb_position', 'bb_width',
                          'momentum', 'volatility', 'roc']

        feature_columns = [col for col in feature_columns if col in data_with_indicators.columns]

        print(f"\nUsing {len(feature_columns)} features:")
        for i, col in enumerate(feature_columns, 1):
            print(f"  {i}. {col}")

        # Extract feature values
        feature_data = data_with_indicators[feature_columns].values

        if feature_data.shape[1] != scaler.n_features_in_:
            raise ValueError(
                f"Feature mismatch! Data has {feature_data.shape[1]} features "
                f"but scaler expects {scaler.n_features_in_}")

        signals_df = pd.DataFrame(index=range(sequence_length, len(feature_data)))

        predictions = []
        current_prices = []
        signals = []
        signal_strengths = []
        regimes = []

        print(f"\nGenerating predictions for {len(feature_data) - sequence_length} time steps...")

        model.eval()

        with torch.no_grad():
            for i in range(sequence_length, len(feature_data)):
                # Prepare input sequence
                sequence = feature_data[i - sequence_length:i]
                scaled_sequence = scaler.transform(sequence)

                # Convert to tensor
                input_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)

                # Generate prediction
                scaled_pred = model(input_tensor).cpu().numpy()

                # Inverse transform prediction
                n_features = scaler.n_features_in_
                dummy_pred = np.zeros((1, n_features))
                dummy_pred[:, 0] = scaled_pred.flatten()
                prediction = scaler.inverse_transform(dummy_pred)[0, 0]

                current_price = data_with_indicators['close'].iloc[i]
                current_volatility = data_with_indicators['volatility'].iloc[i]
                sma_50 = data_with_indicators['sma_50'].iloc[i] if 'sma_50' in data_with_indicators else current_price
                sma_200 = data_with_indicators['sma_200'].iloc[i] if 'sma_200' in data_with_indicators else current_price

                predictions.append(prediction)
                current_prices.append(current_price)

                # Detect market regime
                regime = detect_market_regime(
                    data_with_indicators.iloc[:i+1],
                    self.config.regime_lookback
                )
                regimes.append(regime.value)

                # Calculate expected return
                expected_return = (prediction - current_price) / current_price
                signal_strength = abs(expected_return)
                signal_strengths.append(signal_strength)

                # IMPROVED SIGNAL GENERATION with asymmetric thresholds
                if expected_return > self.config.buy_threshold:
                    preliminary_signal = 1  # Buy
                elif expected_return < -self.config.sell_threshold:
                    preliminary_signal = -1  # Sell
                elif abs(expected_return) < self.config.hold_threshold:
                    preliminary_signal = 0  # Hold (neutral zone)
                else:
                    # In between - default to hold
                    preliminary_signal = 0

                # Apply trend filter
                if self.config.use_trend_filter:
                    trend_aligned = self.check_trend_alignment(
                        current_price, sma_50, sma_200, preliminary_signal
                    )
                    if not trend_aligned:
                        preliminary_signal = 0  # Override to hold if not trend-aligned

                signals.append(preliminary_signal)

        # Create output DataFrame
        signals_df['Price'] = current_prices
        signals_df['Prediction'] = predictions
        signals_df['Signal'] = signals
        signals_df['Expected_Return'] = [(p - c) / c for p, c in zip(predictions, current_prices)]
        signals_df['Signal_Strength'] = signal_strengths
        signals_df['Regime'] = regimes

        # Add volatility
        signals_df['Volatility'] = data_with_indicators['volatility'].iloc[sequence_length:].values

        # Add trend indicators
        if 'sma_50' in data_with_indicators.columns:
            signals_df['SMA_50'] = data_with_indicators['sma_50'].iloc[sequence_length:].values
        if 'sma_200' in data_with_indicators.columns:
            signals_df['SMA_200'] = data_with_indicators['sma_200'].iloc[sequence_length:].values

        print(f"\n✓ Generated {len(signals_df)} trading signals")
        print(f"  Buy signals:  {sum(s == 1 for s in signals):4d} ({sum(s == 1 for s in signals)/len(signals)*100:.1f}%)")
        print(f"  Hold signals: {sum(s == 0 for s in signals):4d} ({sum(s == 0 for s in signals)/len(signals)*100:.1f}%)")
        print(f"  Sell signals: {sum(s == -1 for s in signals):4d} ({sum(s == -1 for s in signals)/len(signals)*100:.1f}%)")

        print(f"\nMarket Regime Distribution:")
        regime_counts = pd.Series(regimes).value_counts()
        for regime, count in regime_counts.items():
            print(f"  {regime:10s}: {count:4d} ({count/len(regimes)*100:.1f}%)")

        return signals_df

    def run_backtest(self, signals_df: pd.DataFrame) -> Tuple[Dict, list, list]:
        """
        Run backtest with advanced risk management.

        Includes:
        - Stop-loss and take-profit
        - Trailing stop
        - Volatility-based position sizing
        - Maximum drawdown protection
        - Extended holding periods

        Returns:
            Tuple of (metrics dict, trades list, portfolio values list)
        """

        print("\n" + "=" * 70)
        print("RUNNING IMPROVED BACKTEST")
        print("=" * 70)

        capital = self.config.initial_capital
        position = 0  # 0: no position, 1: long
        trades = []
        portfolio_values = []

        # Position tracking
        entry_price = 0
        shares = 0
        last_trade_day = -self.config.min_holding_period
        days_in_position = 0
        position_peak_price = 0  # For trailing stop

        # Risk management counters
        stop_loss_exits = 0
        take_profit_exits = 0
        trailing_stop_exits = 0
        max_holding_exits = 0
        signal_exits = 0

        for day_index, (idx, row) in enumerate(signals_df.iterrows()):
            current_price = row['Price']
            current_volatility = row.get('Volatility', 0.02)
            regime = MarketRegime(row.get('Regime', 'sideways'))

            # Calculate current portfolio value
            if position == 1:
                unrealized_pnl = (current_price - entry_price) * shares
                current_value = capital + unrealized_pnl
                position_peak_price = max(position_peak_price, current_price)
            else:
                current_value = capital

            portfolio_values.append(current_value)

            # Update peak and drawdown tracking
            if current_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_value
                self.in_drawdown_protection = False

            self.current_drawdown = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value

            # Check drawdown circuit breaker
            if self.current_drawdown > self.config.max_drawdown_limit:
                if not self.in_drawdown_protection:
                    print(f"\n⚠ WARNING: Maximum drawdown limit reached ({self.current_drawdown*100:.2f}%)")
                    print("  Activating drawdown protection mode")
                    self.in_drawdown_protection = True

                # Force exit if in position
                if position == 1:
                    pnl = (current_price - entry_price) * shares
                    trade_cost = abs(shares * current_price * self.config.transaction_cost)
                    capital += pnl - trade_cost

                    trades.append({
                        'Date': idx,
                        'Type': 'CLOSE',
                        'Reason': 'DRAWDOWN_LIMIT',
                        'Price': current_price,
                        'Shares': shares,
                        'PnL': pnl - trade_cost,
                        'Days_Held': days_in_position,
                        'Return_Pct': (current_price / entry_price - 1) * 100
                    })

                    position = 0
                    shares = 0
                    days_in_position = 0
                    last_trade_day = day_index
                    continue

            # Check if we can trade (minimum holding period)
            can_trade = (day_index - last_trade_day) >= self.config.min_holding_period

            # RISK MANAGEMENT: Check stop-loss and take-profit for open positions
            if position == 1:
                days_in_position += 1

                position_return = (current_price / entry_price) - 1
                trailing_from_peak = (position_peak_price - current_price) / position_peak_price

                exit_triggered = False
                exit_reason = ""

                # 1. Stop-loss check
                if position_return < -self.config.stop_loss_pct:
                    exit_triggered = True
                    exit_reason = "STOP_LOSS"
                    stop_loss_exits += 1

                # 2. Take-profit check
                elif position_return > self.config.take_profit_pct:
                    exit_triggered = True
                    exit_reason = "TAKE_PROFIT"
                    take_profit_exits += 1

                # 3. Trailing stop check
                elif trailing_from_peak > self.config.trailing_stop_pct:
                    exit_triggered = True
                    exit_reason = "TRAILING_STOP"
                    trailing_stop_exits += 1

                # 4. Maximum holding period check
                elif days_in_position >= self.config.max_holding_period:
                    exit_triggered = True
                    exit_reason = "MAX_HOLDING"
                    max_holding_exits += 1

                # Execute exit if triggered
                if exit_triggered and can_trade:
                    pnl = (current_price - entry_price) * shares
                    trade_cost = abs(shares * current_price * self.config.transaction_cost)
                    capital += pnl - trade_cost

                    trades.append({
                        'Date': idx,
                        'Type': 'CLOSE',
                        'Reason': exit_reason,
                        'Price': current_price,
                        'Shares': shares,
                        'PnL': pnl - trade_cost,
                        'Days_Held': days_in_position,
                        'Return_Pct': position_return * 100
                    })

                    position = 0
                    shares = 0
                    days_in_position = 0
                    last_trade_day = day_index
                    continue

            # SIGNAL-BASED TRADING
            if row['Signal'] != position and can_trade:
                # Close existing position
                if position != 0:
                    pnl = (current_price - entry_price) * shares * position
                    trade_cost = abs(shares * current_price * self.config.transaction_cost)
                    capital += pnl - trade_cost

                    trades.append({
                        'Date': idx,
                        'Type': 'CLOSE',
                        'Reason': 'SIGNAL',
                        'Price': current_price,
                        'Shares': shares,
                        'PnL': pnl - trade_cost,
                        'Days_Held': days_in_position,
                        'Return_Pct': (current_price / entry_price - 1) * 100 * position
                    })

                    signal_exits += 1

                # Open new position (only long positions for now)
                if row['Signal'] == 1:  # Buy signal
                    position = 1
                    entry_price = current_price

                    # DYNAMIC POSITION SIZING
                    dynamic_position_size = self.calculate_position_size(current_volatility, regime)
                    shares = int(capital * dynamic_position_size / current_price)

                    if shares > 0:
                        trade_cost = abs(shares * current_price * self.config.transaction_cost)
                        capital -= trade_cost
                        last_trade_day = day_index
                        days_in_position = 0
                        position_peak_price = current_price

                        trades.append({
                            'Date': idx,
                            'Type': 'OPEN',
                            'Signal': position,
                            'Price': entry_price,
                            'Shares': shares,
                            'Position_Size': dynamic_position_size,
                            'Volatility': current_volatility,
                            'Regime': regime.value,
                            'Cost': trade_cost
                        })
                    else:
                        position = 0
                else:
                    # Sell signal or hold - go to cash
                    position = 0
                    shares = 0
                    days_in_position = 0
                    last_trade_day = day_index

        # Close any remaining position at end
        if position == 1 and len(signals_df) > 0:
            final_price = signals_df['Price'].iloc[-1]
            pnl = (final_price - entry_price) * shares
            trade_cost = abs(shares * final_price * self.config.transaction_cost)
            capital += pnl - trade_cost

            trades.append({
                'Date': signals_df.index[-1],
                'Type': 'CLOSE',
                'Reason': 'END_OF_BACKTEST',
                'Price': final_price,
                'Shares': shares,
                'PnL': pnl - trade_cost,
                'Days_Held': days_in_position,
                'Return_Pct': (final_price / entry_price - 1) * 100
            })

        # Calculate performance metrics
        if len(portfolio_values) > 0:
            final_value = portfolio_values[-1]
            total_return = (final_value / self.config.initial_capital - 1) * 100

            portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()

            if len(portfolio_returns) > 0 and portfolio_returns.std() > 0:
                sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0

            cumulative = pd.Series(portfolio_values) / self.config.initial_capital
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min()) * 100

            # Calculate Sortino Ratio (downside deviation)
            negative_returns = portfolio_returns[portfolio_returns < 0]
            if len(negative_returns) > 0 and negative_returns.std() > 0:
                sortino_ratio = portfolio_returns.mean() / negative_returns.std() * np.sqrt(252)
            else:
                sortino_ratio = 0

            # Calculate Calmar Ratio (return / max drawdown)
            calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0

        else:
            final_value = self.config.initial_capital
            total_return = 0
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
            calmar_ratio = 0

        metrics = {
            'Total_Return': total_return,
            'Initial_Capital': self.config.initial_capital,
            'Final_Value': final_value,
            'Number_of_Trades': len([t for t in trades if t['Type'] == 'CLOSE']),
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Max_Drawdown': max_drawdown,
            'Calmar_Ratio': calmar_ratio,

            # Exit reason breakdown
            'Stop_Loss_Exits': stop_loss_exits,
            'Take_Profit_Exits': take_profit_exits,
            'Trailing_Stop_Exits': trailing_stop_exits,
            'Max_Holding_Exits': max_holding_exits,
            'Signal_Exits': signal_exits
        }

        print(f"\n✓ Backtest complete")
        print(f"  Total trades executed: {metrics['Number_of_Trades']}")
        print(f"  Final portfolio value: ${final_value:,.2f}")
        print(f"  Total return: {total_return:.2f}%")
        print(f"\nExit Reasons:")
        print(f"  Stop-loss:      {stop_loss_exits:3d}")
        print(f"  Take-profit:    {take_profit_exits:3d}")
        print(f"  Trailing stop:  {trailing_stop_exits:3d}")
        print(f"  Max holding:    {max_holding_exits:3d}")
        print(f"  Signal change:  {signal_exits:3d}")

        return metrics, trades, portfolio_values


def create_performance_summary(metrics: Dict, trades: list, signals_df: pd.DataFrame,
                               config: BacktestConfig, symbol: str = "STOCK"):
    """Create comprehensive performance summary report with improved metrics."""

    summary_lines = []
    summary_lines.append("\n" + "=" * 80)
    summary_lines.append("                  IMPROVED BACKTESTING PERFORMANCE SUMMARY")
    summary_lines.append("=" * 80)

    # Trading period
    if len(signals_df) > 0:
        start_date = signals_df.index[0]
        end_date = signals_df.index[-1]
        start_price = signals_df['Price'].iloc[0]
        end_price = signals_df['Price'].iloc[-1]
        price_change = ((end_price - start_price) / start_price * 100) if start_price > 0 else 0
    else:
        start_date = "N/A"
        end_date = "N/A"
        start_price = 0
        end_price = 0
        price_change = 0

    summary_lines.append(f"\nStock: {symbol}")
    summary_lines.append(f"Trading Period: {len(signals_df)} days")
    summary_lines.append(f"Start Price: ${start_price:.2f}")
    summary_lines.append(f"End Price: ${end_price:.2f}")
    summary_lines.append(f"Buy & Hold Return: {price_change:.2f}%")

    summary_lines.append(f"\n{'─' * 80}")
    summary_lines.append("STRATEGY PERFORMANCE")
    summary_lines.append(f"{'─' * 80}")

    summary_lines.append(f"Total Return:           {metrics['Total_Return']:>10.2f}%")
    summary_lines.append(f"Initial Capital:        ${metrics.get('Initial_Capital', 100000):>10,.2f}")
    summary_lines.append(f"Final Portfolio Value:  ${metrics['Final_Value']:>10,.2f}")
    summary_lines.append(f"Absolute Gain/Loss:     ${metrics['Final_Value'] - metrics.get('Initial_Capital', 100000):>10,.2f}")
    summary_lines.append(f"")
    summary_lines.append(f"Sharpe Ratio:           {metrics['Sharpe_Ratio']:>10.3f}")
    summary_lines.append(f"Sortino Ratio:          {metrics['Sortino_Ratio']:>10.3f}")
    summary_lines.append(f"Calmar Ratio:           {metrics['Calmar_Ratio']:>10.3f}")
    summary_lines.append(f"Maximum Drawdown:       {metrics['Max_Drawdown']:>10.2f}%")
    summary_lines.append(f"Number of Trades:       {metrics['Number_of_Trades']:>10d}")

    # Strategy vs Buy & Hold
    strategy_outperformance = metrics['Total_Return'] - price_change
    summary_lines.append(f"\n{'─' * 80}")
    summary_lines.append("STRATEGY VS BUY & HOLD")
    summary_lines.append(f"{'─' * 80}")
    summary_lines.append(f"Outperformance:         {strategy_outperformance:>10.2f}%")

    if strategy_outperformance > 0:
        summary_lines.append(f"Result: ✓ Strategy OUTPERFORMED buy & hold")
    else:
        summary_lines.append(f"Result: ✗ Strategy UNDERPERFORMED buy & hold")

    # Signal statistics
    if len(signals_df) > 0:
        signal_counts = signals_df['Signal'].value_counts()
        total_signals = len(signals_df)

        summary_lines.append(f"\n{'─' * 80}")
        summary_lines.append("SIGNAL DISTRIBUTION")
        summary_lines.append(f"{'─' * 80}")
        summary_lines.append(f"Buy Signals:      {signal_counts.get(1, 0):5d} ({signal_counts.get(1, 0) / total_signals * 100:5.1f}%)")
        summary_lines.append(f"Hold Signals:     {signal_counts.get(0, 0):5d} ({signal_counts.get(0, 0) / total_signals * 100:5.1f}%)")
        summary_lines.append(f"Sell Signals:     {signal_counts.get(-1, 0):5d} ({signal_counts.get(-1, 0) / total_signals * 100:5.1f}%)")

    # Trade analysis
    if trades:
        close_trades = [t for t in trades if t['Type'] == 'CLOSE' and 'PnL' in t]

        if close_trades:
            profitable_trades = [t for t in close_trades if t['PnL'] > 0]
            losing_trades = [t for t in close_trades if t['PnL'] < 0]

            win_rate = len(profitable_trades) / len(close_trades) * 100
            avg_trade_pnl = np.mean([t['PnL'] for t in close_trades])

            summary_lines.append(f"\n{'─' * 80}")
            summary_lines.append("TRADE ANALYSIS")
            summary_lines.append(f"{'─' * 80}")
            summary_lines.append(f"Win Rate:             {win_rate:>10.1f}%")
            summary_lines.append(f"Average Trade P&L:    ${avg_trade_pnl:>10,.2f}")

            if profitable_trades:
                avg_profit = np.mean([t['PnL'] for t in profitable_trades])
                max_profit = max([t['PnL'] for t in profitable_trades])
                avg_win_days = np.mean([t.get('Days_Held', 0) for t in profitable_trades])
                summary_lines.append(f"Average Profit:       ${avg_profit:>10,.2f}")
                summary_lines.append(f"Best Trade:           ${max_profit:>10,.2f}")
                summary_lines.append(f"Avg Winning Hold:     {avg_win_days:>10.1f} days")

            if losing_trades:
                avg_loss = np.mean([t['PnL'] for t in losing_trades])
                max_loss = min([t['PnL'] for t in losing_trades])
                avg_loss_days = np.mean([t.get('Days_Held', 0) for t in losing_trades])
                summary_lines.append(f"Average Loss:         ${avg_loss:>10,.2f}")
                summary_lines.append(f"Worst Trade:          ${max_loss:>10,.2f}")
                summary_lines.append(f"Avg Losing Hold:      {avg_loss_days:>10.1f} days")

            # Profit factor
            total_profit = sum([t['PnL'] for t in profitable_trades]) if profitable_trades else 0
            total_loss = abs(sum([t['PnL'] for t in losing_trades])) if losing_trades else 1
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
            summary_lines.append(f"Profit Factor:        {profit_factor:>10.2f}")

    # Risk management statistics
    summary_lines.append(f"\n{'─' * 80}")
    summary_lines.append("RISK MANAGEMENT")
    summary_lines.append(f"{'─' * 80}")
    summary_lines.append(f"Stop-Loss Exits:      {metrics.get('Stop_Loss_Exits', 0):>10d}")
    summary_lines.append(f"Take-Profit Exits:    {metrics.get('Take_Profit_Exits', 0):>10d}")
    summary_lines.append(f"Trailing Stop Exits:  {metrics.get('Trailing_Stop_Exits', 0):>10d}")
    summary_lines.append(f"Max Holding Exits:    {metrics.get('Max_Holding_Exits', 0):>10d}")
    summary_lines.append(f"Signal Exits:         {metrics.get('Signal_Exits', 0):>10d}")

    # Configuration summary
    summary_lines.append(f"\n{'─' * 80}")
    summary_lines.append("CONFIGURATION")
    summary_lines.append(f"{'─' * 80}")
    summary_lines.append(f"Buy Threshold:        {config.buy_threshold*100:>10.2f}%")
    summary_lines.append(f"Sell Threshold:       {config.sell_threshold*100:>10.2f}%")
    summary_lines.append(f"Stop Loss:            {config.stop_loss_pct*100:>10.2f}%")
    summary_lines.append(f"Take Profit:          {config.take_profit_pct*100:>10.2f}%")
    summary_lines.append(f"Trailing Stop:        {config.trailing_stop_pct*100:>10.2f}%")
    summary_lines.append(f"Base Position Size:   {config.base_position_size*100:>10.2f}%")
    summary_lines.append(f"Min Holding Period:   {config.min_holding_period:>10d} days")
    summary_lines.append(f"Max Holding Period:   {config.max_holding_period:>10d} days")

    summary_lines.append("=" * 80)

    for line in summary_lines:
        print(line)


# ==================== Main Execution ====================

if __name__ == "__main__":

    stock = "MSFT"  # Change to your stock symbol
    data_directory = Path(__file__).resolve().parent.parent
    model_path = data_directory / "models" / f"{stock}_optimized_model.pth"
    price_data_path = data_directory / "data" / f"{stock}.csv"
    scaler_path = data_directory / "models" / f"{stock}_scaler_optimized.pkl"

    print("\n" + "=" * 80)
    print(f"IMPROVED LSTM BACKTESTING SYSTEM - {stock}")
    print("=" * 80)
    print(f"\nModel path: {model_path}")
    print(f"Data path: {price_data_path}")
    print(f"Scaler path: {scaler_path}")

    # Default model configuration
    default_config = {
        'input_dim': 12,
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'output_dim': 1
    }

    # Load model
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
        raise ValueError("Model-scaler dimension mismatch")

    # Create improved backtest configuration
    config = BacktestConfig(
        initial_capital=100000,
        base_position_size=0.65,
        transaction_cost=0.001,

        # Asymmetric thresholds
        buy_threshold=0.02,      # 2% for buying
        sell_threshold=0.025,    # 2.5% for selling
        hold_threshold=0.015,    # 1.5% neutral zone

        # Holding periods
        min_holding_period=5,
        max_holding_period=60,

        # Risk management
        stop_loss_pct=0.08,
        trailing_stop_pct=0.10,
        take_profit_pct=0.15,

        # Drawdown protection
        max_drawdown_limit=0.20,

        # Trend filtering
        use_trend_filter=True,
        use_regime_detection=True
    )

    print(f"\n{'─' * 80}")
    print("BACKTEST CONFIGURATION")
    print(f"{'─' * 80}")
    print(f"Buy Threshold:      {config.buy_threshold*100:.1f}%")
    print(f"Sell Threshold:     {config.sell_threshold*100:.1f}%")
    print(f"Position Size:      {config.base_position_size*100:.1f}%")
    print(f"Stop Loss:          {config.stop_loss_pct*100:.1f}%")
    print(f"Take Profit:        {config.take_profit_pct*100:.1f}%")
    print(f"Trailing Stop:      {config.trailing_stop_pct*100:.1f}%")
    print(f"Min Holding:        {config.min_holding_period} days")
    print(f"Trend Filter:       {'Enabled' if config.use_trend_filter else 'Disabled'}")
    print(f"Regime Detection:   {'Enabled' if config.use_regime_detection else 'Disabled'}")

    # Initialize improved backtest engine
    backtest_engine = ImprovedBacktestEngine(config=config)

    # Generate signals and run backtest
    try:
        signals_df = backtest_engine.generate_trading_signals(model, price_data, scaler)
        metrics, trades, portfolio_values = backtest_engine.run_backtest(signals_df)

        # Print performance summary
        create_performance_summary(metrics, trades, signals_df, config, symbol=stock)

        # Save results
        results_dir = data_directory / "analysis" / "backtesting_result"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save signals
        signals_output = results_dir / f"{stock}_improved_signals.csv"
        signals_df.to_csv(signals_output)
        print(f"\n✓ Signals saved to: {signals_output}")

        # Save trades
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_output = results_dir / f"{stock}_improved_trades.csv"
            trades_df.to_csv(trades_output, index=False)
            print(f"✓ Trades saved to: {trades_output}")

        # Save portfolio values
        portfolio_df = pd.DataFrame({
            'Day': range(len(portfolio_values)),
            'Portfolio_Value': portfolio_values
        })
        portfolio_output = results_dir / f"{stock}_improved_portfolio.csv"
        portfolio_df.to_csv(portfolio_output, index=False)
        print(f"✓ Portfolio values saved to: {portfolio_output}")

        print("\n" + "=" * 80)
        print("IMPROVED BACKTEST COMPLETE")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n✗ Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        raise
