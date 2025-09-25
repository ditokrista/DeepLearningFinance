import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from typing import Tuple, Dict, Optional
import warnings
import joblib
from pathlib import Path
warnings.filterwarnings('ignore')


# Add LSTM model class definition
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out




class TradingSignalGenerator:
    """
    Professional-grade trading signal generator for LSTM predictions.
    Implements industry-standard techniques for signal generation and risk management.
    """
    
    def __init__(self, 
                 lookback_window: int = 20,
                 confidence_level: float = 0.95,
                 transaction_cost: float = 0.001,  # 10 bps
                 min_holding_period: int = 1,
                 max_position: float = 1.0,
                 volatility_window: int = 20):
        """
        Initialize signal generator with risk parameters.
        
        Args:
            lookback_window: Period for calculating moving statistics
            confidence_level: Confidence level for prediction intervals
            transaction_cost: Transaction cost as fraction of trade value
            min_holding_period: Minimum holding period in days
            max_position: Maximum position size (1.0 = 100% of capital)
            volatility_window: Window for volatility calculation
        """
        self.lookback_window = lookback_window
        self.confidence_level = confidence_level
        self.transaction_cost = transaction_cost
        self.min_holding_period = min_holding_period
        self.max_position = max_position
        self.volatility_window = volatility_window
        
    def calculate_prediction_intervals(self, 
                                      predictions: np.ndarray,
                                      actual_values: np.ndarray,
                                      window: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals using rolling residuals.
        
        Returns:
            upper_bound, lower_bound: Confidence intervals for predictions
        """
        # Calculate rolling prediction errors
        errors = predictions - actual_values
        rolling_std = pd.Series(errors).rolling(window=window, min_periods=30).std()
        
        # Calculate z-score for confidence level
        z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
        
        # Calculate bounds
        upper_bound = predictions + z_score * rolling_std.values
        lower_bound = predictions - z_score * rolling_std.values
        
        return upper_bound, lower_bound
    
    def generate_base_signals(self,
                             current_price: float,
                             predicted_price: float,
                             upper_bound: float,
                             lower_bound: float) -> int:
        """
        Generate base trading signal based on prediction and confidence.
        
        Returns:
            1 for buy, -1 for sell, 0 for hold
        """
        expected_return = (predicted_price - current_price) / current_price
        
        # Adjust for transaction costs
        threshold = 2 * self.transaction_cost
        
        # Strong buy signal: predicted price above upper bound
        if predicted_price > current_price * (1 + threshold) and predicted_price > upper_bound:
            return 1
        
        # Strong sell signal: predicted price below lower bound  
        elif predicted_price < current_price * (1 - threshold) and predicted_price < lower_bound:
            return -1
        
        # Moderate signals based on expected return
        elif expected_return > threshold:
            return 1
        elif expected_return < -threshold:
            return -1
        else:
            return 0
    
    def calculate_kelly_position_size(self,
                                     expected_return: float,
                                     win_probability: float,
                                     volatility: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Kelly fraction = (p * b - q) / b
        where p = win probability, q = loss probability, b = win/loss ratio
        """
        if expected_return <= 0 or win_probability <= 0.5:
            return 0.0
        
        # Estimate win/loss ratio from expected return and volatility
        win_loss_ratio = abs(expected_return) / volatility
        
        # Kelly fraction
        kelly_fraction = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
        
        # Apply Kelly fraction scaling (typically use 25% of Kelly for safety)
        conservative_kelly = kelly_fraction * 0.25
        
        # Cap at maximum position
        return min(max(conservative_kelly, 0), self.max_position)
    
    def apply_regime_filter(self,
                           prices: pd.Series,
                           sma_short: int = 50,
                           sma_long: int = 200) -> str:
        """
        Determine market regime using moving averages.
        
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        sma_s = prices.rolling(window=sma_short).mean().iloc[-1]
        sma_l = prices.rolling(window=sma_long).mean().iloc[-1]
        current_price = prices.iloc[-1]
        
        if current_price > sma_s > sma_l:
            return 'bullish'
        elif current_price < sma_s < sma_l:
            return 'bearish'
        else:
            return 'neutral'
    
    def calculate_risk_metrics(self,
                              returns: pd.Series,
                              benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Calculate comprehensive risk metrics.
        """
        # Annualization factor (assuming daily returns)
        ann_factor = np.sqrt(252)
        
        metrics = {
            'annual_return': returns.mean() * 252,
            'annual_volatility': returns.std() * ann_factor,
            'sharpe_ratio': (returns.mean() / returns.std()) * ann_factor if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns),
            'var_95': np.percentile(returns, 5),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean()
        }
        
        # Calculate Information Ratio if benchmark provided
        if benchmark_returns is not None:
            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std() * ann_factor
            metrics['information_ratio'] = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def generate_trading_signals(self,
                                model,
                                historical_data: pd.DataFrame,
                                scaler,
                                sequence_length: int = 59) -> pd.DataFrame:
        """
        Generate complete trading signals with position sizing.
        
        Args:
            model: Trained LSTM model
            historical_data: DataFrame with price data
            scaler: Fitted MinMaxScaler
            sequence_length: Sequence length for LSTM input
            
        Returns:
            DataFrame with signals, positions, and metrics
        """
        signals_df = pd.DataFrame(index=historical_data.index[sequence_length:])
        
        # Pre-calculate volatility
        returns = historical_data['close'].pct_change()
        volatility = returns.rolling(window=self.volatility_window).std()
        
        predictions = []
        current_prices = []
        
        # Generate predictions for entire dataset
        for i in range(sequence_length, len(historical_data)):
            # Prepare input sequence
            sequence = historical_data['close'].iloc[i-sequence_length:i].values.reshape(-1, 1)
            scaled_sequence = scaler.transform(sequence)
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)
            
            # Generate prediction
            model.eval()
            with torch.no_grad():
                scaled_pred = model(input_tensor).numpy()
            
            # Inverse transform prediction
            prediction = scaler.inverse_transform(scaled_pred)[0, 0]
            predictions.append(prediction)
            current_prices.append(historical_data['close'].iloc[i])
        
        predictions = np.array(predictions)
        current_prices = np.array(current_prices)
        
        # Calculate prediction intervals
        actual_next_day = historical_data['close'].iloc[sequence_length+1:].values
        if len(actual_next_day) < len(predictions):
            actual_next_day = np.append(actual_next_day, predictions[-1])
        
        upper_bound, lower_bound = self.calculate_prediction_intervals(
            predictions, actual_next_day[:len(predictions)]
        )
        
        # Generate signals
        signals = []
        positions = []
        current_position = 0
        last_trade_idx = -self.min_holding_period
        
        for i in range(len(predictions)):
            # Check if we can trade (minimum holding period)
            can_trade = (i - last_trade_idx) >= self.min_holding_period
            
            # Get current market regime
            regime = self.apply_regime_filter(
                historical_data['close'].iloc[:sequence_length+i+1]
            )
            
            # Generate base signal
            base_signal = self.generate_base_signals(
                current_prices[i],
                predictions[i],
                upper_bound[i] if not np.isnan(upper_bound[i]) else predictions[i],
                lower_bound[i] if not np.isnan(lower_bound[i]) else predictions[i]
            )
            
            # Apply regime filter
            if regime == 'bearish' and base_signal > 0:
                base_signal = 0  # Don't go long in bearish regime
            elif regime == 'bullish' and base_signal < 0:
                base_signal = 0  # Don't go short in bullish regime
            
            # Calculate position size using Kelly
            expected_return = (predictions[i] - current_prices[i]) / current_prices[i]
            win_probability = 0.55  # This should be calibrated from backtest
            current_vol = volatility.iloc[sequence_length+i] if not np.isnan(volatility.iloc[sequence_length+i]) else 0.02
            
            if base_signal != 0 and can_trade:
                position_size = self.calculate_kelly_position_size(
                    abs(expected_return),
                    win_probability,
                    current_vol
                )
                new_position = base_signal * position_size
                
                # Check if trade is worth it after costs
                trade_cost = self.transaction_cost * abs(new_position - current_position)
                expected_profit = abs(expected_return) * position_size
                
                if expected_profit > trade_cost * 2:  # Require 2x cost coverage
                    current_position = new_position
                    last_trade_idx = i
                    signals.append(base_signal)
                else:
                    signals.append(0)
            else:
                signals.append(0)
            
            positions.append(current_position)
        
        # Create output DataFrame
        signals_df['Price'] = current_prices
        signals_df['Prediction'] = predictions
        signals_df['Signal'] = signals
        signals_df['Position'] = positions
        signals_df['Expected_Return'] = (predictions - current_prices) / current_prices
        signals_df['Upper_Bound'] = upper_bound
        signals_df['Lower_Bound'] = lower_bound
        
        return signals_df


# Example usage with your trained model
def implement_trading_strategy(model, df_ibm, scaler, look_back=60):
    """
    Complete implementation of trading strategy using trained LSTM model.
    """
    # Initialize signal generator
    signal_gen = TradingSignalGenerator(
        transaction_cost=0.001,  # 10 basis points
        min_holding_period=2,     # Hold for at least 2 days
        max_position=0.5,         # Maximum 50% position
        volatility_window=20      # 20-day volatility
    )
    
    # Generate trading signals
    signals_df = signal_gen.generate_trading_signals(
        model=model,
        historical_data=df_ibm,
        scaler=scaler,
        sequence_length=look_back-1
    )
    
    # Calculate strategy returns
    signals_df['Returns'] = df_ibm['close'].pct_change().shift(-1)  # Next day returns
    signals_df['Strategy_Returns'] = signals_df['Position'].shift(1) * signals_df['Returns']
    
    # Calculate cumulative performance
    signals_df['Cumulative_Returns'] = (1 + signals_df['Returns']).cumprod()
    signals_df['Cumulative_Strategy_Returns'] = (1 + signals_df['Strategy_Returns']).cumprod()
    
    # Calculate risk metrics
    risk_metrics = signal_gen.calculate_risk_metrics(
        signals_df['Strategy_Returns'].dropna(),
        signals_df['Returns'].dropna()
    )
    
    return signals_df, risk_metrics



# Main execution function
def generate_live_signal(model, current_data, scaler, signal_generator, look_back=60):
    """
    Generate real-time trading signal for today's trading decision.
    
    Args:
        model: Trained LSTM model
        current_data: Recent price history (at least look_back days)
        scaler: Fitted scaler
        signal_generator: Initialized TradingSignalGenerator
        look_back: Sequence length for model
        
    Returns:
        dict: Trading decision with confidence and risk metrics
    """
    # Ensure we have enough data
    if len(current_data) < look_back:
        return {'signal': 'HOLD', 'reason': 'Insufficient data'}
    
    # Prepare the most recent sequence
    recent_sequence = current_data['close'].iloc[-look_back+1:].values.reshape(-1, 1)
    scaled_sequence = scaler.transform(recent_sequence)
    input_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)
    
    # Generate prediction
    model.eval()
    with torch.no_grad():
        scaled_pred = model(input_tensor).numpy()
    
    tomorrow_pred = scaler.inverse_transform(scaled_pred)[0, 0]
    today_price = current_data['close'].iloc[-1]
    
    # Calculate expected return
    expected_return = (tomorrow_pred - today_price) / today_price
    
    # Calculate recent volatility
    recent_returns = current_data['close'].pct_change().dropna()
    volatility = recent_returns.iloc[-20:].std()
    
    # Calculate confidence based on recent prediction accuracy
    # (In production, you'd track actual prediction accuracy)
    confidence = min(0.7, max(0.3, 1 - volatility * 10))  # Simple confidence heuristic
    
    # Generate signal
    signal_threshold = 2 * 0.001  # 2x transaction cost
    
    decision = {
        'date': current_data.index[-1],
        'current_price': today_price,
        'predicted_price': tomorrow_pred,
        'expected_return': expected_return * 100,  # As percentage
        'volatility': volatility * 100,
        'confidence': confidence,
        'signal': 'HOLD',
        'position_size': 0,
        'risk_metrics': {
            'value_at_risk_95': -1.65 * volatility * today_price,
            'expected_profit': expected_return * today_price * 1000,  # Assuming $1000 position
            'risk_reward_ratio': abs(expected_return) / volatility if volatility > 0 else 0
        }
    }
    
    # Determine signal
    if expected_return > signal_threshold and confidence > 0.5:
        decision['signal'] = 'BUY'
        decision['position_size'] = min(0.25, confidence * abs(expected_return) / volatility)
        decision['reasoning'] = f"Bullish: Expected return {expected_return:.2%} exceeds threshold with {confidence:.1%} confidence"
    elif expected_return < -signal_threshold and confidence > 0.5:
        decision['signal'] = 'SELL'
        decision['position_size'] = min(0.25, confidence * abs(expected_return) / volatility)
        decision['reasoning'] = f"Bearish: Expected return {expected_return:.2%} below threshold with {confidence:.1%} confidence"
    else:
        decision['reasoning'] = "Neutral: Insufficient edge after transaction costs or low confidence"
    
    return decision


# Example of how to use for today's decision
if __name__ == "__main__":
    # Assuming you have your trained model and data loaded
    data_directory = Path(__file__).parent.parent
    model_path = data_directory / "models" / "complete_lstm_model.pth"
    price_data_path = data_directory / "data" / "MSFT.csv" # Change file name to your desired stock data
    scaler_path = data_directory / "models" / "scaler.pkl"
    
    model = torch.load(model_path, weights_only=False)
    price_data = pd.read_csv(price_data_path)
    scaler = joblib.load(scaler_path)
    
    print("=" * 50)
    print(f"TRADING SIGNAL GENERATION SYSTEM - {price_data_path.stem}")
    print("=" * 50)
    
    # Initialize signal generator
    signal_gen = TradingSignalGenerator()
    
    # Generate today's signal (example with dummy data)
    # In production, replace with actual current data
    today_signal = generate_live_signal(model, price_data, scaler, signal_gen)
    
    print(f"\nToday's Trading Signal: {today_signal['signal']}")
    print(f"Position Size: {today_signal['position_size']:.2%}")
    print(f"Expected Return: {today_signal['expected_return']:.2%}")
    print(f"Confidence: {today_signal['confidence']:.1%}")
    print(f"Reasoning: {today_signal['reasoning']}")
    
    '''
    print("\nRisk Management Guidelines:")
    print("1. Never risk more than 2% of capital on a single trade")
    print("2. Use stop-loss at 2x ATR below entry for longs")
    print("3. Take partial profits at 1.5x expected return")
    print("4. Reduce position size in high volatility regimes")
    print("5. Monitor correlation with market beta")
    '''