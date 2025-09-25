import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from typing import Tuple, Dict, Optional
import warnings
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
warnings.filterwarnings('ignore')

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

class BacktestEngine:
    """Simple backtesting engine for LSTM trading signals."""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001,
                 volatility_window: int = 20,
                 min_holding_period: int = 1):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction
            volatility_window: Window for volatility calculation
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.volatility_window = volatility_window
        self.min_holding_period = min_holding_period

    def generate_trading_signals(self,
                                model,
                                historical_data: pd.DataFrame,
                                scaler,
                                sequence_length: int = 59) -> pd.DataFrame:
        """Generate simple trading signals based on LSTM predictions."""
        
        signals_df = pd.DataFrame(index=historical_data.index[sequence_length:])
        
        # Pre-calculate volatility
        returns = historical_data['close'].pct_change()
        volatility = returns.rolling(window=self.volatility_window).std()
        
        predictions = []
        current_prices = []
        signals = []
        
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
            current_price = historical_data['close'].iloc[i]
            
            predictions.append(prediction)
            current_prices.append(current_price)
            
            # Simple signal generation
            expected_return = (prediction - current_price) / current_price
            threshold = 2 * self.transaction_cost  # 2x transaction cost
            
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
        
        return signals_df

    def run_backtest(self, signals_df: pd.DataFrame) -> Tuple[Dict, list, list]:
            """Run simple backtest with transaction costs and minimum holding period."""
            
            capital = self.initial_capital
            position = 0  # Current position (0 = no position, 1 = long, -1 = short)
            trades = []
            portfolio_values = []
            entry_price = 0
            shares = 0
            last_trade_day = -self.min_holding_period  # Track when last trade occurred
            
            for day_index, (idx, row) in enumerate(signals_df.iterrows()):
                # Calculate current portfolio value
                if position == 1:  # Long position
                    current_value = capital + (row['Price'] - entry_price) * shares
                elif position == -1:  # Short position (simplified)
                    current_value = capital - (row['Price'] - entry_price) * shares
                else:  # No position
                    current_value = capital
                    
                portfolio_values.append(current_value)
                
                # Check if minimum holding period has passed
                can_trade = (day_index - last_trade_day) >= self.min_holding_period
                
                # Check for new signals (only if we can trade)
                if row['Signal'] != position and can_trade:  # Add can_trade condition
                    
                    # Close existing position if any
                    if position != 0:
                        pnl = (row['Price'] - entry_price) * shares * position
                        trade_cost = abs(shares * row['Price'] * self.transaction_cost)
                        capital += pnl - trade_cost
                        
                        trades.append({
                            'Date': idx,
                            'Type': 'CLOSE',
                            'Price': row['Price'],
                            'PnL': pnl - trade_cost,
                            'Days_Held': day_index - last_trade_day  # Track holding period
                        })
                    
                    # Open new position
                    if row['Signal'] != 0:
                        position = row['Signal']
                        entry_price = row['Price']
                        shares = int(capital * 0.95 / row['Price'])  # Use 95% of capital
                        
                        if shares > 0:  # Only trade if we can afford at least 1 share
                            trade_cost = abs(shares * row['Price'] * self.transaction_cost)
                            capital -= trade_cost
                            last_trade_day = day_index  # Update last trade day
                            
                            trades.append({
                                'Date': idx,
                                'Type': 'OPEN',
                                'Signal': position,
                                'Price': entry_price,
                                'Shares': shares,
                                'Cost': trade_cost
                            })
                        else:
                            position = 0  # Can't afford the trade
                    else:
                        position = 0
                        last_trade_day = day_index  # Update last trade day even for closing
        
            # Calculate final metrics
            if len(portfolio_values) > 0:
                total_return = (portfolio_values[-1] / self.initial_capital - 1) * 100
                portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
                
                if len(portfolio_returns) > 0 and portfolio_returns.std() > 0:
                    sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                else:
                    sharpe_ratio = 0
                    
                # Calculate max drawdown
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
            
            return metrics, trades, portfolio_values


def plot_backtest_results(signals_df: pd.DataFrame, 
                         portfolio_values: list, 
                         trades: list,
                         price_data: pd.DataFrame,
                         symbol: str = "Stock",
                         save_dir: str = "figures"):
    """
    Create comprehensive visualization of backtest results.
    
    Args:
        signals_df: DataFrame with predictions and signals
        portfolio_values: List of portfolio values over time
        trades: List of trade records
        price_data: Original price data with dates
        symbol: Stock symbol for title
        save_dir: Directory to save figures
    """
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Prepare data for plotting
    # Get dates aligned with signals
    signal_dates = signals_df.index
    
    # Convert dates if they're not datetime
    if 'date' in price_data.columns:
        price_data['date'] = pd.to_datetime(price_data['date'])
        dates = price_data['date'].iloc[len(price_data) - len(signals_df):].values
    else:
        dates = range(len(signals_df))
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'{symbol} LSTM Trading Strategy Performance', fontsize=16, fontweight='bold')
    
    # Plot 1: Actual vs Predicted Prices with Signals
    ax1 = axes[0]
    
    # Plot price lines
    ax1.plot(dates, signals_df['Price'], label='Actual Price', color='blue', linewidth=2)
    ax1.plot(dates, signals_df['Prediction'], label='Predicted Price', color='red', linewidth=2, alpha=0.8)
    
    # Add buy/sell signals
    buy_signals = signals_df[signals_df['Signal'] == 1]
    sell_signals = signals_df[signals_df['Signal'] == -1]
    
    if len(buy_signals) > 0:
        buy_dates = [dates[i] for i in range(len(dates)) if signals_df.iloc[i]['Signal'] == 1]
        ax1.scatter(buy_dates, buy_signals['Price'], 
                   color='green', marker='^', s=100, label='Buy Signal', zorder=5)
    
    if len(sell_signals) > 0:
        sell_dates = [dates[i] for i in range(len(dates)) if signals_df.iloc[i]['Signal'] == -1]
        ax1.scatter(sell_dates, sell_signals['Price'], 
                   color='red', marker='v', s=100, label='Sell Signal', zorder=5)
    
    ax1.set_title('Price Prediction and Trading Signals')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis for dates
    if isinstance(dates[0], (datetime, np.datetime64)):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Portfolio Value Over Time
    ax2 = axes[1]
    
    portfolio_dates = dates[:len(portfolio_values)]
    ax2.plot(portfolio_dates, portfolio_values, color='purple', linewidth=2, label='Portfolio Value')
    
    # Add horizontal line for initial capital
    initial_capital = portfolio_values[0] if portfolio_values else 100000
    ax2.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
    
    # Mark trade points
    for trade in trades:
        if trade['Type'] == 'OPEN':
            trade_idx = signals_df.index.get_loc(trade['Date']) if trade['Date'] in signals_df.index else 0
            if trade_idx < len(portfolio_dates):
                color = 'green' if trade['Signal'] == 1 else 'red'
                marker = '^' if trade['Signal'] == 1 else 'v'
                ax2.scatter(portfolio_dates[trade_idx], portfolio_values[trade_idx], 
                           color=color, marker=marker, s=80, alpha=0.8, zorder=5)
    
    ax2.set_title('Portfolio Performance')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis for dates
    if isinstance(portfolio_dates[0], (datetime, np.datetime64)):
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 3: Prediction Accuracy and Returns
    ax3 = axes[2]
    
    # Calculate prediction errors
    prediction_errors = signals_df['Expected_Return'] * 100  # Convert to percentage
    
    # Plot prediction accuracy
    ax3_twin = ax3.twinx()
    
    # Bar plot for expected returns
    colors = ['green' if x > 0 else 'red' for x in prediction_errors]
    ax3.bar(range(len(prediction_errors)), prediction_errors, alpha=0.6, color=colors, width=1)
    ax3.set_ylabel('Expected Return (%)', color='black')
    ax3.set_title('Expected Returns and Signal Distribution')
    
    # Signal distribution on twin axis
    signal_counts = signals_df['Signal'].value_counts()
    ax3_twin.pie([signal_counts.get(1, 0), signal_counts.get(0, 0), signal_counts.get(-1, 0)], 
                labels=['Buy', 'Hold', 'Sell'], 
                colors=['green', 'gray', 'red'],
                autopct='%1.1f%%',
                startangle=90)
    ax3_twin.set_ylabel('Signal Distribution')
    
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the figure
    figure_path = save_path / f"{symbol}_backtest_analysis.png"
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive analysis saved to: {figure_path}")
    
    plt.show()
    
    # Create a separate detailed signal plot
    create_detailed_signal_plot(signals_df, dates, symbol, save_path)

def create_detailed_signal_plot(signals_df: pd.DataFrame, 
                               dates, 
                               symbol: str, 
                               save_path: Path):
    """Create a detailed plot focusing on signals and price movements."""
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot prices
    ax.plot(dates, signals_df['Price'], label='Actual Price', color='blue', linewidth=2)
    ax.plot(dates, signals_df['Prediction'], label='Predicted Price', color='orange', 
            linewidth=2, alpha=0.8, linestyle='--')
    
    # Highlight prediction confidence with fill_between
    prediction_error = abs(signals_df['Prediction'] - signals_df['Price'])
    upper_bound = signals_df['Prediction'] + prediction_error * 0.5
    lower_bound = signals_df['Prediction'] - prediction_error * 0.5
    
    ax.fill_between(dates, lower_bound, upper_bound, 
                   alpha=0.2, color='orange', label='Prediction Confidence')
    
    # Add signals with annotations
    for i, (idx, row) in enumerate(signals_df.iterrows()):
        if row['Signal'] == 1:  # Buy
            ax.annotate('BUY', xy=(dates[i], row['Price']), 
                       xytext=(dates[i], row['Price'] * 1.05),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2),
                       fontsize=10, color='green', fontweight='bold',
                       ha='center')
            ax.scatter(dates[i], row['Price'], color='green', marker='^', 
                      s=150, zorder=5, edgecolor='darkgreen', linewidth=2)
        
        elif row['Signal'] == -1:  # Sell
            ax.annotate('SELL', xy=(dates[i], row['Price']), 
                       xytext=(dates[i], row['Price'] * 0.95),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=10, color='red', fontweight='bold',
                       ha='center')
            ax.scatter(dates[i], row['Price'], color='red', marker='v', 
                      s=150, zorder=5, edgecolor='darkred', linewidth=2)
    
    ax.set_title(f'{symbol} - Detailed Trading Signals', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Format dates on x-axis
    if isinstance(dates[0], (datetime, np.datetime64)):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save detailed signal plot
    signal_figure_path = save_path / f"{symbol}_detailed_signals.png"
    plt.savefig(signal_figure_path, dpi=300, bbox_inches='tight')
    print(f"Detailed signals plot saved to: {signal_figure_path}")
    
    plt.show()

def create_performance_summary(metrics: Dict, trades: list, signals_df: pd.DataFrame, price_data: pd.DataFrame, symbol: str = "STOCK"):
    """Create a text-based performance summary with trading period information and save to file."""
    
    # Create the summary content
    summary_lines = []
    
    summary_lines.append("=" * 70)
    summary_lines.append("                    PERFORMANCE SUMMARY")
    summary_lines.append("=" * 70)
    
    # Trading period information
    start_date = signals_df.index[0] if len(signals_df) > 0 else "N/A"
    end_date = signals_df.index[-1] if len(signals_df) > 0 else "N/A"
    
    # Calculate trading period duration
    if len(signals_df) > 0:
        if hasattr(signals_df.index[0], 'date'):
            # If index has date attribute
            trading_days = len(signals_df)
            trading_period = f"{trading_days} trading days"
        else:
            # Try to get dates from price_data if available
            if 'date' in price_data.columns:
                price_data['date'] = pd.to_datetime(price_data['date'])
                start_price_date = price_data['date'].iloc[-len(signals_df)]
                end_price_date = price_data['date'].iloc[-1]
                trading_days = len(signals_df)
                trading_period = f"{trading_days} trading days"
                start_date = start_price_date.strftime('%Y-%m-%d')
                end_date = end_price_date.strftime('%Y-%m-%d')
            else:
                trading_period = f"{len(signals_df)} periods"
    else:
        trading_period = "N/A"
    
    # Starting and ending prices
    start_price = signals_df['Price'].iloc[0] if len(signals_df) > 0 else 0
    end_price = signals_df['Price'].iloc[-1] if len(signals_df) > 0 else 0
    price_change = ((end_price - start_price) / start_price * 100) if start_price > 0 else 0
    
    summary_lines.append(f"Trading Period:")
    summary_lines.append(f"  Start Date:           {start_date}")
    summary_lines.append(f"  End Date:             {end_date}")
    summary_lines.append(f"  Duration:             {trading_period}")
    summary_lines.append(f"  Start Price:          ${start_price:.2f}")
    summary_lines.append(f"  End Price:            ${end_price:.2f}")
    summary_lines.append(f"  Buy & Hold Return:    {price_change:.2f}%")
    
    summary_lines.append(f"\nStrategy Performance:")
    summary_lines.append(f"  Total Return:         {metrics['Total_Return']:.2f}%")
    summary_lines.append(f"  Initial Capital:      ${metrics.get('Initial_Capital', 100000):,.2f}")
    summary_lines.append(f"  Final Portfolio Value: ${metrics['Final_Value']:,.2f}")
    summary_lines.append(f"  Absolute Gain/Loss:   ${metrics['Final_Value'] - metrics.get('Initial_Capital', 100000):,.2f}")
    summary_lines.append(f"  Sharpe Ratio:         {metrics['Sharpe_Ratio']:.3f}")
    summary_lines.append(f"  Maximum Drawdown:     {metrics['Max_Drawdown']:.2f}%")
    summary_lines.append(f"  Number of Trades:     {metrics['Number_of_Trades']}")
    
    # Strategy vs Buy & Hold comparison
    strategy_outperformance = metrics['Total_Return'] - price_change
    summary_lines.append(f"\nStrategy vs Buy & Hold:")
    summary_lines.append(f"  Outperformance:       {strategy_outperformance:.2f}%")
    summary_lines.append(f"  Alpha Generated:      {strategy_outperformance:.2f}%")
    
    # Signal statistics
    signal_counts = signals_df['Signal'].value_counts()
    total_signals = len(signals_df)
    
    summary_lines.append(f"\nSignal Distribution:")
    summary_lines.append(f"  Buy Signals:    {signal_counts.get(1, 0):3d} ({signal_counts.get(1, 0)/total_signals*100:.1f}%)")
    summary_lines.append(f"  Hold Signals:   {signal_counts.get(0, 0):3d} ({signal_counts.get(0, 0)/total_signals*100:.1f}%)")
    summary_lines.append(f"  Sell Signals:   {signal_counts.get(-1, 0):3d} ({signal_counts.get(-1, 0)/total_signals*100:.1f}%)")
    
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
                
                # Profit factor
                if avg_loss != 0:
                    profit_factor = abs(avg_profit / avg_loss) if profitable_trades else 0
                    summary_lines.append(f"  Profit Factor:      {profit_factor:.2f}")
    
    # Risk metrics
    summary_lines.append(f"\nRisk Metrics:")
    if metrics['Sharpe_Ratio'] > 1.0:
        risk_assessment = "Excellent"
    elif metrics['Sharpe_Ratio'] > 0.5:
        risk_assessment = "Good"
    elif metrics['Sharpe_Ratio'] > 0:
        risk_assessment = "Fair"
    else:
        risk_assessment = "Poor"
    
    summary_lines.append(f"  Risk Assessment:      {risk_assessment}")
    summary_lines.append(f"  Volatility (implied): {(metrics['Total_Return'] / max(metrics['Sharpe_Ratio'], 0.001)):.2f}%")
    
    summary_lines.append("=" * 70)
    
    # Print to console
    for line in summary_lines:
        print(line)
    
    # Save to file
    save_summary_to_file(summary_lines, symbol)

def save_summary_to_file(summary_lines: list, symbol: str):
    """Save performance summary to a text file."""
    from datetime import datetime
    
    # Get current date and time
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M")
    
    # Create directory path
    current_file_dir = Path(__file__).parent  # analysis directory
    results_dir = current_file_dir / "backtesting_result"
    results_dir.mkdir(exist_ok=True)
    
    # Create filename
    filename = f"{symbol}_{date_str}_{time_str}.txt"
    file_path = results_dir / filename
    
    try:
        # Write summary to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"LSTM Trading Strategy Backtest Results\n")
            f.write(f"Generated on: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Symbol: {symbol}\n\n")
            
            for line in summary_lines:
                f.write(line + '\n')
            
            f.write(f"\n\nFile generated automatically by LSTM Trading System")
            f.write(f"\nTimestamp: {now.isoformat()}")
        
        print(f"\n📁 Performance summary saved to: {file_path}")
        print(f"📊 File size: {file_path.stat().st_size} bytes")
        
    except Exception as e:
        print(f"❌ Error saving performance summary: {e}")
        print(f"🔧 Attempted path: {file_path}")

# Update the main execution section
if __name__ == "__main__":
    
    data_directory = Path(__file__).parent.parent
    model_path = data_directory / "models" / "complete_lstm_model.pth"
    price_data_path = data_directory / "data" / "AAPL.csv"  # Change to your stock data
    scaler_path = data_directory / "models" / "scaler.pkl"
    
    # Load model and data
    model = torch.load(model_path, weights_only=False)
    price_data = pd.read_csv(price_data_path)
    scaler = joblib.load(scaler_path)
    
    print("=" * 50)
    print(f"BACKTESTING SYSTEM - {price_data_path.stem}")
    print("=" * 50)

    # Initialize backtest engine
    backtest_engine = BacktestEngine(
        initial_capital=100000,
        transaction_cost=0.001,
        volatility_window=20,
        min_holding_period=2
    )
    
    # Generate signals and run backtest
    signals_df = backtest_engine.generate_trading_signals(model, price_data, scaler)
    metrics, trades, portfolio_value = backtest_engine.run_backtest(signals_df)
    
    # Create comprehensive visualizations
    plot_backtest_results(
        signals_df=signals_df,
        portfolio_values=portfolio_value,
        trades=trades,
        price_data=price_data,
        symbol=price_data_path.stem,
        save_dir="figures"
    )
    
    # Print performance summary
    create_performance_summary(metrics, trades, signals_df, price_data, symbol=price_data_path.stem)
    



    