"""
Script to update Backtesting.py to support 12 features instead of 1 feature.
Run this script to automatically update your Backtesting.py file.
"""

import re
from pathlib import Path

# Technical indicators function to add
TECHNICAL_INDICATORS_CODE = '''
# ==================== Feature Engineering ====================

def calculate_technical_indicators(df):
    """Calculate technical indicators matching PyTorchOptimized.py"""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['momentum'] = df['close'] - df['close'].shift(4)
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
    return df

# ==================== Model Architecture ====================

class ImprovedLSTM(nn.Module):
    """Enhanced LSTM matching PyTorchOptimized.py"""
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, output_dim):
        super(ImprovedLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0, bidirectional=False)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 4)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.layer_norm(out)
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

'''

# New generate_trading_signals method
NEW_GENERATE_SIGNALS = '''    def generate_trading_signals(self,
                                model,
                                historical_data: pd.DataFrame,
                                scaler,
                                sequence_length: int = 60) -> pd.DataFrame:
        """Generate trading signals using 12 features matching PyTorchOptimized.py"""
        
        # Calculate technical indicators
        df_with_indicators = calculate_technical_indicators(historical_data)
        
        # Select features (must match PyTorchOptimized.py)
        feature_columns = ['close', 'returns', 'sma_5', 'sma_20', 'rsi', 
                          'macd', 'macd_signal', 'bb_position', 'bb_width',
                          'momentum', 'volatility', 'roc']
        
        # Filter to available columns
        feature_columns = [col for col in feature_columns if col in df_with_indicators.columns]
        
        # Remove NaN values
        df_with_indicators = df_with_indicators.dropna().reset_index(drop=True)
        
        # Extract feature values
        feature_values = df_with_indicators[feature_columns].values
        
        signals_df = pd.DataFrame()
        
        predictions = []
        current_prices = []
        signals = []
        
        # Generate predictions for entire dataset
        for i in range(sequence_length, len(feature_values)):
            # Prepare input sequence with all features
            sequence = feature_values[i-sequence_length:i]  # Shape: (sequence_length, num_features)
            scaled_sequence = scaler.transform(sequence)
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0)  # Shape: (1, sequence_length, num_features)
            
            # Generate prediction
            model.eval()
            with torch.no_grad():
                scaled_pred = model(input_tensor).cpu().numpy()
            
            # Inverse transform prediction (close price is first feature)
            n_features = scaler.n_features_in_
            dummy_pred = np.zeros((1, n_features))
            dummy_pred[:, 0] = scaled_pred.flatten()
            prediction = scaler.inverse_transform(dummy_pred)[0, 0]
            
            current_price = df_with_indicators['close'].iloc[i]
            
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
        
        return signals_df'''

# New model loading code
NEW_MODEL_LOADING = '''    # Load model and data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with correct architecture (matching PyTorchOptimized.py)
    model = ImprovedLSTM(
        input_dim=12,  # 12 features
        hidden_dim=256,
        num_layers=3,
        dropout=0.3,
        output_dim=1
    ).to(device)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    price_data = pd.read_csv(price_data_path)
    scaler = joblib.load(scaler_path)
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Input dimension: 12 features")
    print(f"Scaler features: {scaler.n_features_in_}")'''

def update_backtesting_file():
    """Update the Backtesting.py file"""
    file_path = Path(__file__).parent / "Backtesting.py"
    
    print(f"Reading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Add technical indicators and ImprovedLSTM after imports
    import_end = content.find("warnings.filterwarnings('ignore')")
    if import_end != -1:
        import_end = content.find('\n', import_end) + 1
        next_class = content.find('\nclass LSTM(nn.Module):', import_end)
        if next_class != -1:
            content = content[:next_class] + '\n' + TECHNICAL_INDICATORS_CODE + content[next_class:]
            print("[OK] Added technical indicators function and ImprovedLSTM class")
    
    # 2. Replace generate_trading_signals method
    pattern = r'    def generate_trading_signals\(self,.*?return signals_df'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        content = content[:match.start()] + NEW_GENERATE_SIGNALS + content[match.end():]
        print("[OK] Updated generate_trading_signals method")
    
    # 3. Replace model loading section
    pattern = r'    # Load model and data\n    model = torch\.load\(model_path.*?scaler = joblib\.load\(scaler_path\)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        content = content[:match.start()] + NEW_MODEL_LOADING + content[match.end():]
        print("[OK] Updated model loading section")
    
    # Write updated content
    backup_path = file_path.with_suffix('.py.backup')
    print(f"\nCreating backup at {backup_path}...")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Writing updated file to {file_path}...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n[SUCCESS] Backtesting.py has been successfully updated!")
    print(f"[SUCCESS] Backup saved to {backup_path}")
    print("\nChanges made:")
    print("1. Added calculate_technical_indicators() function")
    print("2. Added ImprovedLSTM class matching PyTorchOptimized.py")
    print("3. Updated generate_trading_signals() to use 12 features")
    print("4. Updated model loading to use ImprovedLSTM with 12 input features")

if __name__ == "__main__":
    try:
        update_backtesting_file()
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
