"""
Technical Indicators Feature Engineering Module

Functions for calculating technical analysis indicators for stock trading.
"""

import numpy as np
import pandas as pd


def calculate_technical_indicators(df):
    """
    Calculate comprehensive technical indicators for enhanced feature set

    This function computes various technical analysis indicators including:
    - Price-based features (returns, log returns)
    - Moving averages (SMA, EMA)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - RSI (Relative Strength Index)
    - Momentum indicators
    - Volatility measures
    - Volume-based features
    - ATR (Average True Range)

    Args:
        df (pd.DataFrame): DataFrame with columns: 'close', optionally 'high', 'low', 'volume'

    Returns:
        pd.DataFrame: Original DataFrame with additional technical indicator columns
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

    # MACD (Moving Average Convergence Divergence)
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


def get_default_feature_columns(df):
    """
    Get default feature columns for modeling

    Args:
        df (pd.DataFrame): DataFrame with technical indicators

    Returns:
        list: List of feature column names
    """
    # Default feature set (close must be first for prediction target)
    feature_columns = [
        'close', 'returns', 'sma_5', 'sma_20', 'rsi',
        'macd', 'macd_signal', 'bb_position', 'bb_width',
        'momentum', 'volatility', 'roc'
    ]

    # Filter to only columns that exist in dataframe
    available_features = [col for col in feature_columns if col in df.columns]

    return available_features


def calculate_alpha_factors(df):
    """
    Calculate alpha factors for quantitative trading

    Alpha factors are transformations that aim to predict future returns.
    These are inspired by WorldQuant's 101 Alphas.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data

    Returns:
        pd.DataFrame: DataFrame with additional alpha factor columns
    """
    df = df.copy()

    # Alpha 1: Rank-based momentum
    if 'close' in df.columns:
        df['alpha_momentum_rank'] = df['close'].pct_change(5).rolling(20).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
        )

    # Alpha 2: Volume-price correlation
    if 'volume' in df.columns and 'close' in df.columns:
        df['alpha_volume_price_corr'] = df['close'].rolling(10).corr(df['volume'])

    # Alpha 3: Mean reversion (distance from MA)
    if 'close' in df.columns:
        ma_20 = df['close'].rolling(20).mean()
        df['alpha_mean_reversion'] = (df['close'] - ma_20) / ma_20

    # Alpha 4: Volatility-adjusted returns
    if 'returns' in df.columns and 'volatility' in df.columns:
        df['alpha_vol_adj_returns'] = df['returns'] / (df['volatility'] + 1e-8)

    return df


def select_features(df, feature_set='default'):
    """
    Select feature set for modeling

    Args:
        df (pd.DataFrame): DataFrame with features
        feature_set (str): Feature set to use ('default', 'minimal', 'extended', 'alpha')

    Returns:
        list: List of selected feature column names
    """
    if feature_set == 'minimal':
        features = ['close', 'returns', 'sma_20', 'rsi']
    elif feature_set == 'extended':
        features = get_default_feature_columns(df) + [
            'sma_50', 'ema_12', 'ema_26', 'macd_diff',
            'bb_upper', 'bb_lower', 'volume_ratio', 'atr'
        ]
    elif feature_set == 'alpha':
        features = get_default_feature_columns(df) + [
            'alpha_momentum_rank', 'alpha_volume_price_corr',
            'alpha_mean_reversion', 'alpha_vol_adj_returns'
        ]
    else:  # default
        features = get_default_feature_columns(df)

    # Filter to only available columns
    available_features = [col for col in features if col in df.columns]

    return available_features
