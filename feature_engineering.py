import pandas as pd
import numpy as np
import ta

def engineer_features(df, transaction_cost_pct=0.01):
    """Create features for a single ticker using TA library with fixed ATR calculation"""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Make sure we have a proper index before calculations
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Basic price and volume features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volume_change'] = df['Volume'].pct_change()
    
    # Manual ATR calculation (avoiding TA library for this specific indicator)
    high_low = df['High'] - df['Low']
    high_close_prev = df['High'] - df['Close'].shift(1)
    low_close_prev = df['Low'] - df['Close'].shift(1)
    
    # Calculate True Range manually
    ranges = pd.DataFrame({
        'hl': high_low,
        'hcp': high_close_prev.abs(),
        'lcp': low_close_prev.abs()
    })
    
    df['tr'] = ranges.max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['Close']
    
    # Use TA library for other indicators
    # Simple Moving Averages
    for period in [5, 10, 20, 50, 100]:
        df[f'sma_{period}'] = ta.trend.sma_indicator(close=df['Close'], window=period)
    
    # Exponential Moving Averages
    for period in [5, 10, 20, 50, 100]:
        df[f'ema_{period}'] = ta.trend.ema_indicator(close=df['Close'], window=period)
    
    # Moving Average Crossovers
    df['ma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
    df['ma_cross_10_50'] = (df['sma_10'] > df['sma_50']).astype(int)
    df['ma_cross_20_100'] = (df['sma_20'] > df['sma_100']).astype(int)
    
    # Price relative to moving averages
    df['close_over_sma_50'] = (df['Close'] > df['sma_50']).astype(int)
    df['sma_200'] = ta.trend.sma_indicator(close=df['Close'], window=200)
    df['close_over_sma_200'] = (df['Close'] > df['sma_200']).astype(int)
    
    # RSI
    df['rsi_14'] = ta.momentum.rsi(close=df['Close'], window=14)
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['Close']
    df['bb_pct'] = (df['Close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-10)  # avoid div by zero
    
    # MACD
    macd = ta.trend.MACD(close=df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Volume indicators
    df['obv'] = ta.volume.on_balance_volume(close=df['Close'], volume=df['Volume'])
    df['volume_sma_20'] = ta.trend.sma_indicator(close=df['Volume'], window=20)
    df['volume_sma_ratio'] = df['Volume'] / (df['volume_sma_20'] + 1e-10)  # avoid div by zero
    
    # Volume spikes
    df['volume_spike'] = (df['Volume'] > 2 * df['volume_sma_20']).astype(int)
    
    # Date features
    df['day_of_week'] = pd.to_datetime(df.index).dayofweek
    df['day_of_month'] = pd.to_datetime(df.index).day
    df['day_of_year'] = pd.to_datetime(df.index).dayofyear
    df['month'] = pd.to_datetime(df.index).month
    df['quarter'] = pd.to_datetime(df.index).quarter
    
    # Momentum indicators
    df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    # Target variables
    df['target_next_day'] = (df['Close'].shift(-1) > df['Close'] * (1 + transaction_cost_pct)).astype(int)
    future_return_5d = (df['Close'].shift(-5) / df['Close']) - 1
    df['target_5d'] = (future_return_5d > transaction_cost_pct).astype(int)
    future_return_10d = (df['Close'].shift(-10) / df['Close']) - 1
    df['target_10d'] = (future_return_10d > transaction_cost_pct).astype(int)
    
    # Replace any infinite values that might have been created
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df