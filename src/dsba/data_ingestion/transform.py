import numpy as np
from pathlib import Path
import pandas as pd

def determine_market_state(data: pd.DataFrame) -> pd.DataFrame:
    """Safer market state logic with expanding window"""
    data = data.copy()
    # Use expanding mean to prevent lookahead bias
    data["SMA_50"] = data["Adj Close"].expanding(min_periods=50).mean()
    data["SMA_200"] = data["Adj Close"].expanding(min_periods=200).mean()
    data["Market_State"] = np.where(
        data["SMA_50"] > data["SMA_200"], "Bull", "Bear"
    )
    return data

def add_features(clean_path: Path) -> Path:
    features_path = Path(f"data/processed/features/{clean_path.stem}_features.csv")
    data = pd.read_csv(clean_path, parse_dates=['Date'], index_col='Date')
    
    # --- Monte Carlo Prep ---
    data = determine_market_state(data)
    data["Log_Return"] = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
    
    # --- Random Forest Prep ---
    data["Close_Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
    
    # Feature windows aligned with prediction horizon
    windows = {
        'short_term': 5,    # 1 week
        'medium_term': 20,  # 1 month
        'long_term': 60     # 3 months
    }
    
    for name, window in windows.items():
        # Price Features (lagged to prevent leakage)
        data[f'ma_{name}'] = data['Close'].shift(1).rolling(window).mean()
        data[f'close_ratio_{name}'] = data['Close'] / data[f'ma_{name}']
        
        # Volatility (lagged)
        data[f'volatility_{name}'] = (
            data["Close_Log_Return"]
            .shift(1)
            .rolling(window)
            .std()
        )
        
        # Volume (lagged)
        data[f'volume_{name}'] = data["Volume"].shift(1).rolling(window).mean()
    
    # Drop NA values from feature creation
    data = data.dropna()
    
    # Save only necessary columns
    keep_cols = [
        'stock_code', 'Close', 'Adj Close', 'Volume', 'Log_Return', 'Market_State',
        *[f for f in data.columns if f.startswith(('ma_', 'close_ratio_', 'volatility_', 'volume_'))]
    ]
    data[keep_cols].to_csv(features_path)
    
    return features_path


