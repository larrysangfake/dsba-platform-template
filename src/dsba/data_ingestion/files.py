from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import yfinance as yf

def fetch_stock_data(stock_code: str, end_date: datetime = None) -> Path:
    """
    Fetch stock data up to a specific end date (default: yesterday).
    - Uses post-2020 data only
    - Automatically caches to data/raw/
    """
    end_date = end_date or (datetime.now() - timedelta(days=1))
    start_date = max(datetime(2022, 1, 1), end_date - timedelta(days=3*365))
    
    # Check cache first
    raw_path = Path(f"data/raw/{stock_code}_{end_date.date()}.csv")
    if raw_path.exists():
        return raw_path
    
    # Fetch from Yahoo Finance
    data = yf.download(stock_code, start=start_date, end=end_date, auto_adjust=False, multi_level_index=False)

    if data.empty:
        raise ValueError(f"No data found for ticker {stock_code}")
        
    # Cache the data
    raw_path.parent.mkdir(exist_ok=True)
    data['stock_code'] = stock_code
    data.to_csv(raw_path)
    return raw_path

def get_holdout_data(stock_code: str, days: int = 30) -> pd.DataFrame:
    """Get the most recent [days] as holdout data"""
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)
    return fetch_stock_data(stock_code, end_date=end_date).loc[start_date:end_date]
