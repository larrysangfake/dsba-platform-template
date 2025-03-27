from pathlib import Path
from .files import fetch_stock_data
from .clean import clean_data
from .transform import add_features
from .validate import validate_data

def run_pipeline(stock_code: str):
    # 1. Collect
    raw_path = fetch_stock_data(stock_code)
    
    # 2. Curate
    clean_path = clean_data(raw_path)
    
    # 3. Transform
    features_path = add_features(clean_path)
    
    # 4. Validate
    if validate_data(features_path):
        print(f"Pipeline succeeded for {stock_code}")
        return features_path
    raise ValueError(f"Validation failed for {stock_code}")
