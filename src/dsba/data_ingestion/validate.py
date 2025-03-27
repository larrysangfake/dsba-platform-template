from pathlib import Path
import pandas as pd

def validate_data(features_path: Path) -> bool:
    """Runs data quality checks."""
    df = pd.read_csv(features_path)
    assert not df["Close"].isna().any(), "Missing prices"
    assert (df["Close"] > 0).all(), "Negative prices"
    return True  # Or raise exceptions
