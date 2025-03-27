from pathlib import Path
import pandas as pd

def clean_data(raw_path: Path) -> Path:
    """Handles missing values."""
    clean_path = Path(f"data/processed/cleaned/{raw_path.stem}_clean.csv")
    
    df = pd.read_csv(raw_path)
    #drop missing values and missing entries
    df.dropna(inplace=True)
    df.to_csv(clean_path)
    return clean_path
