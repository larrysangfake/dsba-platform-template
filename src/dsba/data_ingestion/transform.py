import numpy as np
from pathlib import Path
from arch import arch_model

def determine_market_state(data: pd.DataFrame) -> pd.DataFrame:
  """ market state logic"""
  data["SMA_50"] = data["Adj Close"].rolling(window=50).mean()
  data["SMA_200"] = data["Adj Close"].rolling(window=200).mean()
  data["Market_State"] = np.where(
        data["SMA_50"] > data["SMA_200"], "Bull", "Bear"
    )
  return data

def add_features(clean_path: Path) -> Path:
  features_path = Path(f"data/processed/features/{clean_path.stem}_features.csv")
  df = pd.read_csv(clean_path)
  
  """Generate features for Monte Carlo"""
  # --- Monte Carlo Prep ---
  data["Log_Return"] = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
  data = determine_market_state(data)
  data.dropna()

  df.to_csv(features_path)

  return features_path


