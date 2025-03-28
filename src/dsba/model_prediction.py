# model_prediction.py
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Union
import logging
from sklearn.base import ClassifierMixin

class StockPredictor:
    def __init__(self, registry_path: Path = Path("models/registry")):
        self.registry = registry_path
        self.models_cache = {}  # {stock_code: (model, metadata)}

    def load_model(self, stock_code: str) -> tuple[ClassifierMixin, dict]:
        """Load model and metadata with caching"""
        if stock_code not in self.models_cache:
            model_path = self._find_latest_model(stock_code)
            if not model_path:
                raise ValueError(f"No model found for {stock_code}")
            
            model = joblib.load(model_path / "model.pkl")
            with open(model_path / "metadata.json") as f:
                metadata = json.load(f)
            
            self.models_cache[stock_code] = (model, metadata)
        
        return self.models_cache[stock_code]

    def _find_latest_model(self, stock_code: str) -> Path:
        """Locate most recent model version"""
        stock_dir = self.registry / stock_code
        if not stock_dir.exists():
            return None
            
        versions = sorted(stock_dir.glob("v*"))
        return versions[-1] if versions else None

    def predict_stock(
        self,
        stock_code: str,
        latest_data: pd.DataFrame,
    ) -> Dict[str, Union[float, str]]:
        """
        Generate prediction for the next trading day
        
        Args:
            stock_code: Stock ticker symbol
            latest_data: DataFrame with recent market data
            
        Returns:
            Dictionary with prediction and metadata
        """
        model, metadata = self.load_model(stock_code)
        
        # Validate prediction horizon
        required_horizon = 1
        if metadata["prediction_horizon"] != required_horizon:
            logging.error(
                f"Model trained for {metadata['prediction_horizon']} day horizon "
                f"but required horizon is {required_horizon}"
            )
            raise ValueError("Model horizon mismatch")
        
        # Prepare features (should match training)
        features = self._prepare_features(latest_data, metadata["feature_columns"])
        
        # Generate prediction
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]  # Probability of price increase
        
        return {
            "stock_code": stock_code,
            "prediction": "Hold" if prediction == 1 else "Sell",
            "confidence": float(proba),
            "model_version": metadata["id"],
            "last_train_date": metadata["train_date_range"]["end"],
            "horizon_days": required_horizon
        }

    def _prepare_features(self, data: pd.DataFrame, required_features: list) -> pd.DataFrame:
        """Ensure input data has correct features"""
        missing = set(required_features) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return data[required_features]

# Professor's template compatibility
def classify_dataframe(
    model: ClassifierMixin,
    df: pd.DataFrame,
    target_column: str = "prediction"
) -> pd.DataFrame:
    """Wrapper for professor's expected function"""
    predictor = StockPredictor()
    results = []
    
    for stock_code in df["stock_code"].unique():
        stock_data = df[df["stock_code"] == stock_code]
        pred = predictor.predict_stock(stock_code, stock_data)
        results.append(pred)
    
    return pd.DataFrame(results)

def classify_record(
    model: ClassifierMixin,
    record: dict,
    target_column: str = "prediction"
) -> dict:
    """Wrapper for professor's expected function"""
    predictor = StockPredictor()
    return predictor.predict_stock(
        record["stock_code"],
        pd.DataFrame([record])
    )
