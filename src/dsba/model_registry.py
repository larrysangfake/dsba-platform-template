import joblib
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
from sklearn.base import BaseEstimator
import hashlib

@dataclass
class StockModelMetadata:
    """Extended metadata for stock prediction models"""
    id: str
    stock_code: str
    created_at: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    target_column: str
    description: str
    performance_metrics: Dict[str, float]
    feature_columns: List[str]
    train_date_range: Dict[str, str]  # {"start": "2020-01-01", "end": "2023-06-30"}
    prediction_horizon: int

class StockModelRegistry:
    def __init__(self, registry_root: Path = Path("models/registry")):
        self.registry_root = registry_root
        self.registry_root.mkdir(exist_ok=True)
    
    def save_model(self, model: BaseEstimator, metadata: StockModelMetadata):
        """Stock-aware model saving with versioning"""
        # Create versioned directory
        stock_dir = self.registry_root / metadata.stock_code
        stock_dir.mkdir(exist_ok=True)
        
        version = f"v{len(list(stock_dir.glob('v*'))) + 1}"
        version_dir = stock_dir / version
        version_dir.mkdir()
        
        # Save artifacts
        model_path = version_dir / "model.pkl"
        metadata_path = version_dir / "metadata.json"
        
        joblib.dump(model, model_path)
        metadata.id = f"{metadata.stock_code}_{version}"
        
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        logging.info(f"Saved {metadata.stock_code} model {version}")

    def load_model(self, stock_code: str, version: str = "latest") -> BaseEstimator:
        """Load specific model version"""
        if version == "latest":
            versions = sorted((self.registry_root / stock_code).glob("v*"))
            if not versions:
                raise FileNotFoundError(f"No models found for {stock_code}")
            version_dir = versions[-1]
        else:
            version_dir = self.registry_root / stock_code / version
        
        return joblib.load(version_dir / "model.pkl")

    def get_metadata(self, stock_code: str, version: str) -> StockModelMetadata:
        """Retrieve model metadata"""
        metadata_path = self.registry_root / stock_code / version / "metadata.json"
        with open(metadata_path) as f:
            return StockModelMetadata(**json.load(f))
