# model_registry.py
import joblib
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
from sklearn.base import BaseEstimator

@dataclass
class ModelMetadata:
    stock_code: str
    created_at: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    feature_columns: list[str]
    performance_metrics: Dict[str, float]
    prediction_horizon: int
    last_train_date: str

class ModelRegistry:
    def __init__(self, registry_path: str = None):
        """
        Initialize model registry
        
        Args:
            registry_path: Optional custom path for model storage
                         Defaults to STOCK_MODELS_PATH environment variable
        """
        self.registry_path = self._get_registry_path(registry_path)
        self.logger = logging.getLogger("model_registry")

    def save_model(self, model: BaseEstimator, metadata: ModelMetadata) -> None:
        """Save model and metadata to registry"""
        model_path = self._get_model_path(metadata.stock_code)
        metadata_path = self._get_metadata_path(metadata.stock_code)
        
        self.logger.info(f"Saving model for {metadata.stock_code} to {model_path}")
        
        # Save model and metadata
        joblib.dump(model, model_path)
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

    def load_model(self, stock_code: str) -> BaseEstimator:
        """Load model from registry"""
        model_path = self._get_model_path(stock_code)
        self.logger.info(f"Loading model for {stock_code} from {model_path}")
        return joblib.load(model_path)

    def load_metadata(self, stock_code: str) -> ModelMetadata:
        """Load model metadata from registry"""
        metadata_path = self._get_metadata_path(stock_code)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return ModelMetadata(**metadata)

    def list_models(self) -> list[str]:
        """List all registered stock models"""
        model_files = [f for f in os.listdir(self.registry_path) if f.endswith(".pkl")]
        return [os.path.splitext(f)[0] for f in model_files]

    def _get_model_path(self, stock_code: str) -> Path:
        """Get full path to model file"""
        return self.registry_path / f"{stock_code}.pkl"

    def _get_metadata_path(self, stock_code: str) -> Path:
        """Get full path to metadata file"""
        return self.registry_path / f"{stock_code}_metadata.json"

    def _get_registry_path(self, custom_path: str = None) -> Path:
        """Resolve registry directory path"""
        if custom_path:
            path = Path(custom_path)
        else:
            env_path = os.getenv("STOCK_MODELS_PATH")
            if not env_path:
                raise ValueError(
                    "STOCK_MODELS_PATH environment variable not set. "
                    "Please set it or provide custom registry_path"
                )
            path = Path(env_path)
        
        # Create directory if it doesn't exist
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()
