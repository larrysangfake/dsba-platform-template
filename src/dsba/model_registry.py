# model_registry.py
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import shutil
import hashlib

@dataclass
class ModelVersion:
    version: str  # e.g., "v1"
    stock_code: str
    training_date: str
    metrics: Dict[str, float]
    model_path: Path
    feature_columns: List[str]
    data_fingerprint: str  # Hash of training data

class StockModelRegistry:
    def __init__(self, registry_root: Path = Path("models/registry")):
        self.registry_root = registry_root
        self.registry_root.mkdir(exist_ok=True)
    
    def _get_stock_registry(self, stock_code: str) -> Path:
        """Get path to stock-specific registry file"""
        return self.registry_root / f"{stock_code}.json"
    
    def _generate_version(self, stock_code: str) -> str:
        """Auto-increment version number"""
        registry_file = self._get_stock_registry(stock_code)
        if registry_file.exists():
            versions = json.loads(registry_file.read_text())
            return f"v{len(versions) + 1}"
        return "v1"
    
    def register_model(self, model_dir: Path) -> ModelVersion:
        """
        Register a trained model from training directory
        
        Args:
            model_dir: Path containing:
                - {stock_code}_model.pkl
                - {stock_code}_train_metadata.json
                - {stock_code}_evaluation.json
        """
        # Load metadata
        stock_code = self._extract_stock_code(model_dir)
        train_meta = json.load(open(model_dir / f"{stock_code}_train_metadata.json"))
        eval_meta = json.load(open(model_dir / f"{stock_code}_evaluation.json"))
        
        # Create versioned artifact
        version = self._generate_version(stock_code)
        version_dir = self.registry_root / stock_code / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        shutil.copy(
            model_dir / f"{stock_code}_model.pkl",
            version_dir / "model.pkl"
        )
        
        # Create version record
        model_version = ModelVersion(
            version=version,
            stock_code=stock_code,
            training_date=train_meta["training_date"],
            metrics=eval_meta["evaluation_metrics"],
            model_path=version_dir / "model.pkl",
            feature_columns=train_meta["feature_columns"],
            data_fingerprint=self._hash_data(model_dir / f"{stock_code}_test_data.csv")
        )
        
        # Update registry
        self._update_stock_registry(stock_code, model_version)
        return model_version
    
    def _extract_stock_code(self, model_dir: Path) -> str:
        """Extract stock code from model files"""
        for f in model_dir.glob("*_model.pkl"):
            return f.stem.split("_")[0]
        raise ValueError("No valid model files found")
    
    def _hash_data(self, data_path: Path) -> str:
        """Create MD5 hash of data file for versioning"""
        return hashlib.md5(data_path.read_bytes()).hexdigest()
    
    def _update_stock_registry(self, stock_code: str, version: ModelVersion):
        """Append new version to stock's registry"""
        registry_file = self._get_stock_registry(stock_code)
        versions = []
        
        if registry_file.exists():
            versions = json.loads(registry_file.read_text())
        
        versions.append(asdict(version))
        registry_file.write_text(json.dumps(versions, indent=2))
    
    def get_latest_version(self, stock_code: str) -> Optional[ModelVersion]:
        """Retrieve the latest registered version"""
        registry_file = self._get_stock_registry(stock_code)
        if not registry_file.exists():
            return None
            
        versions = json.loads(registry_file.read_text())
        latest = max(versions, key=lambda x: x["training_date"])
        return ModelVersion(**latest)

# CLI Interface
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Path to trained model directory")
    args = parser.parse_args()
    
    registry = StockModelRegistry()
    result = registry.register_model(Path(args.model_dir))
    print(f"Registered {result.stock_code} version {result.version}")
