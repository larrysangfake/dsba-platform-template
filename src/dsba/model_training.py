# model_train.py
import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import json

class StockModelTrainer:
    def __init__(self, preprocessor, stock_code: str):
        """
        Simplified time-series aware model trainer
        
        Args:
            preprocessor: Initialized StockPreprocessor
            stock_code: Stock ticker symbol
        """
        self.preprocessor = preprocessor
        self.stock_code = stock_code
        self.model = None
        self.logger = logging.getLogger(f"trainer.{stock_code}")

    def train(self, features_path: Path) -> Dict[str, Any]:
        processed_data = self.preprocessor.process(features_path)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced',
            n_jobs=-1,
            max_features=0.3
        )
        
        tscv = TimeSeriesSplit(n_splits=5)
        X = processed_data['X_train']
        y = processed_data['y_train']
        dates = processed_data['train_dates']
        
        # Initialize collectors
        test_preds = []
        test_dates = []
        test_y_true = []
        
        for train_idx, test_idx in tscv.split(X):
            fold_model = clone(self.model)
            fold_model.fit(X[train_idx], y[train_idx])
            test_preds.extend(fold_model.predict_proba(X[test_idx])[:, 1])
            test_dates.extend(dates[test_idx])
            test_y_true.extend(y[test_idx])  # Collect actual values
        
        # Final training
        self.model.fit(X, y)
        
        return {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'validation_results': {
                'dates': test_dates,
                'y_true': test_y_true,  # Use collected values
                'y_pred': test_preds,
                'auc': roc_auc_score(test_y_true, test_preds)
            },
            'metadata': {
                'stock_code': self.stock_code,
                'trained_at': datetime.now().isoformat(),
                'last_train_date': dates[-1],
                'features': processed_data['feature_names']
            }
        }
    
    def save_artifacts(self, artifacts: dict, output_dir: Path):
        """Save minimal required artifacts"""
        output_dir.mkdir(exist_ok=True)
        
        # Save combined model and preprocessor
        joblib.dump(
            {
                'model': artifacts['model'],
                'preprocessor': artifacts['preprocessor']
            },
            output_dir / f"{self.stock_code}_model.pkl"
        )
        
        # Save validation results
        pd.DataFrame(artifacts['validation_results']).to_csv(
            output_dir / f"{self.stock_code}_validation.csv",
            index=False
        )
        
        # Save metadata
        with open(output_dir / f"{self.stock_code}_metadata.json", 'w') as f:
            json.dump(artifacts['metadata'], f)

def train_stock_model(
    preprocessor: StockPreprocessor,
    stock_code: str,
    features_path: Path,
    output_dir: Path
) -> bool:
    """Simplified training pipeline"""
    try:
        trainer = StockModelTrainer(preprocessor, stock_code)
        artifacts = trainer.train(features_path)
        trainer.save_artifacts(artifacts, output_dir)
        return True
    except Exception as e:
        logging.error(f"Training failed for {stock_code}: {str(e)}")
        return False
