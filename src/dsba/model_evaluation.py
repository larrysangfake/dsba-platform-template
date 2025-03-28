# model_evaluation.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import json
import logging
from typing import Dict, Any
import joblib

class StockModelEvaluator:
    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.logger = logging.getLogger(f"evaluator.{stock_code}")

    def _load_artifacts(self, model_dir: Path) -> Dict[str, Any]:
        """Load model and preprocessor"""
        model_path = model_dir / f"{self.stock_code}_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file missing for {self.stock_code}")
        
        pipeline = joblib.load(model_path)
        return {
            'model': pipeline['model'],
            'preprocessor': pipeline['preprocessor']
        }

    def evaluate(self, model_dir: Path, test_data: Dict) -> Dict[str, Any]:
        """
        Evaluate model performance on test set
        
        Args:
            model_dir: Directory containing model artifacts
            test_data: Preprocessed test data from StockPreprocessor
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Load model and preprocessor
            artifacts = self._load_artifacts(model_dir)
            model = artifacts['model']
            
            # Get predictions
            X_test = test_data['X_test']
            y_test = test_data['y_test']
            test_dates = test_data['test_dates']
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'test_date_range': {
                    'start': test_dates[0].strftime('%Y-%m-%d'),
                    'end': test_dates[-1].strftime('%Y-%m-%d')
                },
                'evaluation_date': pd.Timestamp.now().isoformat()
            }
            
            # Load training metadata
            metadata_path = model_dir / f"{self.stock_code}_metadata.json"
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Create full report
            full_report = {
                'stock_code': self.stock_code,
                'training_metadata': metadata,
                'evaluation_metrics': metrics
            }
            
            # Save evaluation report
            eval_path = model_dir / f"{self.stock_code}_evaluation.json"
            with open(eval_path, 'w') as f:
                json.dump(full_report, f, indent=2)
                
            return full_report
            
        except Exception as e:
            self.logger.error(f"Evaluation failed for {self.stock_code}: {str(e)}")
            raise

def evaluate_model(
    stock_code: str,
    model_dir: Path,
    test_data: Dict
) -> bool:
    """
    Pipeline integration point
    
    Args:
        stock_code: Stock ticker symbol
        model_dir: Directory containing model artifacts
        test_data: Preprocessed test data from StockPreprocessor
        
    Returns:
        bool: True if evaluation succeeded
    """
    try:
        evaluator = StockModelEvaluator(stock_code)
        report = evaluator.evaluate(model_dir, test_data)
        logging.info(f"Evaluation completed for {stock_code}")
        logging.info(f"ROC AUC: {report['evaluation_metrics']['roc_auc']:.4f}")
        return True
    except Exception as e:
        logging.error(f"Evaluation failed for {stock_code}: {str(e)}")
        return False
