# model_evaluation.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import json
import logging
from typing import Dict, Any

class StockModelEvaluator:
    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.logger = logging.getLogger(f"evaluator.{stock_code}")

    def _load_test_data(self, model_dir: Path) -> Dict[str, Any]:
        """Load test data saved during training"""
        test_data_path = model_dir / f"{self.stock_code}_test_data.csv"
        if not test_data_path.exists():
            raise FileNotFoundError(f"Test data missing for {self.stock_code}")
        
        test_df = pd.read_csv(test_data_path, parse_dates=['Date'])
        return {
            'dates': test_df['Date'].values,
            'y_true': test_df['y_true'].values
        }

    def evaluate(self, model_dir: Path, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            model_dir: Path containing training artifacts
            predictions: Model predictions on test data
            
        Returns:
            Dictionary of evaluation metrics and metadata
        """
        try:
            # Load ground truth
            test_data = self._load_test_data(model_dir)
            y_true = test_data['y_true']
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_true, predictions),
                'precision': precision_score(y_true, predictions),
                'recall': recall_score(y_true, predictions),
                'f1': f1_score(y_true, predictions),
                'confusion_matrix': confusion_matrix(y_true, predictions).tolist(),
                'evaluation_date': pd.Timestamp.now().isoformat(),
                'test_date_range': {
                    'start': pd.to_datetime(test_data['dates'].min()).strftime('%Y-%m-%d'),
                    'end': pd.to_datetime(test_data['dates'].max()).strftime('%Y-%m-%d')
                }
            }
            
            # Load training metadata
            with open(model_dir / f"{self.stock_code}_train_metadata.json") as f:
                metadata = json.load(f)
            
            # Merge with evaluation results
            full_report = {
                'stock_code': self.stock_code,
                'training_metadata': metadata,
                'evaluation_metrics': metrics,
                'model_path': str(model_dir / f"{self.stock_code}_model.pkl")
            }
            
            # Save evaluation report
            eval_path = model_dir / f"{self.stock_code}_evaluation.json"
            with open(eval_path, 'w') as f:
                json.dump(full_report, f, indent=2)
                
            return full_report
            
        except Exception as e:
            self.logger.error(f"Evaluation failed for {self.stock_code}: {str(e)}")
            raise

def evaluate_model(stock_code: str, model_dir: Path, predictions: np.ndarray) -> bool:
    """
    Pipeline integration point
    
    Args:
        stock_code: Stock ticker symbol
        model_dir: Directory containing model artifacts
        predictions: Model predictions on test data
        
    Returns:
        bool: True if evaluation succeeded
    """
    try:
        evaluator = StockModelEvaluator(stock_code)
        report = evaluator.evaluate(model_dir, predictions)
        logging.info(f"Evaluation completed for {stock_code}: {report['evaluation_metrics']}")
        return True
    except Exception as e:
        logging.error(f"Evaluation failed for {stock_code}: {str(e)}")
        return False
