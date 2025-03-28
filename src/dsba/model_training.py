# model_training.py
import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime
from typing import Dict, Any
import logging

class StockModelTrainer:
    def __init__(self, stock_code: str):
        """
        Stock-specific model trainer
        
        Args:
            stock_code: The stock ticker symbol (e.g., 'AAPL')
        """
        self.stock_code = stock_code
        self.model = None
        self.logger = logging.getLogger(f"trainer.{stock_code}")
        
    def _validate_inputs(self, features_path: Path) -> pd.DataFrame:
        """Load and validate training data"""
        if not features_path.exists():
            raise FileNotFoundError(f"Features file missing: {features_path}")
            
        df = pd.read_csv(features_path, parse_dates=['Date'])
        
        # Verify stock code matches
        if 'stock_code' not in df.columns or df['stock_code'].iloc[0] != self.stock_code:
            raise ValueError(f"Stock code mismatch in {features_path}")
            
        return df.sort_values('Date')

    def train(self, features_path: Path) -> Dict[str, Any]:
        """
        Train a stock-specific model
        
        Returns:
            Dictionary with:
            - model: Trained model object
            - train_metadata: Training context
            - test_data: Holdout data for evaluation
        """
        df = self._validate_inputs(features_path)
        
        # Time-based split
        split_idx = int(0.8 * len(df))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Prepare data
        X_train = train_df.drop(columns=['target', 'stock_code', 'Date'])
        y_train = train_df['target']
        X_test = test_df.drop(columns=['target', 'stock_code', 'Date'])
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced'
        )
        self.model.fit(X_train, y_train)
        
        # Prepare metadata
        train_metadata = {
            'stock_code': self.stock_code,
            'training_date': datetime.now().isoformat(),
            'feature_columns': list(X_train.columns),
            'train_date_range': {
                'start': train_df['Date'].min().strftime('%Y-%m-%d'),
                'end': train_df['Date'].max().strftime('%Y-%m-%d')
            },
            'model_type': 'RandomForestClassifier',
            'hyperparameters': self.model.get_params()
        }
        
        return {
            'model': self.model,
            'train_metadata': train_metadata,
            'test_data': {
                'X': X_test,
                'y': test_df['target'].values,
                'dates': test_df['Date'].values
            }
        }

    def save_model(self, model_output_dir: Path, artifacts: Dict[str, Any]):
        """Save trained artifacts"""
        model_output_dir.mkdir(exist_ok=True)
        
        # Save model
        joblib.dump(
            artifacts['model'],
            model_output_dir / f"{self.stock_code}_model.pkl"
        )
        
        # Save training metadata
        with open(model_output_dir / f"{self.stock_code}_train_metadata.json", 'w') as f:
            json.dump(artifacts['train_metadata'], f)
        
        # Save test data for evaluation
        pd.DataFrame({
            'Date': artifacts['test_data']['dates'],
            'y_true': artifacts['test_data']['y']
        }).to_csv(model_output_dir / f"{self.stock_code}_test_data.csv", index=False)

def train_stock_model(stock_code: str, features_path: Path, output_dir: Path) -> bool:
    """
    Pipeline integration point
    
    Returns:
        bool: True if training succeeded
    """
    try:
        trainer = StockModelTrainer(stock_code)
        artifacts = trainer.train(features_path)
        trainer.save_model(output_dir, artifacts)
        return True
    except Exception as e:
        logging.error(f"Training failed for {stock_code}: {str(e)}")
        return False
