# preprocessing.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple

class StockPreprocessor:
    def __init__(self, 
                 prediction_horizon: int = 1,  # Predict the next trading day
                 test_size: float = 0.2,
                 holdout_days: int = 30):
        """
        Args:
            prediction_horizon: Predict the next trading day 
            test_size: Fraction of data for validation (excludes holdout)
            holdout_days: Most recent days to reserve for final testing
        """
        self.prediction_horizon = prediction_horizon
        self.test_size = test_size
        self.holdout_days = holdout_days
        self.scaler = RobustScaler()
        self.feature_columns = None

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable for classification"""
        df = df.copy()
        # Binary target: 1 if price increases within horizon
        df['target'] = (
            df['Close'].shift(-self.prediction_horizon) > df['Close']
        ).astype(int)
        return df

    def _temporal_split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Time-aware data splitting"""
        # Holdout set (most recent data)
        holdout = df.iloc[-self.holdout_days:]
        
        # Test set (before holdout)
        test_size = int(len(df) * self.test_size)
        test = df.iloc[-(self.holdout_days + test_size):-self.holdout_days]
        
        # Training data (the rest)
        train = df.iloc[:-(self.holdout_days + test_size)]
        
        return {
            'train': train,
            'test': test,
            'holdout': holdout
        }

    def process(self, features_path: Path) -> Dict:
        """
        Main preprocessing pipeline
        
        Returns:
            {
                'X_train': scaled training features,
                'y_train': training targets,
                'X_test': scaled test features,
                'y_test': test targets,
                'X_holdout': holdout features (unscaled),
                'y_holdout': holdout targets,
                'feature_names': list of feature names,
                'monte_carlo_data': raw data needed for simulations
            }
        """
        # Load and prepare data
        df = pd.read_csv(features_path, parse_dates=['Date'], index_col='Date')
        df = self._create_target(df)
        
        # Temporal split
        splits = self._temporal_split(df)
        
        # Identify feature columns (exclude targets and raw prices)
        self.feature_columns = [
            col for col in df.columns 
            if col.startswith(('ma_', 'close_ratio_', 'volatility_', 'volume_'))
        ]
        
        # Scale features
        X_train = self.scaler.fit_transform(splits['train'][self.feature_columns])
        X_test = self.scaler.transform(splits['test'][self.feature_columns])
        
        # Prepare Monte Carlo data (unscaled)
        mc_data = splits['holdout'][['Close', 'Log_Return', 'Market_State']]
        
        return {
            'X_train': X_train,
            'y_train': splits['train']['target'].values,
            'X_test': X_test,
            'y_test': splits['test']['target'].values,
            'X_holdout': splits['holdout'][self.feature_columns].values,
            'y_holdout': splits['holdout']['target'].values,
            'feature_names': self.feature_columns,
            'monte_carlo_data': mc_data
        }

    def save_artifacts(self, output_dir: Path):
        """Save preprocessing state for inference"""
        output_dir.mkdir(exist_ok=True)
        pd.to_pickle({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'prediction_horizon': self.prediction_horizon
        }, output_dir / 'preprocessor.pkl')
        
