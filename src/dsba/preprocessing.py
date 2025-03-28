# preprocessing.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Tuple

class StockPreprocessor:
    def __init__(self, 
                 stock_code: str,
                 prediction_horizon: int = 1,  # Predict the next trading day
                 test_size: float = 0.2,
                 holdout_days: int = 30):
        """
        Args:
            prediction_horizon: Predict the next trading day 
            test_size: Fraction of data for validation (excludes holdout)
            holdout_days: Most recent days to reserve for final testing
        """
        self.stock_code = stock_code
        self.prediction_horizon = prediction_horizon
        self.test_size = test_size
        self.holdout_days = holdout_days
        self.scaler = RobustScaler()
        self.feature_columns: List[str] = []
        self.logger = logging.getLogger(f"preprocessor.{stock_code}")

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable for classification"""
        if 'Close' not in df.columns:
            raise ValueError("Missing 'Close' column for target creation")
            
        df = df.copy()
        df['target'] = (
            df['Close'].shift(-self.prediction_horizon) > df['Close']
        ).astype(int)
        return df.dropna()

    def _temporal_split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Time-aware data splitting with length checks"""
        total_samples = len(df)
        required = self.holdout_days + int(total_samples * self.test_size) + 10
        if total_samples < required:
            raise ValueError(f"Need at least {required} samples, got {total_samples}")
        
        # Split indices
        holdout_start = -self.holdout_days
        test_start = holdout_start - int(total_samples * self.test_size)
        
        return {
            'train': df.iloc[:test_start],
            'test': df.iloc[test_start:holdout_start],
            'holdout': df.iloc[holdout_start:]
        }

    def process(self, features_path: Path) -> Dict:
        """Main preprocessing pipeline with validation"""
        df = pd.read_csv(features_path, parse_dates=['Date'], index_col='Date')
        df = df.sort_index().asfreq('D')  # Ensure daily frequency
        
        # Validate core columns
        required_cols = {'Close', 'Volume', 'High', 'Low'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
            
        df = self._create_target(df)
        splits = self._temporal_split(df)

        split_dates = {
        'train_dates': splits['train'].index,
        'test_dates': splits['test'].index,
        'holdout_dates': splits['holdout'].index
        }
        
        # Feature selection
        self.feature_columns = [
            col for col in df.columns 
            if col.startswith(('ma_', 'close_ratio_', 'volatility_', 'volume_'))
            and col != 'target'
        ]
        
        # Scaling
        self.scaler.fit(splits['train'][self.feature_columns])
        return {
            'X_train': self.scaler.transform(splits['train'][self.feature_columns]),
            'y_train': splits['train']['target'].values,
            'X_test': self.scaler.transform(splits['test'][self.feature_columns]),
            'y_test': splits['test']['target'].values,
            'X_holdout': splits['holdout'][self.feature_columns].values,
            'y_holdout': splits['holdout']['target'].values,
            'feature_names': self.feature_columns,
            'monte_carlo_data': splits['holdout'][['Close', 'Log_Return']],  # Removed Market_State
            split_dates
        }

    def transform(self, live_data: pd.DataFrame) -> np.ndarray:
        """Production data transformation"""
        self._validate_features(live_data)
        return self.scaler.transform(live_data[self.feature_columns])

    def _validate_features(self, df: pd.DataFrame):
        """Strict feature validation"""
        missing = set(self.feature_columns) - set(df.columns)
        if missing:
            self.logger.error(f"Missing features: {missing}")
            raise ValueError(f"Missing {len(missing)} features")
            
        if df[self.feature_columns].isna().any().any():
            self.logger.error("NaN values in features")
            raise ValueError("Input contains NaN values")

    def save_artifacts(self, output_dir: Path):
        """Versioned artifact storage"""
        output_dir.mkdir(exist_ok=True)
        artifacts = {
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'prediction_horizon': self.prediction_horizon,
            'stock_code': self.stock_code,
            'feature_prefixes': ('ma_', 'close_ratio_', 'volatility_', 'volume_')
        }
        pd.to_pickle(artifacts, output_dir / 'preprocessor.pkl')
        
