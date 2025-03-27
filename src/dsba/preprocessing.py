from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler 

class TimeSeriesPreprocessor:
    def __init__(self, 
                 n_splits: int = 5,
                 test_size: int = 30,  # days
                 holdout_size: int = 30):
        self.scaler = RobustScaler()
        self.test_size = test_size
        self.holdout_size = holdout_size
        self.n_splits = n_splits
        self.feature_columns = None

    def create_ts_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-aware feature engineering"""
        df = df.sort_index()
        
        # Lag features (avoid future leakage)
        for lag in [1, 2, 5, 20]:
            df[f'return_{lag}d'] = df['Close'].pct_change(lag)
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
        
        # Volatility (expanding window)
        df['volatility'] = df['Close'].pct_change().rolling(21, min_periods=1).std()
        
        # Volume features
        df['volume_z'] = (df['Volume'] - df['Volume'].rolling(30).mean()) / df['Volume'].rolling(30).std()
        
        # Target: Next day's return (binary classification)
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        return df.dropna()

    def temporal_split(self, df: pd.DataFrame) -> dict:
        """Chronological split with holdout"""
        holdout = df.iloc[-self.holdout_size:]
        test = df.iloc[-(self.holdout_size + self.test_size):-self.holdout_size]
        train_val = df.iloc[:-(self.holdout_size + self.test_size)]
        
        # TimeSeriesCV for validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        splits = []
        for train_idx, val_idx in tscv.split(train_val):
            splits.append({
                'X_train': train_val.iloc[train_idx],
                'X_val': train_val.iloc[val_idx]
            })
            
        return {
            'splits': splits,
            'test': test,
            'holdout': holdout
        }

    def scale_features(self, X: pd.DataFrame) -> np.ndarray:
        """Fit scaler on training only"""
        return self.scaler.fit_transform(X)
