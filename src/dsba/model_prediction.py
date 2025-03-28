import numpy as np
import pandas as pd
from typing import Dict

class StockPredictionEngine:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        
    def predict(self, live_data: pd.DataFrame) -> Dict:
        """Main prediction workflow"""
        # Validate input
        self._validate_live_data(live_data)
        
        # Transform features
        features = self.preprocessor.transform(live_data)
        
        # Get prediction outputs
        prediction = self.model.predict(features)[0]
        confidence = self.model.predict_proba(features)[0][1]
        
        # Extract market context
        volatility = live_data['volatility_30d'].values[0]
        current_price = live_data['Close'].values[0]
        
        return {
            "prediction": int(prediction),
            "confidence": float(confidence),
            "volatility": float(volatility),
            "current_price": float(current_price)
        }

    def _validate_live_data(self, data: pd.DataFrame):
        """Ensure required fields exist"""
        required = ['Close', 'volatility_30d'] + self.preprocessor.feature_columns
        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
