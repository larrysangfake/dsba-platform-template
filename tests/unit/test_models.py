import pytest
import joblib
import numpy as np
from src.dsba.model_prediction import StockPredictionEngine
from src.dsba.model_registry import ModelRegistry

@pytest.fixture
def sample_model():
    # Mock a trained model
    class MockModel:
        def predict(self, X):
            return np.array([1])
        def predict_proba(self, X):
            return np.array([[0.3, 0.7]])
    return MockModel()

@pytest.fixture 
def sample_preprocessor():
    class MockPreprocessor:
        feature_columns = ['close', 'volume']
        def transform(self, X):
            return np.array([[1.0, 2.0]])
    return MockPreprocessor()

def test_prediction_engine(sample_model, sample_preprocessor):
    engine = StockPredictionEngine(sample_model, sample_preprocessor)
    result = engine.predict(pd.DataFrame({'close': [150], 'volume': [1000]}))
    
    assert 'prediction' in result
    assert 'confidence' in result
    assert result['confidence'] == 0.7

def test_model_registry(tmp_path):
    # Test registry with temp directory
    registry = ModelRegistry(tmp_path)
    test_data = {'model': 'test', 'preprocessor': 'test'}
    joblib.dump(test_data, tmp_path / "AAPL_model.pkl")
    
    loaded = registry.load_model("AAPL")
    assert loaded['model'] == 'test'
