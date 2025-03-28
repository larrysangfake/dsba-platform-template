import pytest
import pandas as pd

@pytest.fixture
def sample_stock_data():
    return pd.DataFrame({
        'Close': [150, 151, 149, 152],
        'Volume': [1000, 1200, 800, 1500],
        'Log_Return': [0.01, -0.02, 0.03, -0.01]
    })
