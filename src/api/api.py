from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from src.dsba.model_registry import ModelRegistry
from src.dsba.model_prediction import StockPredictionEngine
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional

app = FastAPI()

# CORS setup for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockRequest(BaseModel):
    ticker: str
    cost_basis: float  # Price per share when acquired
    shares: int
    min_gain: float  # Minimum acceptable gain (%)
    expected_gain: float  # Expected average gain (%)
    lookback_days: Optional[int] = 365  # Historical data period

@app.post("/analyze")
async def analyze_stock(request: StockRequest):
    """Main analysis endpoint"""
    try:
        # 1. Get market data
        hist_data = get_historical_data(request.ticker, request.lookback_days)
        current_price = hist_data['Close'].iloc[-1]
        
        # 2. Get model prediction
        registry = ModelRegistry()
        pipeline = registry.load_model(request.ticker)
        predictor = StockPredictionEngine(pipeline['model'], pipeline['preprocessor'])
        
        live_features = prepare_features(hist_data)
        prediction = predictor.predict(live_features)
        
        # 3. Monte Carlo Simulation
        simulations = run_monte_carlo(
            current_price=current_price,
            confidence=prediction['confidence'],
            volatility=hist_data['Log_Return'].std()
        )
        
        # 4. Generate recommendations
        analysis = generate_recommendation(
            request=request,
            current_price=current_price,
            simulations=simulations,
            confidence=prediction['confidence']
        )
        
        # 5. Prepare visualization data
        charts = {
            "historical": build_historical_chart(hist_data),
            "monte_carlo": build_monte_carlo_chart(simulations)
        }
        
        return {
            **analysis,
            "charts": charts,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Helper functions
def get_historical_data(ticker: str, lookback_days: int) -> pd.DataFrame:
    """Fetch and prepare historical data"""
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    data = yf.download(ticker, start=start, end=end)
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    return data.dropna()

def prepare_features(hist_data: pd.DataFrame) -> pd.DataFrame:
    """Create features for model prediction"""
    return pd.DataFrame({
        'Close': [hist_data['Close'].iloc[-1]],
        'volatility_30d': hist_data['Log_Return'].rolling(30).std().iloc[-1],
    })

def run_monte_carlo(current_price: float, confidence: float, volatility: float, 
                   n_simulations=1000) -> np.ndarray:
    """Run Monte Carlo simulation"""
    directions = np.random.choice([1, -1], size=n_simulations, p=[confidence, 1-confidence])
    magnitudes = np.random.normal(0, volatility, n_simulations)
    return current_price * (1 + directions * magnitudes)

def generate_recommendation(request: StockRequest, current_price: float, 
                          simulations: np.ndarray, confidence: float) -> dict:
    """Core recommendation logic"""
    # Calculate potential gains
    potential_gains = (simulations - current_price) * request.shares
    total_investment = request.cost_basis * request.shares
    
    # Probability calculations
    prob_profit = (simulations > current_price).mean()
    prob_min_gain = (potential_gains >= request.min_gain/100 * total_investment).mean()
    prob_expected_gain = (potential_gains >= request.expected_gain/100 * total_investment).mean()
    
    # Recommendation logic
    if prob_expected_gain >= 0.7:
        action = "strong_hold"
    elif prob_min_gain >= 0.5:
        action = "hold"
    else:
        action = "consider_selling"
    
    return {
        "current_value": current_price * request.shares,
        "potential_gain": {
            "mean": float(np.mean(potential_gains)),
            "min": float(np.min(potential_gains)),
            "max": float(np.max(potential_gains)),
            "prob_profit": float(prob_profit),
            "prob_min_gain": float(prob_min_gain),
            "prob_expected_gain": float(prob_expected_gain)
        },
        "recommendation": action,
        "confidence": confidence
    }

def build_historical_chart(data: pd.DataFrame) -> dict:
    """Generate historical price chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close'],
        mode='lines',
        name='Price'
    ))
    return fig.to_dict()

def build_monte_carlo_chart(simulations: np.ndarray) -> dict:
    """Generate Monte Carlo distribution chart"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=simulations,
        nbinsx=50,
        name='Price Distribution'
    ))
    return fig.to_dict()
