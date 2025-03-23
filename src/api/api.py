import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dsba.model_registry import list_models_ids, load_model, load_model_metadata
from dsba.model_prediction import predict_record

class StockAnalysisRequest(BaseModel):
    company_code: str  # Company name or stock code (e.g., "AAPL")
    num_shares: int    # Number of shares owned
    acquisition_price: float  # Price at which shares were acquired
    min_gain: float    # Minimum acceptable gain (e.g., $X or Y%)
    expected_gain: float  # Expected average gain (e.g., $Z or W%)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S,",
)

app = FastAPI()


# using FastAPI with defaults is very convenient
# we just add this "decorator" with the "route" we want.
# If I deploy this app on "https//mywebsite.com", this function can be called by visiting "https//mywebsite.com/models/"
@app.get("/models/")
async def list_models():
    return list_models_ids()

from fastapi import FastAPI, HTTPException
import yfinance as yf
import logging
from dsba.model_registry import load_model, load_model_metadata
from dsba.model_prediction import predict_five_day_average  # New function for prediction

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI()

# Predict stock price and calculate gain
@app.post("/predict/")
async def predict(request: StockAnalysisRequest):
    """
    Predict the next five working days' average stock price and calculate potential gain.
    """
    try:
        # Fetch historical stock data
        stock_data = yf.download(request.company_code, period="1mo")  # Last month's data
        historical_prices = stock_data["Close"].tolist()

        # Load the model and metadata
        model = load_model("arima_model")  # Example model ID
        metadata = load_model_metadata("arima_model")

        # Predict the next five working days' average price
        predicted_avg_price = predict_five_day_average(model, historical_prices, metadata.target_column)

        # Calculate potential gain
        acquisition_cost = request.num_shares * request.acquisition_price
        predicted_value = request.num_shares * predicted_avg_price
        potential_gain = predicted_value - acquisition_cost

        # Compare gain to user expectations
        if potential_gain >= request.expected_gain:
            recommendation = "Sell now"
        elif potential_gain >= request.min_gain:
            recommendation = "Consider selling"
        else:
            recommendation = "Hold"

        # Return the results
        return {
            "predicted_avg_price": predicted_avg_price,
            "potential_gain": potential_gain,
            "recommendation": recommendation,
        }
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

import pandas as pd

def predict_five_day_average(model, historical_prices, target_column):
    """
    Predict the next five working days' average stock price.
    """
    # Convert historical prices to a DataFrame
    input_data = pd.DataFrame(historical_prices, columns=["price"])

    # Preprocess the data (e.g., calculate moving averages)
    input_data["moving_avg"] = input_data["price"].rolling(window=5).mean()

    # Make predictions for the next five days
    predictions = model.predict(input_data[["price", "moving_avg"]])

    # Calculate the average of the next five days' predictions
    predicted_avg_price = predictions[-5:].mean()

    return predicted_avg_price


@app.api_route("/predict/", methods=["GET", "POST"])
async def predict(query: str, model_id: str):
    """
    Predict the target column of a record using a model.
    The query should be a json string representing a record.
    """
    # This function is a bit naive and focuses on the logic.
    # To make it more production-ready you would want to validate the input, manage authentication,
    # process the various possible errors and raise an appropriate HTTP exception, etc.
    try:
        record = json.loads(query)
        model = load_model(model_id)
        metadata = load_model_metadata(model_id)
        prediction = classify_record(model, record, metadata.target_column)
        return {"prediction": prediction}
    except Exception as e:
        # We do want users to be able to see the exception message in the response
        # FastAPI will by default block the Exception and send a 500 status code
        # (In the HTTP protocol, a 500 status code just means "Internal Server Error" aka "Something went wrong but we're not going to tell you what")
        # So we raise an HTTPException that contains the same details as the original Exception and FastAPI will send to the client.
        raise HTTPException(status_code=500, detail=str(e))
