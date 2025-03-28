# 📊 Stock Advisory App

A machine-learning-powered stock advisory platform designed to help stockholders make informed decisions on whether to **hold** or **sell** their stocks.

## 🧠 Overview
The **Stock Advisory App** predicts next-day price movements using a **Random Forest** classifier and provides actionable recommendations based on probabilistic forecasts. Users input stock details such as:

- Stock ticker
- Purchase cost/price & amount
- Minimum desired gain
- Expected average gain

The app calculates potential gains using **Monte Carlo simulations** and compares them against user-defined thresholds to generate personalized **hold or sell** recommendations.

## 🚀 Features

### Core Functionality
- **Next-Day Prediction**: Predicts whether a stock's price will rise using a **Random Forest** model.
- **Monte Carlo Forecasting**: Simulates future price movements based on historical volatility and model predictions.
- **Custom Recommendations**: Provides personalized **hold** or **sell** advice based on user-input gain thresholds.

### Insights & Visualization
- Historical price trends via interactive charts.
- Monte Carlo simulation outputs with confidence intervals.
- Real-time estimation of potential gains based on market conditions.

### Monitoring & Automation
- Automated model retraining via **GitHub Actions**.
- System health and model drift monitoring using **Prometheus** and **Grafana**.

## 🛠️ Installation & Setup

### Requirements
Ensure you have the following installed:

- Python 3.12
- Docker & Kubernetes
- Git

### Clone the Repository

```bash
git clone <your-repo-url>
cd stock-advisory-app
```

### Set Up Environment

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate  # For Windows: .venv\Scripts\activate
```

Install dependencies:

```bash
pip install -e .
```

### Configure Environment Variables
Set the `DSBA_MODELS_ROOT_PATH` for model storage:

```bash
export DSBA_MODELS_ROOT_PATH="/path/to/models"
```

### Run Tests

```bash
pytest
```

## 📊 Usage

### 1. API Interface

Start the application (local development):

```bash
docker-compose up
```

Example API Call:

```bash
curl -X POST "http://localhost:8000/predict" -d '{"ticker":"AAPL","cost":150,"amount":10,"min_gain":5,"avg_gain":10}'
```

### 2. CLI Interface

Check available models:

```bash
src/cli/dsba_cli list
```

Run a prediction from a CSV file:

```bash
src/cli/dsba_cli predict --input /path/to/data.csv --output /path/to/output.csv --model-id your_model_id
```

### 3. Monitoring Setup

Set up monitoring using **Prometheus** and **Grafana**:

```bash
# Apply Prometheus configuration
kubectl apply -f monitor/prometheus_config.yml

# Install Grafana
kubectl apply -f https://raw.githubusercontent.com/grafana/helm-charts/main/charts/grafana/values.yaml

# Access Grafana dashboard
kubectl port-forward svc/grafana 3000:80
```
Access Grafana at [http://localhost:3000](http://localhost:3000) (default: admin/admin).

## 📅 Roadmap

Check our full development roadmap in [docs/roadmap.md](./docs/roadmap.md).

## 📚 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## 👥 Owner
Baichuan DU

Binong HAN

Linhui SANG



