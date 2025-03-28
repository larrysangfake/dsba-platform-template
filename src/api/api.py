import click
import yaml
from pathlib import Path
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import json
import subprocess
import uvicorn
from typing import Optional

# Configuration
CONFIG_DIR = Path.home() / ".stockml"
MODEL_REGISTRY = CONFIG_DIR / "models"
CONFIG_DIR.mkdir(exist_ok=True)
MODEL_REGISTRY.mkdir(exist_ok=True)

@click.group()
def cli():
    """Stock Analysis MLOps CLI"""
    pass

@cli.command()
@click.argument("ticker")
@click.option("--lookback-days", default=365, help="Days of historical data to fetch")
def fetch_data(ticker, lookback_days):
    """Fetch historical stock data"""
    try:
        end = datetime.now()
        start = end - timedelta(days=lookback_days)
        data = yf.download(ticker, start=start, end=end)
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()
        
        output_file = f"data/{ticker}_{datetime.now().date()}.csv"
        data.to_csv(output_file)
        click.echo(f"Data saved to {output_file}")
    except Exception as e:
        click.echo(f"Error fetching data: {str(e)}", err=True)

@cli.command()
@click.argument("ticker")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--description", help="Model description")
@click.option("--metrics", help="Path to metrics JSON file")
def register(ticker, model_path, description, metrics):
    """Register a trained model"""
    try:
        model_dir = MODEL_REGISTRY / ticker
        model_dir.mkdir(exist_ok=True)
        
        version = datetime.now().strftime("%Y%m%d%H%M%S")
        version_dir = model_dir / version
        version_dir.mkdir()
        
        # Copy model files
        subprocess.run(["cp", "-r", model_path, str(version_dir / "model")])
        
        # Create metadata
        metadata = {
            "ticker": ticker,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "description": description,
        }
        
        if metrics:
            with open(metrics) as f:
                metadata["metrics"] = json.load(f)
        
        with open(version_dir / "metadata.yaml", "w") as f:
            yaml.safe_dump(metadata, f)
            
        click.echo(f"Registered {ticker} model version {version}")
    except Exception as e:
        click.echo(f"Error registering model: {str(e)}", err=True)

@cli.command()
def list_models():
    """List all registered models"""
    try:
        for model_dir in MODEL_REGISTRY.iterdir():
            versions = [v.name for v in model_dir.iterdir() if v.is_dir()]
            click.echo(f"{model_dir.name}: {len(versions)} versions")
            for version in versions:
                metadata_file = model_dir / version / "metadata.yaml"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = yaml.safe_load(f)
                    click.echo(f"  {version}: {metadata.get('description', '')}")
    except Exception as e:
        click.echo(f"Error listing models: {str(e)}", err=True)

@cli.command()
@click.argument("ticker")
@click.option("--version", help="Specific model version")
def info(ticker, version):
    """Show model information"""
    try:
        model_dir = MODEL_REGISTRY / ticker
        if not model_dir.exists():
            raise ValueError(f"No models found for {ticker}")
            
        if version:
            version_dir = model_dir / version
            metadata_file = version_dir / "metadata.yaml"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = yaml.safe_load(f)
                click.echo(yaml.dump(metadata))
            else:
                click.echo(f"No metadata found for version {version}")
        else:
            versions = sorted([v.name for v in model_dir.iterdir() if v.is_dir()], reverse=True)
            if versions:
                click.echo(f"Available versions for {ticker}:")
                for v in versions:
                    click.echo(f"  {v}")
            else:
                click.echo(f"No versions found for {ticker}")
    except Exception as e:
        click.echo(f"Error getting model info: {str(e)}", err=True)

@cli.command()
@click.argument("ticker")
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8000, help="Server port")
def serve(host, port):
    """Start the API server"""
    click.echo(f"Starting API server on {host}:{port}")
    uvicorn.run("api:app", host=host, port=port, reload=True)

@cli.command()
@click.argument("ticker")
@click.option("--days", default=30, help="Days to simulate")
def simulate(ticker, days):
    """Run Monte Carlo simulation"""
    try:
        # This would integrate with your existing simulation code
        click.echo(f"Running simulation for {ticker} over {days} days")
        # Actual implementation would call your existing functions
    except Exception as e:
        click.echo(f"Error running simulation: {str(e)}", err=True)

if __name__ == "__main__":
    cli()
