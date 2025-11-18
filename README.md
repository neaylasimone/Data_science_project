# Stock Prediction Dashboard

A comprehensive stock prediction application using machine learning and technical analysis, built with Streamlit.

## Features

- **Stock Price Prediction**: Predict stock price movements for 5-day, 20-day, and 60-day horizons
- **Interactive Visualizations**: 
  - Real-time price charts with moving averages
  - Technical indicators (RSI, MACD)
  - Prediction probability charts
  - Feature importance visualization
- **Multiple Models**: Separate Random Forest models for different time horizons
- **50+ Pre-configured Tickers**: Including major tech, finance, retail, and energy stocks

## Installation

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Run the Streamlit app:**
```bash
streamlit run stock_app.py
```

The app will open in your default web browser. On first run, it will automatically train the machine learning models (this may take a few minutes).

## How It Works

1. **Model Training**: The app trains Random Forest classifiers and regressors on historical data from 50+ stocks
2. **Feature Engineering**: Creates technical indicators including:
   - Simple Moving Averages (SMA 20, SMA 50)
   - Exponential Moving Average (EMA 20)
   - Relative Strength Index (RSI)
   - MACD and MACD Signal
   - Volatility metrics
3. **Predictions**: Provides:
   - Probability of price increase for 5, 20, and 60-day horizons
   - Predicted future prices
   - Expected returns

## Project Structure

- `stock_app.py` - Main Streamlit application
- `newStockProjectCode.py` - Original command-line version with 3-horizon predictions
- `Stock_project.py` - Original command-line version with single-horizon predictions
- `requirements.txt` - Python dependencies

## Dependencies

- streamlit - Web interface
- yfinance - Stock data
- pandas - Data manipulation
- numpy - Numerical computing
- scikit-learn - Machine learning models
- plotly - Interactive visualizations