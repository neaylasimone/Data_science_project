# stock_probability_model_v3.py

import yfinance as yf
import pandas as pd
import numpy as np

# Safe tqdm import
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# === List of tickers ===
tickers = [
    'AAPL','MSFT','AMZN','GOOGL','NVDA','META','TSLA','AMD','ORCL','INTC',
    'CSCO','IBM','ADBE','CRM','QCOM','JPM','BAC','WFC','C','GS','MS','BK',
    'V','MA','PYPL','WMT','COST','KO','PEP','MCD','SBUX','NKE','HD','LOW',
    'TGT','CVS','PG','PM','JNJ','PFE','MRK','ABBV','AMGN','BMY','LLY','UNH',
    'MDT','DHR','XOM','CVX','COP','CAT','DE','BA','GE','HON','LMT','RTX','AMT',
    'SPG'
]

# ========= Feature Engineering =========
def create_features(df, symbol, horizons=[5, 20, 60], pct_thresholds=[0.01, 0.05, 0.10]):
    df = df.sort_index()
    close = df["Close"]

    # --- Technical Indicators ---
    df['SMA_20'] = close.rolling(20).mean()
    df['SMA_50'] = close.rolling(50).mean()
    df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    df['Volatility'] = close.pct_change().rolling(20).std()
    
    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    RS = roll_up / roll_down
    df['RSI_14'] = 100 - (100 / (1 + RS))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # --- Multi-horizon targets ---
    for h, pct in zip(horizons, pct_thresholds):
        future_col = f'future_close_{h}d'
        return_col = f'return_{h}d'
        target_col = f'target_{h}d'
        
        df[future_col] = close.shift(-h)
        df[return_col] = (df[future_col] - close) / close
        df[target_col] = (df[return_col] > pct).astype(int)
    
    df['symbol'] = symbol
    return df.dropna()


# ========= Download and Prepare Dataset =========
all_data = []
failed_tickers = []

print("Fetching data from Yahoo Finance...\n")
for symbol in tqdm(tickers):
    try:
        df = yf.download(symbol, period="10y", auto_adjust=True)
        if df.empty:
            failed_tickers.append(symbol)
            continue

        # Multi-index fix
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(symbol, level=1, axis=1)

        df = create_features(df, symbol)
        all_data.append(df)

    except Exception as e:
        failed_tickers.append(symbol)
        print(f"Error downloading {symbol}: {e}")

print("\nFailed tickers:", failed_tickers)

if len(all_data) == 0:
    print("\nFATAL ERROR: No data loaded. Cannot train model.")
    exit()


# ========= Build dataset =========
dataset = pd.concat(all_data)
features = ['SMA_20','SMA_50','EMA_20','Volatility','RSI_14','MACD','MACD_Signal']

# Separate classifiers and regressors for 3 horizons
X = dataset[features]

# Targets
y_5d = dataset['target_5d']
y_20d = dataset['target_20d']
y_60d = dataset['target_60d']

price_5d = dataset['future_close_5d']
price_20d = dataset['future_close_20d']
price_60d = dataset['future_close_60d']

# ========= Train/Test split =========
X_train, X_test, y_train_5d, y_test_5d, price_train_5d, price_test_5d = train_test_split(
    X, y_5d, price_5d, test_size=0.2, shuffle=True, random_state=42
)
_, _, y_train_20d, y_test_20d, price_train_20d, price_test_20d = train_test_split(
    X, y_20d, price_20d, test_size=0.2, shuffle=True, random_state=42
)
_, _, y_train_60d, y_test_60d, price_train_60d, price_test_60d = train_test_split(
    X, y_60d, price_60d, test_size=0.2, shuffle=True, random_state=42
)

# ========= Train models =========
print("\nTraining classifiers...")
model_5d = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model_20d = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model_60d = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

model_5d.fit(X_train, y_train_5d)
model_20d.fit(X_train, y_train_20d)
model_60d.fit(X_train, y_train_60d)

print("\nTraining regressors for price prediction...")
reg_5d = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
reg_20d = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
reg_60d = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

reg_5d.fit(X_train, price_train_5d)
reg_20d.fit(X_train, price_train_20d)
reg_60d.fit(X_train, price_train_60d)

# ========= Evaluation =========
print("\n=== Model Evaluation (5d) ===")
preds_5d = model_5d.predict(X_test)
print("Accuracy:", accuracy_score(y_test_5d, preds_5d))
print(classification_report(y_test_5d, preds_5d))

print("\n=== Model Evaluation (20d) ===")
preds_20d = model_20d.predict(X_test)
print("Accuracy:", accuracy_score(y_test_20d, preds_20d))
print(classification_report(y_test_20d, preds_20d))

print("\n=== Model Evaluation (60d) ===")
preds_60d = model_60d.predict(X_test)
print("Accuracy:", accuracy_score(y_test_60d, preds_60d))
print(classification_report(y_test_60d, preds_60d))


# ========= Prediction Mode =========
while True:
    symbol = input("\nEnter a stock to predict (or 'quit'): ").upper()
    if symbol == "QUIT":
        break

    df = yf.download(symbol, period="1y", auto_adjust=True)
    if df.empty:
        print("No data for this ticker.")
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, level=1, axis=1)

    df = create_features(df, symbol)
    latest = df.tail(1)

    # Probabilities
    prob_5d = model_5d.predict_proba(latest[features])[:,1][0]
    prob_20d = model_20d.predict_proba(latest[features])[:,1][0]
    prob_60d = model_60d.predict_proba(latest[features])[:,1][0]

    # Predicted prices
    price_pred_5d = reg_5d.predict(latest[features])[0]
    price_pred_20d = reg_20d.predict(latest[features])[0]
    price_pred_60d = reg_60d.predict(latest[features])[0]

    print(f"\n📈 Probability {symbol} will go up:")
    print(f"  1 week: {prob_5d:.2%}, predicted price: ${price_pred_5d:.2f}")
    print(f"  1 month: {prob_20d:.2%}, predicted price: ${price_pred_20d:.2f}")
    print(f"  3 months: {prob_60d:.2%}, predicted price: ${price_pred_60d:.2f}")
