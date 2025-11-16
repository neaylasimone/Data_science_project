# stock_probability_model_v2.py

import yfinance as yf
import pandas as pd
import numpy as np

# Safe tqdm import
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# === List of tickers ===
tickers = [
    'AAPL','MSFT','AMZN','GOOGL','NVDA','META','TSLA','AMD','ORCL','INTC',
    'CSCO','IBM','ADBE','CRM','QCOM','JPM','BAC','WFC','C','GS','MS','BK',
    'V','MA','PYPL','WMT','COST','KO','PEP','MCD','SBUX','NKE','HD','LOW',
    'TGT','CVS','PG','PM','JNJ','PFE','MRK','ABBV','AMGN','BMY','LLY','UNH',
    'MDT','DHR','XOM','CVX','COP','CAT','DE','BA','GE','HON','LMT','RTX','AMT',
    'SPG']


# ========= Feature Engineering =========
def create_features(df, symbol, target_days=30, target_pct=0.05):
    df = df.sort_index()
    close = df["Close"]

    # Moving averages
    df['SMA_20'] = close.rolling(20).mean()
    df['SMA_50'] = close.rolling(50).mean()
    df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    df['Volatility'] = close.pct_change().rolling(20).std()

    # RSI calculation
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    RS = roll_up / roll_down
    df['RSI_14'] = 100 - (100 / (1 + RS))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Target
    df['future_close'] = close.shift(-target_days)
    df['return_future'] = (df['future_close'] - close) / close
    df['target'] = (df['return_future'] > target_pct).astype(int)

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
X = dataset[features]
y = dataset['target']

# ========= Train Test Split =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# ========= Train Model =========
print("\nTraining model...\n")
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ========= Evaluation =========
print("\n=== Model Evaluation ===")
preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:")
print(classification_report(y_test, preds))


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

    prob = model.predict_proba(latest[features])[:,1][0]
    print(f"\n📈 Probability {symbol} will go up in the next 30 days: {prob:.2%}")
