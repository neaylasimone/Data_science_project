# stock_probability_model.py

import yfinance as yf
import pandas as pd
import numpy as np

# Attempt to import tqdm; if missing, fallback
try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# === List of tickers ===
tickers = [
    'AAPL','ABBV','ABT','ACN','ADBE','AIG','AMD','AMGN','AMT','AMZN','AVGO','AXP',
    'BA','BAC','BK','BKNG','BLK','BMY','BRK-B','C','CAT','CHTR','CL','CMCSA','COF','COP',
    'COST','CRM','CSCO','CVS','CVX','DD','DE','DHR','DIS','DOW','DUK','EMR','EXC','F',
    'FDX','GD','GE','GILD','GM','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM',
    'KO','LIN','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','META','MMM','MO','MRK',
    'MS','MSFT','NEE','NFLX','NKE','NVDA','ORCL','PEP','PFE','PG','PM','PYPL','QCOM',
    'RTX','SBUX','SO','SPG','T','TGT','TMO','TMUS','TSLA','TXN','UNH','UNP','UPS','USB',
    'V','VZ','WFC','WMT','XOM'
]


# ========= Feature Engineering =========
def create_features(df, symbol):
    df = df.sort_index()

    close = df["Close"]

    df['SMA_20'] = close.rolling(20).mean()
    df['SMA_50'] = close.rolling(50).mean()
    df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    df['Volatility'] = close.pct_change().rolling(20).std()

    df['future_close'] = close.shift(-30)
    df['return_30d'] = (df['future_close'] - close) / close
    df['target'] = (df['return_30d'] > 0.05).astype(int)

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

        # FIX MULTI-INDEX PROBLEM
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(symbol, level=1, axis=1)

        df = create_features(df, symbol)
        all_data.append(df)

    except Exception as e:
        failed_tickers.append(symbol)


# ---- After loop: check failures ----
print("\nFailed tickers:", failed_tickers)

if len(all_data) == 0:
    print("\nFATAL ERROR: No data loaded. Cannot train model.")
    exit()


# ========= Build dataset =========
dataset = pd.concat(all_data)

features = ['SMA_20', 'SMA_50', 'EMA_20', 'Volatility']
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

    prob = model.predict_proba(latest[features])[:, 1][0]
    print(f"\n📈 Probability {symbol} will do well in next 30 days: {prob:.2%}")
