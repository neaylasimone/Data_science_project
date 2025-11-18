import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

# Page configuration
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Constants ===
TICKERS = [
    'AAPL','MSFT','AMZN','GOOGL','NVDA','META','TSLA','AMD','ORCL','INTC',
    'CSCO','IBM','ADBE','CRM','QCOM','JPM','BAC','WFC','C','GS','MS','BK',
    'V','MA','PYPL','WMT','COST','KO','PEP','MCD','SBUX','NKE','HD','LOW',
    'TGT','CVS','PG','PM','JNJ','PFE','MRK','ABBV','AMGN','BMY','LLY','UNH',
    'MDT','DHR','XOM','CVX','COP','CAT','DE','BA','GE','HON','LMT','RTX','AMT',
    'SPG'
]

FEATURES = ['SMA_20','SMA_50','EMA_20','Volatility','RSI_14','MACD','MACD_Signal']

def rate_stock(prob_20d, rsi, current_price, predicted_20d):
    summary = ""

    # Simple interpretation of probabilities
    if prob_20d >= 0.60:
        summary += "🟢 **Bullish Outlook (Positive)**\n"
        summary += "- The model predicts a >60% chance of gaining at least 5% in 20 days.\n"
    elif prob_20d >= 0.45:
        summary += "🟡 **Neutral Outlook**\n"
        summary += "- The model sees mixed signals — about 50/50 chance of a 5%+ gain.\n"
    else:
        summary += "🔴 **Bearish Outlook (Weak)**\n"
        summary += "- The model predicts <45% chance of gaining 5% in 20 days.\n"
    
    # RSI interpretation
    if rsi > 70:
        summary += "- RSI indicates the stock may be **overbought** (price may be high).\n"
    elif rsi < 30:
        summary += "- RSI indicates the stock may be **oversold** (price may be low).\n"

    # Expected return
    exp_return = (predicted_20d - current_price) / current_price
    if exp_return > 0.05:
        summary += f"- Expected return over the next month: **{exp_return:.1%}** (positive)\n"
    elif exp_return < -0.05:
        summary += f"- Expected return over the next month: **{exp_return:.1%}** (negative)\n"
    
    return summary

@st.cache_data
def create_features(df, symbol, horizons=[5, 20, 60], pct_thresholds=[0.01, 0.05, 0.10]):
    """Create technical features for the stock data"""
    df = df.sort_index()
    close = df["Close"]
    
    # Technical Indicators
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
    
    # Multi-horizon targets for training
    for h, pct in zip(horizons, pct_thresholds):
        future_col = f'future_close_{h}d'
        return_col = f'return_{h}d'
        target_col = f'target_{h}d'
        
        df[future_col] = close.shift(-h)
        df[return_col] = (df[future_col] - close) / close
        df[target_col] = (df[return_col] > pct).astype(int)
    
    df['symbol'] = symbol
    return df.dropna()

@st.cache_data
def get_company_info(symbol):
    """Get company name and info for a ticker symbol"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        company_name = info.get('longName') or info.get('shortName') or info.get('name') or symbol
        return company_name, info
    except Exception:
        return symbol, None

@st.cache_data
def build_company_ticker_map():
    """Build a mapping of company names to ticker symbols from known tickers"""
    mapping = {}
    reverse_mapping = {}  # ticker -> company name
    
    for ticker in TICKERS:
        try:
            company_name, _ = get_company_info(ticker)
            # Store multiple variations for better matching
            company_upper = company_name.upper()
            ticker_upper = ticker.upper()
            
            # Map company name to ticker
            mapping[company_upper] = ticker
            # Map ticker to ticker (for direct lookup)
            mapping[ticker_upper] = ticker
            # Store reverse mapping
            reverse_mapping[ticker] = company_name
            
            # Also add shortened versions (first word or key words)
            words = company_upper.split()
            if len(words) > 1:
                # Map first word if it's meaningful
                if words[0] not in ['THE', 'INC', 'CORP', 'LTD', 'LLC']:
                    mapping[words[0]] = ticker
        except Exception:
            # If we can't get the name, just use the ticker
            mapping[ticker.upper()] = ticker
            reverse_mapping[ticker] = ticker
    
    return mapping, reverse_mapping

def search_ticker_by_input(user_input):
    """Search for ticker symbol from user input (can be ticker or company name)"""
    if not user_input:
        return None, None
    
    user_input = user_input.strip().upper()
    
    # Build mapping
    company_map, reverse_map = build_company_ticker_map()
    
    # First, check if it's an exact match in our mapping
    if user_input in company_map:
        ticker = company_map[user_input]
        company_name = reverse_map.get(ticker, ticker)
        return ticker, company_name
    
    # Check if it's a direct ticker match (short, all caps, no spaces)
    if len(user_input) <= 5 and user_input.replace('.', '').replace('-', '').isalnum():
        # Try it as a ticker directly
        try:
            test_df = yf.download(user_input, period="5d", auto_adjust=True, progress=False)
            if not test_df.empty:
                # It's a valid ticker, get company name
                company_name, _ = get_company_info(user_input)
                return user_input, company_name
        except Exception:
            pass
    
    # Search for partial matches in company names
    matches = []
    for ticker in TICKERS:
        try:
            company_name = reverse_map.get(ticker)
            if not company_name or company_name == ticker:
                company_name, _ = get_company_info(ticker)
            
            company_upper = company_name.upper()
            
            # Check for partial matches
            if user_input in company_upper or company_upper in user_input:
                matches.append((ticker, company_name))
            # Check word-by-word matching
            elif any(word in company_upper for word in user_input.split() if len(word) > 2):
                matches.append((ticker, company_name))
        except Exception:
            continue
    
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Return best match (exact contains match preferred)
        for ticker, company_name in matches:
            if user_input in company_name.upper():
                return ticker, company_name
        # If no exact match, return first one
        return matches[0]
    
    # Try searching yfinance directly with the input
    try:
        test_df = yf.download(user_input, period="5d", auto_adjust=True, progress=False)
        if not test_df.empty:
            company_name, _ = get_company_info(user_input)
            return user_input, company_name
    except Exception:
        pass
    
    return None, None

@st.cache_data
def download_stock_data(symbol, period="10y"):
    """Download stock data from Yahoo Finance"""
    try:
        df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return None
        
        # Multi-index fix
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(symbol, level=1, axis=1)
        
        return df
    except Exception as e:
        st.error(f"Error downloading {symbol}: {e}")
        return None

@st.cache_resource
def train_models():
    """Train all models using cached data"""
    with st.spinner("Training models... This may take a few minutes on first run."):
        all_data = []
        failed_tickers = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, symbol in enumerate(TICKERS):
            status_text.text(f"Fetching data for {symbol}... ({idx+1}/{len(TICKERS)})")
            df = download_stock_data(symbol, period="10y")
            
            if df is None or df.empty:
                failed_tickers.append(symbol)
                continue
            
            df = create_features(df, symbol)
            all_data.append(df)
            progress_bar.progress((idx + 1) / len(TICKERS))
        
        status_text.text("Preparing dataset...")
        progress_bar.progress(1.0)
        
        if len(all_data) == 0:
            st.error("No data loaded. Cannot train model.")
            return None
        
        dataset = pd.concat(all_data)
        
        # Sort dataset by date (chronological order) for time-based split
        dataset = dataset.sort_index()
        
        # Time-based split: first 80% for training, last 20% for testing
        split_idx = int(len(dataset) * 0.8)
        train_dataset = dataset.iloc[:split_idx]
        test_dataset = dataset.iloc[split_idx:]
        
        # Extract features and targets from train portion
        X_train = train_dataset[FEATURES]
        y_train_5d = train_dataset['target_5d']
        y_train_20d = train_dataset['target_20d']
        y_train_60d = train_dataset['target_60d']
        price_train_5d = train_dataset['future_close_5d']
        price_train_20d = train_dataset['future_close_20d']
        price_train_60d = train_dataset['future_close_60d']
        
        # Extract features and targets from test portion
        X_test = test_dataset[FEATURES]
        y_test_5d = test_dataset['target_5d']
        y_test_20d = test_dataset['target_20d']
        y_test_60d = test_dataset['target_60d']
        price_test_5d = test_dataset['future_close_5d']
        price_test_20d = test_dataset['future_close_20d']
        price_test_60d = test_dataset['future_close_60d']
        
        status_text.text("Training classifiers...")
        model_5d = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        model_20d = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        model_60d = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        
        model_5d.fit(X_train, y_train_5d)
        model_20d.fit(X_train, y_train_20d)
        model_60d.fit(X_train, y_train_60d)
        
        status_text.text("Training regressors...")
        reg_5d = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        reg_20d = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        reg_60d = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        
        reg_5d.fit(X_train, price_train_5d)
        reg_20d.fit(X_train, price_train_20d)
        reg_60d.fit(X_train, price_train_60d)
        
        # Calculate accuracies
        acc_5d = accuracy_score(y_test_5d, model_5d.predict(X_test))
        acc_20d = accuracy_score(y_test_20d, model_20d.predict(X_test))
        acc_60d = accuracy_score(y_test_60d, model_60d.predict(X_test))
        
        progress_bar.empty()
        status_text.empty()
        
        return {
            'classifiers': {'5d': model_5d, '20d': model_20d, '60d': model_60d},
            'regressors': {'5d': reg_5d, '20d': reg_20d, '60d': reg_60d},
            'accuracies': {'5d': acc_5d, '20d': acc_20d, '60d': acc_60d},
            'failed_tickers': failed_tickers
        }

def plot_price_chart(df, symbol):
    """Create an interactive price chart with moving averages"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Moving averages
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_20'],
            name='SMA 20',
            line=dict(color='orange', width=1)
        ))
    
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            name='SMA 50',
            line=dict(color='blue', width=1)
        ))
    
    if 'EMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA_20'],
            name='EMA 20',
            line=dict(color='purple', width=1)
        ))
    
    fig.update_layout(
        title=f'{symbol} Stock Price',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig

def plot_technical_indicators(df, symbol):
    """Create subplot with RSI and MACD"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f'{symbol} Price', 'RSI', 'MACD')
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')),
        row=1, col=1
    )
    
    # RSI
    if 'RSI_14' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI_14'], name='RSI', line=dict(color='orange')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='red')),
            row=3, col=1
        )
    
    fig.update_layout(height=800, showlegend=True, hovermode='x unified')
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig

def plot_predictions(probabilities, prices, current_price, symbol):
    """Create bar chart for predictions"""
    horizons = ['5 days', '20 days', '60 days']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Probability of Price Increase (%)', 'Predicted Prices'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Probability chart
    fig.add_trace(
        go.Bar(
            x=horizons,
            y=[p * 100 for p in probabilities],
            name='Probability',
            marker_color=colors,
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='auto'
        ),
        row=1, col=1
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=1, col=1)
    
    # Price prediction chart
    all_prices = [current_price] + prices
    price_labels = ['Current'] + horizons
    fig.add_trace(
        go.Scatter(
            x=price_labels,
            y=all_prices,
            mode='lines+markers+text',
            name='Price',
            line=dict(color='blue', width=2),
            marker=dict(size=10),
            text=[f'${p:.2f}' for p in all_prices],
            textposition='top center'
        ),
        row=1, col=2
    )
    
    fig.update_layout(height=400, showlegend=False)
    fig.update_xaxes(title_text="Time Horizon", row=1, col=1)
    fig.update_xaxes(title_text="Time Horizon", row=1, col=2)
    fig.update_yaxes(title_text="Probability (%)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=2)
    
    return fig

def plot_feature_importance(models_dict):
    """Plot feature importance for all models"""
    fig = go.Figure()
    
    horizons = ['5d', '20d', '60d']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, horizon in enumerate(horizons):
        model = models_dict['classifiers'][horizon]
        importances = model.feature_importances_
        fig.add_trace(go.Bar(
            x=FEATURES,
            y=importances,
            name=f'{horizon} model',
            marker_color=colors[i]
        ))
    
    fig.update_layout(
        title='Feature Importance Across Models',
        xaxis_title='Features',
        yaxis_title='Importance',
        barmode='group',
        height=400
    )
    
    return fig

# Main App
def main():
    st.title("📈 Stock Prediction Dashboard")
    st.markdown("### AI-Powered Stock Price Prediction with Technical Analysis")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model training section
    st.sidebar.subheader("Model Training")
    if st.sidebar.button("Train/Retrain Models", type="primary"):
        st.session_state.models = train_models()
    
    # Load or train models
    if 'models' not in st.session_state:
        with st.spinner("Initializing models..."):
            st.session_state.models = train_models()
    
    if st.session_state.models is None:
        st.error("Failed to load models. Please retrain.")
        return
    
    # Display model accuracies
    with st.sidebar.expander("Model Performance", expanded=False):
        st.write(f"**5-day model accuracy:** {st.session_state.models['accuracies']['5d']:.2%}")
        st.write(f"**20-day model accuracy:** {st.session_state.models['accuracies']['20d']:.2%}")
        st.write(f"**60-day model accuracy:** {st.session_state.models['accuracies']['60d']:.2%}")
        if st.session_state.models['failed_tickers']:
            st.warning(f"Failed tickers: {', '.join(st.session_state.models['failed_tickers'])}")
    
    # Stock selection
    st.sidebar.subheader("Stock Selection")
    selected_ticker = st.sidebar.selectbox(
        "Select a stock ticker:",
        options=TICKERS,
        index=0
    )
    
    custom_input = st.sidebar.text_input(
        "Or enter ticker or company name:", 
        value="",
        placeholder="e.g., AAPL, Apple, Microsoft..."
    )
    
    # Determine which ticker to use
    ticker = None
    company_name = None
    
    if custom_input:
        # Search for ticker using the input (can be ticker or company name)
        resolved_ticker, resolved_name = search_ticker_by_input(custom_input)
        
        if resolved_ticker:
            ticker = resolved_ticker
            company_name = resolved_name if resolved_name and resolved_name != resolved_ticker else None
            if company_name:
                st.sidebar.success(f"✓ Found: **{company_name}** ({ticker})")
            else:
                st.sidebar.success(f"✓ Using ticker: **{ticker}**")
        else:
            st.sidebar.warning(f"⚠ Could not find ticker for '{custom_input}'. Using dropdown selection.")
            ticker = selected_ticker
    else:
        ticker = selected_ticker
        # Get company name for selected ticker
        try:
            company_name, _ = get_company_info(ticker)
            if company_name == ticker:
                company_name = None
        except Exception:
            company_name = None
    
    period = st.sidebar.selectbox(
        "Data Period:",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
        index=3
    )
    
    # Download and process data
    if st.sidebar.button("Analyze Stock", type="primary") or 'analyze' not in st.session_state:
        st.session_state.analyze = True
    
    if st.session_state.get('analyze', False):
        with st.spinner(f"Fetching data for {ticker}..."):
            df = download_stock_data(ticker, period=period)
            
            if df is None or df.empty:
                st.error(f"No data available for {ticker}. Please try another ticker.")
                st.session_state.analyze = False
            else:
                df = create_features(df, ticker, horizons=[5, 20, 60], pct_thresholds=[0.01, 0.05, 0.10])
                
                if df.empty or len(df) < 1:
                    st.error("Not enough data to create features.")
                    st.session_state.analyze = False
                else:
                    latest = df.tail(1)
                    current_price = df['Close'].iloc[-1]
                    
                    # Get predictions
                    probabilities = []
                    prices = []
                    
                    for horizon in ['5d', '20d', '60d']:
                        prob = st.session_state.models['classifiers'][horizon].predict_proba(
                            latest[FEATURES]
                        )[:,1][0]
                        price = st.session_state.models['regressors'][horizon].predict(
                            latest[FEATURES]
                        )[0]
                        probabilities.append(prob)
                        prices.append(price)
                    
                    # Display results
                    if company_name and company_name != ticker:
                        st.header(f"Analysis for {company_name} ({ticker})")
                    else:
                        st.header(f"Analysis for {ticker}")

                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        st.metric("5-Day Probability", f"{probabilities[0]:.1%}")
                    with col3:
                        st.metric("20-Day Probability", f"{probabilities[1]:.1%}")
                    with col4:
                        st.metric("60-Day Probability", f"{probabilities[2]:.1%}")

                    # Beginner-friendly summary
                    st.subheader("📊 Easy Summary (Beginner Friendly)")
                    beginner_text = rate_stock(
                        prob_20d=probabilities[1],
                        rsi=df['RSI_14'].iloc[-1],
                        current_price=current_price,
                        predicted_20d=prices[1]
                    )
                    st.markdown(beginner_text)

                    
                    # Prediction charts
                    st.subheader("Predictions")
                    pred_fig = plot_predictions(probabilities, prices, current_price, ticker)
                    st.plotly_chart(pred_fig, use_container_width=True)
                    
                    # Price chart
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Price Chart with Moving Averages")
                        price_fig = plot_price_chart(df, ticker)
                        st.plotly_chart(price_fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Current Technical Indicators")
                        indicator_data = {
                            'Indicator': ['RSI (14)', 'MACD', 'MACD Signal', 'Volatility', 
                                        'SMA 20', 'SMA 50', 'EMA 20'],
                            'Value': [
                                df['RSI_14'].iloc[-1],
                                df['MACD'].iloc[-1],
                                df['MACD_Signal'].iloc[-1],
                                df['Volatility'].iloc[-1],
                                df['SMA_20'].iloc[-1],
                                df['SMA_50'].iloc[-1],
                                df['EMA_20'].iloc[-1]
                            ]
                        }
                        indicator_df = pd.DataFrame(indicator_data)
                        st.dataframe(indicator_df, use_container_width=True, hide_index=True)
                    
                    # Technical indicators chart
                    st.subheader("Technical Indicators")
                    tech_fig = plot_technical_indicators(df, ticker)
                    st.plotly_chart(tech_fig, use_container_width=True)
                    
                    # Feature importance
                    st.subheader("Model Feature Importance")
                    importance_fig = plot_feature_importance(st.session_state.models)
                    st.plotly_chart(importance_fig, use_container_width=True)
                    
                    # Detailed predictions table
                    st.subheader("Detailed Predictions")
                    predictions_data = {
                        'Horizon': ['5 days', '20 days', '60 days'],
                        'Probability of Increase': [f"{p:.2%}" for p in probabilities],
                        'Predicted Price': [f"${p:.2f}" for p in prices],
                        'Expected Return': [f"{((p - current_price) / current_price):.2%}" 
                                          for p in prices]
                    }
                    predictions_df = pd.DataFrame(predictions_data)
                    st.dataframe(predictions_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()

