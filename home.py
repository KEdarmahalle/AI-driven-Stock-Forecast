import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import ta
import time
import logging
from dotenv import load_dotenv
from google import genai
import json

# Load environment variables
load_dotenv()

# Setup
logging.basicConfig(level=logging.INFO)
t = logging.getLogger(__name__)

# Constants
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_DIR = 'modelsaved'
SEQUENCE_LENGTH = 60
RATE_LIMIT = 5  # req/min
CACHE_TTL = 60

# Page Configuration
st.set_page_config(page_title="AI-Powered Stock Analysis", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3em;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1em;
    }
    .sub-header {
        font-size: 1.5em;
        color: #9E9E9E;
        text-align: center;
        margin-bottom: 2em;
    }
    .feature-section {
        background-color: rgba(49, 51, 63, 0.2);
        padding: 1em;
        border-radius: 5px;
        margin-bottom: 2em;
    }
    .feature-header {
        color: #4CAF50;
        font-size: 1.2em;
    }
    </style>
    """, unsafe_allow_html=True)

# Welcome Section
st.markdown('<h1 class="main-header">Welcome to AI-Powered Stock Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Harness the Power of Machine Learning for Smarter Trading Decisions</p>', unsafe_allow_html=True)

# Introduction
st.markdown("""
This advanced stock analysis platform combines real-time market data, machine learning predictions, 
and AI-driven market insights to help you make informed trading decisions.
""")

# Feature Highlights
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-section">
        <h3 class="feature-header">🤖 Dual ML Models</h3>
        <p>Compare predictions from both LSTM and GRU models for more reliable forecasting</p>
        <ul>
            <li>Advanced time series analysis</li>
            <li>Multiple prediction horizons</li>
            <li>Real-time updates</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-section">
        <h3 class="feature-header">📊 Technical Indicators</h3>
        <p>Comprehensive technical analysis tools</p>
        <ul>
            <li>Moving Averages</li>
            <li>RSI Indicator</li>
            <li>Bollinger Bands</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-section">
        <h3 class="feature-header">🔍 AI Market Insights</h3>
        <p>AI-powered market analysis and sentiment</p>
        <ul>
            <li>Real-time sentiment analysis</li>
            <li>News impact assessment</li>
            <li>Risk factor identification</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Quick Start Guide
st.markdown("""
---
### 📈 Getting Started
1. Select a sector and stock symbol from the sidebar
2. Choose your preferred timeframe
3. Set your prediction horizon
4. Click "Load Data" to view analysis
5. Use "Get Market Analysis" for AI insights

---
""")

# Initialize Gemini API client with caching to prevent multiple initializations
@st.cache_resource
def get_gemini_client():
    """Get a cached instance of the Gemini client to prevent recreation on each rerun"""
    if GEMINI_API_KEY:
        return genai.Client(api_key=GEMINI_API_KEY)
    return None

# Stock categories for dropdown
STOCK_CATEGORIES = {
    'Tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
    'Finance': ['JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 'WFC'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV', 'LLY'],
    'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'MCD', 'NKE', 'SBUX'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG']
}

# Session state for API call tracking
if 'call_count' not in st.session_state:
    st.session_state.call_count = 0
    st.session_state.last_reset = time.time()
if 'prediction_horizon' not in st.session_state:
    st.session_state.prediction_horizon = 6
if 'current_predictions' not in st.session_state:
    st.session_state.current_predictions = None
if 'current_predictions_gru' not in st.session_state:
    st.session_state.current_predictions_gru = None

# Rate limiter
def check_rate():
    now = time.time()
    if now - st.session_state.last_reset > CACHE_TTL:
        st.session_state.call_count = 0
        st.session_state.last_reset = now
    if st.session_state.call_count >= RATE_LIMIT:
        st.warning(f"Rate limit of {RATE_LIMIT}/min hit. Try again later.")
        return False
    st.session_state.call_count += 1
    return True

# Fetch from Polygon
def fetch_bars(symbol, timespan):
    if not check_rate():
        return None
    to_dt = datetime.utcnow()
    from_dt = to_dt - timedelta(days=30)
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/"
        f"{from_dt.strftime('%Y-%m-%d')}/{to_dt.strftime('%Y-%m-%d')}"
        f"?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_API_KEY}"
    )
    r = requests.get(url)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
        return None
    data = r.json().get('results', [])
    if not data:
        st.error("No data returned.")
        return None
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]

# Load models
@st.cache_resource
def load_models():
    lstm = load_model(os.path.join(MODEL_DIR, 'best_lstm_model_20250420-113831.keras'))
    gru = load_model(os.path.join(MODEL_DIR, 'best_gru_model_20250420-113831.keras'))
    return lstm, gru

# Add indicators
def add_indicators(df):
    df['SMA20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    bb = ta.volatility.BollingerBands(df['close'])
    df['BBU'] = bb.bollinger_hband()
    df['BBL'] = bb.bollinger_lband()
    return df.ffill().bfill()

# ML prep
def prepare_ml(df):
    arr = df[['open', 'high', 'low', 'close', 'volume']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr)
    X = scaled[-SEQUENCE_LENGTH:].reshape(1, SEQUENCE_LENGTH, 5)
    return X, scaler

# Predict
def predict(model, X, scaler, horizon):
    preds = []
    seq = X.copy()
    for _ in range(horizon):
        next_step = model.predict(seq, verbose=0)[0]
        preds.append(next_step[3])  # close
        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, :] = next_step
    dummy = np.zeros((len(preds), 5))
    dummy[:, 3] = preds
    inv = scaler.inverse_transform(dummy)
    return inv[:, 3]

# Plotting
def plot_line_chart(df, lstm_preds=None, gru_preds=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close', line=dict(color='white')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', yaxis='y2', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BBU'], name='BB Upper', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BBL'], name='BB Lower', fill='tonexty'))

    # Add predictions if available
    if lstm_preds is not None or gru_preds is not None:
        last_time = df.index[-1]
        delta = df.index[-1] - df.index[-2]
        
        if lstm_preds is not None:
            pred_times = [last_time + (i + 1) * delta for i in range(len(lstm_preds))]
            full_times = [last_time] + pred_times
            full_prices = [df['close'].iloc[-1]] + list(lstm_preds)
            fig.add_trace(go.Scatter(x=full_times, y=full_prices, mode='lines+markers',
                                    name='LSTM Prediction', line=dict(color='red', dash='dot')))
        
        if gru_preds is not None:
            pred_times_gru = [last_time + (i + 1) * delta for i in range(len(gru_preds))]
            full_times_gru = [last_time] + pred_times_gru
            full_prices_gru = [df['close'].iloc[-1]] + list(gru_preds)
            fig.add_trace(go.Scatter(x=full_times_gru, y=full_prices_gru, mode='lines+markers',
                                    name='GRU Prediction', line=dict(color='green', dash='dot')))
    
    fig.update_layout(
        title='Price & Indicators',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Price'),
        yaxis2=dict(title='RSI', overlaying='y', side='right', showgrid=False),
        xaxis_rangeslider_visible=False,
        template="plotly_dark"  # Dark theme to make white lines more visible
    )
    st.plotly_chart(fig, use_container_width=True)

# Update predictions when horizon changes
def update_predictions():
    if 'df_loaded' in st.session_state and st.session_state.df_loaded is not None:
        df = st.session_state.df_loaded
        X, scaler = prepare_ml(df)
        lstm, gru = load_models()
        horizon = st.session_state.prediction_horizon
        
        # Get predictions from both models
        lstm_preds = predict(lstm, X, scaler, horizon)
        gru_preds = predict(gru, X, scaler, horizon)
        
        st.session_state.current_predictions = lstm_preds
        st.session_state.current_predictions_gru = gru_preds

# Get market insights function
def get_market_insights(symbol):
    """Get market insights using Gemini API"""
    client = get_gemini_client()
    if not client:
        return {
            "error": "Gemini API key not found. Please add GEMINI_API_KEY to your .env file."
        }
    
    try:
        # Create prompt for analysis
        prompt = f"""
        Provide a concise market analysis for {symbol} stock including:
        1. Current market sentiment (bullish/bearish/neutral)
        2. Key price levels and recent performance
        3. Latest news affecting the stock
        4. Risk factors to watch
        5. Potential opportunities

        Format response as JSON with these keys:
        - sentiment: overall market sentiment (bullish/bearish/neutral)
        - summary: brief 1-2 sentence summary of current situation
        - recent_news: array of 2-3 recent news items
        - risk_factors: array of key risk factors
        - opportunities: array of potential positive catalysts
        
        Respond ONLY with the JSON, no other text.
        """
        
        # Configure Google Search for Gemini
        google_search_tool = genai.types.Tool(
            google_search=genai.types.GoogleSearch()
        )
        
        # Generate content
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                tools=[google_search_tool],
                response_modalities=["TEXT"],
            )
        )
        
        # Extract JSON content by parsing the response
        try:
            # First try to parse the entire text as JSON
            result = json.loads(response.text)
            return result
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the text
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response.text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    return result
                except:
                    pass
            
            # Return raw text as fallback
            return {
                "summary": response.text,
                "error": "Failed to parse JSON response"
            }
            
    except Exception as e:
        return {
            "error": f"Error generating market insights: {str(e)}"
        }

# Sidebar
st.sidebar.title("Stock Settings")

# Stock selection dropdown
category = st.sidebar.selectbox("Sector", list(STOCK_CATEGORIES.keys()))
symbol = st.sidebar.selectbox("Stock Symbol", STOCK_CATEGORIES[category])

# Custom symbol option
custom_symbol = st.sidebar.text_input("Or enter custom symbol", "")
if custom_symbol:
    symbol = custom_symbol.upper()

timeframe = st.sidebar.selectbox("Timeframe", ["hour", "day"])

# Horizon slider with callback
def on_horizon_change():
    if 'df_loaded' in st.session_state and st.session_state.df_loaded is not None:
        update_predictions()

horizon = st.sidebar.slider(
    "Predict Hours Ahead", 1, 24, 6, 
    key="prediction_horizon", 
    on_change=on_horizon_change
)

# MAIN PAGE CONTENT
st.title(f"Stock Analysis & Prediction: {symbol}")

# Trigger fetch and prediction only on button
if st.sidebar.button("Load Data"):
    df = fetch_bars(symbol, timeframe)
    if df is not None:
        df = add_indicators(df)
        st.session_state.df_loaded = df
        
        # Initialize predictions right away with both models
        X, scaler = prepare_ml(df)
        lstm, gru = load_models()
        horizon = st.session_state.prediction_horizon
        lstm_preds = predict(lstm, X, scaler, horizon)
        gru_preds = predict(gru, X, scaler, horizon)
        
        st.session_state.current_predictions = lstm_preds
        st.session_state.current_predictions_gru = gru_preds
        
        # Display initial chart
        plot_line_chart(df)
    else:
        st.session_state.df_loaded = None
        st.session_state.current_predictions = None
        st.session_state.current_predictions_gru = None

# Always show predictions if we have data
if 'df_loaded' in st.session_state and st.session_state.df_loaded is not None:
    df = st.session_state.df_loaded
    lstm_preds = st.session_state.current_predictions
    gru_preds = st.session_state.current_predictions_gru
    
    # Show chart with predictions if available
    if lstm_preds is not None or gru_preds is not None:
        plot_line_chart(df, lstm_preds, gru_preds)
        
        # Create columns for predictions
        col1, col2 = st.columns(2)
        
        with col1:
            if lstm_preds is not None:
                st.metric("LSTM Next Prediction", f"${lstm_preds[0]:.2f}")
        
        with col2:
            if gru_preds is not None:
                st.metric("GRU Next Prediction", f"${gru_preds[0]:.2f}")
    else:
        plot_line_chart(df)
    
    # Separate button to refresh predictions
    if st.sidebar.button("Update Prediction"):
        update_predictions()
        st.experimental_rerun()
    
    # Market Insights Section
    st.markdown("---")
    st.header("AI Market Insights")
    
    if st.button("Get Market Analysis"):
        with st.spinner("Generating market insights..."):
            insights = get_market_insights(symbol)
            
            if "error" in insights:
                st.error(insights["error"])
            else:
                # Display sentiment
                if "sentiment" in insights:
                    sentiment = insights["sentiment"]
                    if sentiment.lower() == "bullish":
                        sentiment_color = "green"
                    elif sentiment.lower() == "bearish":
                        sentiment_color = "red"
                    else:
                        sentiment_color = "orange"
                    
                    st.markdown(f"### Market Sentiment: :{sentiment_color}[{sentiment}]")
                
                # Summary
                if "summary" in insights:
                    st.markdown(f"**Summary:** {insights['summary']}")
                
                # Create columns for different sections
                col1, col2 = st.columns(2)
                
                # Recent news
                with col1:
                    if "recent_news" in insights:
                        st.subheader("Recent News")
                        news_items = insights["recent_news"]
                        if isinstance(news_items, list):
                            for news in news_items:
                                if isinstance(news, dict):
                                    st.markdown(f"- **{news.get('date', '')}**: {news.get('headline', news.get('item', str(news)))}")
                                else:
                                    st.markdown(f"- {news}")
                        else:
                            st.write(news_items)
                
                # Risk factors
                with col2:
                    if "risk_factors" in insights:
                        st.subheader("Risk Factors")
                        risk_factors = insights["risk_factors"]
                        if isinstance(risk_factors, list):
                            for risk in risk_factors:
                                st.markdown(f"- {risk}")
                        else:
                            st.write(risk_factors)
                
                # Opportunities
                st.subheader("Opportunities")
                if "opportunities" in insights:
                    opportunities = insights["opportunities"]
                    if isinstance(opportunities, list):
                        for opp in opportunities:
                            st.markdown(f"- {opp}")
                    else:
                        st.write(opportunities)
