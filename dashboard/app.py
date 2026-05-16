import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import os
import yfinance as yf

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="StockAI | Market Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        font-size: 15px;
    }

    .main { background-color: #0b0f1a; }

    /* Bigger base text */
    p, li, .stMarkdown { font-size: 15px !important; line-height: 1.7; }

    /* Metric cards with colored left border */
    [data-testid="stMetric"] {
        background: #111827;
        padding: 20px 22px !important;
        border-radius: 12px;
        border: 1px solid #1f2937;
        border-left: 4px solid #6366f1 !important;
    }

    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #9ca3af !important; text-transform: uppercase; letter-spacing: 0.08em; }
    [data-testid="stMetricValue"] { font-size: 1.7rem !important; font-weight: 700 !important; color: #f9fafb !important; }
    [data-testid="stMetricDelta"] { font-size: 0.9rem !important; }

    /* Section headers */
    .section-header {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #6366f1;
        font-weight: 600;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 2px solid #1f2937;
    }

    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #111827 60%);
        padding: 28px;
        border-radius: 16px;
        border: 1px solid #312e81;
        color: #f9fafb;
        margin-bottom: 20px;
    }

    .direction-bullish {
        color: #34d399;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -1px;
        text-shadow: 0 0 30px rgba(52, 211, 153, 0.3);
    }

    .direction-bearish {
        color: #f87171;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -1px;
        text-shadow: 0 0 30px rgba(248, 113, 113, 0.3);
    }

    .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6b7280;
        font-weight: 600;
    }

    .value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f9fafb;
        margin-top: 3px;
    }

    .value-accent { color: #818cf8; }

    /* Stat row inside prediction card */
    .stat-box {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 12px 16px;
        flex: 1;
    }

    /* Sentiment badge */
    .sentiment-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .positive { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
    .negative { background: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
    .neutral  { background: rgba(156,163,175,0.1);  color: #9ca3af; border: 1px solid rgba(156,163,175,0.2); }

    /* Sentiment result card */
    .sentiment-card {
        background: #111827;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #1f2937;
        margin-top: 14px;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 28px;
        border: 1px solid #312e81;
    }

    .app-title {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
    }

    .app-subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        margin-top: 4px;
    }

    .version-tag {
        display: inline-block;
        margin-left: 14px;
        padding: 3px 10px;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.4);
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 700;
        color: #818cf8;
        letter-spacing: 0.08em;
        vertical-align: middle;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b0f1a 0%, #111827 100%);
        border-right: 1px solid #1f2937;
    }

    section[data-testid="stSidebar"] .stSelectbox label {
        font-size: 0.8rem !important;
        color: #9ca3af !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 10px 20px !important;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Expander */
    .streamlit-expanderHeader {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #d1d5db !important;
    }

    /* Divider */
    hr { border-color: #1f2937 !important; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
    <div class='app-header'>
        <div>
            <span class='app-title'>StockAI</span>
            <span class='version-tag'>v1.0 LIVE</span>
        </div>
        <div class='app-subtitle'>
            Market Intelligence Platform &mdash; LSTM Price Forecasting &amp; FinBERT Sentiment Analysis
        </div>
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<div class='section-header'>Symbol</div>", unsafe_allow_html=True)

    symbol = st.selectbox(
        "Trading Symbol",
        ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        help="NIFTY blue-chip stocks supported.",
        label_visibility="collapsed"
    )

    st.markdown(f"""
        <div style='background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2);
                    border-radius: 10px; padding: 14px 16px; margin: 12px 0;'>
            <div style='font-size:0.75rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.08em;'>Selected</div>
            <div style='font-size:1.3rem; font-weight:800; color:#818cf8; margin-top:2px;'>{symbol}</div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<div class='section-header'>Model Training</div>", unsafe_allow_html=True)
    st.caption("LSTM uses a 60-day rolling window on 5 years of price history to forecast next-day movement.")

    if st.button("Retrain Model", use_container_width=True):
        with st.spinner("Training on 5 years of historical data..."):
            res = requests.post(f"{API_URL}/train?symbol={symbol}")
            if res.status_code == 200:
                st.success("Training complete.")
            else:
                st.error("Training failed. Check backend connection.")

# --- HOW IT WORKS EXPANDER ---
with st.expander("How StockAI works"):
    st.markdown("""
    StockAI uses a **three-layer hybrid decision system**:

    1. **Quantitative Analysis** — An LSTM deep learning model trained on 5 years of price data forecasts the next closing price.
    2. **Technical Analysis** — RSI (momentum) and MACD (trend) indicators provide additional signals.
    3. **Sentiment Analysis** — FinBERT, a transformer model fine-tuned on financial news, scores market headlines as Positive, Neutral, or Negative.
    """)

# --- MAIN LAYOUT ---
col1, col2 = st.columns([2.2, 1], gap="large")

with col1:
    st.markdown("<div class='section-header'>5-Year Price History</div>", unsafe_allow_html=True)

    with st.spinner(f"Loading market data for {symbol}..."):
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5y")

        if df.empty:
            st.error(f"No data found for {symbol}.")
            df = pd.DataFrame(columns=['Close', 'MA20', 'MA50'])
        else:
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df = df.reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Close'],
        name="Close Price",
        line=dict(color='#818cf8', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(129, 140, 248, 0.05)',
        hovertemplate="<b>Date:</b> %{x}<br><b>Close:</b> ₹%{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['MA20'],
        name="MA 20",
        line=dict(color='#34d399', dash='dot', width=1.8),
        hovertemplate="<b>Date:</b> %{x}<br><b>MA20:</b> ₹%{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['MA50'],
        name="MA 50",
        line=dict(color='#fbbf24', dash='dot', width=1.8),
        hovertemplate="<b>Date:</b> %{x}<br><b>MA50:</b> ₹%{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=13, color='#9ca3af')
        ),
        dragmode="pan",
        xaxis=dict(
            rangeslider=dict(visible=False),
            type='date',
            gridcolor='#1f2937',
            color='#6b7280',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            gridcolor='#1f2937',
            color='#6b7280',
            tickfont=dict(size=12)
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})

    st.divider()

    # Indicators section
    st.markdown("<div class='section-header'>Technical Indicators</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("RSI", "62.4", "+2.1", help="Relative Strength Index: Above 70 is Overbought, Below 30 is Oversold.")
    c2.metric("MACD", "12.5", "-0.5", help="Moving Average Convergence Divergence: Measures trend momentum.")
    c3.metric("Volatility", "1.8%", "Stable")

with col2:
    # --- AI PREDICTION ---
    st.markdown("<div class='section-header'>AI Price Forecast</div>", unsafe_allow_html=True)
    try:
        res = requests.post(f"{API_URL}/predict", json={
            "symbol": symbol,
            "headlines": []
        }, timeout=60)

        if res.status_code == 200:
            pred = res.json()
            direction = pred.get('predicted_direction', 'N/A').upper()
            price = pred.get('predicted_next_price', 0)
            current = pred.get('current_price', 0)
            conf = pred.get('confidence', 85)
            direction_class = "direction-bullish" if direction == "BULLISH" else "direction-bearish"
            change = price - current
            change_pct = (change / current * 100) if current else 0
            change_color = "#34d399" if change >= 0 else "#f87171"
            change_sign = "+" if change >= 0 else ""

            st.markdown(f"""
                <div class='prediction-card'>
                    <div class='label'>LSTM Model Signal</div>
                    <div class='{direction_class}'>{direction}</div>
                    <div style='margin-top: 20px; display: flex; gap: 12px;'>
                        <div class='stat-box'>
                            <div class='label'>Current</div>
                            <div class='value'>&#8377;{current:.2f}</div>
                        </div>
                        <div class='stat-box'>
                            <div class='label'>Forecast</div>
                            <div class='value'>&#8377;{price:.2f}</div>
                        </div>
                    </div>
                    <div style='margin-top: 12px; display: flex; gap: 12px;'>
                        <div class='stat-box'>
                            <div class='label'>Expected Move</div>
                            <div class='value' style='color:{change_color};'>{change_sign}{change:.2f} ({change_sign}{change_pct:.1f}%)</div>
                        </div>
                        <div class='stat-box'>
                            <div class='label'>Confidence</div>
                            <div class='value value-accent'>{conf}%</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Model not trained. Click **Retrain Model** in the sidebar.")
    except Exception:
        st.error("Prediction service unavailable. Ensure the backend is running.")

    st.divider()

    # --- FEATURE IMPORTANCE ---
    st.markdown("<div class='section-header'>Feature Importance</div>", unsafe_allow_html=True)
    try:
        imp_res = requests.get(f"{API_URL}/feature_importance?symbol={symbol}")
        if imp_res.status_code == 200:
            data = imp_res.json()
            if "error" in data and data["error"] == "Random Forest model not trained.":
                st.caption("Train the model to see feature importance.")
            else:
                importances = data.get('importances', {})
                if importances:
                    total = sum(importances.values())
                    imp_df = pd.DataFrame({
                        'Feature': list(importances.keys()),
                        'Importance': [(v/total)*100 for v in importances.values()]
                    }).sort_values('Importance', ascending=True)

                    colors = ['#6366f1', '#818cf8', '#38bdf8', '#34d399']
                    fig_imp = go.Figure(go.Bar(
                        x=imp_df['Importance'],
                        y=imp_df['Feature'],
                        orientation='h',
                        text=[f"{val:.1f}%" for val in imp_df['Importance']],
                        textposition='auto',
                        textfont=dict(size=13, color='white'),
                        marker=dict(
                            color=colors[:len(imp_df)],
                            line=dict(color='rgba(0,0,0,0)', width=0)
                        )
                    ))
                    fig_imp.update_layout(
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(showticklabels=False, showgrid=False),
                        yaxis=dict(color='#9ca3af', tickfont=dict(size=13)),
                        height=210
                    )
                    st.plotly_chart(fig_imp, use_container_width=True, config={'displayModeBar': False})

                    top_feature = imp_df.iloc[-1]['Feature']
                    st.caption(f"Top signal: **{top_feature}**")
                else:
                    st.caption("No importance data available.")
    except Exception:
        st.caption("Feature importance temporarily unavailable.")

    st.divider()

    # --- SENTIMENT ANALYSIS ---
    st.markdown("<div class='section-header'>News Sentiment Analysis</div>", unsafe_allow_html=True)
    news_input = st.text_area(
        "Headlines",
        "Reliance Industries announces major green energy investment\nTCS expects strong growth in cloud services next quarter",
        height=130,
        label_visibility="collapsed",
        placeholder="Enter one headline per line..."
    )
    st.caption("Paste news headlines above, one per line.")

    if st.button("Run Sentiment Analysis", use_container_width=True):
        with st.spinner("Running FinBERT NLP analysis..."):
            headlines = [h.strip() for h in news_input.split("\n") if h.strip()]
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"symbol": symbol, "headlines": headlines},
                    timeout=120
                )

                if response.status_code == 200:
                    res = response.json()
                    sent_val = res['sentiment_analysis']['sentiment']
                    score = res['sentiment_analysis']['score']

                    badge_class = "positive" if "Pos" in sent_val else "negative" if "Neg" in sent_val else "neutral"
                    score_color = "#34d399" if "Pos" in sent_val else "#f87171" if "Neg" in sent_val else "#9ca3af"

                    # Score bar width (normalize -1 to 1 → 0% to 100%)
                    bar_width = int((score + 1) / 2 * 100)

                    st.markdown(f"""
                        <div class='sentiment-card'>
                            <div class='label'>Market Mood</div>
                            <div style='margin-top: 8px;'>
                                <span class='sentiment-badge {badge_class}'>{sent_val.upper()}</span>
                            </div>
                            <div style='margin-top: 16px;'>
                                <div class='label'>Sentiment Score</div>
                                <div style='font-size: 1.5rem; font-weight: 800; color: {score_color}; margin: 4px 0;'>{score:+.3f}</div>
                                <div style='background: #1f2937; border-radius: 6px; height: 8px; overflow: hidden;'>
                                    <div style='width: {bar_width}%; height: 100%; background: linear-gradient(90deg, #6366f1, {score_color}); border-radius: 6px;'></div>
                                </div>
                                <div style='display: flex; justify-content: space-between; margin-top: 4px;'>
                                    <span style='font-size:0.72rem; color:#6b7280;'>Negative</span>
                                    <span style='font-size:0.72rem; color:#6b7280;'>Positive</span>
                                </div>
                            </div>
                            <div style='font-size: 0.8rem; color: #6b7280; margin-top: 14px; border-top: 1px solid #1f2937; padding-top: 12px;'>
                                Powered by <strong style='color:#818cf8;'>FinBERT</strong> &mdash; transformer model fine-tuned on financial news.
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                elif response.status_code == 404:
                    st.warning("Model not trained. Click **Retrain Model** in the sidebar first.")
                else:
                    st.error(f"Sentiment analysis failed (HTTP {response.status_code}): {response.text[:200]}")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The model may still be loading — try again in a moment.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend service. Ensure Docker containers are running.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

st.divider()
st.caption("StockAI v1.0  |  Data provided by Yahoo Finance  |  2026")
