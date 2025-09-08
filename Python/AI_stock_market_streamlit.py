import streamlit as st
from plotly.subplots import make_subplots
import pandas as pd
import ta
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split


# ===================== CSS Styling =====================
def add_custom_css():
    st.markdown("""
        <style>
            body {
                background: linear-gradient(135deg, #f8f9fa 0%, #e3f2fd 100%);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .hero {
                padding: 2rem;
                text-align: center;
                border-radius: 15px;
                background: linear-gradient(135deg, #2196F3, #21CBF3);
                color: white;
                margin-bottom: 2rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            }
            .upload-box {
                border: 2px dashed #2196F3;
                padding: 1.5rem;
                border-radius: 15px;
                background-color: #ffffffcc;
                margin-bottom: 1.5rem;
            }
            .stButton>button {
                background-color: #2196F3;
                color: white;
                border-radius: 10px;
                padding: 0.6rem 1.2rem;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)


# ===================== HERO SECTION =====================
def show_hero_section():
    st.markdown("""
        <div class="hero">
            <h1>📈 AI Stock Market Predictor</h1>
            <p>Upload your stock data (Excel or CSV) and get insights into trends, 
            next-day percentage movement, and AI-based future forecasts.</p>
            <p><b>⚠️ Disclaimer:</b> This tool is for <b>educational purposes only</b>. 
            It does not provide financial advice. Invest at your own risk.</p>
        </div>
    """, unsafe_allow_html=True)


# ===================== FILE UPLOAD =====================
def upload_file():
    st.markdown("<div class='upload-box'><h3>📤 Upload your Stock Data</h3></div>", unsafe_allow_html=True)
    return st.file_uploader("", type=["csv", "xlsx"])


# ===================== CLEANING =====================
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess stock data."""
    df = df.dropna().reset_index(drop=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Convert to datetime and sort
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Convert numeric cols
    for col in df.columns:
        if col not in ["date", "series"]:
            df[col] = df[col].replace(",", "", regex=True).astype(float)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and target labels."""
    # SMA
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["sma_200"] = df["close"].rolling(window=200).mean()

    # RSI
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # Volatility
    df["returns"] = df["close"].pct_change()
    df["volatility_5"] = df["returns"].rolling(window=5).std()

    # Momentum
    df["roc_5"] = df["close"].pct_change(periods=5)
    df["relative_strength"] = df["close"] / df["close"].rolling(window=5).mean()

    # Lag features
    df["close_lag1"] = df["close"].shift(1)
    df["close_lag2"] = df["close"].shift(2)

    # Targets
    def classify_trend(r):
        if r > 0.01:
            return 2  # Bullish
        elif r < -0.01:
            return 0  # Bearish
        else:
            return 1  # Neutral

    df["target_price"] = df["close"].shift(-1)
    df["target_return"] = (df["target_price"] - df["close"]) / df["close"]
    df["trend"] = (df["target_price"] > df["close"]).astype(int)
    df["multiclass_trend"] = df["target_return"].apply(classify_trend)

    return df.dropna().reset_index(drop=True)


def train_random_forest(df: pd.DataFrame):
    """Train RandomForest and predict next day trend with confidence."""
    features = [
        "close", "vwap", "volume", "sma_50", "rsi_14",
        "macd", "macd_hist",
        "returns", "volatility_5", "roc_5", "relative_strength",
        "close_lag1"
    ]
    X = df[features]
    y = df["trend"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=250, random_state=42)
    model.fit(X_train, y_train)

    # Predict the last available row (latest day)
    latest = X.iloc[[-1]]
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0][pred]

    trend = "📈 Uptrend" if pred == 1 else "📉 Downtrend"

    return trend, round(prob * 100, 2)


def plot_candlestick(df: pd.DataFrame):
    """Plot candlestick chart with SMA overlays."""
    fig = go.Figure(data=[
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Candlestick"
        ),
        go.Scatter(x=df["date"], y=df["sma_20"], line=dict(color="blue", width=1), name="SMA 20"),
        go.Scatter(x=df["date"], y=df["sma_50"], line=dict(color="orange", width=1), name="SMA 50"),
        go.Scatter(x=df["date"], y=df["sma_200"], line=dict(color="purple", width=1.5, dash='dot'), name="SMA 200"),
    ])
    fig.update_layout(template="plotly_dark", title="Stock Price with SMA", xaxis_rangeslider_visible=False)
    return fig


# Plot RSI and MACD using Plotly
def plot_rsi_chart(df: pd.DataFrame):
    """Plot RSI 14 chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["rsi_14"], name="RSI 14", line=dict(color="purple")
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(
        template="plotly_dark",
        title="RSI (14)",
        yaxis_title="RSI",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_macd_chart(df: pd.DataFrame):
    """Plot MACD chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["macd"], name="MACD", line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["macd_signal"], name="Signal", line=dict(color="orange")
    ))
    fig.add_trace(go.Bar(
        x=df["date"], y=df["macd_hist"], name="Histogram", marker_color="grey", opacity=0.5
    ))
    fig.update_layout(
        template="plotly_dark",
        title="MACD (12,26,9)",
        yaxis_title="MACD",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


# ===================== FORECASTING =====================
def plot_daily_forecast_with_volume(df: pd.DataFrame):
    """Plots daily forecast with volume on a secondary axis."""
    # Prepare data for Prophet
    daily_df = df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})

    # Initialize model
    daily_model = Prophet(daily_seasonality=True)
    daily_model.fit(daily_df)

    # Make future dataframe (next 30 days)
    daily_future = daily_model.make_future_dataframe(periods=30, freq='D')
    daily_forecast = daily_model.predict(daily_future)

    # Plot Prophet forecast using prophet's own plotting function
    fig = plot_plotly(daily_model, daily_forecast)

    # Add volume (red line) to the plot
    volume_trace = go.Scatter(
        x=df['date'],
        y=df['volume'],
        mode='lines',
        name='Volume',
        line=dict(color='red'),
        yaxis='y2'  # assign volume to secondary axis
    )
    fig.add_trace(volume_trace)

    # Add secondary y-axis for volume
    fig.update_layout(
        title="Daily Forecast with Volume",
        yaxis=dict(title="Close Price"),
        yaxis2=dict(
            title="Volume",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        template="plotly_dark"  # Match the theme
    )
    return fig


def forecast_with_prophet(df: pd.DataFrame, periods=30):
    """Forecast future prices using Prophet."""
    prophet_df = df[["date", "close"]].rename(columns={"date": "ds", "close": "y"})
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast"))
    fig.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], name="Historical"))
    fig.update_layout(template="plotly_dark", title="Forecast with Prophet")
    return fig


def analyze_rsi_for_traders(df: pd.DataFrame):
    """Analyzes RSI and provides conclusions for different trading styles based on new user requirements."""

    # --- Short-Term Trading Analysis (4 Weeks) ---
    short_term_df = df.iloc[-20:]  # Approx 4 weeks
    short_term_conclusion = ""
    if not short_term_df.empty:
        last_rsi = short_term_df['rsi_14'].iloc[-1]
        avg_rsi_short = short_term_df['rsi_14'].mean()
        if avg_rsi_short > 70:
            short_term_conclusion = (
                f"📉 **Overbought ({last_rsi:.2f})**: The stock is currently overbought. Over the last 4 weeks, the average RSI was {avg_rsi_short:.2f}, "
                "suggesting a potential pullback. Short-term traders might consider taking profits."
            )
        elif avg_rsi_short < 30:
            short_term_conclusion = (
                f"📈 **Oversold ({last_rsi:.2f})**: The stock is oversold. With a 4-week average RSI of {avg_rsi_short:.2f}, "
                "it may be undervalued and poised for a short-term bounce."
            )
        else:
            short_term_conclusion = (
                f"🔵 **Neutral ({last_rsi:.2f})**: The RSI is neutral. The 4-week average of {avg_rsi_short:.2f} suggests no immediate "
                "strong pressure. Look for moves towards 30 or 70."
            )
    else:
        short_term_conclusion = "ℹ️ Not enough data for a 4-week analysis."

    # --- Swing Trading Analysis (2 Weeks) ---
    swing_df = df.iloc[-10:]  # Approx 2 weeks
    swing_conclusion = ""
    if not swing_df.empty:
        avg_rsi_swing = swing_df['rsi_14'].mean()
        if avg_rsi_swing > 55:
            swing_conclusion = (
                f"🟢 **Bullish Momentum**: Over the last 2 weeks, the RSI has averaged **{avg_rsi_swing:.2f}**. "
                "This suggests underlying strength suitable for swing traders looking to buy on dips."
            )
        elif avg_rsi_swing < 45:
            swing_conclusion = (
                f"🟠 **Bearish Momentum**: The average RSI over the last 2 weeks is **{avg_rsi_swing:.2f}**. "
                "This indicates a prevailing bearish trend. Swing traders might be cautious."
            )
        else:
            swing_conclusion = (
                f"⚪ **Sideways Momentum**: The 2-week average RSI of **{avg_rsi_swing:.2f}** suggests a "
                "ranging market, ideal for traders buying at support and selling at resistance."
            )
    else:
        swing_conclusion = "ℹ️ Not enough data for a 2-week analysis."

    # --- Long-Term Investing Analysis (1 Year) ---
    long_term_df = df.iloc[-252:]  # Approx 1 year
    long_term_conclusion = ""
    if not long_term_df.empty:
        avg_rsi_long_term = long_term_df['rsi_14'].mean()

        if avg_rsi_long_term < 30:
            long_term_conclusion = (
                f"✅ **Long-Term Buy Zone**: The 1-year average RSI is {avg_rsi_long_term:.2f}, "
                "which is below 30. This suggests the stock has been in oversold territory "
                "and may offer good long-term value if fundamentals are strong."
            )
        elif avg_rsi_long_term > 70:
            long_term_conclusion = (
                f"⚠️ **Long-Term Overheated**: The 1-year average RSI is {avg_rsi_long_term:.2f}, "
                "which is above 70. The stock looks overheated, so long-term investors "
                "might wait for a correction before buying."
            )
        else:
            long_term_conclusion = (
                f"⚖️ **Neutral Long-Term**: The 1-year average RSI is {avg_rsi_long_term:.2f}, "
                "which is in the normal 30–70 range. No strong buy/sell signal here; "
                "investors should wait for a clearer setup."
            )
    else:
        long_term_conclusion = "ℹ️ Not enough data for a 1-year RSI analysis."

    return short_term_conclusion, swing_conclusion, long_term_conclusion


def analyze_sma_for_traders(df: pd.DataFrame):
    """Analyzes SMA for different trading styles using average conditions instead of 1-day checks."""

    # --- Swing Trading Analysis (8-10 Weeks) ---
    swing_df = df.iloc[-50:]
    if len(swing_df) >= 50:
        bullish_ratio = (swing_df['sma_20'] > swing_df['sma_50']).mean()
        if bullish_ratio > 0.6:
            swing_conclusion = (
                f"🟢 **Bullish Swing**: In the last 10 weeks, SMA20 stayed above SMA50 "
                f"about {bullish_ratio * 100:.1f}% of the time. "
                "This means short-term momentum is stronger than the medium-term trend, "
                "a common sign of buying interest suitable for swing traders."
            )
        elif bullish_ratio < 0.4:
            swing_conclusion = (
                f"📉 **Bearish Swing**: In the last 10 weeks, SMA20 was below SMA50 "
                f"about {(1 - bullish_ratio) * 100:.1f}% of the time. "
                "This shows weakening momentum, suggesting swing traders should be cautious or consider bearish setups."
            )
        else:
            swing_conclusion = (
                "⚖️ **Neutral Swing**: SMA20 and SMA50 have been crossing frequently in the last 10 weeks. "
                "Momentum is unclear, and the market may be consolidating. Swing traders should wait for a clearer trend."
            )
    else:
        swing_conclusion = "ℹ️ Not enough data for a 10-week swing analysis."

    # --- Short-Term Trading Analysis (3-4 Weeks) ---
    short_term_df = df.iloc[-20:]
    if len(short_term_df) >= 20:
        bullish_ratio = (short_term_df['close'] > short_term_df['sma_20']).mean()
        if bullish_ratio > 0.6:
            short_term_conclusion = (
                f"🟢 **Bullish Short-Term**: In the past 4 weeks, price closed above SMA20 "
                f"about {bullish_ratio * 100:.1f}% of the time. "
                "This shows buyers are in control in the immediate term."
            )
        elif bullish_ratio < 0.4:
            short_term_conclusion = (
                f"📉 **Bearish Short-Term**: In the past 4 weeks, price closed below SMA20 "
                f"about {(1 - bullish_ratio) * 100:.1f}% of the time. "
                "This indicates selling pressure is dominating short-term moves."
            )
        else:
            short_term_conclusion = (
                "⚖️ **Neutral Short-Term**: Price has been moving around SMA20 in the past 4 weeks. "
                "No clear momentum, so short-term traders should be patient."
            )
    else:
        short_term_conclusion = "ℹ️ Not enough data for a 4-week short-term analysis."

    # --- Long-Term Investing Analysis (40+ Weeks) ---
    long_term_df = df.iloc[-200:]
    if len(long_term_df) >= 200:
        bullish_ratio = (long_term_df['sma_50'] > long_term_df['sma_200']).mean()
        if bullish_ratio > 0.6:
            long_term_conclusion = (
                f"⭐ **Strong Long-Term Bullish**: Over the past 40 weeks, SMA50 stayed above SMA200 "
                f"about {bullish_ratio * 100:.1f}% of the time. "
                "This is the classic 'Golden Cross' type setup, showing a strong, sustained uptrend suitable for long-term investors."
            )
        elif bullish_ratio < 0.4:
            long_term_conclusion = (
                f"⚠️ **Caution Long-Term Bearish**: Over the past 40 weeks, SMA50 stayed below SMA200 "
                f"about {(1 - bullish_ratio) * 100:.1f}% of the time. "
                "This is similar to a 'Death Cross', signaling weakness and possible prolonged downtrend for investors."
            )
        else:
            long_term_conclusion = (
                "⚖️ **Neutral Long-Term**: SMA50 and SMA200 have been mixed in the past 40 weeks. "
                "The long-term trend is uncertain, so investors may want to wait for stronger confirmation."
            )
    else:
        long_term_conclusion = "ℹ️ Not enough data for a 40-week long-term analysis."

    return swing_conclusion, short_term_conclusion, long_term_conclusion


# def analyze_macd(df: pd.DataFrame):
#     """Analyzes the latest MACD crossover and returns a conclusion."""
#     last_day = df.iloc[-1]
#     prev_day = df.iloc[-2]
#
#     # Bullish Crossover: MACD crosses above Signal line
#     if prev_day['macd'] < prev_day['macd_signal'] and last_day['macd'] > last_day['macd_signal']:
#         return "success", (
#             "📈 **Bullish MACD Crossover!**\n\n"
#             "The MACD line has just crossed **above** the Signal line. This is a classic **buy signal**, "
#             "suggesting that momentum is shifting upwards."
#         )
#     # Bearish Crossover: MACD crosses below Signal line
#     elif prev_day['macd'] > prev_day['macd_signal'] and last_day['macd'] < last_day['macd_signal']:
#         return "error", (
#             "📉 **Bearish MACD Crossover!**\n\n"
#             "The MACD line has just crossed **below** the Signal line. This is a classic **sell signal**, "
#             "suggesting that momentum is shifting downwards."
#         )
#     # No recent crossover, report current state
#     elif last_day['macd'] > last_day['macd_signal']:
#         return "info", (
#             "🔵 **Bullish Momentum Active**\n\n"
#             "The MACD line is currently **above** the Signal line, indicating that positive momentum is in place."
#         )
#     else:
#         return "warning", (
#             "🟠 **Bearish Momentum Active**\n\n"
#             "The MACD line is currently **below** the Signal line, indicating that negative momentum is in place."
#         )

def analyze_macd(df: pd.DataFrame, lookback: int = 20):
    """Analyzes MACD over multiple weeks with detailed explanations."""

    recent = df.iloc[-lookback:]  # last X days
    last_day = recent.iloc[-1]

    conclusion = []

    # --- Bullish ratio (consistency) ---
    bullish_ratio = (recent['macd'] > recent['macd_signal']).mean()
    if bullish_ratio > 0.6:
        conclusion.append(
            f"📈 **Bullish Bias:** In the past {lookback} trading days, the MACD line stayed above the Signal line "
            f"on {bullish_ratio * 100:.1f}% of days. This shows that momentum has mostly favored buyers, "
            "indicating sustained strength in the stock’s upward moves."
        )
    elif bullish_ratio < 0.4:
        conclusion.append(
            f"📉 **Bearish Bias:** In the past {lookback} trading days, the MACD line stayed below the Signal line "
            f"on {(1 - bullish_ratio) * 100:.1f}% of days. This means sellers have dominated for most of this period, "
            "suggesting consistent downward pressure."
        )
    else:
        conclusion.append(
            f"⚖️ **Neutral / Choppy Market:** Over the last {lookback} days, the MACD and Signal lines crossed back "
            "and forth frequently. This indicates indecision in the market — neither bulls nor bears have strong control, "
            "and price may be consolidating."
        )

    # --- Trend direction (slope) ---
    macd_trend = recent['macd'].iloc[-1] - recent['macd'].iloc[0]
    if macd_trend > 0:
        conclusion.append(
            "🔵 **MACD Trending Upward:** The MACD line has moved higher compared to 20 days ago. "
            "This means momentum is gradually strengthening, a positive sign that buying pressure "
            "is building in the background."
        )
    elif macd_trend < 0:
        conclusion.append(
            "🟠 **MACD Trending Downward:** The MACD line has dropped compared to 20 days ago. "
            "This suggests momentum is fading, and sellers are slowly gaining the upper hand."
        )
    else:
        conclusion.append(
            "⚖️ **Flat MACD Trend:** The MACD line has hardly changed in the last 20 days. "
            "This usually points to sideways movement without a clear momentum shift."
        )

    # --- Histogram bias ---
    avg_hist = recent['macd_hist'].mean()
    if avg_hist > 0:
        conclusion.append(
            "🚀 **Positive Histogram:** On average, the MACD histogram has been above zero during this period. "
            "This means bullish momentum has been stronger than bearish, confirming upward pressure."
        )
    else:
        conclusion.append(
            "⬇️ **Negative Histogram:** On average, the MACD histogram has stayed below zero during this period. "
            "This highlights that bearish momentum has been heavier, pointing to continued selling pressure."
        )

    # --- Zero line check ---
    if last_day['macd'] > 0:
        conclusion.append(
            "✅ **MACD Above Zero:** The MACD line is currently above the zero line. "
            "This is important because it signals the stock is trading with bullish momentum overall, "
            "often seen during medium- to long-term uptrends."
        )
    else:
        conclusion.append(
            "⚠️ **MACD Below Zero:** The MACD line is currently below the zero line. "
            "This shows the stock is trading with bearish momentum overall, "
            "often seen during sustained downtrends."
        )

    return "\n\n".join(conclusion)


# =====================
# Streamlit App
# =====================
def main():
    st.set_page_config(page_title="AI Stock Market Prediction", layout="wide")

    st.markdown(
        """
        <style>
        .hero {
            background: linear-gradient(90deg, #4f46e5, #9333ea);
            padding: 20px;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 20px;
        }
        /* The .bordered-section class is no longer needed with st.container(border=True) */
        </style>
        <div class="hero">
            <h1>📊 AI Stock Market Prediction</h1>
            <p>Upload your stock data (CSV/Excel) and get AI-powered trend prediction & forecast.<br>
            ⚠️ Disclaimer: This is for educational purposes only. Not financial advice. Use at your own risk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("📂 Upload your stock data file", type=["csv", "xlsx"])

    df = None

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return
    else:
        # Load default data from GitHub if no file is uploaded
        default_url = r'https://raw.githubusercontent.com/anishkatoch/Time-Series-Analysis-Anomaly-Detection/main/Data/Equity-TATAMOTORS.csv'
        try:
            df = pd.read_csv(default_url)
            st.info("Showing default data for TATAMOTORS. Upload a file to see your own data.")
        except Exception as e:
            st.error(f"Failed to load default data from GitHub: {e}")
            return

    if df is not None:
        # Preprocess & features
        df = preprocess_data(df)
        df = feature_engineering(df)

        # Data Preview
        st.subheader("Data Preview")
        with st.container(border=True):
            st.dataframe(df.head())

        # Model prediction
        st.subheader("📈 Next Day Trend Prediction")
        with st.container(border=True):
            trend, confidence = train_random_forest(df)
            st.success(f"Prediction: {trend} with {confidence}% confidence")

        # Candlestick Chart
        st.subheader("📉 Candlestick Chart with SMA 20 & SMA 50")
        with st.container(border=True):
            st.plotly_chart(plot_candlestick(df), use_container_width=True)
            # Add the SMA analysis for different trading styles
            swing, short, long = analyze_sma_for_traders(df)
            st.info(f"**Swing Trading (8-10 weeks):** {swing}")
            st.info(f"**Short-Term Trading (3-4 weeks):** {short}")
            st.info(f"**Long-Term Investing (40+ weeks):** {long}")

        # RSI Chart
        st.subheader("📊 RSI (14) Analysis")
        with st.container(border=True):
            st.plotly_chart(plot_rsi_chart(df), use_container_width=True)

            st.markdown("---")  # Visual separator
            st.markdown("##### Trading Strategy Insights")

            # Get the three RSI conclusions
            short, swing, long = analyze_rsi_for_traders(df)

            st.info(f"**Short-Term Trading (5-6 weeks):**\n\n{short}")
            st.info(f"**Swing Trading (~6 months):**\n\n{swing}")
            st.info(f"**Long-Term Investing (1+ year):**\n\n{long}")

        # MACD Chart
        st.subheader("📊 MACD (12,26,9)")
        with st.container(border=True):
            st.plotly_chart(plot_macd_chart(df), use_container_width=True)
            message = analyze_macd(df)
            st.info(message)

        # Daily Forecast with Volume
        st.subheader("📈 Daily Forecast with Volume")
        with st.container(border=True):
            st.plotly_chart(plot_daily_forecast_with_volume(df), use_container_width=True)

        # Prophet Forecast
        st.subheader("🔮 Forecast")
        with st.container(border=True):
            st.plotly_chart(forecast_with_prophet(df, periods=30), use_container_width=True)

        # Model Performance Metrics
        st.subheader("📊 Model Performance Metrics")
        with st.container(border=True):
            # Re-create prophet dataframes to calculate metrics
            daily_df = df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})
            daily_model = Prophet(daily_seasonality=True)
            daily_model.fit(daily_df)
            daily_future = daily_model.make_future_dataframe(periods=30, freq='D')
            daily_forecast = daily_model.predict(daily_future)

            comparisons = daily_forecast.merge(daily_df, on='ds', how='left')
            comparisons.rename(columns={'y': 'actual_close'}, inplace=True)

            st.write("Forecast vs Actuals (historical data):")

            # Dynamic period selection
            period_unit = st.radio("Select period unit", ('Days', 'Weeks', 'Months'), horizontal=True)
            if period_unit == 'Days':
                period_value = st.number_input("Enter number of days", min_value=1, value=5)
                delta = pd.Timedelta(days=period_value)
            elif period_unit == 'Weeks':
                period_value = st.number_input("Enter number of weeks", min_value=1, value=2)
                delta = pd.Timedelta(weeks=period_value)
            else:  # Months
                period_value = st.number_input("Enter number of months", min_value=1, value=1)
                delta = pd.DateOffset(months=period_value)

            # Filter data for display
            historical_comparisons = comparisons.copy()
            end_date = historical_comparisons['ds'].max()
            start_date = end_date - delta



            display_df = historical_comparisons[historical_comparisons['ds'] >= start_date]
            st.dataframe(
                display_df[['ds', 'actual_close', 'yhat']].rename(columns={'yhat': 'prediction_close'})
            )
            # st.dataframe(display_df[['ds', 'actual_close', 'yhat']])

            comparison = comparisons.copy()
            comparison = comparison.dropna().reset_index(drop=True)

            # Actual and predicted values
            y_true = comparison['actual_close']
            y_pred = comparison['yhat']

            # Calculate RMSE and MAE
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            import numpy as np

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)

            # Calculate percentage errors
            rmse_pct = (rmse / y_true.mean()) * 100
            mae_pct = (mae / y_true.mean()) * 100

            # st.markdown("---")
        st.markdown("#### Interpretation:")
        with st.container(border=True):
            st.info(f"""
            **RMSE = {rmse:.2f}** → On average, your predictions are about ₹{rmse:.2f} away from actual prices.

            **MAE = {mae:.2f}** → On average, error is ₹{mae:.2f} per prediction.

            **RMSE% ≈ {rmse_pct:.1f}%** → Model’s average prediction error is ~{rmse_pct:.1f}% of stock price.

            **MAE% ≈ {mae_pct:.1f}%** → More intuitive: predictions are ~{mae_pct:.1f}% off on average.
            """)


if __name__ == "__main__":
    main()
