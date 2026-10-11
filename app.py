import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="StockSense AI | Market Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# PROFESSIONAL DARK THEME
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #080D18;
    color: #F1F5F9;
}

[data-testid="stSidebar"] {
    background-color: #0D1525;
    border-right: 1px solid #1E293B;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #F8FAFC !important;
}

p, label, span {
    color: #CBD5E1;
}

.metric-card {
    background: linear-gradient(135deg, #111C30, #0D1525);
    border: 1px solid #25344D;
    padding: 20px;
    border-radius: 14px;
    min-height: 125px;
}

.metric-label {
    color: #94A3B8;
    font-size: 13px;
    margin-bottom: 10px;
}

.metric-value {
    color: #F8FAFC;
    font-size: 27px;
    font-weight: 700;
}

.metric-caption {
    color: #2DD4BF;
    font-size: 12px;
    margin-top: 8px;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #F8FAFC;
    margin-top: 20px;
    margin-bottom: 12px;
}

div.stButton > button {
    background-color: #14B8A6;
    color: #06121D;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 42px;
    width: 100%;
}

div.stButton > button:hover {
    background-color: #2DD4BF;
    color: #06121D;
    border: none;
}

[data-testid="stDataFrame"] {
    border: 1px solid #25344D;
    border-radius: 10px;
}

hr {
    border-color: #25344D;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown("# 📈 StockSense AI")

    st.caption("INTELLIGENT MARKET ANALYTICS")

    st.divider()

    st.markdown("### Market Configuration")

    ticker = st.text_input(
        "Stock Symbol",
        value="AAPL",
        help="Examples: AAPL, MSFT, GOOGL, TSLA, RELIANCE.NS"
    ).strip().upper()

    period = st.selectbox(
        "Historical Period",
        ["6mo", "1y", "2y", "5y"],
        index=1
    )

    interval = st.selectbox(
        "Data Interval",
        ["1d", "1wk"],
        index=0,
        format_func=lambda x: {
            "1d": "Daily",
            "1wk": "Weekly"
        }[x]
    )

    st.divider()

    st.markdown("### Analysis Modules")

    st.markdown("""
    - 📊 Market Overview
    - 📉 Price Analytics
    - 🧮 Technical Indicators
    - 🤖 ML Forecast
    - 🎯 Model Evaluation
    """)

    st.divider()

    load_button = st.button(
        "↻ Load Market Data",
        use_container_width=True
    )

    st.caption(
        "Market data requires an internet connection. "
        "Forecasts are experimental and not financial advice."
    )


# ==========================================
# HEADER
# ==========================================

header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        "# StockSense AI"
    )

    st.markdown(
        "### Market intelligence. Data-driven decisions."
    )

    st.caption(
        "Historical market analytics and experimental machine-learning forecasts."
    )

with header_right:
    st.markdown("")

    st.markdown(
        "<div style='text-align:right;color:#2DD4BF;"
        "font-weight:700;'>● ANALYTICS DASHBOARD</div>",
        unsafe_allow_html=True
    )

st.divider()


# ==========================================
# SESSION STATE
# ==========================================

if "market_data" not in st.session_state:
    st.session_state.market_data = None

if "loaded_ticker" not in st.session_state:
    st.session_state.loaded_ticker = None


# ==========================================
# DOWNLOAD MARKET DATA
# ==========================================

@st.cache_data(ttl=300, show_spinner=False)
def download_market_data(symbol, selected_period, selected_interval):

    data = yf.download(
        symbol,
        period=selected_period,
        interval=selected_interval,
        auto_adjust=True,
        progress=False
    )

    if data is None or data.empty:
        return pd.DataFrame()

    # Handle newer yfinance MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.loc[:, ~data.columns.duplicated()]

    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    for column in required_columns:
        if column not in data.columns:
            return pd.DataFrame()

    data = data[required_columns].copy()

    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data.dropna(subset=["Close"], inplace=True)

    return data


# ==========================================
# LOAD BUTTON
# ==========================================

if load_button:

    if not ticker:
        st.error("Please enter a valid stock symbol.")

    else:
        with st.spinner(
            f"Retrieving market data for {ticker}..."
        ):

            try:
                downloaded_data = download_market_data(
                    ticker,
                    period,
                    interval
                )

                if downloaded_data.empty:

                    st.session_state.market_data = None

                    st.error(
                        "No market data found. Check the symbol "
                        "or try again later."
                    )

                else:
                    st.session_state.market_data = downloaded_data
                    st.session_state.loaded_ticker = ticker

                    st.success(
                        f"Market data loaded for {ticker}."
                    )

            except Exception as error:

                st.error(
                    "Unable to retrieve market data. "
                    "Check your internet connection and try again."
                )

                st.caption(str(error))


# ==========================================
# DEFAULT INITIAL LOAD
# ==========================================

if st.session_state.market_data is None:

    try:
        with st.spinner("Loading initial market data..."):

            initial_data = download_market_data(
                ticker,
                period,
                interval
            )

            if not initial_data.empty:
                st.session_state.market_data = initial_data
                st.session_state.loaded_ticker = ticker

    except Exception:
        pass


# ==========================================
# DASHBOARD CONTENT
# ==========================================

data = st.session_state.market_data

if data is None or data.empty:

    st.warning(
        "Market data is not available right now. "
        "Check your internet connection or enter another stock symbol "
        "and click Load Market Data."
    )

    st.stop()


# Prevent confusion when the sidebar symbol changes
# without loading the newly selected symbol.

active_ticker = st.session_state.loaded_ticker

if ticker != active_ticker:

    st.info(
        f"Currently displaying {active_ticker}. "
        "Click **Load Market Data** to load {ticker}."
    )


# ==========================================
# CALCULATE INDICATORS
# ==========================================

data = data.copy()

data["SMA_20"] = data["Close"].rolling(20).mean()

data["SMA_50"] = data["Close"].rolling(50).mean()

data["EMA_20"] = data["Close"].ewm(
    span=20,
    adjust=False
).mean()

delta = data["Close"].diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

average_gain = gain.rolling(14).mean()

average_loss = loss.rolling(14).mean()

relative_strength = (
    average_gain / average_loss.replace(0, np.nan)
)

data["RSI"] = 100 - (
    100 / (1 + relative_strength)
)

data["RSI"] = data["RSI"].fillna(50)

ema_12 = data["Close"].ewm(
    span=12,
    adjust=False
).mean()

ema_26 = data["Close"].ewm(
    span=26,
    adjust=False
).mean()

data["MACD"] = ema_12 - ema_26

data["MACD_Signal"] = data["MACD"].ewm(
    span=9,
    adjust=False
).mean()


# ==========================================
# MARKET METRICS
# ==========================================

latest_close = float(data["Close"].iloc[-1])

previous_close = (
    float(data["Close"].iloc[-2])
    if len(data) > 1
    else latest_close
)

price_change = latest_close - previous_close

percentage_change = (
    price_change / previous_close * 100
    if previous_close != 0
    else 0
)

average_volume = data["Volume"].tail(20).mean()

average_volume = (
    0 if pd.isna(average_volume)
    else float(average_volume)
)

latest_rsi = float(data["RSI"].iloc[-1])

currency_symbol = "₹" if active_ticker.endswith(".NS") else "$"


# ==========================================
# METRIC CARDS
# ==========================================

st.markdown(
    f"## Market Overview — {active_ticker}"
)

st.caption(
    f"Displaying {len(data)} historical observations."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">LATEST CLOSE</div>
            <div class="metric-value">
                {currency_symbol}{latest_close:,.2f}
            </div>
            <div class="metric-caption">
                Latest available observation
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    change_color = (
        "#2DD4BF" if price_change >= 0
        else "#FB7185"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">LAST PERIOD CHANGE</div>
            <div class="metric-value" style="color:{change_color};">
                {price_change:+,.2f}
            </div>
            <div class="metric-caption" style="color:{change_color};">
                {percentage_change:+.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">RSI (14)</div>
            <div class="metric-value">
                {latest_rsi:.1f}
            </div>
            <div class="metric-caption">
                Momentum indicator
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">20-PERIOD AVG VOLUME</div>
            <div class="metric-value">
                {average_volume:,.0f}
            </div>
            <div class="metric-caption">
                Historical average
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# PRICE CHART
# ==========================================

st.markdown(
    '<div class="section-title">Price Analytics</div>',
    unsafe_allow_html=True
)

show_sma20 = st.checkbox(
    "Show SMA 20",
    value=True
)

show_sma50 = st.checkbox(
    "Show SMA 50",
    value=True
)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.72, 0.28],
    subplot_titles=(
        "Historical Closing Price",
        "Trading Volume"
    )
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Close"],
        name="Closing Price",
        line=dict(color="#2DD4BF", width=2),
        fill="tozeroy",
        fillcolor="rgba(45,212,191,0.06)"
    ),
    row=1,
    col=1
)

if show_sma20:
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA_20"],
            name="SMA 20",
            line=dict(color="#FBBF24", width=1.5)
        ),
        row=1,
        col=1
    )

if show_sma50:
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["SMA_50"],
            name="SMA 50",
            line=dict(color="#A78BFA", width=1.5)
        ),
        row=1,
        col=1
    )

fig.add_trace(
    go.Bar(
        x=data.index,
        y=data["Volume"],
        name="Volume",
        marker_color="#38BDF8",
        opacity=0.65
    ),
    row=2,
    col=1
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#080D18",
    plot_bgcolor="#0D1525",
    font=dict(color="#CBD5E1"),
    height=570,
    margin=dict(l=15, r=15, t=50, b=15),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    hovermode="x unified"
)

fig.update_xaxes(
    showgrid=True,
    gridcolor="#1E293B"
)

fig.update_yaxes(
    showgrid=True,
    gridcolor="#1E293B"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==========================================
# TECHNICAL INDICATORS
# ==========================================

st.markdown(
    '<div class="section-title">Technical Indicators</div>',
    unsafe_allow_html=True
)

indicator1, indicator2 = st.columns(2)

with indicator1:

    rsi_fig = go.Figure()

    rsi_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI"],
            name="RSI",
            line=dict(color="#A78BFA", width=2)
        )
    )

    rsi_fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="#FB7185"
    )

    rsi_fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="#2DD4BF"
    )

    rsi_fig.update_layout(
        title="Relative Strength Index",
        template="plotly_dark",
        paper_bgcolor="#080D18",
        plot_bgcolor="#0D1525",
        height=320,
        yaxis=dict(range=[0, 100]),
        margin=dict(l=10, r=10, t=50, b=10)
    )

    st.plotly_chart(
        rsi_fig,
        use_container_width=True
    )

with indicator2:

    macd_fig = go.Figure()

    macd_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD"],
            name="MACD",
            line=dict(color="#38BDF8", width=2)
        )
    )

    macd_fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD_Signal"],
            name="Signal",
            line=dict(color="#FBBF24", width=2)
        )
    )

    macd_fig.update_layout(
        title="MACD Momentum",
        template="plotly_dark",
        paper_bgcolor="#080D18",
        plot_bgcolor="#0D1525",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h")
    )

    st.plotly_chart(
        macd_fig,
        use_container_width=True
    )


# ==========================================
# MACHINE-LEARNING FORECAST
# ==========================================

st.markdown(
    '<div class="section-title">AI Price Forecast</div>',
    unsafe_allow_html=True
)

st.caption(
    "Experimental Random Forest regression using historical closing-price "
    "lags. Predictions are estimates, not guaranteed future prices."
)

if len(data) < 80:

    st.warning(
        "At least 80 observations are recommended for this demonstration. "
        "Select a longer historical period."
    )

else:

    forecast_days = st.slider(
        "Forecast horizon (trading periods)",
        min_value=1,
        max_value=10,
        value=5
    )

    prices = data["Close"].astype(float).reset_index(drop=True)

    feature_frame = pd.DataFrame({
        "lag_1": prices.shift(1),
        "lag_2": prices.shift(2),
        "lag_3": prices.shift(3),
        "lag_5": prices.shift(5),
        "lag_10": prices.shift(10)
    })

    feature_frame["target"] = prices

    feature_frame = feature_frame.dropna()

    X = feature_frame.drop(columns=["target"])

    y = feature_frame["target"]

    split_index = int(len(feature_frame) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions
        )
    )

    latest_features = np.array([
        prices.iloc[-1],
        prices.iloc[-2],
        prices.iloc[-3],
        prices.iloc[-5],
        prices.iloc[-10]
    ], dtype=float).reshape(1, -1)

    future_predictions = []

    current_features = latest_features.copy()

    for _ in range(forecast_days):

        prediction = float(
            model.predict(current_features)[0]
        )

        future_predictions.append(prediction)

        current_features = np.array([[
            prediction,
            current_features[0, 0],
            current_features[0, 1],
            current_features[0, 3],
            current_features[0, 4]
        ]])

    forecast1, forecast2, forecast3 = st.columns(3)

    with forecast1:
        st.metric(
            "Last Observed Close",
            f"{currency_symbol}{latest_close:,.2f}"
        )

    with forecast2:
        st.metric(
            f"Estimated Close in {forecast_days} Period(s)",
            f"{currency_symbol}{future_predictions[-1]:,.2f}",
            delta=(
                f"{(future_predictions[-1] / latest_close - 1) * 100:+.2f}%"
                if latest_close else None
            )
        )

    with forecast3:
        st.metric(
            "Test MAE",
            f"{currency_symbol}{mae:,.2f}"
        )

    forecast_chart = go.Figure()

    recent_count = min(60, len(prices))

    recent_dates = data.index[-recent_count:]

    recent_prices = prices.iloc[-recent_count:]

    forecast_chart.add_trace(
        go.Scatter(
            x=recent_dates,
            y=recent_prices,
            name="Historical Price",
            line=dict(color="#2DD4BF", width=2)
        )
    )

    last_date = data.index[-1]

    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days
    )

    forecast_chart.add_trace(
        go.Scatter(
            x=[last_date] + list(future_dates),
            y=[latest_close] + future_predictions,
            name="Experimental Forecast",
            line=dict(
                color="#FBBF24",
                width=2,
                dash="dash"
            ),
            mode="lines+markers"
        )
    )

    forecast_chart.update_layout(
        title="Historical Price vs Experimental Forecast",
        template="plotly_dark",
        paper_bgcolor="#080D18",
        plot_bgcolor="#0D1525",
        height=430,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h")
    )

    st.plotly_chart(
        forecast_chart,
        use_container_width=True
    )

    with st.expander("View model evaluation"):

        eval1, eval2 = st.columns(2)

        eval1.metric(
            "Mean Absolute Error (MAE)",
            f"{currency_symbol}{mae:,.2f}"
        )

        eval2.metric(
            "Root Mean Squared Error (RMSE)",
            f"{currency_symbol}{rmse:,.2f}"
        )

        st.caption(
            "Evaluation uses a chronological holdout set. "
            "The model is a demonstration and has not been established "
            "as a reliable trading strategy."
        )


# ==========================================
# HISTORICAL DATA TABLE
# ==========================================

st.markdown(
    '<div class="section-title">Historical Market Data</div>',
    unsafe_allow_html=True
)

display_data = data[
    ["Open", "High", "Low", "Close", "Volume"]
].sort_index(ascending=False)

st.dataframe(
    display_data.head(20).round(2),
    use_container_width=True
)

csv_data = data.to_csv().encode("utf-8")

st.download_button(
    label="⬇ Download Historical Data (CSV)",
    data=csv_data,
    file_name=f"{active_ticker}_historical_data.csv",
    mime="text/csv"
)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748B; padding:15px;">
        <b style="color:#2DD4BF;">StockSense AI</b>
        &nbsp;|&nbsp; Intelligent Market Analytics
        <br><br>
        Built with Python, Streamlit, Plotly and Scikit-learn.
        <br>
        Educational demonstration only — not financial advice.
    </div>
    """,
    unsafe_allow_html=True
)