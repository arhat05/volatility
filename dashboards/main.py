"""
Main Streamlit dashboard for volatility forecasting visualization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from etl.data_loader import DataLoader
from models.volatility_model import GARCHModel


st.set_page_config(
    page_title="Volatility Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)


st.title("📈 Volatility Forecasting Dashboard")

st.sidebar.header("Configuration")

asset = st.sidebar.selectbox(
    "Select Asset",
    ["S&P 500", "NASDAQ", "BTC", "ETH"]
)


end_date = datetime.now()
start_date = end_date - timedelta(days=365)
date_range = st.sidebar.date_input(
    "Date Range",
    value=(start_date, end_date)
)


st.sidebar.subheader("Model Parameters")
p = st.sidebar.slider("ARCH Order (p)", 1, 5, 1)
q = st.sidebar.slider("GARCH Order (q)", 1, 5, 1)
confidence_level = st.sidebar.slider("VaR Confidence Level", 0.90, 0.99, 0.95)

# Main content
@st.cache_data
def load_data(asset: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load and process data."""
    loader = DataLoader()
    data = loader.fetch_stock_data(asset, start_date, end_date)
    returns = loader.calculate_returns(data['Close'])
    return returns

@st.cache_resource
def train_model(returns: pd.Series, p: int, q: int) -> GARCHModel:
    """Train GARCH model."""
    model = GARCHModel(p=p, q=q)
    model.fit(returns)
    return model


returns = load_data(asset, date_range[0], date_range[1])

model = train_model(returns, p, q)

forecast_vol, params = model.predict(returns)
var = model.calculate_var(returns, confidence_level)


col1, col2 = st.columns(2)

with col1:
    st.subheader("Volatility Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=returns.index,
        y=returns.rolling(window=21).std() * np.sqrt(252),
        name="Realized Volatility"
    ))
    fig.add_trace(go.Scatter(
        x=[returns.index[-1]],
        y=[forecast_vol],
        mode='markers',
        name="Forecast",
        marker=dict(size=10)
    ))
    fig.update_layout(
        title="Volatility Forecast",
        xaxis_title="Date",
        yaxis_title="Volatility"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Value at Risk")
    st.metric(
        "VaR (95%)",
        f"{var:.2%}",
        delta=f"{(var - returns.std()):.2%}"
    )
    
    # Model parameters
    st.subheader("Model Parameters")
    for param, value in params.items():
        st.text(f"{param}: {value:.4f}")


st.subheader("Model Performance Metrics")
col3, col4, col5 = st.columns(3)

with col3:
    st.metric("Mean Absolute Error", "0.XX%")
with col4:
    st.metric("Root Mean Squared Error", "0.XX%")
with col5:
    st.metric("VaR Coverage", "XX%") 