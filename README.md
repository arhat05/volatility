# Financial Volatility Forecasting

A comprehensive project for forecasting volatility and modeling financial risk using modern machine learning techniques.

## Overview

This project implements a daily GARCH-style model using modern machine learning techniques to predict volatility and assess downside risk (VaR or Expected Shortfall) for financial assets.

### Key Features

- Daily volatility forecasting for major indices and crypto assets
- Value-at-Risk (VaR) and Expected Shortfall predictions
- Regime-switching models for volatility clustering
- Automated ETL pipeline for data ingestion
- Interactive dashboards for visualization
- MLOps integration for model monitoring and deployment

## Project Structure

```
financial-volatility-forecasting/
├── data/                  # Data storage
│   ├── raw/              # Raw data from sources
│   └── processed/        # Processed and cleaned data
├── notebooks/            # Jupyter notebooks for analysis
├── src/                  # Source code
│   ├── etl/             # Data pipeline code
│   ├── models/          # ML model implementations
│   ├── visualization/   # Plotting and dashboard code
│   └── pipeline/        # Pipeline orchestration
├── dashboards/          # Streamlit/Dash applications
├── docker/              # Docker configuration files
├── airflow_dags/        # Airflow DAG definitions
├── requirements.txt     # Python dependencies
└── .github/workflows/   # CI/CD configuration
```

## Getting Started

### Prereqs

- Python 3.8+
- Docker
- PostgreSQL (optional)
- Airflow (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/arhat05/volatility
cd volatility
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

1. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

2. Initialize the database (if using PostgreSQL):
```bash
python src/etl/init_db.py
```

## Usage

### Data Pipeline

Run the ETL pipeline:
```bash
python src/pipeline/run_pipeline.py
```

### Model Training

Train the volatility model:
```bash
python src/models/train.py
```

### Dashboard

Launch the dashboard:
```bash
streamlit run dashboards/main.py
```

## Features

- **Data Collection**: Automated daily data pulls from Yahoo Finance
- **Feature Engineering**: Technical indicators and macroeconomic features
- **Model Training**: GARCH, ML models (XGBoost, LSTM)
- **Risk Metrics**: VaR, Expected Shortfall calculations
- **Visualization**: Interactive dashboards and reports
- **MLOps**: Model monitoring and automated retraining
