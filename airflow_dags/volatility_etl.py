"""
Airflow DAG for volatility forecasting ETL pipeline.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from etl.data_loader import DataLoader

# Default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'volatility_etl',
    default_args=default_args,
    description='ETL pipeline for volatility forecasting',
    schedule_interval='0 0 * * *',  # Daily at midnight
    start_date=datetime(2024, 1, 1),
    catchup=False
)

# fetch data
def fetch_data(**kwargs):
    """Fetch financial data from Yahoo Finance."""
    loader = DataLoader()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2) 
    
    assets = ['^GSPC', '^IXIC', 'BTC-USD', 'ETH-USD']  # S&P 500, NASDAQ, BTC, ETH
    for asset in assets:
        data = loader.fetch_stock_data(asset, start_date, end_date)
        returns = loader.calculate_returns(data['Close'])
        
        pg_hook = PostgresHook(postgres_conn_id='postgres_default')
        returns.to_sql(
            f'{asset.lower().replace("-", "_")}_returns',
            pg_hook.get_sqlalchemy_engine(),
            if_exists='append',
            index=True
        )

# calculate volatility
def calculate_volatility(**kwargs):
    """Calculate volatility metrics."""
    pg_hook = PostgresHook(postgres_conn_id='postgres_default')
    
    assets = ['^GSPC', '^IXIC', 'BTC-USD', 'ETH-USD']
    for asset in assets:
        table_name = f'{asset.lower().replace("-", "_")}_returns'
        
        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name}_volatility AS
        SELECT 
            date,
            STDDEV(returns) * SQRT(252) as volatility
        FROM {table_name}
        GROUP BY date
        ORDER BY date;
        """
        
        pg_hook.run(sql)


fetch_data_task = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_data,
    dag=dag,
)

calculate_volatility_task = PythonOperator(
    task_id='calculate_volatility',
    python_callable=calculate_volatility,
    dag=dag,
)

# Set task dependencies
fetch_data_task >> calculate_volatility_task 