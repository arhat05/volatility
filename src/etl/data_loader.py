"""
Data loader module for fetching financial data from various sources.
"""

import yfinance as yf
import pandas as pd
from typing import List, Optional, Union
from datetime import datetime, timedelta
import numpy as np

class DataLoader:
    
    def __init__(self):
        pass
    
    def fetch_stock_data(
        self,
        tickers: Union[str, List[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            tickers: Single ticker or list of tickers
            start_date: Start date for data fetch
            end_date: End date for data fetch
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with stock data
        """
        if isinstance(tickers, str):
            tickers = [tickers]
            
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
            
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            interval=interval,
            group_by='ticker'
        )
        
        return data
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate log returns from price data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with log returns
        """
        return pd.DataFrame(np.log(data / data.shift(1)))
    
    def calculate_volatility(
        self,
        data: pd.DataFrame,
        window: int = 21
    ) -> pd.DataFrame:
        """
        Calculate rolling volatility.
        
        Args:
            data: DataFrame with returns data
            window: Rolling window size
            
        Returns:
            DataFrame with rolling volatility
        """
        return data.rolling(window=window).std() * np.sqrt(252)