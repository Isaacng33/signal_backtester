"""
To Fetch Historical Stock Data using yfinance
This script fetches historical stock data using the yfinance library and saves it to a DataFrame.
"""

import os
import yfinance as yf
import pandas as pd
import config 

def get_data(ticker=config.TICKER, start_date=config.START_DATE, end_date=config.END_DATE):
    """
    Downloads historical stock data using yfinance.

    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format. If None, fetches up to the latest data.

    Returns:
        pd.DataFrame: DataFrame containing historical stock data (OHLC, Volume, Adj Close).
                      Returns an empty DataFrame if download fails.
    """
    try:
        print(f"Fetching data for {ticker} from {start_date} to {end_date or 'today'}...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=True, auto_adjust=False)
        if data.empty:
            print(f"Warning: No data downloaded for {ticker}. Check ticker symbol and date range.")
            return pd.DataFrame()
        print(f"Data fetched successfully for {ticker}.")
        
        # Save data with pickle
        if isinstance(data.columns, pd.MultiIndex):
            print("Detected MultiIndex columns, attempting to flatten...")
            if len(data.columns.levels) > 1 and len(data.columns.get_level_values(1).unique()) == 1:
                data.columns = data.columns.droplevel(1)
                print("Flattened columns by dropping ticker level.")

        if config.DATA_DIR:
            # Ensure the directory exists
            if not os.path.exists(config.DATA_DIR):
                os.makedirs(config.DATA_DIR)
            pickle_path = os.path.join(config.DATA_DIR, f'{ticker}_data.pkl')
            data.to_pickle(pickle_path)
            print(f"Data saved to {pickle_path}.")
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    stock_data = get_data()
    if not stock_data.empty:
        print("\nSample Data:")
        print(stock_data.head())
        print("\nData Info:")
        stock_data.info()