"""
Fetches Historical Stock Data using yfinance for multiple tickers.
"""
import os
import yfinance as yf
import pandas as pd
import config

def get_data(tickers=config.TICKERS, start_date=config.START_DATE, end_date=config.END_DATE, interval=config.INTERVAL):
    """
    Downloads or loads historical stock data for multiple tickers,
    fetching/saving each ticker individually.

    Args:
        tickers (list): List of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format. If None, fetches up to the latest data.
        interval (str): Data interval (e.g., '1d', '1h').

    Returns:
        dict: A dictionary where keys are ticker symbols (uppercase) and values are
              DataFrames containing historical stock data for that ticker.
              Returns an empty dictionary if fetching/loading fails for all tickers.
    """
    all_data_dict = {}
    print(f"--- Getting data for tickers: {', '.join(tickers)} ---")

    # Ensure the save directory exists
    save_dir = config.DATA_DIR
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created data directory: {save_dir}")

    for ticker in tickers:
        ticker_upper = ticker.upper()
        pickle_path = os.path.join(save_dir, f'{ticker_upper}_{interval}_data.pkl') if save_dir else None # Include interval in filename

        # Fetch data
        try:
            print(f"Fetching data for {ticker_upper} from {start_date} to {end_date or 'today'} ({interval})...")
            ticker_obj = yf.Ticker(ticker)
            single_df = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False
            )

            if single_df.empty:
                print(f"Warning: No data downloaded for {ticker_upper}. Check ticker symbol, date range, and interval.")
                continue # Skip to next ticker

            print(f"Data fetched successfully for {ticker_upper} ({len(single_df)} rows).")

            # Save data to pickle file if directory is configured
            if pickle_path:
                single_df.to_pickle(pickle_path)
                print(f"Data for {ticker_upper} saved to {pickle_path}.")

        except Exception as fetch_e:
            print(f"Error fetching data for {ticker_upper}: {fetch_e}")
            continue # Skip to next ticker

        # Final check for necessary columns before adding to dict
        required_cols = ['Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in single_df.columns for col in required_cols):
            print(f"Warning: DataFrame for {ticker_upper} is missing standard columns after fetch/load. Skipping.")
            continue

        # Add the DataFrame to the dictionary
        all_data_dict[ticker_upper] = single_df

    print(f"--- Finished getting data for {len(all_data_dict)} tickers ---")
    return all_data_dict


if __name__ == '__main__':
    all_stock_data = get_data()
    if all_stock_data:
        print(f"\nGot data for {len(all_stock_data)} tickers: {list(all_stock_data.keys())}")
        # Print first ticket for checking purposes
        if all_stock_data:
            first_ticker = next(iter(all_stock_data))
            print(f"\nSample Data for {first_ticker}:")
            print(all_stock_data[first_ticker].head())
