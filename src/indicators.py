# src/indicators.py
"""
Functions to calculate technical indicators.
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import config

print(ta.version)

def calculate_sma(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculates the Simple Moving Average (SMA).

    Args:
        data (pd.DataFrame): DataFrame with 'Adj Close' column.
        window (int): The rolling window period.

    Returns:
        pd.Series: Series containing the SMA values. Returns empty Series if input is invalid.
    """
    if 'Adj Close' not in data.columns:
        print("Error: DataFrame must contain 'Adj Close' column for SMA.")
        return pd.Series(dtype=float)
    if not isinstance(window, int) or window <= 0:
        print("Error: Window must be a positive integer for SMA.")
        return pd.Series(dtype=float)
    # Ensure window is not larger than data length to avoid errors with min_periods=window
    if window > len(data):
        print(f"Warning: SMA window ({window}) is larger than data length ({len(data)}). Calculating with available data.")
        effective_window = len(data)
    else:
        effective_window = window

    # Use min_periods=effective_window ensures enough data points for a valid average
    # Or use min_periods=1 if you want partial calculations at the beginning
    return data['Adj Close'].rolling(window=effective_window, min_periods=effective_window).mean()


def calculate_ema(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculates the Exponential Moving Average (EMA).

    Args:
        data (pd.DataFrame): DataFrame with 'Adj Close' column.
        window (int): The smoothing period (span). Typically corresponds to SMA window length.

    Returns:
        pd.Series: Series containing the EMA values. Returns empty Series if input is invalid.
    """
    if 'Adj Close' not in data.columns:
        print("Error: DataFrame must contain 'Adj Close' column for EMA.")
        return pd.Series(dtype=float)
    if not isinstance(window, int) or window <= 0:
        print("Error: Window must be a positive integer for EMA.")
        return pd.Series(dtype=float)
    if window > len(data):
         print(f"Warning: EMA window ({window}) is larger than data length ({len(data)}). Calculating with available data.")
         effective_window = len(data)
    else:
        effective_window = window

    # adjust=False uses the formula commonly used in trading platforms
    # min_periods=effective_window ensures the EMA calculation starts only when enough data is present
    return data['Adj Close'].ewm(span=effective_window, adjust=False, min_periods=effective_window).mean()


def calculate_rsi(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        data (pd.DataFrame): DataFrame with 'Adj Close' column.
        window (int): The RSI calculation period.

    Returns:
        pd.Series: Series containing the RSI values. Returns empty Series if input is invalid.
    """
    if 'Adj Close' not in data.columns:
        print("Error: DataFrame must contain 'Adj Close' column for RSI.")
        return pd.Series(dtype=float)
    if not isinstance(window, int) or window <= 0:
        print("Error: Window must be a positive integer for RSI.")
        return pd.Series(dtype=float)
    # Need at least window+1 points for diff and initial rolling mean/ewm
    if window + 1 > len(data):
         print(f"Warning: RSI window ({window}) requires more data points ({window+1}) than available ({len(data)}).")
         return pd.Series(dtype=float)


    delta = data['Adj Close'].diff(1) # Difference between consecutive days

    # Separate gains and losses
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0) # Loss as a positive value

    # Calculate average gains and losses using Exponential Moving Average (common practice for RSI)
    # Wilder's smoothing method (alpha = 1 / window) equivalent to ewm(com=window-1)
    avg_gain = gain.ewm(com=window - 1, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, adjust=False, min_periods=window).mean()

    # Calculate Relative Strength (RS)
    # Avoid division by zero: if avg_loss is 0, RS is effectively infinite
    rs = avg_gain / avg_loss.replace(0, np.nan) # Replace 0 with NaN to handle division

    # Calculate RSI
    # Formula: RSI = 100 - (100 / (1 + RS))
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle specific cases arising from division or initial NaNs
    rsi.fillna(100, inplace=True) # Fill NaN RSIs (where avg_loss was 0) with 100
    rsi.loc[(avg_gain == 0) & (avg_loss == 0)] = 50 # If no change, RSI is neutral (can be debated, 50 is common)
    rsi.loc[(avg_gain > 0) & (avg_loss == 0)] = 100 # If only gains, RSI is 100
    rsi.loc[(avg_gain == 0) & (avg_loss > 0)] = 0 # If only losses, RSI is 0

    return rsi


if __name__ == '__main__':
    TICKER = 'AAPL' # Or use config.TICKER
    # Use config for path consistency
    DATA_PATH = os.path.join(config.DATA_DIR, f'{TICKER}_data.pkl')

    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        try:
            test_data = pd.read_pickle(DATA_PATH)

            if not test_data.empty and 'Adj Close' in test_data.columns:

                # --- Parameters ---
                sma_window = 50
                ema_window = 50 # Use same window as SMA for comparison if desired
                rsi_window = 14

                # --- Calculate using YOUR functions ---
                print(f"\nCalculating SMA_{sma_window} using your function...")
                test_data[f'SMA_{sma_window}_custom'] = calculate_sma(test_data, sma_window)

                print(f"\nCalculating EMA_{ema_window} using your function...")
                test_data[f'EMA_{ema_window}_custom'] = calculate_ema(test_data, ema_window)

                print(f"\nCalculating RSI_{rsi_window} using your function...")
                test_data[f'RSI_{rsi_window}_custom'] = calculate_rsi(test_data, rsi_window)

                # --- Calculate using pandas_ta ---
                print(f"\nCalculating indicators using pandas_ta...")
                # Use the 'close' argument to specify which column to use
                # Note: pandas_ta might default to 'close', ensure you use 'Adj Close' if needed
                # The 'append=True' adds the columns directly to test_data
                test_data.ta.sma(length=sma_window, close='Adj Close', append=True) # Will create 'SMA_50'
                test_data.ta.ema(length=ema_window, close='Adj Close', append=True) # Will create 'EMA_50'
                test_data.ta.rsi(length=rsi_window, close='Adj Close', append=True) # Will create 'RSI_14'

                # --- Compare Results ---
                print("\n--- Comparison (Tail) ---")
                # Select relevant columns for comparison
                comparison_cols = [
                    'Adj Close',
                    f'SMA_{sma_window}_custom', f'SMA_{sma_window}', # Compare SMA
                    f'EMA_{ema_window}_custom', f'EMA_{ema_window}', # Compare EMA
                    f'RSI_{rsi_window}_custom', f'RSI_{rsi_window}'  # Compare RSI
                ]
                # Use .filter to safely select columns that exist
                print(test_data.filter(items=comparison_cols).tail(10))

            elif test_data.empty:
                print(f"Pickle file '{DATA_PATH}' is empty.")
            else:
                print("Error: Pickle file loaded but missing 'Adj Close' column.")

        except ModuleNotFoundError:
             print("\nError: pandas_ta library not found.")
             print("Please install it: pip install pandas-ta")
        except Exception as e:
            # Make error message more specific for pickle
            print(f"Error processing pickle file '{DATA_PATH}': {e}")

    else:
         # Make error message more specific for pickle
        print(f"Error: Pickle file '{DATA_PATH}' not found.")
