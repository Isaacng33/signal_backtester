"""
Logic for generating trading signals based on indicators.
"""
import pandas as pd
import numpy as np
from src import config
from src import indicators

def generate_sma_crossover_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generates buy/sell signals based on SMA crossover strategy.

    Adds 'SMA_Short', 'SMA_Long', 'Signal', and 'Position' columns to the DataFrame.
    - Signal: 1 if short SMA > long SMA, 0 otherwise.
    - Position: 1 for buy (signal changes 0->1), -1 for sell (signal changes 1->0), 0 otherwise.

    Args:
        data (pd.DataFrame): DataFrame containing 'Adj Close' prices.

    Returns:
        pd.DataFrame: Original DataFrame with added signal columns.
    """
    df = data.copy() # Work on a copy

    # Calculate SMAs using the indicators module
    df['SMA_Short'] = indicators.calculate_sma(df, window=config.SMA_SHORT_WINDOW)
    df['SMA_Long'] = indicators.calculate_sma(df, window=config.SMA_LONG_WINDOW)

    # Generate the signal based on SMA comparison
    # Wait until both SMAs have valid values (past the longest window)
    min_periods_long = config.SMA_LONG_WINDOW
    df['Signal'] = 0.0
    # Use .loc to avoid SettingWithCopyWarning, ensuring modification happens on the DataFrame directly
    df.loc[df.index[min_periods_long-1]:, 'Signal'] = np.where(
        df['SMA_Short'].iloc[min_periods_long-1:] > df['SMA_Long'].iloc[min_periods_long-1:], 1.0, 0.0
    )


    # Generate trading positions based on changes in the signal
    # .diff() calculates the difference between the current and prior element
    df['Position'] = df['Signal'].diff()

    # Fill NaN values in Position column (at the start) with 0
    df['Position'] = df['Position'].fillna(0)

    print("SMA Crossover Signals generated.")
    return df

# --- Example: Add RSI Signal Generation ---
# def generate_rsi_signals(data: pd.DataFrame) -> pd.DataFrame:
#     """
#     Generates signals based on RSI thresholds (example).
#     Adds 'RSI' and 'RSI_Position' columns.
#     - RSI_Position: 1 for buy (oversold), -1 for sell (overbought), 0 otherwise.
#     """
#     df = data.copy()
#     df['RSI'] = indicators.calculate_rsi(df, window=config.RSI_WINDOW)

#     df['RSI_Signal'] = 0 # 0: Hold, 1: Potential Buy (Oversold), -1: Potential Sell (Overbought)
#     df.loc[df['RSI'] < config.RSI_OVERSOLD, 'RSI_Signal'] = 1
#     df.loc[df['RSI'] > config.RSI_OVERBOUGHT, 'RSI_Signal'] = -1

#     # You might want more complex logic here, e.g., only trigger on crossovers
#     df['RSI_Position'] = df['RSI_Signal'].diff().fillna(0)
#     # Filter: only take action when crossing threshold
#     df['RSI_Position'] = df['RSI_Position'].apply(lambda x: x if abs(x) > 0.5 else 0) # Adjust logic as needed

#     print("RSI Signals generated.")
#     return df


if __name__ == '__main__':
     # Example usage when running this script directly
    from src import data_fetcher # Assuming data_fetcher is in the same directory or PYTHONPATH is set
    stock_data = data_fetcher.get_data()
    if not stock_data.empty:
        signal_data = generate_sma_crossover_signals(stock_data)
        print("\nData with SMA Signals:")
        # Display rows where a position change occurred
        print(signal_data.loc[signal_data['Position'] != 0, ['Adj Close', 'SMA_Short', 'SMA_Long', 'Signal', 'Position']].head(10))


        # Example RSI signals (if functions uncommented)
        # rsi_signal_data = generate_rsi_signals(stock_data)
        # print("\nData with RSI Signals:")
        # print(rsi_signal_data[['Adj Close', 'RSI', 'RSI_Signal', 'RSI_Position']].tail(10))