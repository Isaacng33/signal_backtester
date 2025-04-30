"""
Generates trading signals based on the EMA Crossover with RSI Filter strategy.
"""
import pandas as pd
import numpy as np
import os
import config

# Import the necessary indicator calculation functions
from indicators import calculate_ema, calculate_rsi

def generate_ema_rsi_signals(
        data: pd.DataFrame,
        short_window: int,
        medium_window: int,
        long_window: int,
        rsi_window: int,
        rsi_overbought: int
    ) -> pd.DataFrame:
    """
    Calculates required indicators (EMAs, RSI) and generates trading signals based on:
    - Trend Filter: Close > Long EMA
    - Entry: Short EMA crosses above Medium EMA, confirmed by RSI < Overbought
    - Exit: Short EMA crosses below Medium EMA

    Args:
        data (pd.DataFrame): DataFrame containing price data. Requires 'Adj Close'.
        short_window (int): Short EMA period.
        medium_window (int): Medium EMA period.
        long_window (int): Long EMA period (for trend).
        rsi_window (int): RSI period.
        rsi_overbought (int): RSI overbought threshold.

    Returns:
        pd.DataFrame: Input DataFrame with added indicator, 'Signal', and 'Position' columns.
                      Returns an empty DataFrame if input is insufficient after indicator calculation.
                      Signal: 1 for Buy entry, -1 for Sell entry, 0 otherwise.
                      Position: 1 meaning currently having a long position, 0 for Flat.
    """
    adj_close_col = 'Adj Close'
    if adj_close_col not in data.columns:
         raise ValueError(f"Input DataFrame missing required column: '{adj_close_col}'")

    df = data.copy()

    # --- Calculate Indicators Internally ---
    print(f"\nCalculating EMA_{short_window}...")
    df[f'EMA_{short_window}'] = calculate_ema(df, short_window)

    print(f"\nCalculating EMA_{medium_window}...")
    df[f'EMA_{medium_window}'] = calculate_ema(df, medium_window)

    print(f"\nCalculating EMA_{long_window}...")
    df[f'EMA_{long_window}'] = calculate_ema(df, long_window)

    print(f"\nCalculating RSI_{rsi_window}...")
    df[f'RSI_{rsi_window}'] = calculate_rsi(df, rsi_window)

    # --- Handle NaNs from Indicator Calculation ---
    print(f"\nData length before indicator dropna: {len(df)}")
    df.dropna(inplace=True)
    print(f"Data length after indicator dropna: {len(df)}")

    if df.empty:
        print("Warning: No data remaining after calculating indicators and dropping NaNs.")
        # Return the empty DataFrame with expected columns for consistency
        df['Position'] = np.nan
        df['Signal'] = np.nan
        return df

    # --- Define Column Names (now that they exist) ---
    short_ema_col = f'EMA_{short_window}'
    medium_ema_col = f'EMA_{medium_window}'
    long_ema_col = f'EMA_{long_window}'
    rsi_col = f'RSI_{rsi_window}'

    # --- Strategy Logic Conditions ---
    # 1. Trend Filter
    trend_up = df[adj_close_col] > df[long_ema_col]

    # 2. Entry Crossover Signal (Short EMA crosses above Medium EMA)
    ema_cross_above = (df[short_ema_col] > df[medium_ema_col]) & \
                      (df[short_ema_col].shift(1) <= df[medium_ema_col].shift(1))

    # 3. RSI Confirmation Filter (RSI is not overbought)
    rsi_ok = df[rsi_col] < rsi_overbought

    # 4. Exit Crossover Signal (Short EMA crosses below Medium EMA)
    ema_cross_below = (df[medium_ema_col] < df[long_ema_col]) & \
                      (df[medium_ema_col].shift(1) >= df[long_ema_col].shift(1))

    # --- Combine Conditions for Buy/Sell Triggers ---
    buy_trigger = trend_up & ema_cross_above & rsi_ok
    sell_trigger = ema_cross_below

    # --- Determine Position (using vectorized ffill method) ---
    df['Position'] = np.nan
    df.loc[buy_trigger, 'Position'] = 1
    df.loc[sell_trigger, 'Position'] = 0
    df['Position']  = df['Position'].ffill()
    df['Position'] = df['Position'].fillna(0)
    df['Position'] = df['Position'].astype(int)

    # --- Generate Signals (entry/exit points only) ---
    df['Signal'] = 0
    buy_entry_mask = (df['Position'] == 1) & (df['Position'].shift(1) == 0)
    sell_entry_mask = (df['Position'] == 0) & (df['Position'].shift(1) == 1)
    df.loc[buy_entry_mask, 'Signal'] = 1
    df.loc[sell_entry_mask, 'Signal'] = -1

    print(f"\nSignals generated based on EMA {short_window}/{medium_window}/{long_window} crossover with RSI {rsi_window} < {rsi_overbought} filter.")
    print(f"Total Buy Signals: {df['Signal'].value_counts().get(1, 0)}")
    print(f"Total Sell Signals: {df['Signal'].value_counts().get(-1, 0)}")

    return df


# --- Example Usage ---
if __name__ == '__main__':
    # --- Load Data ---
    TICKER = config.TICKER
    DATA_PATH = os.path.join(config.DATA_DIR, f'{TICKER}_data.pkl')

    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        try:
            test_data = pd.read_pickle(DATA_PATH)

            if not test_data.empty and 'Adj Close' in test_data.columns:
                # --- Generate Signals ---
                print("\nGenerating signals ...")
                data_with_signals = generate_ema_rsi_signals(
                    test_data, # Pass the raw data
                    short_window=config.EMA_SHORT_WINDOW,
                    medium_window=config.EMA_MEDIUM_WINDOW,
                    long_window=config.EMA_LONG_WINDOW,
                    rsi_window=config.RSI_WINDOW,
                    rsi_overbought=config.RSI_OVERBOUGHT
                )

                if not data_with_signals.empty:
                    print("\nData with Signals (Tail):")
                    display_cols = [
                        'Adj Close',
                        f'EMA_{config.EMA_SHORT_WINDOW}',
                        f'EMA_{config.EMA_MEDIUM_WINDOW}',
                        f'EMA_{config.EMA_LONG_WINDOW}',
                        f'RSI_{config.RSI_WINDOW}',
                        'Position',
                        'Signal'
                    ]
                    print(data_with_signals.filter(items=display_cols).tail(20))
                else:
                     print("\nSignal generation resulted in empty DataFrame (likely due to NaNs).")

            elif test_data.empty:
                 print("Loaded data file is empty.")
            else:
                 print("Loaded data missing 'Adj Close' column.")

        except Exception as e:
            print(f"An error occurred in the signals test block: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging

    else:
        print(f"Error: Data file '{DATA_PATH}' not found.")