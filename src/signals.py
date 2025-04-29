"""
Generates trading signals based on the EMA Crossover with RSI Filter strategy.
"""
import pandas as pd
import numpy as np
import os
# Import config to use parameters easily in the main function and test block
import config

def generate_ema_rsi_signals(
        data: pd.DataFrame,
        short_window: int,
        medium_window: int,
        long_window: int,
        rsi_window: int,
        rsi_overbought: int
    ) -> pd.DataFrame:
    """
    Generates trading signals based on:
    - Trend Filter: Close > Long EMA
    - Entry: Short EMA crosses above Medium EMA, confirmed by RSI < Overbought
    - Exit: Short EMA crosses below Medium EMA

    Args:
        data (pd.DataFrame): DataFrame containing price and indicator columns.
                             Requires 'Adj Close', f'EMA_{short_window}',
                             f'EMA_{medium_window}', f'EMA_{long_window}',
                             and f'RSI_{rsi_window}'.
        short_window (int): Short EMA period.
        medium_window (int): Medium EMA period.
        long_window (int): Long EMA period (for trend).
        rsi_window (int): RSI period.
        rsi_overbought (int): RSI overbought threshold.

    Returns:
        pd.DataFrame: Input DataFrame with added 'Signal' and 'Position' columns.
                      Signal: 1 for Buy entry, -1 for Sell entry, 0 otherwise.
                      Position: 1 meaning currently having a long position, 0 for Flat.
    """
    # Define required column names
    adj_close_col = 'Adj Close'
    short_ema_col = f'EMA_{short_window}'
    medium_ema_col = f'EMA_{medium_window}'
    long_ema_col = f'EMA_{long_window}'
    rsi_col = f'RSI_{rsi_window}'

    required_cols = [adj_close_col, short_ema_col, medium_ema_col, long_ema_col, rsi_col]
    # Check if all required columns exist
    if not all(col in data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data.columns]
        raise ValueError(f"DataFrame missing required columns: {missing}")

    df = data.copy()

    # --- Strategy Logic Conditions ---
    # 1. Trend Filter
    trend_up = df[adj_close_col] > df[long_ema_col]

    # 2. Entry Crossover Signal (Short EMA crosses above Medium EMA)
    # Checks the current day and the previous day to confirm crossover
    ema_cross_above = (df[short_ema_col] > df[medium_ema_col]) & \
                      (df[short_ema_col].shift(1) <= df[medium_ema_col].shift(1))

    # 3. RSI Confirmation Filter (RSI is not overbought)
    rsi_ok = df[rsi_col] < rsi_overbought

    # 4. Exit Crossover Signal (Short EMA crosses below Medium EMA)
    # Checks the current day and the previous day to confirm crossover
    ema_cross_below = (df[short_ema_col] < df[medium_ema_col]) & \
                      (df[short_ema_col].shift(1) >= df[medium_ema_col].shift(1))

    # --- Combine Conditions for Buy/Sell Triggers ---
    buy_trigger = trend_up & ema_cross_above & rsi_ok
    sell_trigger = ema_cross_below # Exit regardless of trend or RSI once short crosses below medium

    # --- Determine Position ---
    # Initialize position column
    df['Position'] = np.nan
    df.loc[buy_trigger, 'Position'] = 1  # Mark with 1 on days with potential buy signals
    df.loc[sell_trigger, 'Position'] = 0  # Mark with 0 on days with potential exit signals
    df['Position']  = df['Position'].ffill()  # Forward fill to maintain position until exit
    df['Position'] = df['Position'].fillna(0)  # Fill remaining NaNs with 0 (no position)
    df['Position'] = df['Position'].astype(int)  # Ensure Position is integer type

    # --- Generate Signals ---
    df['Signal'] = 0  # Initialize signal column
    buy_entry_mask = (df['Position'] == 1) & (df['Position'].shift(1) == 0)  # Buy entry signal
    sell_entry_mask = (df['Position'] == 0) & (df['Position'].shift(1) == 1)  # Sell entry signal
    df.loc[buy_entry_mask, 'Signal'] = 1  # Mark buy entry with 1
    df.loc[sell_entry_mask, 'Signal'] = -1  # Mark sell entry with -1
    
    print(f"Signals generated based on EMA {short_window}/{medium_window}/{long_window} crossover with RSI {rsi_window} < {rsi_overbought} filter.")
    return df

if __name__ == '__main__':
    from indicators import calculate_ema, calculate_rsi

    # --- Load Data ---
    TICKER = config.TICKER
    DATA_PATH = os.path.join(config.DATA_DIR, f'{TICKER}_data.pkl') # Using pickle path

    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        try:
            test_data = pd.read_pickle(DATA_PATH)

            if not test_data.empty:
                # --- Calculate Required Indicators ---
                # Get window sizes and parameters from config
                short_w = config.EMA_SHORT_WINDOW
                medium_w = config.EMA_MEDIUM_WINDOW
                long_w = config.EMA_LONG_WINDOW
                rsi_w = config.RSI_WINDOW
                rsi_ob = config.RSI_OVERBOUGHT

                print(f"\nCalculating EMA_{short_w}...")
                test_data[f'EMA_{short_w}'] = calculate_ema(test_data, short_w)

                print(f"\nCalculating EMA_{medium_w}...")
                test_data[f'EMA_{medium_w}'] = calculate_ema(test_data, medium_w)

                print(f"\nCalculating EMA_{long_w}...")
                test_data[f'EMA_{long_w}'] = calculate_ema(test_data, long_w)

                print(f"\nCalculating RSI_{rsi_w}...")
                test_data[f'RSI_{rsi_w}'] = calculate_rsi(test_data, rsi_w)

                # Drop rows with NaN values created by indicator calculations
                print(f"\nOriginal data length: {len(test_data)}")
                test_data.dropna(inplace=True)
                print(f"Data length after dropna: {len(test_data)}")


                # --- Generate Signals ---
                if not test_data.empty: # Check if data remains
                    data_with_signals = generate_ema_rsi_signals(
                        test_data,
                        short_window=short_w,
                        medium_window=medium_w,
                        long_window=long_w,
                        rsi_window=rsi_w,
                        rsi_overbought=rsi_ob
                    )

                    print("\nData with Signals (Tail):")
                    # Display the last 20 rows of relevant columns for checking purposes
                    display_cols = [
                        'Adj Close',
                        f'EMA_{short_w}',
                        f'EMA_{medium_w}',
                        f'EMA_{long_w}',
                        f'RSI_{rsi_w}',
                        'Position',
                        'Signal'
                    ]
                    print(data_with_signals.filter(items=display_cols).tail(20))

                    # Check signal counts
                    print("\nSignal Counts:")
                    print(f"Total Buy Signals: {data_with_signals['Signal'].value_counts().get(1, 0)}")
                    print(f"Total Sell Signals: {data_with_signals['Signal'].value_counts().get(-1, 0)}")
                    print(f"Total No Action: {data_with_signals['Signal'].value_counts().get(0, 0)}")
                else:
                    print("\nError: No data remaining after removing NaNs from indicator calculation.")

        except Exception as e:
            print(f"An error occurred in the test block: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging

    else:
        print(f"Error: Data file '{DATA_PATH}' not found.")