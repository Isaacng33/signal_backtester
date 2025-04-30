"""
Main script to run the investment signals backtester workflow for multiple tickers.
"""

import pandas as pd
import config
from data_fetcher import get_data
from signals import generate_ema_rsi_signals 
from backtester import run_event_driven_backtest
from visualiser import print_summary_and_trades


def main():
    """Backtesting workflow for single/multiple tickers."""
    print("--- Starting Backtesting Workflow ---")
    print(f"Configuration:")
    print(f"  Tickers: {config.TICKERS}")
    print(f"  Start Date: {config.START_DATE}")
    print(f"  End Date: {config.END_DATE or 'Latest'}")
    interval = config.INTERVAL
    print(f"  Interval: {interval}")
    print(f"  Initial Capital: ${config.INITIAL_CAPITAL:,.2f} (per ticker)")
    print(f"  Commission: ${config.COMMISSION:.2f}")
    print(f"  Position Size: {config.POSITION_SIZE_PERCENT:.0%}")

    # --- 1. Load/Fetch Data for ALL Tickers ---
    all_hist_data =  get_data(
        tickers=config.TICKERS,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        interval=interval
    )

    if not all_hist_data:
        print("Error: No data available for any ticker. Exiting.")
        return

    all_results = {} # Dictionary to store results per ticker

    # --- Loop Through Each Ticker ---
    for ticker, hist_data_raw in all_hist_data.items():
        print(f"\n--- Processing Ticker: {ticker} ---")

        # Make sure data is not empty and has required columns
        if hist_data_raw.empty or not all(col in hist_data_raw.columns for col in ['Adj Close', 'Open']):
            print(f"Error: Data for {ticker} is empty or missing required price columns. Skipping.")
            continue

        # --- 1. Copy Data ---
        hist_data_ticker = hist_data_raw.copy()

        # --- 2. Generate Signals ---
        print(f"Generating signals for {ticker}...")
        try:
            data_with_signals = generate_ema_rsi_signals(
                hist_data_ticker,
                short_window=config.EMA_SHORT_WINDOW,
                medium_window=config.EMA_MEDIUM_WINDOW,
                long_window=config.EMA_LONG_WINDOW,
                rsi_window=config.RSI_WINDOW,
                rsi_overbought=config.RSI_OVERBOUGHT
            )
        except Exception as e:
            print(f"Error during signal generation for {ticker}: {e}")
            import traceback; traceback.print_exc()
            continue # Skip to next ticker

        # --- 3. Prepare Data for Backtest ---
        print(f"Starting backtest for {ticker}...")
        backtest_input_data = hist_data_ticker[['Adj Close', 'Open']].copy()
        # Use .loc to safely align based on matching index dates
        backtest_input_data.loc[data_with_signals.index, 'Signal'] = data_with_signals['Signal']
        # Fill signals that might be NaN due to date range differences or initial indicator NaNs
        backtest_input_data['Signal'] = backtest_input_data['Signal'].fillna(0)
        # Drop rows only if essential columns for backtest have NaN AFTER alignment
        backtest_input_data = backtest_input_data.dropna(subset=['Adj Close', 'Open', 'Signal'])

        if backtest_input_data.empty:
            print(f"Error: No data available for {ticker} backtest after signal generation. Skipping.")
            continue

        # --- 4. Run Backtest ---
        try:
            backtest_results = run_event_driven_backtest(
                data=backtest_input_data,
                initial_capital=config.INITIAL_CAPITAL,
                commission_per_trade=config.COMMISSION,
                position_size_percent=config.POSITION_SIZE_PERCENT,
                trade_on_close=True
            )
            all_results[ticker] = backtest_results # Store results for this ticker
        except Exception as e:
            print(f"Error during backtest execution for {ticker}: {e}")
            import traceback; traceback.print_exc()
            continue # Skip to next ticker

    # --- 5. Display Aggregate/Individual Results ---
    for ticker, results in all_results.items():
        print_summary_and_trades(results, ticker)

    print("\n--- Backtesting Workflow Complete ---")

if __name__ == "__main__":
    main()
