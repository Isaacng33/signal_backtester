"""
Main script to run the investment signals backtester workflow.
"""
import os
import pandas as pd
import config
from data_fetcher import get_data
from signals import generate_ema_rsi_signals
from backtester import run_event_driven_backtest

def main():
    """Orchestrates the backtesting process."""
    print("--- Starting Backtesting Workflow ---")

    # --- 1. Fetch Data ---
    ticker = config.TICKER
    hist_data_raw = get_data(ticker=ticker, start_date=config.START_DATE, end_date=config.END_DATE)
    if hist_data_raw.empty:
        print("Error: No data fetched. Exiting.")
        return
    print(f"\nFetched {len(hist_data_raw)} rows of data for {ticker}.")

    # --- 2. Generate Signals ---
    print("\nGenerating signals...")
    try:
        data_with_signals = generate_ema_rsi_signals(
            hist_data_raw.copy(),
            short_window=config.EMA_SHORT_WINDOW,
            medium_window=config.EMA_MEDIUM_WINDOW,
            long_window=config.EMA_LONG_WINDOW,
            rsi_window=config.RSI_WINDOW,
            rsi_overbought=config.RSI_OVERBOUGHT
        )
    except Exception as e:
        print(f"Error during signal generation: {e}")
        return

    # --- 3. Prepare Data for Backtest ---
    backtest_input_data = hist_data_raw[['Adj Close', 'Open']].copy()
    backtest_input_data['Signal'] = data_with_signals['Signal']
    backtest_input_data['Signal'] = backtest_input_data['Signal'].fillna(0)
    backtest_input_data = backtest_input_data.dropna(subset=['Adj Close', 'Open', 'Signal'])
    print(f"\nData ready for backtest (rows: {len(backtest_input_data)})")

    if backtest_input_data.empty:
        print("Error: No data available for backtest after signal alignment and cleaning. Exiting.")
        return

    # --- 4. Run Backtest ---
    print("\nRunning backtest...")
    try:
        backtest_results = run_event_driven_backtest(
            data=backtest_input_data,
            initial_capital=config.INITIAL_CAPITAL,
            commission_per_trade=config.COMMISSION,
            position_size_percent=config.POSITION_SIZE_PERCENT,
            trade_on_close=False
        )
    except Exception as e:
         print(f"Error during backtest execution: {e}")
         return

    # --- 5. Display Summary Results ---
    if backtest_results:
        print("\n--- Backtest Performance Summary ---")
        # Reuse the printing logic from backtester.py's test block or customize
        for key, value in backtest_results.items():
            if key not in ["Trades Log", "Daily Portfolio Value", "Performance Data"]: # Don't print DataFrame/Series summary
                if isinstance(value, float):
                    if "Value" in key or "Capital" in key or "PnL" in key: print(f"  {key}: ${value:,.2f}")
                    elif "%" in key or "Rate" in key: print(f"  {key}: {value:.2f}%")
                    elif "Factor" in key : print(f"  {key}: {value:.2f}")
                    else: print(f"  {key}: {value:.4f}")
                else: print(f"  {key}: {value}")
        print("\nTrade Log:")
        pd.set_option('display.width', 1000)
        print(backtest_results["Trades Log"])

        print("\n--- Backtesting Workflow Complete ---")
    else:
         print("\nBacktest did not produce results.")


if __name__ == "__main__":
    main()
