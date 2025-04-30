# src/visualizer.py (Revised for Aggregated Summary)

import pandas as pd
from tabulate import tabulate


# Set pandas display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)

DEFAULT_METRICS_TO_PRINT = [
    "Trade Method",
    "Start Date",
    "End Date",
    "Initial Capital",
    "Final Portfolio Value",
    "Total Return %",
    "Buy & Hold Return %",
    "Max Drawdown %",
    "Max Drawdown Date",
    "Worst Daily Return %",
    "Worst Daily Return Date",
    "Profit Factor",
    "Total Trades",
    "Win Rate %",
    "Average Win PnL",
    "Average Loss PnL",
    "Commission Per Trade",
    "Total Commission Paid",
]

def print_summary_and_trades(all_results: dict):
    """
    Prints individual trade logs and a final aggregated summary table
    for all tickers in the results dictionary.

    Args:
        all_results (dict): Dictionary where keys are ticker symbols and values
                            are the results dictionaries returned by the backtester.
    """
    if not all_results:
        print("No backtest results to display.")
        return

    # --- 1. Print Individual Trade Logs ---
    print("\n--- === Trade Logs Per Ticker === ---")
    any_trades = False
    for ticker, results in all_results.items():
        if not results:
            print(f"\n--- No valid results dictionary for {ticker} ---")
            continue

        trades_log = results.get("Trades Log")
        print(f"\n--- Trade Log for {ticker} ---")
        if trades_log is not None and isinstance(trades_log, pd.DataFrame):
            if not trades_log.empty:
                any_trades = True
                headers = ["Entry Date", "Entry Price", "Exit Date", "Exit Price", "Shares", "PnL", "Return %"]
                log_copy = trades_log.copy()
                headers_present = [h for h in headers if h in log_copy.columns]

                # Format dates
                if 'Entry Date' in log_copy.columns: log_copy['Entry Date'] = pd.to_datetime(log_copy['Entry Date']).dt.strftime('%Y-%m-%d')
                if 'Exit Date' in log_copy.columns: log_copy['Exit Date'] = pd.to_datetime(log_copy['Exit Date']).dt.strftime('%Y-%m-%d')

                print(tabulate(log_copy[headers_present], headers='keys', tablefmt='psql', showindex=False, floatfmt=".2f"))
            else:
                print("No trades were executed.")
        else:
            print("Trade log not available or not a DataFrame in results.")

    if not any_trades and len(all_results) > 0:
         print("\nNote: No trades were executed for any ticker in this backtest.")

    # --- 2. Create and Print Aggregated Summary Table ---
    print("\n--- === Overall Backtest Summary === ---")

    summary_data = []
    specified_metrics = [
        "Trade Method",
        "Start Date",
        "End Date",
        "Initial Capital",
        "Final Portfolio Value",
        "Total Return %",
        "Buy & Hold Return %",
        "Max Drawdown %",
        "Worst Daily Return %",
        "Profit Factor",
        "Win Rate %",
        "Average Win PnL",
        "Average Loss PnL",
        "Total Trades",
    ]

    summary_columns = ["Ticker"] + specified_metrics

    for ticker, results in all_results.items():
        row_data = {"Ticker": ticker}
        if results:
            for metric_key in specified_metrics:
                value = results.get(metric_key, 'N/A')
                if value is None: 
                    value = 'N/A'
                row_data[metric_key] = value
        else: # Handle case where backtest failed entirely for a ticker
            for metric_key in specified_metrics:
                row_data[metric_key] = 'ERROR'
        summary_data.append(row_data)

    # Create DataFrame using the defined columns for order
    summary_df = pd.DataFrame(summary_data, columns=summary_columns)
    print(tabulate(summary_df, headers='keys', tablefmt='psql', showindex=False, floatfmt=".2f", missingval="N/A"))
