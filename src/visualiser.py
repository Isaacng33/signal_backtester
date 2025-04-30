# src/visualizer.py (Corrected Formatting Logic)

import pandas as pd
import numpy as np
from tabulate import tabulate

# Set pandas display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)


DEFAULT_METRICS_TO_PRINT = {
    "Method": "string",
    "Start Date": "string",
    "End Date": "string",
    "Initial Capital": "currency",
    "Final Portfolio Value": "currency",
    "Total Return %": "percent",
    "Buy & Hold Final Value": "currency",
    "Buy & Hold Return %": "percent",
    "Max Drawdown %": "percent",
    "Max Drawdown Date": "string",
    "Worst Daily Return %": "percent",
    "Worst Daily Return Date": "string",
    "Profit Factor": "factor", 
    "Total Trades": "integer",
    "Win Rate %": "percent",
    "Average Win PnL": "currency",
    "Average Loss PnL": "currency",
    "Commission Per Trade": "currency",
    "Total Commission Paid": "currency",
}

def print_summary_and_trades(results: dict, ticker: str):
    """
    Prints a formatted summary of backtest metrics and the trade log for a single ticker.

    Args:
        results (dict): The dictionary returned by the backtester function for one ticker.
        ticker (str): The ticker symbol.
    """
    if not results:
        print(f"\n--- No results available for {ticker} ---")
        return

    print(f"\n--- === Detailed Summary for {ticker} === ---")

    # --- Print Key Metrics ---
    print("Performance Metrics:")
    for display_key, format_type in DEFAULT_METRICS_TO_PRINT.items():
        results_key = display_key
        value = results.get(results_key, 'N/A')

        print(f"  {display_key:<25}: ", end="") # Left align key

        # Formatting based on format_type
        if value == 'N/A' or value is None:
            print("N/A")
        elif isinstance(value, (int, float)) and np.isinf(value):
            print("Inf")
        elif isinstance(value, (int, float)) and np.isnan(value):
            print("NaN")
        elif format_type == "string":
            print(f"{value}")
        elif format_type == "integer":
            try: print(f"{int(value):,d}") # Integer with comma
            except (ValueError, TypeError): print(value)
        elif format_type == "currency":
            try: print(f"${float(value):,.2f}") # Currency format
            except (ValueError, TypeError): print(value)
        elif format_type == "percent":
            try: print(f"{float(value):.2f}%") # Percent format
            except (ValueError, TypeError): print(value)
        elif format_type == "factor":
            try: print(f"{float(value):.2f}") # Factor/Ratio format (2 decimals)
            except (ValueError, TypeError): print(value)
        else: # Default print for any other case
            print(f"{value}")


    # --- Print Trade Log ---
    trades_log = results.get("Trades Log")
    print(f"\nTrade Log for {ticker}:")
    if trades_log is not None and isinstance(trades_log, pd.DataFrame):
        if not trades_log.empty:
            headers = ["Entry Date", "Entry Price", "Exit Date", "Exit Price", "Shares", "PnL", "Return %"]
            log_copy = trades_log.copy()
            headers_present = [h for h in headers if h in log_copy.columns]

            # Format dates if columns exist
            if 'Entry Date' in log_copy.columns:
                 log_copy['Entry Date'] = pd.to_datetime(log_copy['Entry Date']).dt.strftime('%Y-%m-%d')
            if 'Exit Date' in log_copy.columns:
                 log_copy['Exit Date'] = pd.to_datetime(log_copy['Exit Date']).dt.strftime('%Y-%m-%d')

            print(tabulate(log_copy[headers_present], headers='keys', tablefmt='psql', showindex=False, floatfmt=".2f"))
        else:
            print("No trades were executed.")
    else:
        print("Trade log not available or not a DataFrame in results.")
