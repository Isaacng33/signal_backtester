"""
Main script to run the investment signal generation and backtesting workflow.
"""
from src import config
from src import data_fetcher
from src import signals
# Import other modules like backtester, visualizer when ready

def run_strategy():
    """
    Executes the main workflow: fetch data, generate signals.
    (Later: run backtest, visualize results).
    """
    print("--- Starting Strategy Workflow ---")

    # 1. Fetch Data
    stock_data = data_fetcher.get_data(
        ticker=config.TICKER,
        start_date=config.START_DATE,
        end_date=config.END_DATE
    )

    if stock_data.empty:
        print("Workflow aborted: Could not fetch data.")
        return

    # 2. Generate Signals (Example: SMA Crossover)
    data_with_signals = signals.generate_sma_crossover_signals(stock_data)

    print("\n--- Signals Generated ---")
    print(f"Displaying last 5 rows for ticker: {config.TICKER}")
    print(data_with_signals[['Adj Close', 'SMA_Short', 'SMA_Long', 'Signal', 'Position']].tail())

    # --- Future Steps ---
    # 3. Run Backtester
    # results = backtester.run_backtest(data_with_signals, config.INITIAL_CAPITAL)
    # print("\n--- Backtesting Results ---")
    # print(results) # Display performance metrics

    # 4. Visualize Results
    # visualizer.plot_strategy(data_with_signals, results)
    # print("Plots generated.")


    print("\n--- Strategy Workflow Finished ---")

if __name__ == '__main__':
    # This block executes when the script is run directly
    run_strategy()