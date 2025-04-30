"""
Core backtesting engine to simulate strategy performance against buy and hold strategy.
"""
import pandas as pd
import numpy as np
import config
import os

# --- Event-Driven Backtest Function ---
def run_event_driven_backtest(
        data: pd.DataFrame,
        initial_capital: float,
        commission_per_trade: float,
        position_size_percent: float,
        trade_on_close: bool = False
    ):
    """
    Runs an event-driven (trade-by-trade) backtest simulation.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex.")
    required_cols = ['Adj Close', 'Open', 'Signal']
    if not all(col in data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data.columns]
        raise ValueError(f"Data must contain required columns: {missing}")
    if not (0 < position_size_percent <= 1):
         raise ValueError("position_size_percent must be between 0 (exclusive) and 1 (inclusive).")
    if data.empty:
        print("Warning: Input data for event-driven backtest is empty. Cannot run.")
        return None

    df = data.copy()
    n_days = len(df)

    # State variables
    cash = initial_capital
    shares_held = 0
    in_market = False
    entry_price = 0.0
    entry_date = None
    portfolio_value = initial_capital

    # Logging
    trades = []
    daily_portfolio_values = pd.Series(index=df.index, dtype=float)
    daily_portfolio_values.iloc[0] = initial_capital

    print(f"\nRunning Event-Driven Backtest ({'Trade on Close' if trade_on_close else 'Trade on Next Open'})...")

    for i in range(n_days):
        current_date = df.index[i]
        signal = df['Signal'].iloc[i]
        current_close = df['Adj Close'].iloc[i]

        execution_price = np.nan
        execution_date = None
        if not trade_on_close:
            if i + 1 < n_days:
                execution_price = df['Open'].iloc[i+1]
                execution_date = df.index[i+1]
        else:
            execution_price = current_close
            execution_date = current_date

        # --- Handle Trading Logic ---
        if signal == -1 and in_market: # Exit Logic
            if not np.isnan(execution_price):
                proceeds = (shares_held * execution_price) - commission_per_trade
                cash += proceeds
                trade_pnl = (execution_price - entry_price) * shares_held - (commission_per_trade * 2) # Apply commission to PnL calc too
                trade_return_pct = ((execution_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                trades.append({
                    "Entry Date": entry_date, "Entry Price": entry_price,
                    "Exit Date": execution_date, "Exit Price": execution_price,
                    "Shares": shares_held, "PnL": trade_pnl, "Return %": trade_return_pct
                })
                print(f"{current_date.date()}: Sell Signal -> Executing exit of {shares_held} shares on {execution_date.date()} at {execution_price:.2f}")
                shares_held = 0
                in_market = False
                entry_price = 0.0
                entry_date = None
            else:
                print(f"{current_date.date()}: Sell Signal -> Cannot execute on last day.")

        elif signal == 1 and not in_market: # Entry Logic
            if not np.isnan(execution_price):
                target_investment = cash * position_size_percent
                shares_to_buy = int(target_investment // execution_price)
                if shares_to_buy > 0:
                    cost = (shares_to_buy * execution_price) + commission_per_trade
                    if cost <= cash:
                        cash -= cost
                        shares_held = shares_to_buy
                        in_market = True
                        entry_price = execution_price
                        entry_date = execution_date
                        print(f"{current_date.date()}: Buy Signal -> Executing entry of {shares_held} shares on {execution_date.date()} at {execution_price:.2f}")
                    else:
                        print(f"{current_date.date()}: Buy Signal -> Not enough cash ({cash:.2f}) to buy {shares_to_buy} shares at {execution_price:.2f} on {execution_date.date()} (cost {cost:.2f})")
                else:
                    print(f"{current_date.date()}: Buy Signal -> Cannot afford any shares at {execution_price:.2f} on {execution_date.date()}")
            else:
                print(f"{current_date.date()}: Buy Signal -> Cannot execute on last day.")

        # --- Update Portfolio Value for the Day ---
        holdings_value = shares_held * current_close
        portfolio_value = cash + holdings_value
        daily_portfolio_values[current_date] = portfolio_value

    # --- Final Calculations & Metrics ---
    if in_market: # Close final position if still held
        print(f"Note: Still in market at end of period. Closing position at last price: {current_close:.2f}")
        proceeds = (shares_held * current_close) - commission_per_trade
        cash += proceeds
        trade_pnl = (current_close - entry_price) * shares_held - (commission_per_trade * 2)
        trade_return_pct = ((current_close / entry_price) - 1) * 100 if entry_price > 0 else 0
        trades.append({
            "Entry Date": entry_date, "Entry Price": entry_price,
            "Exit Date": current_date, "Exit Price": current_close,
            "Shares": shares_held, "PnL": trade_pnl, "Return %": trade_return_pct
        })
        shares_held = 0
        holdings_value = 0

    final_portfolio_value = cash + holdings_value
    total_return_pct = ((final_portfolio_value / initial_capital) - 1) * 100
    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    total_commission_paid = 2 * (total_trades * commission_per_trade) # For entry and exit

    first_price_bh = df['Adj Close'].iloc[0]
    last_price_bh = df['Adj Close'].iloc[-1]
    buy_hold_return_pct_bh = (last_price_bh / first_price_bh) - 1
    buy_hold_final_value_bh = initial_capital * (1 + buy_hold_return_pct_bh)

    winning_trades = trades_df[trades_df['PnL'] > 0] if not trades_df.empty else pd.DataFrame()
    losing_trades = trades_df[trades_df['PnL'] <= 0] if not trades_df.empty else pd.DataFrame()
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    avg_win_pnl = winning_trades['PnL'].mean() if not winning_trades.empty else 0
    avg_loss_pnl = losing_trades['PnL'].mean() if not losing_trades.empty else 0
    total_loss = abs(losing_trades['PnL'].sum())
    profit_factor = winning_trades['PnL'].sum() / total_loss if total_loss > 0 else np.inf

    # --- Daily Returns Calculation & Max Drawdown --
    portfolio_daily_returns = daily_portfolio_values.pct_change().fillna(0)
    cumulative_max = daily_portfolio_values.cummax()
    drawdown = (daily_portfolio_values - cumulative_max) / cumulative_max
    max_drawdown_pct = drawdown.min() * 100 if not drawdown.empty else 0
    max_drawdown_date = drawdown.idxmin()

    # Worst Daily Return
    worst_daily_return_pct = portfolio_daily_returns.min() * 100 if not portfolio_daily_returns.empty else 0
    worst_daily_return_date = portfolio_daily_returns.idxmin()

    results = {
        "Method": f"Event-Driven ({'Close' if trade_on_close else 'Next Open'})",
        "Start Date": df.index[0].strftime('%Y-%m-%d'),
        "End Date": df.index[-1].strftime('%Y-%m-%d'),
        "Initial Capital": initial_capital,
        "Final Portfolio Value": final_portfolio_value,
        "Total Return %": total_return_pct,
        "Buy & Hold Final Value": buy_hold_final_value_bh, "Buy & Hold Return %": buy_hold_return_pct_bh * 100,
        "Max Drawdown %": max_drawdown_pct,
        "Max Drawdown Date": max_drawdown_date.strftime('%Y-%m-%d') if pd.notna(max_drawdown_date) else 'N/A',
        "Worst Daily Return %": worst_daily_return_pct,
        "Worst Daily Return Date": worst_daily_return_date.strftime('%Y-%m-%d') if pd.notna(worst_daily_return_date) else 'N/A',
        "Profit Factor": profit_factor,
        "Total Trades": total_trades,
        "Win Rate %": win_rate,
        "Average Win PnL": avg_win_pnl,
        "Average Loss PnL": avg_loss_pnl,
        "Commission Per Trade": commission_per_trade,
        "Total Commission Paid": total_commission_paid,
        "Trades Log": trades_df,
        "Daily Portfolio Value": daily_portfolio_values # Keep this for plotting
    }
    print("\nEvent-Driven Backtest Simulation Complete.")
    return results


# --- Example Usage  ---
if __name__ == '__main__':
    from signals import generate_ema_rsi_signals

    # --- Use Parameters from Config ---
    TICKER = config.TICKER
    START_DATE = config.START_DATE
    END_DATE = config.END_DATE
    INITIAL_CAPITAL = config.INITIAL_CAPITAL
    COMMISSION = config.COMMISSION # Use commission from config
    POSITION_SIZE_PERCENT = config.POSITION_SIZE_PERCENT # Use position size from config
    EMA_SHORT = config.EMA_SHORT_WINDOW
    EMA_MEDIUM = config.EMA_MEDIUM_WINDOW
    EMA_LONG = config.EMA_LONG_WINDOW
    RSI_W = config.RSI_WINDOW
    RSI_OB = config.RSI_OVERBOUGHT

    DATA_PATH = os.path.join(config.DATA_DIR, f'{TICKER}_data.pkl')

    # --- 1. Load Data ---
    if os.path.exists(DATA_PATH):
        print(f"Loading data from {DATA_PATH}...")
        try:
            test_data = pd.read_pickle(DATA_PATH)

            if not test_data.empty and 'Adj Close' in test_data.columns and 'Open' in test_data.columns:

                # --- 2. Generate Signals ---
                print("\nGenerating signals...")
                data_with_signals = generate_ema_rsi_signals(
                    test_data.copy(),
                    short_window=EMA_SHORT,
                    medium_window=EMA_MEDIUM,
                    long_window=EMA_LONG,
                    rsi_window=RSI_W,
                    rsi_overbought=RSI_OB
                )

                # --- 3. Prepare Data for Backtest ---
                backtest_input_data = test_data[['Adj Close', 'Open']].copy()
                backtest_input_data['Signal'] = data_with_signals['Signal']
                backtest_input_data['Signal'] = backtest_input_data['Signal'].fillna(0)
                backtest_input_data = backtest_input_data.dropna(subset=['Adj Close', 'Open', 'Signal'])
                print(f"Data length for backtest after aligning signals: {len(backtest_input_data)}")


                # --- 4. Run Event-Driven Backtest ---
                if not backtest_input_data.empty:
                    print("\nRunning Event-Driven backtest (Trade on Next Open)...")
                    event_backtest_results = run_event_driven_backtest(
                        data=backtest_input_data,
                        initial_capital=INITIAL_CAPITAL,
                        commission_per_trade=COMMISSION, # Pass COMMISSION from config
                        position_size_percent=POSITION_SIZE_PERCENT, # Pass position size from config
                        trade_on_close=False
                    )

                    # --- 5. Display Results ---
                    print("\n--- Event-Driven Backtest Performance Summary ---")
                    if event_backtest_results:
                        for key, value in event_backtest_results.items():
                            if key not in ["Trades Log", "Daily Portfolio Value"]:
                                if isinstance(value, float):
                                    if "Value" in key or "Capital" in key or "PnL" in key: print(f"  {key}: ${value:,.2f}")
                                    elif "%" in key or "Rate" in key: print(f"  {key}: {value:.2f}%")
                                    elif "Factor" in key : print(f"  {key}: {value:.2f}")
                                    else: print(f"  {key}: {value:.4f}")
                                else: print(f"  {key}: {value}")
                        print("\nTrade Log:")
                        pd.set_option('display.width', 1000)
                        print(event_backtest_results["Trades Log"])
                else:
                     print("\nError: No data available for backtest after aligning signals.")

            else:
                print("Error: Loaded data missing 'Adj Close' or 'Open' columns.")

        except Exception as e:
            print(f"An error occurred in the backtester test block: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Error: Data file '{DATA_PATH}' not found.")