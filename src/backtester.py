"""
Core backtesting engine to simulate strategy performance against buy and hold strategy.
"""
import pandas as pd
import numpy as np
import config
import os

# --- Event-Driven Backtest Function ---
# --- Event-Driven Backtest Function (Modified for % Take Profit / Stop Loss) ---
def run_event_driven_backtest(
        data: pd.DataFrame,
        initial_capital: float,
        commission_per_trade: float,
        position_size_percent: float,
        trade_on_close: bool = False
    ):
    """
    Runs an event-driven backtest with optional fixed percentage take profit and stop loss.

    Args:
        data (pd.DataFrame): DataFrame with 'Adj Close', 'Open', 'Signal'.
        initial_capital (float): Starting capital.
        commission_per_trade (float): Cost per trade.
        position_size_percent (float): Fraction of capital per trade.
        trade_on_close (bool): True to trade at Close, False to trade at next Open.

    Returns:
        dict: Dictionary containing backtest results and statistics.
    """

    take_profit_percent = config.TAKE_PROFIT_PERCENT
    stop_loss_percent = config.STOP_LOST_PERCENT

    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Data index must be a DatetimeIndex.")
    required_cols = ['Adj Close', 'Open', 'Signal']
    if not all(col in data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data.columns]
        raise ValueError(f"Data must contain required columns: {missing}")
    if not (0 < position_size_percent <= 1):
         raise ValueError("position_size_percent must be between 0 (exclusive) and 1 (inclusive).")
    if take_profit_percent is not None and take_profit_percent <= 0:
        raise ValueError("take_profit_percent must be positive if set.")
    if stop_loss_percent is not None and stop_loss_percent <= 0:
         raise ValueError("stop_loss_percent must be positive if set.")
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
    if take_profit_percent is not None: print(f"  Take Profit Target: {take_profit_percent:.1%}")
    if stop_loss_percent is not None: print(f"  Stop Loss Target: {stop_loss_percent:.1%}")


    for i in range(n_days):
        current_date = df.index[i]
        original_signal = df['Signal'].iloc[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        current_close = df['Adj Close'].iloc[i]

        # Determine potential execution price/date for triggers occurring *today*
        entry_execution_price = np.nan
        entry_execution_date = None
        signal_exit_price = np.nan
        signal_exit_date = None

        if not trade_on_close: # Trade on Next Open
            if i + 1 < n_days:
                entry_execution_price = df['Open'].iloc[i+1]
                entry_execution_date = df.index[i+1]
                signal_exit_price = df['Open'].iloc[i+1]
                signal_exit_date = df.index[i+1]
        else: # Trade on Current Close
            entry_execution_price = current_close
            entry_execution_date = current_date
            signal_exit_price = current_close
            signal_exit_date = current_date

        # --- Initialize Variables for Possible Current Day's Action ---
        action_taken_today = False
        exit_triggered_today = False
        todays_exit_price = np.nan
        todays_exit_date = None
        todays_exit_reason = ""

        # --- Process Exits (Priority: TP/SL Intra-day, then Original Signal) ---
        if in_market:
            target_price_tp = entry_price * (1 + take_profit_percent) if take_profit_percent is not None else np.inf
            target_price_sl = entry_price * (1 - stop_loss_percent) if stop_loss_percent is not None else 0.0

            # 1. Check Stop Loss Intra-day (using Low) - Highest Priority Exit
            if stop_loss_percent is not None and current_low <= target_price_sl:
                exit_triggered_today = True
                todays_exit_price = target_price_sl
                todays_exit_date = current_date
                todays_exit_reason = f"Stop Loss ({stop_loss_percent:.1%})"
                print(f"{current_date.date()}: Stop Loss condition met intra-day (Entry: {entry_price:.2f}, Target: <= {target_price_sl:.2f}, Low: {current_low:.2f})")

            # 2. Check Take Profit Intra-day (using High) - Only if SL wasn't hit
            elif take_profit_percent is not None and current_high >= target_price_tp:
                exit_triggered_today = True
                 # Approx execution at TP price. More complex: max(current_open, target_price_tp) if gapped up
                todays_exit_price = target_price_tp
                todays_exit_date = current_date # Occurred during this day
                todays_exit_reason = f"Take Profit ({take_profit_percent:.1%})"
                print(f"{current_date.date()}: Take Profit condition met intra-day (Entry: {entry_price:.2f}, Target: >= {target_price_tp:.2f}, High: {current_high:.2f})")

            # 3. Check Original Signal Exit - Only if TP/SL wasn't hit intra-day
            elif original_signal == -1:
                if not np.isnan(signal_exit_price):
                    exit_triggered_today = True
                    todays_exit_price = signal_exit_price
                    todays_exit_date = signal_exit_date
                    todays_exit_reason = "Signal"
                    print(f"{current_date.date()}: Original Sell Signal -> Will execute on {todays_exit_date.date()} at {todays_exit_price:.2f}")
                else:
                    # Cannot execute signal exit (e.g., last day and trade_on_close=False)
                    print(f"{current_date.date()}: Original Sell Signal -> Cannot execute (likely end of data).")
                    # Position remains open until end-of-period handling

            # --- Execute Exit if Triggered ---
            if exit_triggered_today and not np.isnan(todays_exit_price):
                proceeds = (shares_held * todays_exit_price) - commission_per_trade
                cash += proceeds
                trade_pnl = (todays_exit_price - entry_price) * shares_held - (commission_per_trade * 2)
                trade_return_pct = ((todays_exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0

                trades.append({
                    "Entry Date": entry_date, "Entry Price": entry_price,
                    "Exit Date": todays_exit_date, "Exit Price": todays_exit_price,
                    "Shares": shares_held, "PnL": trade_pnl, "Return %": trade_return_pct,
                    "Exit Reason": todays_exit_reason
                })
                print(f"  -> Executed exit of {shares_held} shares on {todays_exit_date.date()} at {todays_exit_price:.2f} ({todays_exit_reason})")

                shares_held = 0
                in_market = False
                entry_price = 0.0
                entry_date = None
                action_taken_today = True # Mark that an action (exit) occurred

        # --- Process Entry ---
        if not in_market and not action_taken_today and original_signal == 1:
            if not np.isnan(entry_execution_price):
                target_investment = cash * position_size_percent
                shares_to_buy = int(target_investment // entry_execution_price)

                if shares_to_buy > 0:
                    cost = (shares_to_buy * entry_execution_price) + commission_per_trade
                    if cost <= cash:
                        cash -= cost
                        shares_held = shares_to_buy
                        in_market = True
                        entry_price = entry_execution_price
                        entry_date = entry_execution_date
                        action_taken_today = True # Mark that an action (entry) occurred
                        print(f"{current_date.date()}: Buy Signal -> Executing entry of {shares_held} shares on {entry_execution_date.date()} at {entry_execution_price:.2f}")
                    else:
                        print(f"{current_date.date()}: Buy Signal -> Not enough cash ({cash:.2f}) to buy {shares_to_buy} shares at {entry_execution_price:.2f} on {entry_execution_date.date()} (cost {cost:.2f})")
                else:
                     print(f"{current_date.date()}: Buy Signal -> Cannot afford any shares at {entry_execution_price:.2f} on {entry_execution_date.date()}")
            else:
                # Cannot execute entry (e.g., last day and trade_on_close=False)
                 print(f"{current_date.date()}: Buy Signal -> Cannot execute (likely end of data).")

        # --- Update Daily Portfolio Value (always based on current day's close) ---
        holdings_value = shares_held * current_close
        portfolio_value = cash + holdings_value
        if current_date in daily_portfolio_values.index:
             daily_portfolio_values[current_date] = portfolio_value
        else:
             print(f"Warning: Date {current_date} not in daily_portfolio_values index. Skipping update.")


    # --- Final Calculations & Metrics (End of Backtest) ---
    if in_market: # Close final position if still held
        print(f"Note: Still in market at end of period. Closing position at last price: {current_close:.2f}")
        proceeds = (shares_held * current_close) - commission_per_trade
        cash += proceeds
        trade_pnl = (current_close - entry_price) * shares_held - (commission_per_trade * 2)
        trade_return_pct = ((current_close / entry_price) - 1) * 100 if entry_price > 0 else 0
        trades.append({
            "Entry Date": entry_date, "Entry Price": entry_price,
            "Exit Date": current_date, "Exit Price": current_close, # Use last close price
            "Shares": shares_held, "PnL": trade_pnl, "Return %": trade_return_pct,
            "Exit Reason": "End of Period"
        })
        shares_held = 0
        holdings_value = 0 # Ensure holdings value is zeroed

    final_portfolio_value = cash # Final value is cash after closing position
    total_return_pct = ((final_portfolio_value / initial_capital) - 1) * 100 if initial_capital > 0 else 0
    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)

    # Recalculate total commission based on trades log
    total_commission_paid = commission_per_trade * total_trades * 2 if total_trades > 0 else 0 # Rough estimate: 2 per trade

    first_price_bh = df['Adj Close'].iloc[0]
    last_price_bh = df['Adj Close'].iloc[-1]
    buy_hold_return_pct_bh = ((last_price_bh / first_price_bh) - 1) if first_price_bh > 0 else 0
    buy_hold_final_value_bh = initial_capital * (1 + buy_hold_return_pct_bh)

    # --- Performance Metrics ---
    winning_trades = trades_df[trades_df['PnL'] > 0] if not trades_df.empty else pd.DataFrame()
    losing_trades = trades_df[trades_df['PnL'] <= 0] if not trades_df.empty else pd.DataFrame()
    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    avg_win_pnl = winning_trades['PnL'].mean() if not winning_trades.empty else 0
    avg_loss_pnl = losing_trades['PnL'].mean() if not losing_trades.empty else 0
    total_profit = winning_trades['PnL'].sum()
    total_loss = abs(losing_trades['PnL'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else np.inf

    # Daily Returns, Drawdown, etc.
    daily_portfolio_values = daily_portfolio_values.dropna() # Drop any NaNs if dates were missed
    portfolio_daily_returns = daily_portfolio_values.pct_change().fillna(0)
    cumulative_max = daily_portfolio_values.cummax()
    drawdown = (daily_portfolio_values - cumulative_max) / cumulative_max
    max_drawdown_pct = drawdown.min() * 100 if not drawdown.empty and not drawdown.min() >= 0 else 0
    max_drawdown_date = drawdown.idxmin() if not drawdown.empty and not drawdown.min() >= 0 else (df.index[0] if not df.empty else None)

    worst_daily_return_pct = portfolio_daily_returns.min() * 100 if not portfolio_daily_returns.empty else 0
    worst_daily_return_date = portfolio_daily_returns.idxmin() if not portfolio_daily_returns.empty else (df.index[0] if not df.empty else None)


    results = {
        "Take Profit %": f"{take_profit_percent:.1%}" if take_profit_percent is not None else "None",
        "Stop Loss %": f"{stop_loss_percent:.1%}" if stop_loss_percent is not None else "None",
        "Start Date": df.index[0].strftime('%Y-%m-%d') if not df.empty else 'N/A',
        "End Date": df.index[-1].strftime('%Y-%m-%d') if not df.empty else 'N/A',
        "Initial Capital": initial_capital,
        "Final Portfolio Value": final_portfolio_value,
        "Total Return %": total_return_pct,
        "Buy & Hold Final Value": buy_hold_final_value_bh,
        "Buy & Hold Return %": buy_hold_return_pct_bh * 100,
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
        "Total Commission Paid": total_commission_paid, # Note: Rough estimate
        "Trades Log": trades_df,
        "Daily Portfolio Value": daily_portfolio_values
    }
    print("\nEvent-Driven Backtest Simulation Complete.")
    return results