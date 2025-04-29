# Configuration settings for backtesting stock trading strategies
# --- Data Settings ---
TICKER = 'GOOGL'
START_DATE = '2010-01-01'
END_DATE = None  # Use None for the latest data

# --- Strategy Parameters ---
# Use None for unused values
# Exponential Moving Average (EMA) Crossover Strategy
EMA_SHORT_WINDOW = 9  # Short-term EMA period for entry/exit signal
EMA_MEDIUM_WINDOW = 21 # Medium-term EMA period for entry/exit signal
EMA_LONG_WINDOW = 50   # Long-term EMA period for trend filter

# Relative Strength Index (RSI) Filter
RSI_WINDOW = 14        # RSI calculation period
RSI_OVERBOUGHT = 70    # RSI level above which we consider the stock overbought (filter out buys)
RSI_OVERSOLD = 30      # RSI level below which we consider the stock oversold (not used in this simple long-only exit)

# --- Backtesting Settings ---
INITIAL_CAPITAL = 100000.0

# --- Visualization Settings ---
PLOT_WIDTH = 12
PLOT_HEIGHT = 8

# --- File Paths ---
DATA_DIR = 'data/historical/' # For storing .pkl data
RESULTS_DIR = 'data/results/'