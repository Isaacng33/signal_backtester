# Configuration settings for backtesting stock trading strategies
# --- Data Settings ---
TICKER = 'AAPL'
START_DATE = '2024-01-01'
END_DATE = None  # Use None for the latest data

# --- Strategy Parameters ---
# Simple Moving Average (SMA)
SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200

# Exponential  Moving Average (EMA) 
EMA_SHORT_WINDOW = 50
EMA_LONG_WINDOW = 200

# Relative Strength Index (RSI)
# RSI_WINDOW = 14
# RSI_OVERSOLD = 30
# RSI_OVERBOUGHT = 70

# --- Backtesting Settings ---
INITIAL_CAPITAL = 100000.0

# --- Visualization Settings ---
PLOT_WIDTH = 12
PLOT_HEIGHT = 8

# --- File Paths ---
DATA_DIR = 'data/historical/'
RESULTS_DIR = 'data/results/'
