# Configuration settings for backtesting stock trading strategies
# --- Data Settings ---
TICKERS = ['GOOGL', 'AAPL', 'MSFT']
START_DATE = '2020-01-01'
END_DATE = None  # Use None for the latest data
INTERVAL = '1d'  # Data interval (e.g., '1m', '5m', '1h', '1d')

# --- Strategy Parameters ---
# Use None for unused values
EMA_SHORT_WINDOW = 9
EMA_MEDIUM_WINDOW = 21
EMA_LONG_WINDOW = 50 

SMA_SHORT_WINDOW = 9
SMA_MEDIUM_WINDOW = 21
ESA_LONG_WINDOW = 50 

# Relative Strength Index (RSI) Filter
RSI_WINDOW = 14        # RSI calculation period
RSI_OVERBOUGHT = 70    # RSI level above which we consider the stock overbought (filter out buys)
RSI_OVERSOLD = 30      # RSI level below which we consider the stock oversold (not used in this simple long-only exit)

# --- Backtesting Settings ---
INITIAL_CAPITAL = 100000.0
COMMISSION = 0.0  # FIXED commission per trade
POSITION_SIZE_PERCENT = 1.0  # Fraction of capital to invest in each trade

# --- File Paths ---
DATA_DIR = 'data/historical/' # For storing .pkl data
RESULTS_DIR = 'data/results/'