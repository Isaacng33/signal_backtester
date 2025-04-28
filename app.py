import yfinance as yf
import pandas as pd

ticker = 'AAPL' # Example: Apple
start_date = '2020-01-01'
end_date = '2024-12-31'

data = yf.download(ticker, start=start_date, end=end_date)
print(data.head())