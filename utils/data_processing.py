import pandas as pd
from ta.volatility import BollingerBands
from ta.trend import CCIIndicator
from ta.momentum import RSIIndicator, PercentagePriceOscillator
import os


# Testing on AAPL, MSFT, AMZN, INTC, and NVDA
stock_list = ['AAPL', 'MSFT', 'AMZN', 'INTC', 'NVDA'] 
start_date = '2000-01-01'
end_date = '2016-12-31'

# Hyperparams for calculating technical indicators 
sma_window = 10 
ppo_slow_ema = 26
ppo_fast_ema = 12
ppo_signal = 9

def process_data(stock_list, start_date, end_date):
  folder_name = 'data'
  try:
    os.makedirs(folder_name, exist_ok=True)  # Use exist_ok=True to prevent an error if the folder already exists
    print(f"Directory '{folder_name}' created successfully.")
  except Exception as e:
    print(f"Failed to create directory '{folder_name}'. Error: {e}")
    return

  for stock in stock_list:
    df = pd.read_csv(f'{stock}.csv')
    df = df.iloc[::-1]
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    # 10 day SMA
    sma = df["adj close"].rolling(window=sma_window).mean()
    df["SMA"] = sma

    # Bollinger Bands
    indicator_bb = BollingerBands(close=df["adj close"])
    df['BB%'] = (df['adj close'] - indicator_bb.bollinger_lband()) / (indicator_bb.bollinger_hband() - indicator_bb.bollinger_lband())

    # RSI
    rsi_indicator = RSIIndicator(close=df['adj close'])
    df['RSI'] = rsi_indicator.rsi()

    # CCI
    cci_indicator = CCIIndicator(high=df['high'], low=df['low'], close=df['adj close'])
    df['CCI'] = cci_indicator.cci()

    # PPO and PPO Signal
    indicator_ppo = PercentagePriceOscillator(df['adj close'], window_slow=ppo_slow_ema, window_fast=ppo_fast_ema, window_sign=ppo_signal)

    # Create a DataFrame to store PPO and its signal line
    df['PPO'] = indicator_ppo.ppo()
    df['PPO_signal'] = indicator_ppo.ppo_signal()

    # Get trading signal
    df_labels = assign_trading_signals(df, stock)
    df_labels = df_labels[["date", "adj close", "SMA", "BB%", "RSI", "CCI", "PPO", "PPO_signal", "Signal"]]

    file_path = os.path.join(folder_name, f"{stock}_processed.csv")
    df_labels.to_csv(file_path, index=False)

# Get labels - theoretically the optimal moves to make
def assign_trading_signals(df, stock):
  # Initialize a column for the signals, default to 2 (Hold)
  df['Signal'] = 2  # 2 represents 'Hold'

  # Iterate through DataFrame rows
  prices = pd.read_csv(f'{stock}.csv')[['date', 'adj close']]
  prices = prices[(prices['date'] >= start_date) & (prices['date'] <= end_date)] 

  prices = prices.iloc[::-1]
  prices['date'] = pd.to_datetime(prices['date'])
  prices.set_index('date', inplace=True)

  # columns = np.append(stocks, "Cash")
  holdings = 1000
  for i in range(len(prices) - 1):
    next_price = prices['adj close'].iloc[i + 1]
    curr_price = prices['adj close'].iloc[i]
    if next_price > curr_price:
      if holdings == 0:
        df['Signal'].iloc[i] = 0
        holdings += 1000
      elif holdings == -1000:
        df['Signal'].iloc[i] = 0
        holdings += 2000
    elif next_price < curr_price:
      if holdings == 1000:
        df['Signal'].iloc[i] = 1
        holdings -= 2000
      elif holdings == 0:
        df['Signal'].iloc[i] = 1
        holdings -= 1000
  return df
