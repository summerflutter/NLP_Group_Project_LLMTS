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
  # TODO: create new folder 
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
    df = assign_trading_signals(df) 
    df = df[["date", "adj close", "SMA", "BB%", "RSI", "CCI", "PPO", "PPO_signal", "Signal"]]
    
    file_path = os.path.join(folder_name, f"{stock}_processed.csv")
    df.to_csv(file_path, index=False)

def assign_trading_signals(df):
  # Initialize a column for the signals, default to 2 (Hold)
  df['Signal'] = 2  # 2 represents 'Hold'
    
  # Iterate through DataFrame rows
  for i in range(1, len(df)):
    buy_signals = 0
    sell_signals = 0

    # PPO rule
    if df['PPO'].iloc[i] > df['PPO_signal'].iloc[i] and df['PPO'].iloc[i-1] <= df['PPO_signal'].iloc[i-1]:
      buy_signals += 1
    elif df['PPO'].iloc[i] < df['PPO_signal'].iloc[i] and df['PPO'].iloc[i-1] >= df['PPO_signal'].iloc[i-1]:
      sell_signals += 1

    # CCI rule
    if df['CCI'].iloc[i] > -100 and df['CCI'].iloc[i-1] <= -100:
      buy_signals += 1
    elif df['CCI'].iloc[i] < 100 and df['CCI'].iloc[i-1] >= 100:
      sell_signals += 1

    # Bollinger Bands % rule
    if df['BB%'].iloc[i] > 0.2 and df['BB%'].iloc[i-1] <= 0.2:
      buy_signals += 1
    elif df['BB%'].iloc[i] < 0.8 and df['BB%'].iloc[i-1] >= 0.8:
      sell_signals += 1

    # RSI rule
    if df['RSI'].iloc[i] > 30 and df['RSI'].iloc[i-1] <= 30:
      buy_signals += 1
    elif df['RSI'].iloc[i] < 70 and df['RSI'].iloc[i-1] >= 70:
      sell_signals += 1

    # Assign signals based on majority
    if buy_signals > sell_signals:
      df.loc[i, 'Signal'] = 0  # 0 represents 'Buy'
    elif sell_signals > buy_signals:
      df.loc[i, 'Signal'] = 1  # 1 represents 'Sell'
  return df
