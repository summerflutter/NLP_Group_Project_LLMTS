# Runs a market simulation on the strategy 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def evaluate_strategy(stock, output):
  # Get stock prices for AAPL
  prices = pd.read_csv(f'{stock}.csv')[['date', 'adj close']]
  prices = prices[(prices['date'] >= "2017-01-01") & (prices['date'] <= "2019-12-31")] # Testing split 

  prices = prices.iloc[::-1]
  prices['date'] = pd.to_datetime(prices['date'])
  prices.set_index('date', inplace=True)
  
  output.index = prices.index

  # Simulation 
  start_val = 1000000
  trades_df = pd.DataFrame(index=prices.index, columns=[stock, 'Cash'])
  trades_df.fillna(0, inplace=True)
  trades_df['Cash'] = 0  # Set initial cash

  # Simulate trades based on signals
  current_pos = 0 # Intial Position 
  max_shares = 1000  # Maximum shares to buy or sell
  for date, signal in output['Signal'].iteritems():
    if date in trades_df.index:
      price = prices.loc[date, 'adj close']  # Use .loc for label-based indexing
      if signal == 0 and current_pos < max_shares:  # Buy signal
        shares_to_buy = min(max_shares - current_pos, 1000)
        trades_df.loc[date, stock] += shares_to_buy
        trades_df.loc[date, "Cash"] -= price * shares_to_buy
        current_pos += shares_to_buy
      elif signal == 1 and current_pos > -max_shares:  # Sell signal
        shares_to_sell = min(max_shares + current_pos, 1000)
        trades_df.loc[date, stock] -= shares_to_sell
        trades_df.loc[date, "Cash"] += price * shares_to_sell
        current_pos -= shares_to_sell 
  
  # Calculate cumulative holdings
  holdings_df = pd.DataFrame(0, index=trades_df.index, columns=trades_df.columns)
  holdings_df.iloc[0, :-1] = trades_df.iloc[0, :-1]
  holdings_df.iloc[0,-1] = start_val + trades_df.iloc[0, -1]
  for i in range(1, len(holdings_df)):
    holdings_df.iloc[i] = holdings_df.iloc[i - 1] + trades_df.iloc[i]        

  # Calculate portfolio value
  values_df = pd.DataFrame(index=prices.index, columns=["Portfolio Value"])
  values_df['Portfolio Value'] = holdings_df[stock] * prices['adj close'] + holdings_df['Cash']
  return trades_df, values_df  	   		  		 		  		  		    	 		 		   		 		  

# Gets the return between two dates
def get_benchmark(stock, sv=100000):
    """
    Retrieves the benchmark performance of a given stock using historical price data.
    This function simulates a basic investment strategy where a fixed number of shares are 
    bought at the beginning of the period and held until the end. It provides a simple 
    benchmark for evaluating the performance of more complex trading strategies.

    Parameters:
    - stock (str): The ticker symbol for the stock to be analyzed.
    - sv (int, optional): The starting value of the portfolio in dollars. Defaults to 100,000.

    Returns:
    - pd.DataFrame: A DataFrame indexed by date, with a single column 'Portfolio Value' that
      shows the value of the portfolio over time based on the performance of the stock.

    The function calculates the cumulative return, the standard deviation, and the average of 
    daily returns of the portfolio. It also prints these statistics for quick reference.

    Usage:
    >>> values_df = get_benchmark('AAPL')
    >>> print(values_df)

    Details:
    - The function buys 1000 shares of the stock at the start of the period and calculates the 
      portfolio value across the selected date range from "2017-01-01" to "2019-12-31".
    - The portfolio's daily values are calculated based on the closing prices of the stock, 
      and the remaining cash (if any) after the initial stock purchase is added to the 
      portfolio value.
    - Cumulative return, standard deviation of daily returns, and average daily returns are 
      calculated to provide a statistical summary of the stock's performance over the period.
    """
    
    prices = pd.read_csv(f'{stock}.csv')[['date', 'adj close']]
    prices = prices[(prices['date'] >= "2017-01-01") & (prices['date'] <= "2019-12-31")] # Testing split 

    prices = prices.iloc[::-1]
    prices['date'] = pd.to_datetime(prices['date'])
    prices.set_index('date', inplace=True)
    
    values_df = pd.DataFrame(0, index=prices.index, columns=["Portfolio Value"])
    
    leftover = sv - 1000 * prices['adj close'].iloc[0]    
    values_df["Portfolio Value"] = 1000 * prices["adj close"] + leftover

    # Portfolio value stats  
    cum_ret = values_df['Portfolio Value'].iloc[-1] / values_df['Portfolio Value'].iloc[0] - 1
    daily_rets = values_df['Portfolio Value']/ values_df['Portfolio Value'].shift(1) - 1
    std_daily_rets = daily_rets.std()
    avg_daily_rets = daily_rets.mean()
    print(f"Benchmark cumulative return: {cum_ret}, std: {std_daily_rets}, avg: {avg_daily_rets}")
    return values_df

# Function to calculate cumulature return, std of daily returns, average daily returns. Run this with the output of evaluate_strategy()
def calculate_stats(values_df):
    """
    Calculate key statistics from a DataFrame of portfolio values. This function computes the 
    cumulative return, the standard deviation of daily returns, and the average daily returns
    of the portfolio. It is intended to be run with the output DataFrame from the 
    `evaluate_strategy()` function, which provides daily portfolio values.

    Parameters:
    - values_df (pd.DataFrame): DataFrame with a single column containing the daily portfolio 
      values indexed by date.

    Returns:
    - cum_ret (float): The cumulative return of the portfolio over the period. Calculated as 
      the percentage change from the first to the last value of the portfolio.
    - std_daily_rets (float): The standard deviation of the daily returns, which measures the 
      volatility of the daily portfolio returns.
    - avg_daily_rets (float): The average of the daily returns, indicating the typical daily 
      return of the portfolio.

    Usage Example:
    >>> portfolio_values = evaluate_strategy('AAPL', output)
    >>> cumulative_return, daily_return_std, daily_return_avg = calculate_stats(portfolio_values)
    """

    cum_ret = values_df.iloc[-1] / values_df.iloc[0] - 1
    daily_rets = values_df/ values_df.shift(1) - 1
    std_daily_rets = daily_rets.std()
    avg_daily_rets = daily_rets.mean()
    return cum_ret, std_daily_rets, avg_daily_rets 	  

def plot(trades_df, values_df, stock):
    
    benchmark = get_benchmark(stock)

    # Normalize port_val and benchmark to 1.0 at the start
    port_val_normalized = values_df / values_df.iloc[0]
    benchmark_normalized = benchmark['Portfolio Value'] / benchmark['Portfolio Value'].iloc[0]

    # Create a plot
    plt.figure(figsize=(12, 6))

    # Plotting the normalized port_val and benchmark
    plt.plot(port_val_normalized.index, port_val_normalized, color='red', label='DQN')
    plt.plot(benchmark_normalized.index, benchmark_normalized, color='purple', label='Benchmark')

    # Adding vertical lines for LONG and SHORT entry points
    for date, row in trades_df.iterrows():
        if row[stock] > 0:  # LONG entry
            plt.axvline(x=date, color='blue', linestyle='--', alpha=0.7)
        elif row[stock] < 0:  # SHORT entry
            plt.axvline(x=date, color='black', linestyle='--', alpha=0.7)

    # Adding labels and title
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.title(f'Trading Strategy vs. Benchmark Performance with Trade Entry Points')

    # Create custom lines for the legend
    custom_lines = [Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='purple', lw=2),
                    Line2D([0], [0], color='blue', lw=2, linestyle='--'),
                    Line2D([0], [0], color='black', lw=2, linestyle='--')]

    # Create the legend with all elements
    plt.legend(custom_lines, ['Trading Strategy', 'Benchmark', 'LONG Entry Points', 'SHORT Entry Points'])

    # plt.legend()
    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{stock}.png')
    plt.show()
