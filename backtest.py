import pandas as pd
import numpy as np
from datetime import datetime

def run_backtest(ticker, predictions, initial_capital, transaction_cost_pct, share_increment):
    """
    Run backtest on a single ticker using predictions
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol
    predictions : pandas.DataFrame
        DataFrame with predictions
    initial_capital : float
        Initial capital
    transaction_cost_pct : float
        Transaction cost as percentage
    share_increment : int
        Share increment size
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with backtest results
    """
    # Sort predictions by date
    predictions = predictions.sort_index()
    
    # Initialize backtest variables
    capital = initial_capital
    position_size = 0
    available_capital = initial_capital
    buy_hold_shares = 0
    
    results = []
    
    for idx, row in predictions.iterrows():
        # Extract values
        date = idx
        close = row['Close']
        predicted = row['predicted']
        predicted_prob = row['predicted_prob']
        
        # Initialize variables for this day
        action = 'hold'
        shares_traded = 0
        
        # Calculate buy & hold shares (buy as many as possible on first day)
        if buy_hold_shares == 0:
            # Calculate how many shares we can buy with initial capital (including transaction costs)
            max_shares = int(initial_capital / (close * (1 + transaction_cost_pct)))
            # Round down to nearest share_increment
            buy_hold_shares = (max_shares // share_increment) * share_increment
        
        # Calculate buy & hold value
        buy_hold_value = buy_hold_shares * close
        
        # Trading logic based on prediction
        if predicted == 1 and position_size == 0:
            # Buy signal with no position - calculate how many shares we can buy
            max_shares = int(available_capital / (close * (1 + transaction_cost_pct)))
            # Round down to nearest share_increment
            shares_to_buy = (max_shares // share_increment) * share_increment
            
            if shares_to_buy >= share_increment:
                # Execute buy
                cost = shares_to_buy * close * (1 + transaction_cost_pct)
                available_capital -= cost
                position_size = shares_to_buy
                action = 'buy'
                shares_traded = shares_to_buy
        
        elif predicted == 0 and position_size > 0:
            # Sell signal with position - sell all shares
            proceeds = position_size * close * (1 - transaction_cost_pct)
            available_capital += proceeds
            shares_traded = position_size
            position_size = 0
            action = 'sell'
        
        # Calculate portfolio value
        portfolio_value = available_capital + (position_size * close)
        
        # Store results
        results.append({
            'date': date,
            'close': close,
            'predicted': predicted,
            'predicted_prob': predicted_prob,
            'action': action,
            'shares_traded': shares_traded,
            'position_size': position_size,
            'available_capital': available_capital,
            'portfolio_value': portfolio_value,
            'buy_hold_value': buy_hold_value
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def calculate_metrics(backtest_results):
    """
    Calculate performance metrics from backtest results
    
    Parameters:
    -----------
    backtest_results : pandas.DataFrame
        DataFrame with backtest results
        
    Returns:
    --------
    dict
        Dictionary with performance metrics
    """
    # Calculate returns
    initial_value = backtest_results['portfolio_value'].iloc[0]
    final_value = backtest_results['portfolio_value'].iloc[-1]
    total_return = (final_value / initial_value) - 1
    
    # Calculate buy & hold returns
    buy_hold_initial = backtest_results['buy_hold_value'].iloc[0]
    buy_hold_final = backtest_results['buy_hold_value'].iloc[-1]
    buy_hold_return = (buy_hold_final / buy_hold_initial) - 1 if buy_hold_initial > 0 else 0
    
    # Calculate annualized returns
    start_date = backtest_results['date'].iloc[0]
    end_date = backtest_results['date'].iloc[-1]
    days = (end_date - start_date).days
    years = days / 365.25
    
    annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
    annualized_buy_hold = ((1 + buy_hold_return) ** (1 / years)) - 1 if years > 0 else 0
    
    # Calculate drawdown
    portfolio_values = backtest_results['portfolio_value']
    max_drawdown = calculate_max_drawdown(portfolio_values)
    
    # Calculate trade statistics
    buys = backtest_results[backtest_results['action'] == 'buy']
    sells = backtest_results[backtest_results['action'] == 'sell']
    num_trades = len(buys)
    
    # Calculate average trade duration
    if num_trades > 0 and len(sells) > 0:
        trade_durations = []
        for i in range(min(len(buys), len(sells))):
            if i < len(buys) and i < len(sells):
                buy_date = buys.iloc[i]['date']
                sell_date = sells.iloc[i]['date']
                if sell_date > buy_date:  # Ensure chronological order
                    duration = (sell_date - buy_date).days
                    trade_durations.append(duration)
            
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
    else:
        avg_trade_duration = 0
    
    # Percentage of winning trades
    if num_trades > 0:
        profitable_trades = 0
        for i in range(min(len(buys), len(sells))):
            if i < len(buys) and i < len(sells):
                buy_price = buys.iloc[i]['close']
                sell_price = sells.iloc[i]['close']
                if sell_price > buy_price * (1 + 2 * backtest_results['transaction_cost_pct'].iloc[0]):
                    profitable_trades += 1
        
        win_rate = profitable_trades / num_trades if num_trades > 0 else 0
    else:
        win_rate = 0
    
    # Sharpe ratio (assuming 0% risk-free rate)
    daily_returns = backtest_results['portfolio_value'].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 else 0
    
    # Compile metrics
    metrics = {
        'Total Return': f"{total_return:.2%}",
        'Buy & Hold Return': f"{buy_hold_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Annualized Buy & Hold': f"{annualized_buy_hold:.2%}",
        'Alpha': f"{annualized_return - annualized_buy_hold:.2%}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Number of Trades': num_trades,
        'Win Rate': f"{win_rate:.2%}",
        'Average Trade Duration (days)': f"{avg_trade_duration:.1f}",
    }
    
    return metrics

def calculate_max_drawdown(equity_curve):
    """
    Calculate maximum drawdown from an equity curve
    
    Parameters:
    -----------
    equity_curve : pandas.Series
        Series of portfolio values
        
    Returns:
    --------
    float
        Maximum drawdown as a percentage
    """
    # Calculate the max drawdown
    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    return abs(max_drawdown)