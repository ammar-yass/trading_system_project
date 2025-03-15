#!/usr/bin/env python3
"""
Example script to run the trading system with some sample parameters.
This demonstrates how to use the system with multiple correlated assets.
"""

from trading_system import TradingSystem

def main():
    # Define parameters
    tickers = ['SPY', 'QQQ', 'IWM']  # S&P 500, Nasdaq 100, Russell 2000 ETFs
    start_date = '2018-01-01'
    end_date = '2024-01-01'
    initial_capital = 10000
    transaction_cost_pct = 0.01  # 1% transaction cost
    share_increment = 50  # Buy/sell in increments of 50 shares
    
    # Create the trading system
    system = TradingSystem(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        transaction_cost_pct=transaction_cost_pct,
        share_increment=share_increment,
        train_window_days=365,  # 1 year training window
        test_window_days=90     # 3 month testing window
    )
    
    # Run the full system
    print("Running trading system...")
    system.run()
    
    # Plot results
    print("Plotting results...")
    system.plot_results()
    
    print("Done!")

if __name__ == '__main__':
    main()
