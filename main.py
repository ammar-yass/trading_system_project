import argparse
from datetime import datetime
from trading_system import TradingSystem

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run a trading system with walk-forward validation.')
    
    parser.add_argument('--tickers', nargs='+', required=True,
                        help='List of ticker symbols to trade')
    parser.add_argument('--start_date', required=True,
                        help='Start date for data collection (YYYY-MM-DD)')
    parser.add_argument('--end_date', required=True,
                        help='End date for data collection (YYYY-MM-DD)')
    parser.add_argument('--initial_capital', type=float, default=10000,
                        help='Starting capital')
    parser.add_argument('--transaction_cost_pct', type=float, default=0.01,
                        help='Transaction cost as percentage (0.01 = 1%%)')
    parser.add_argument('--share_increment', type=int, default=50,
                        help='Share increment size (e.g., 50)')
    parser.add_argument('--train_window_days', type=int, default=365,
                        help='Number of days in training window')
    parser.add_argument('--test_window_days', type=int, default=90,
                        help='Number of days in testing window')
    
    return parser.parse_args()

def main():
    """Main function to run the trading system."""
    args = parse_arguments()
    
    # Create and run the trading system
    system = TradingSystem(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        transaction_cost_pct=args.transaction_cost_pct,
        share_increment=args.share_increment,
        train_window_days=args.train_window_days,
        test_window_days=args.test_window_days
    )
    
    # Run the full system
    system.run()
    
    # Plot results
    system.plot_results()

if __name__ == '__main__':
    main()
