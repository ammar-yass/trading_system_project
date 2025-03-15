import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from feature_engineering import engineer_features
from model_manager import train_model, predict
from backtest import run_backtest, calculate_metrics

class TradingSystem:
    def __init__(self, tickers, start_date, end_date, initial_capital=10000, 
                 transaction_cost_pct=0.01, share_increment=50, 
                 train_window_days=365, test_window_days=90):
        """
        Initialize the trading system with parameters
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols to trade
        start_date : str
            Start date for data collection (YYYY-MM-DD)
        end_date : str
            End date for data collection (YYYY-MM-DD)
        initial_capital : float
            Starting capital
        transaction_cost_pct : float
            Transaction cost as percentage (0.01 = 1%)
        share_increment : int
            Share increment size (e.g., 50)
        train_window_days : int
            Number of days in training window
        test_window_days : int
            Number of days in testing window
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.share_increment = share_increment
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        
        # Storage for data
        self.data = {}
        self.features = {}
        self.models = {}
        self.predictions = {}
        self.positions = {ticker: 0 for ticker in tickers}
        self.capital = initial_capital
        self.trade_log = []
        self.daily_portfolio_value = []
        self.backtest_results = {}
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
    
    def fetch_data(self):
        """Fetch data for all tickers"""
        print(f"Fetching data for {self.tickers} from {self.start_date} to {self.end_date}")
        
        # Add extra days for feature calculation (indicators need lookback)
        extended_start = (datetime.strptime(self.start_date, '%Y-%m-%d') - 
                         timedelta(days=200)).strftime('%Y-%m-%d')
        
        for ticker in self.tickers:
            # Get data from yfinance
            self.data[ticker] = yf.download(ticker, start=extended_start, 
                                           end=self.end_date, progress=False)
            
            if self.data[ticker].empty:
                raise ValueError(f"No data found for {ticker}")
                
            print(f"Downloaded {len(self.data[ticker])} rows for {ticker}")
    
    def generate_features(self):
        """Generate features for all tickers"""
        for ticker in self.tickers:
            self.features[ticker] = engineer_features(
                self.data[ticker].copy(), 
                self.transaction_cost_pct
            )
            
            print(f"Created features for {ticker}, total rows: {len(self.features[ticker])}")
    
    def run_walk_forward_validation(self):
        """Run walk-forward validation for all tickers"""
        all_predictions = {}
        
        # Convert date strings to datetime
        start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        # Calculate training and testing dates
        train_start = start_date
        
        while train_start + timedelta(days=self.train_window_days) < end_date:
            train_end = train_start + timedelta(days=self.train_window_days)
            test_start = train_end + timedelta(days=1)
            test_end = min(test_start + timedelta(days=self.test_window_days), end_date)
            
            print(f"\nWalk-forward window:")
            print(f"  Train: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
            print(f"  Test: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
            
            # For each ticker
            for ticker in self.tickers:
                df = self.features[ticker]
                
                # Get training and testing data
                train_data = df[(df.index >= train_start) & (df.index <= train_end)]
                test_data = df[(df.index >= test_start) & (df.index <= test_end)]
                
                if len(train_data) < 50 or len(test_data) < 5:
                    print(f"Skipping {ticker} due to insufficient data")
                    continue
                
                print(f"\nProcessing {ticker}:")
                print(f"  Training data rows: {len(train_data)}")
                print(f"  Testing data rows: {len(test_data)}")
                
                # Train the model
                self.models[ticker] = train_model(
                    ticker, 
                    train_data, 
                    self.tickers, 
                    self.features
                )
                
                # Make predictions
                test_with_preds = predict(
                    ticker, 
                    test_data, 
                    self.models[ticker], 
                    self.tickers, 
                    self.features
                )
                
                # Store the predictions
                if ticker not in all_predictions:
                    all_predictions[ticker] = test_with_preds
                else:
                    all_predictions[ticker] = pd.concat([all_predictions[ticker], test_with_preds])
            
            # Move to the next window
            train_start = test_start
        
        self.predictions = all_predictions
        
        return all_predictions
    
    def run_backtest(self):
        """Run backtest using the predictions"""
        # Run backtest for each ticker
        for ticker in self.tickers:
            if ticker not in self.predictions or self.predictions[ticker].empty:
                print(f"Skipping backtest for {ticker}: no predictions available")
                continue
                
            print(f"\nRunning backtest for {ticker}")
            
            # Get the predictions
            predictions = self.predictions[ticker]
            
            # Run backtest
            backtest_results = run_backtest(
                ticker,
                predictions,
                self.initial_capital,
                self.transaction_cost_pct,
                self.share_increment
            )
            
            # Store backtest results
            self.backtest_results[ticker] = backtest_results
        
        return self.backtest_results
    
    def calculate_performance_metrics(self):
        """Calculate performance metrics for all tickers"""
        for ticker in self.tickers:
            if ticker not in self.backtest_results:
                continue
                
            backtest_results = self.backtest_results[ticker]
            
            # Calculate metrics
            metrics = calculate_metrics(backtest_results)
            
            print(f"\nPerformance metrics for {ticker}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
    
    def plot_results(self):
        """Plot the results of the backtest"""
        if not self.backtest_results:
            print("No backtest results to plot")
            return
            
        plt.figure(figsize=(14, 8))
        
        # Plot equity curves for each ticker
        for ticker, results in self.backtest_results.items():
            plt.plot(results['date'], results['portfolio_value'], label=f"{ticker} Strategy")
            
            # Plot buy and hold
            buy_hold = results['buy_hold_value']
            plt.plot(results['date'], buy_hold, '--', label=f"{ticker} Buy & Hold")
        
        plt.title('Portfolio Equity Curves')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('portfolio_equity_curves.png')
        plt.show()
        
        # Plot individual ticker performance
        for ticker, results in self.backtest_results.items():
            plt.figure(figsize=(14, 12))
            
            # Plot portfolio value
            plt.subplot(3, 1, 1)
            plt.plot(results['date'], results['portfolio_value'], label='Strategy')
            plt.plot(results['date'], results['buy_hold_value'], '--', label='Buy & Hold')
            plt.title(f'{ticker} Portfolio Value')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.legend()
            plt.grid(True)
            
            # Plot price and positions
            plt.subplot(3, 1, 2)
            plt.plot(results['date'], results['close'], label='Close Price')
            
            # Plot buy signals
            buys = results[results['action'] == 'buy']
            sells = results[results['action'] == 'sell']
            
            plt.scatter(buys['date'], buys['close'], color='green', marker='^', label='Buy')
            plt.scatter(sells['date'], sells['close'], color='red', marker='v', label='Sell')
            
            plt.title(f'{ticker} Price and Signals')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True)
            
            # Plot position size
            plt.subplot(3, 1, 3)
            plt.plot(results['date'], results['position_size'], label='Position Size (Shares)')
            plt.title(f'{ticker} Position Size')
            plt.xlabel('Date')
            plt.ylabel('Shares')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{ticker}_performance.png')
            plt.show()
    
    def run(self):
        """Run the complete trading system"""
        # Fetch data
        self.fetch_data()
        
        # Generate features
        self.generate_features()
        
        # Run walk-forward validation
        self.run_walk_forward_validation()
        
        # Run backtest
        self.run_backtest()
        
        # Calculate performance metrics
        self.calculate_performance_metrics()
