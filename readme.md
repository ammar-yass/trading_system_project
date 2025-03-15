# Long-Only Daily Trading System with Walk-Forward Validation

This project implements a multi-asset long-only daily trading system that uses machine learning for predicting favorable market conditions. The system follows a walk-forward validation approach to adaptively train models using historical data and test on out-of-sample periods.

## Key Features

- **Long-only trading** with the ability to hold positions from 1 day to several months
- **Multiple correlated assets** trading with cross-asset information used as features
- **Incremental position sizing** (buying in multiples of 50 shares)
- **Partial sell support** rather than all-or-nothing trading
- **Transaction cost modeling** (1% for slippage/fees)
- **Walk-forward validation** with rolling window retraining
- **Comprehensive feature engineering** (price, volume, technical indicators, date features)
- **Performance comparison** against buy-and-hold strategy

## System Components

The system is organized into the following Python modules:

1. **trading_system.py**: Main class that orchestrates the entire system
2. **feature_engineering.py**: Functions for creating technical and date-based features
3. **model_manager.py**: Functions for training, predicting, and evaluating models
4. **backtest.py**: Engine for running backtests and calculating performance metrics
5. **main.py**: Command-line interface for running the system with custom parameters
6. **run_example.py**: Example script with predefined parameters

## Model and Methods

The system uses Random Forest classifiers for predicting favorable market conditions. This approach:

- Handles non-linear market relationships
- Provides feature importance rankings
- Reduces overfitting risk through ensemble methods
- Works well with both categorical and numerical features
- Is robust to outliers and missing data

## Usage

### Basic Example

```python
from trading_system import TradingSystem

# Create system
system = TradingSystem(
    tickers=['SPY', 'QQQ', 'IWM'],
    start_date='2018-01-01',
    end_date='2024-01-01',
    initial_capital=10000,
    transaction_cost_pct=0.01,
    share_increment=50
)

# Run the system
system.run()

# Plot results
system.plot_results()
```

### Command Line Interface

You can also run the system from the command line:

```bash
python main.py --tickers SPY QQQ IWM --start_date 2018-01-01 --end_date 2024-01-01 --initial_capital 10000
```

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- yfinance
- ta (Technical Analysis library)

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Performance Metrics

The system evaluates trading performance using:

- Total Return
- Annualized Return
- Alpha vs. Buy & Hold
- Maximum Drawdown
- Sharpe Ratio
- Number of Trades
- Win Rate
- Average Trade Duration

## Extending the System

To add custom features or modify the trading strategy:

1. Add new indicators in `feature_engineering.py`
2. Modify the model training in `model_manager.py`
3. Adjust the trading logic in `backtest.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# trading_system_project
