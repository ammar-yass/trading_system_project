import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

def prepare_features_for_model(ticker, df, tickers, features_dict):
    """
    Prepare features for the model
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol
    df : pandas.DataFrame
        DataFrame containing the features
    tickers : list
        List of all ticker symbols
    features_dict : dict
        Dictionary of features for all tickers
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with prepared features for modeling
    """
    # Define the columns to use as features
    feature_columns = [
        'returns', 'log_returns', 'volume_change', 'gap_up',
        'atr', 'atr_pct', 
        'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100',
        'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100',
        'ma_cross_5_20', 'ma_cross_10_50', 'ma_cross_20_100',
        'close_over_sma_50', 'close_over_sma_200',
        'rsi_14', 'bb_width', 'bb_pct',
        'macd', 'macd_signal', 'macd_diff',
        'volume_sma_ratio', 'volume_spike',
        'momentum_10', 'momentum_20',
        'day_of_week', 'day_of_month', 'month', 'quarter'
    ]
    
    # For other assets, add their returns as features to capture correlations
    for other_ticker in tickers:
        if other_ticker != ticker and other_ticker in features_dict:
            other_df = features_dict[other_ticker]
            # Make sure the indices align
            common_idx = df.index.intersection(other_df.index)
            if len(common_idx) > 0:
                # Add other asset returns
                df.loc[common_idx, f'{other_ticker}_returns'] = other_df.loc[common_idx, 'returns']
                feature_columns.append(f'{other_ticker}_returns')
    
    # Select only the features we want to use
    X = df[feature_columns]
    
    # Convert boolean columns to int
    for col in X.select_dtypes(include=[bool]).columns:
        X[col] = X[col].astype(int)
        
    return X

def train_model(ticker, train_data, tickers, features_dict, target_column='target_next_day'):
    """
    Train a model for a single ticker
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol
    train_data : pandas.DataFrame
        Training data
    tickers : list
        List of all ticker symbols
    features_dict : dict
        Dictionary of features for all tickers
    target_column : str
        Target column name
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Trained model
    """
    X_train = prepare_features_for_model(ticker, train_data, tickers, features_dict)
    y_train = train_data[target_column]
    
    # Create a pipeline with scaling and model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate training metrics
    y_pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    
    print(f"{ticker} model training metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Get feature importances
    if hasattr(model[-1], 'feature_importances_'):
        importances = model[-1].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("Top 10 important features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model

def predict(ticker, test_data, model, tickers, features_dict, target_column='target_next_day'):
    """
    Make predictions for a single ticker
    
    Parameters:
    -----------
    ticker : str
        Ticker symbol
    test_data : pandas.DataFrame
        Test data
    model : sklearn.pipeline.Pipeline
        Trained model
    tickers : list
        List of all ticker symbols
    features_dict : dict
        Dictionary of features for all tickers
    target_column : str
        Target column name
        
    Returns:
    --------
    pandas.DataFrame
        Test data with predictions
    """
    X_test = prepare_features_for_model(ticker, test_data, tickers, features_dict)
    
    # Get binary predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class
    
    # Add predictions to the test data
    test_data_with_preds = test_data.copy()
    test_data_with_preds['predicted'] = y_pred
    test_data_with_preds['predicted_prob'] = y_prob
    
    return test_data_with_preds
