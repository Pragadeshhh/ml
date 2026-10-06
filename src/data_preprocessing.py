"""
Data Preprocessing Module
Economic Data Analysis: Predicting Unemployment Rate Using Economic Indicators

This module handles:
1. Loading raw World Bank data from CSV.
2. Cleaning missing values and duplicates.
3. Feature engineering (log transform of GDP per capita, optional lag features).
4. Data splitting (standard random split and time-aware split).
5. Feature scaling using StandardScaler.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Default feature list and target
CORE_FEATURES = [
    'GDP_Growth',
    'Inflation_Rate',
    'Real_Interest_Rate',
    'GDP_Per_Capita'
]

ENGINEERED_FEATURES = [
    'GDP_Growth',
    'Inflation_Rate',
    'Real_Interest_Rate',
    'GDP_Per_Capita',
    'Log_GDP_Per_Capita'
]

TARGET = 'Unemployment_Rate'


def load_data(filepath='data/economic_data.csv'):
    """
    Loads raw CSV dataset from given filepath.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def engineer_features(df):
    """
    Adds methodologically sound engineered features:
    1. Log_GDP_Per_Capita: Logarithmic transformation of GDP per capita
       to compress the heavily right-skewed income distribution.
    2. Sorts data cleanly by Country and Year.
    """
    df = df.copy()
    
    # Ensure proper ordering by Country and Year
    if 'Country' in df.columns and 'Year' in df.columns:
        df = df.sort_values(by=['Country', 'Year']).reset_index(drop=True)
    
    # Log transformation for GDP per Capita (skewed distribution)
    if 'GDP_Per_Capita' in df.columns:
        df['Log_GDP_Per_Capita'] = np.log1p(df['GDP_Per_Capita'].clip(lower=0))
        
    return df


def clean_data(df, feature_cols=None, target_col=TARGET):
    """
    Cleans dataset:
    - Drops duplicate rows.
    - Selects rows where model features and target are non-null.
    - Keeps documented track of dropped observations.
    """
    if feature_cols is None:
        feature_cols = CORE_FEATURES
        
    df_clean = df.copy()
    
    # Remove duplicate rows
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_rows - len(df_clean)
    
    # Drop missing values for required features and target
    required_cols = list(set(feature_cols + [target_col]))
    # Filter to columns that exist
    existing_required = [c for c in required_cols if c in df_clean.columns]
    
    before_dropna = len(df_clean)
    df_clean = df_clean.dropna(subset=existing_required).reset_index(drop=True)
    nulls_removed = before_dropna - len(df_clean)
    
    return df_clean, {
        'initial_rows': initial_rows,
        'duplicates_removed': duplicates_removed,
        'nulls_removed': nulls_removed,
        'final_rows': len(df_clean)
    }


def split_data(df, feature_cols=None, target_col=TARGET, strategy='standard', test_size=0.2, random_state=42, split_year=2018):
    """
    Splits data into train and test sets.
    
    Strategies:
    - 'standard': 80/20 random stratified/shuffle split (standard ML baseline).
    - 'time_aware': Train on past years (Year < split_year), test on future years (Year >= split_year).
                   Prevents temporal data leakage across time series.
    """
    if feature_cols is None:
        feature_cols = CORE_FEATURES
        
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    if strategy == 'time_aware' and 'Year' in df.columns:
        train_mask = df['Year'] < split_year
        test_mask = df['Year'] >= split_year
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Fits StandardScaler on X_train and transforms both X_train and X_test.
    Prevents leakage by fitting only on training data.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


def get_preprocessed_data(filepath='data/economic_data.csv', strategy='standard', test_size=0.2, random_state=42):
    """
    End-to-end preprocessing helper function.
    Returns:
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, df_clean
    """
    df_raw = load_data(filepath)
    df_feat = engineer_features(df_raw)
    df_clean, stats = clean_data(df_feat, feature_cols=CORE_FEATURES)
    
    X_train, X_test, y_train, y_test = split_data(
        df_clean, feature_cols=CORE_FEATURES, target_col=TARGET,
        strategy=strategy, test_size=test_size, random_state=random_state
    )
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'df_clean': df_clean,
        'stats': stats
    }


if __name__ == '__main__':
    print("Testing data preprocessing pipeline...")
    data = get_preprocessed_data()
    print("Clean rows:", len(data['df_clean']))
    print("X_train shape:", data['X_train'].shape)
    print("X_test shape:", data['X_test'].shape)
    print("Preprocessing completed successfully.")
