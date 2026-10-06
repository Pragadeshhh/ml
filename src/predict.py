"""
Prediction Module
Economic Data Analysis: Predicting Unemployment Rate Using Economic Indicators

This module provides:
1. predict_unemployment(): Takes user economic indicators and returns the predicted unemployment rate.
2. Input validation for economic indicator ranges.
3. Command-line interface for running one-off predictions.
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np

from data_preprocessing import CORE_FEATURES


DEFAULT_MODEL_PATH = 'models/final_model.pkl'
DEFAULT_LINEAR_PATH = 'models/linear_model.pkl'
DEFAULT_SCALER_PATH = 'models/scaler.pkl'


def load_prediction_artifacts(model_type='final'):
    """
    Loads saved model and scaler.
    """
    model_path = DEFAULT_MODEL_PATH if model_type == 'final' else DEFAULT_LINEAR_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")
        
    model = joblib.load(model_path)
    scaler = None
    if model_type == 'linear':
        if not os.path.exists(DEFAULT_SCALER_PATH):
            raise FileNotFoundError(f"Scaler file not found at {DEFAULT_SCALER_PATH}.")
        scaler = joblib.load(DEFAULT_SCALER_PATH)
        
    return model, scaler


def validate_inputs(gdp_growth, inflation_rate, real_interest_rate, gdp_per_capita):
    """
    Validates user input values against realistic macroeconomic bounds.
    Returns cleaned float values or raises ValueError.
    """
    try:
        gdp_growth = float(gdp_growth)
        inflation_rate = float(inflation_rate)
        real_interest_rate = float(real_interest_rate)
        gdp_per_capita = float(gdp_per_capita)
    except (ValueError, TypeError):
        raise ValueError("All economic indicators must be valid numeric values.")

    if gdp_per_capita < 0:
        raise ValueError("GDP per Capita cannot be negative.")
        
    if abs(gdp_growth) > 100:
        raise ValueError("GDP Growth must be a realistic annual rate between -100% and 100%.")

    return gdp_growth, inflation_rate, real_interest_rate, gdp_per_capita


def predict_unemployment(gdp_growth, inflation_rate, real_interest_rate, gdp_per_capita, model_type='final'):
    """
    Predicts Unemployment Rate (%) given the 4 key economic indicators.
    
    Parameters:
        gdp_growth (float): Annual GDP Growth rate in %
        inflation_rate (float): Annual Inflation rate in %
        real_interest_rate (float): Real Interest Rate in %
        gdp_per_capita (float): GDP per capita in current USD
        model_type (str): 'final' (Random Forest) or 'linear' (Linear Regression)
        
    Returns:
        dict: {
            'predicted_unemployment_rate': float,
            'formatted_prediction': str,
            'model_used': str,
            'input_data': dict
        }
    """
    # 1. Validate inputs
    g_val, inf_val, int_val, gdp_cap = validate_inputs(
        gdp_growth, inflation_rate, real_interest_rate, gdp_per_capita
    )
    
    # 2. Format into DataFrame
    input_df = pd.DataFrame([{
        'GDP_Growth': g_val,
        'Inflation_Rate': inf_val,
        'Real_Interest_Rate': int_val,
        'GDP_Per_Capita': gdp_cap
    }])[CORE_FEATURES]
    
    # 3. Load model
    model, scaler = load_prediction_artifacts(model_type=model_type)
    
    # 4. Predict
    if model_type == 'linear' and scaler is not None:
        scaled_input = pd.DataFrame(scaler.transform(input_df), columns=CORE_FEATURES)
        pred_val = model.predict(scaled_input)[0]
    else:
        pred_val = model.predict(input_df)[0]
        
    # Economic constraint: Unemployment cannot be strictly below 0
    pred_val_clipped = max(0.1, round(float(pred_val), 2))
    
    return {
        'predicted_unemployment_rate': pred_val_clipped,
        'formatted_prediction': f"{pred_val_clipped:.2f}%",
        'model_used': 'Random Forest (Tuned)' if model_type == 'final' else 'Linear Regression',
        'input_data': {
            'GDP Growth': f"{g_val:.2f}%",
            'Inflation Rate': f"{inf_val:.2f}%",
            'Real Interest Rate': f"{int_val:.2f}%",
            'GDP Per Capita': f"${gdp_cap:,.2f}"
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Unemployment Rate from Economic Indicators")
    parser.add_argument('--gdp_growth', type=float, default=2.5, help="GDP Growth rate (%)")
    parser.add_argument('--inflation', type=float, default=3.2, help="Inflation rate (%)")
    parser.add_argument('--interest', type=float, default=2.0, help="Real interest rate (%)")
    parser.add_argument('--gdp_per_capita', type=float, default=35000.0, help="GDP per Capita (USD)")
    parser.add_argument('--model', type=str, default='final', choices=['final', 'linear'], help="Model type to use")
    
    args = parser.parse_args()
    result = predict_unemployment(args.gdp_growth, args.inflation, args.interest, args.gdp_per_capita, model_type=args.model)
    
    print("\n" + "=" * 50)
    print("UNEMPLOYMENT RATE PREDICTION RESULT")
    print("=" * 50)
    print(f"Model Used: {result['model_used']}")
    print(f"Inputs: {result['input_data']}")
    print(f"Predicted Unemployment Rate: {result['formatted_prediction']}")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()
