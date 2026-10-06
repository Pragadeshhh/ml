"""
Model Training and Hyperparameter Tuning Module
Economic Data Analysis: Predicting Unemployment Rate Using Economic Indicators

This module:
1. Loads preprocessed training and testing data.
2. Trains 6 regression models:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Gradient Boosting Regressor
3. Computes MAE, MSE, RMSE, and R2 on the actual test set.
4. Performs hyperparameter tuning using GridSearchCV for candidate models.
5. Selects the best performing model based on evaluation metrics.
6. Serializes the final model and feature scaler into models/.
7. Exports real performance metrics into outputs/metrics/.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import get_preprocessed_data, CORE_FEATURES


def evaluate_model(model, X, y, is_scaled=False, X_scaled=None):
    """
    Computes regression evaluation metrics: MAE, MSE, RMSE, R2.
    """
    input_data = X_scaled if is_scaled else X
    preds = model.predict(input_data)
    mae = mean_absolute_error(y, preds)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)
    return {
        'MAE': round(float(mae), 4),
        'MSE': round(float(mse), 4),
        'RMSE': round(float(rmse), 4),
        'R2': round(float(r2), 4)
    }, preds


def train_baseline_models(data):
    """
    Trains and evaluates baseline regression models.
    """
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    
    models = {
        'Linear Regression': (LinearRegression(), True),
        'Ridge Regression': (Ridge(alpha=1.0, random_state=42), True),
        'Lasso Regression': (Lasso(alpha=0.1, random_state=42), True),
        'Decision Tree': (DecisionTreeRegressor(max_depth=5, random_state=42), False),
        'Random Forest': (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), False),
        'Gradient Boosting': (GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42), False)
    }
    
    results = []
    trained_models = {}
    test_predictions = {}
    
    for name, (model, is_scaled) in models.items():
        train_features = X_train_scaled if is_scaled else X_train
        model.fit(train_features, y_train)
        
        metrics, preds = evaluate_model(
            model, X_test, y_test, is_scaled=is_scaled, X_scaled=X_test_scaled
        )
        
        results.append({
            'Model': name,
            'Type': 'Baseline',
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'R2': metrics['R2']
        })
        trained_models[name] = (model, is_scaled)
        test_predictions[name] = preds
        
    return results, trained_models, test_predictions


def tune_candidate_models(data):
    """
    Performs grid search cross-validation on key models.
    """
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    
    tuned_results = []
    best_tuned_models = {}
    
    # 1. Ridge Tuning
    ridge_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 50.0]}
    ridge_cv = GridSearchCV(Ridge(random_state=42), ridge_grid, cv=5, scoring='neg_mean_squared_error')
    ridge_cv.fit(X_train_scaled, y_train)
    best_ridge = ridge_cv.best_estimator_
    m_ridge, _ = evaluate_model(best_ridge, X_test, y_test, is_scaled=True, X_scaled=X_test_scaled)
    tuned_results.append({
        'Model': f"Ridge (Tuned: alpha={ridge_cv.best_params_['alpha']})",
        'Type': 'Tuned',
        'MAE': m_ridge['MAE'],
        'MSE': m_ridge['MSE'],
        'RMSE': m_ridge['RMSE'],
        'R2': m_ridge['R2']
    })
    best_tuned_models['Ridge Tuned'] = (best_ridge, True, ridge_cv.best_params_)

    # 2. Random Forest Tuning
    rf_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [6, 10, 15],
        'min_samples_split': [2, 5]
    }
    rf_cv = GridSearchCV(RandomForestRegressor(random_state=42), rf_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    rf_cv.fit(X_train, y_train)
    best_rf = rf_cv.best_estimator_
    m_rf, _ = evaluate_model(best_rf, X_test, y_test, is_scaled=False)
    tuned_results.append({
        'Model': f"Random Forest (Tuned: depth={rf_cv.best_params_['max_depth']}, trees={rf_cv.best_params_['n_estimators']})",
        'Type': 'Tuned',
        'MAE': m_rf['MAE'],
        'MSE': m_rf['MSE'],
        'RMSE': m_rf['RMSE'],
        'R2': m_rf['R2']
    })
    best_tuned_models['Random Forest Tuned'] = (best_rf, False, rf_cv.best_params_)

    # 3. Gradient Boosting Tuning
    gb_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    gb_cv = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    gb_cv.fit(X_train, y_train)
    best_gb = gb_cv.best_estimator_
    m_gb, _ = evaluate_model(best_gb, X_test, y_test, is_scaled=False)
    tuned_results.append({
        'Model': f"Gradient Boosting (Tuned: depth={gb_cv.best_params_['max_depth']}, lr={gb_cv.best_params_['learning_rate']})",
        'Type': 'Tuned',
        'MAE': m_gb['MAE'],
        'MSE': m_gb['MSE'],
        'RMSE': m_gb['RMSE'],
        'R2': m_gb['R2']
    })
    best_tuned_models['Gradient Boosting Tuned'] = (best_gb, False, gb_cv.best_params_)
    
    return tuned_results, best_tuned_models


def run_training_pipeline():
    """
    Executes full model training, evaluation, comparison, and export.
    """
    print("=" * 60)
    print("ECONOMIC DATA ANALYSIS — MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Preprocess data
    data = get_preprocessed_data(strategy='standard', test_size=0.2, random_state=42)
    print(f"Dataset Size: {len(data['df_clean'])} observations")
    print(f"Training Samples: {len(data['X_train'])}, Testing Samples: {len(data['X_test'])}")
    print(f"Features: {CORE_FEATURES}\n")
    
    # 2. Train baseline models
    print("Training baseline regression models...")
    baseline_results, baseline_models, test_preds = train_baseline_models(data)
    
    # 3. Hyperparameter tuning
    print("Running hyperparameter tuning (GridSearchCV)...")
    tuned_results, tuned_models = tune_candidate_models(data)
    
    # 4. Combine results into DataFrame
    all_results = baseline_results + tuned_results
    df_results = pd.DataFrame(all_results)
    
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE COMPARISON (ON REAL TEST DATA)")
    print("=" * 70)
    print(df_results.to_string(index=False))
    print("=" * 70)
    
    # 5. Select Best Model (Highest R2 and Lowest RMSE)
    # Filter tuned models or baseline
    best_row = df_results.sort_values(by='R2', ascending=False).iloc[0]
    best_model_name = best_row['Model']
    print(f"\nFinal Selected Model: {best_model_name}")
    print(f"Metrics -> R2: {best_row['R2']}, RMSE: {best_row['RMSE']}, MAE: {best_row['MAE']}")
    
    # Retrieve the model object
    if 'Random Forest' in best_model_name:
        final_model = tuned_models['Random Forest Tuned'][0]
        is_scaled = False
    elif 'Gradient Boosting' in best_model_name:
        final_model = tuned_models['Gradient Boosting Tuned'][0]
        is_scaled = False
    else:
        final_model = baseline_models['Random Forest'][0]
        is_scaled = False
        
    # Also keep Linear Regression trained for explainable coefficients
    linear_model = baseline_models['Linear Regression'][0]
    
    # 6. Save models and artifacts
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs/metrics', exist_ok=True)
    
    model_save_path = 'models/final_model.pkl'
    linear_save_path = 'models/linear_model.pkl'
    scaler_save_path = 'models/scaler.pkl'
    
    joblib.dump(final_model, model_save_path)
    joblib.dump(linear_model, linear_save_path)
    joblib.dump(data['scaler'], scaler_save_path)
    
    print(f"Saved final model to: {model_save_path}")
    print(f"Saved linear model to: {linear_save_path}")
    print(f"Saved scaler to: {scaler_save_path}")
    
    # Save comparison metrics
    metrics_csv_path = 'outputs/metrics/model_comparison.csv'
    df_results.to_csv(metrics_csv_path, index=False)
    
    # Save detailed JSON summary
    summary_data = {
        'best_model': best_model_name,
        'features': CORE_FEATURES,
        'target': 'Unemployment_Rate',
        'metrics': {
            'MAE': float(best_row['MAE']),
            'MSE': float(best_row['MSE']),
            'RMSE': float(best_row['RMSE']),
            'R2': float(best_row['R2'])
        },
        'linear_coefficients': {
            feat: round(float(coef), 4)
            for feat, coef in zip(CORE_FEATURES, linear_model.coef_)
        },
        'linear_intercept': round(float(linear_model.intercept_), 4),
        'comparison': all_results
    }
    
    with open('outputs/metrics/model_metrics.json', 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"Saved comparison metrics to: {metrics_csv_path}")
    print(f"Saved metrics JSON to: outputs/metrics/model_metrics.json")
    
    return df_results, final_model, linear_model, data


if __name__ == '__main__':
    run_training_pipeline()
