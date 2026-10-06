"""
Evaluation and Visualization Module
Economic Data Analysis: Predicting Unemployment Rate Using Economic Indicators

This module:
1. Loads trained models and evaluation data.
2. Generates comprehensive evaluation visualizations in outputs/figures/:
   - Model comparison chart (R2, RMSE, MAE)
   - Actual vs. Predicted values plot
   - Residual distribution plot
   - Random Forest Feature Importance chart
   - Linear Regression Coefficients chart
   - Economic correlation heatmap
3. Saves feature importance rankings in outputs/metrics/feature_importance.csv.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import get_preprocessed_data, CORE_FEATURES, TARGET


# Set clean, aesthetic plotting style suitable for a student project
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
PEACH_COLOR = '#F4B6A6'
DARK_TEXT = '#3D302C'
SECONDARY_PEACH = '#E89882'


def plot_model_comparison(comparison_df, output_dir='outputs/figures'):
    """
    Plots bar charts comparing R2 and RMSE across all models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. R2 Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_df = comparison_df.sort_values(by='R2', ascending=True)
    bars = ax.barh(sorted_df['Model'], sorted_df['R2'], color=PEACH_COLOR, edgecolor=DARK_TEXT, linewidth=0.8)
    ax.set_title('Model Comparison — R² Score (Higher is Better)', fontsize=13, fontweight='bold', color=DARK_TEXT, pad=12)
    ax.set_xlabel('R² Score', fontsize=11, color=DARK_TEXT)
    ax.set_xlim(0, max(sorted_df['R2'].max() * 1.25, 0.25))
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f'{width:.3f}',
                va='center', ha='left', fontsize=9, color=DARK_TEXT, fontweight='bold')
        
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'model_comparison_r2.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")

    # 2. RMSE & MAE Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sorted_df))
    width = 0.35
    
    ax.bar(x - width/2, sorted_df['RMSE'], width, label='RMSE', color='#E89882', edgecolor=DARK_TEXT)
    ax.bar(x + width/2, sorted_df['MAE'], width, label='MAE', color='#FCE4DC', edgecolor=DARK_TEXT)
    ax.set_title('Model Comparison — Prediction Errors (Lower is Better)', fontsize=13, fontweight='bold', color=DARK_TEXT, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_df['Model'], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Error (% points)', fontsize=11, color=DARK_TEXT)
    ax.legend()
    
    plt.tight_layout()
    fig_path_err = os.path.join(output_dir, 'model_comparison_errors.png')
    plt.savefig(fig_path_err, dpi=300)
    plt.close()
    print(f"Saved: {fig_path_err}")


def plot_actual_vs_predicted(model, X_test, y_test, output_dir='outputs/figures'):
    """
    Plots Actual vs. Predicted Unemployment Rate scatter plot with 45-degree reference line.
    """
    os.makedirs(output_dir, exist_ok=True)
    preds = model.predict(X_test)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, preds, alpha=0.5, color=SECONDARY_PEACH, edgecolors=DARK_TEXT, linewidths=0.5, s=40)
    
    # 45-degree perfect prediction line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], color=DARK_TEXT, linestyle='--', linewidth=1.5, label='Perfect Prediction (y = x)')
    
    ax.set_title('Actual vs. Predicted Unemployment Rate (%)', fontsize=13, fontweight='bold', color=DARK_TEXT, pad=12)
    ax.set_xlabel('Actual Unemployment Rate (%)', fontsize=11, color=DARK_TEXT)
    ax.set_ylabel('Predicted Unemployment Rate (%)', fontsize=11, color=DARK_TEXT)
    ax.legend()
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'actual_vs_predicted.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")
    
    # Residuals Plot
    residuals = y_test - preds
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(residuals, kde=True, color=PEACH_COLOR, edgecolor=DARK_TEXT, ax=ax)
    ax.axvline(0, color=DARK_TEXT, linestyle='--', linewidth=1.5)
    ax.set_title('Residuals Distribution (Actual - Predicted)', fontsize=13, fontweight='bold', color=DARK_TEXT, pad=12)
    ax.set_xlabel('Residual Error (% points)', fontsize=11, color=DARK_TEXT)
    ax.set_ylabel('Frequency', fontsize=11, color=DARK_TEXT)
    
    plt.tight_layout()
    res_path = os.path.join(output_dir, 'residuals_distribution.png')
    plt.savefig(res_path, dpi=300)
    plt.close()
    print(f"Saved: {res_path}")


def plot_feature_importance(model, linear_model, feature_names, output_dir='outputs/figures'):
    """
    Plots feature importances from Random Forest and normalized coefficients from Linear Regression.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Random Forest importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)
        
        df_imp.to_csv('outputs/metrics/feature_importance.csv', index=False)
        print("Saved: outputs/metrics/feature_importance.csv")
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.barh(df_imp['Feature'], df_imp['Importance'], color=PEACH_COLOR, edgecolor=DARK_TEXT)
        ax.set_title('Random Forest Feature Importance (MDI)', fontsize=13, fontweight='bold', color=DARK_TEXT, pad=12)
        ax.set_xlabel('Importance Score (Sums to 1.0)', fontsize=11, color=DARK_TEXT)
        ax.set_xlim(0, max(df_imp['Importance'].max() * 1.25, 0.5))
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.3f}',
                    va='center', ha='left', fontsize=10, color=DARK_TEXT, fontweight='bold')
            
        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved: {fig_path}")
        
    # Linear Regression coefficients
    if hasattr(linear_model, 'coef_'):
        coefs = linear_model.coef_
        df_coef = pd.DataFrame({
            'Feature': feature_names,
            'Standardized_Coefficient': coefs
        }).sort_values(by='Standardized_Coefficient', ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = [PEACH_COLOR if c >= 0 else '#E89882' for c in df_coef['Standardized_Coefficient']]
        bars = ax.barh(df_coef['Feature'], df_coef['Standardized_Coefficient'], color=colors, edgecolor=DARK_TEXT)
        ax.axvline(0, color=DARK_TEXT, linestyle='-', linewidth=1)
        ax.set_title('Linear Regression Standardized Coefficients', fontsize=13, fontweight='bold', color=DARK_TEXT, pad=12)
        ax.set_xlabel('Standardized Coefficient (Effect on Unemployment Rate)', fontsize=11, color=DARK_TEXT)
        
        for bar in bars:
            width = bar.get_width()
            offset = 0.03 if width >= 0 else -0.12
            ax.text(width + offset, bar.get_y() + bar.get_height() / 2, f'{width:.3f}',
                    va='center', ha='left', fontsize=9, color=DARK_TEXT, fontweight='bold')
            
        plt.tight_layout()
        fig_path_coef = os.path.join(output_dir, 'linear_coefficients.png')
        plt.savefig(fig_path_coef, dpi=300)
        plt.close()
        print(f"Saved: {fig_path_coef}")


def plot_correlation_matrix(df, output_dir='outputs/figures'):
    """
    Plots correlation matrix heatmap for modeling features and target.
    """
    os.makedirs(output_dir, exist_ok=True)
    cols = CORE_FEATURES + [TARGET]
    corr = df[cols].corr()
    
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = sns.diverging_palette(20, 220, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, cbar=True, square=True,
                linewidths=0.5, linecolor='white', ax=ax)
    ax.set_title('Correlation Heatmap: Economic Indicators & Unemployment', fontsize=12, fontweight='bold', color=DARK_TEXT, pad=12)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved: {fig_path}")


def run_evaluation_pipeline():
    """
    Runs full evaluation and saves all figures.
    """
    print("=" * 60)
    print("RUNNING EVALUATION AND VISUALIZATION PIPELINE")
    print("=" * 60)
    
    # Load comparison metrics
    comparison_df = pd.read_csv('outputs/metrics/model_comparison.csv')
    plot_model_comparison(comparison_df)
    
    # Load models
    final_model = joblib.load('models/final_model.pkl')
    linear_model = joblib.load('models/linear_model.pkl')
    
    # Load preprocessed data
    data = get_preprocessed_data()
    
    # Plot actual vs predicted & residuals
    plot_actual_vs_predicted(final_model, data['X_test'], data['y_test'])
    
    # Plot feature importance
    plot_feature_importance(final_model, linear_model, CORE_FEATURES)
    
    # Plot correlation matrix
    plot_correlation_matrix(data['df_clean'])
    
    print("\nAll evaluation figures generated successfully.")


if __name__ == '__main__':
    run_evaluation_pipeline()
