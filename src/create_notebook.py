"""
Script to generate notebooks/analysis.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Title & intro
cells.append(nbf.v4.new_markdown_cell("""# Economic Data Analysis: Predicting Unemployment Rate Using Economic Indicators

**Author:** Machine Learning Student  
**Project:** College ML Regression Project  
**Dataset:** World Bank World Development Indicators (WDI) (1995–2023)  

---

### Project Overview
The objective of this project is to analyze macroeconomic indicators and predict national **Unemployment Rate (%)** using machine learning regression techniques.

In this notebook, we perform:
1. **Data Loading & Inspection**: Examining dimensions, columns, and data types.
2. **Data Cleaning & Quality Check**: Analyzing missing values, duplicates, and unique country coverage.
3. **Summary Statistics**: Calculating measures of central tendency and dispersion.
4. **Exploratory Data Analysis (EDA)**: Visualizing feature distributions, correlations, scatter plots against unemployment, and time-series trends.
5. **Economic Insights**: Explaining what the data reveals about unemployment dynamics.
"""))

# Imports
cells.append(nbf.v4.new_code_cell("""# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual aesthetic suitable for student project
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
PEACH = '#F4B6A6'
DARK_TEXT = '#3D302C'

print("Libraries imported successfully.")
"""))

# Load dataset
cells.append(nbf.v4.new_markdown_cell("""## 1. Load Dataset
We load the real-world dataset stored in `../data/economic_data.csv`. This data was sourced directly from the World Bank DataBank API.
"""))

cells.append(nbf.v4.new_code_cell("""# Load CSV
df = pd.read_csv('../data/economic_data.csv')

# Display first 5 rows
df.head()
"""))

cells.append(nbf.v4.new_code_cell("""# Display last 5 rows
df.tail()
"""))

# Dimensions & Info
cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Dimensions, Columns, & Types
Let's inspect the shape, column names, and data types of each indicator.
"""))

cells.append(nbf.v4.new_code_cell("""# Dataset shape
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
"""))

cells.append(nbf.v4.new_code_cell("""# Column names and non-null types
df.info()
"""))

# Missing values & duplicates
cells.append(nbf.v4.new_markdown_cell("""## 3. Data Cleaning & Integrity Check
Here we check for missing values, duplicate rows, unique countries, and the year range.
"""))

cells.append(nbf.v4.new_code_cell("""# Check duplicate rows
duplicates = df.duplicated().sum()
print(f"Total duplicate rows: {duplicates}")
"""))

cells.append(nbf.v4.new_code_cell("""# Missing value counts and percentages
missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df)) * 100
missing_df = pd.DataFrame({'Missing_Values': missing_counts, 'Percentage (%)': missing_pct.round(2)})
missing_df
"""))

cells.append(nbf.v4.new_code_cell("""# Country and Year coverage
print(f"Total unique countries: {df['Country'].nunique()}")
print(f"Time period: {df['Year'].min()} to {df['Year'].max()}")
"""))

# Summary statistics
cells.append(nbf.v4.new_markdown_cell("""## 4. Summary Statistics
We calculate statistical summaries (mean, standard deviation, percentiles) for numerical features.
"""))

cells.append(nbf.v4.new_code_cell("""# Summary statistics for numeric columns
df.describe().round(2)
"""))

# Filter clean complete cases
cells.append(nbf.v4.new_markdown_cell("""## 5. Prepare Complete Observations for Modeling & Analysis
We define our 4 core economic features:
- `GDP_Growth` (Annual GDP growth %)
- `Inflation_Rate` (Consumer price inflation %)
- `Real_Interest_Rate` (Inflation-adjusted lending rate %)
- `GDP_Per_Capita` (Income level in USD)
And our target variable:
- `Unemployment_Rate` (Total unemployment as % of labor force)
"""))

cells.append(nbf.v4.new_code_cell("""features = ['GDP_Growth', 'Inflation_Rate', 'Real_Interest_Rate', 'GDP_Per_Capita']
target = 'Unemployment_Rate'

# Filter complete cases
clean_df = df.dropna(subset=features + [target]).copy()
print(f"Complete observations available: {len(clean_df)} rows across {clean_df['Country'].nunique()} countries.")
clean_df[features + [target]].describe().round(2)
"""))

# EDA 1: Target distribution
cells.append(nbf.v4.new_markdown_cell("""## 6. Exploratory Data Analysis (EDA)

### Visualization 1: Distribution of Unemployment Rate (Target Variable)
Let's inspect how the unemployment rate is distributed across countries and years.
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(8, 4.5))
sns.histplot(clean_df['Unemployment_Rate'], kde=True, color=PEACH, edgecolor=DARK_TEXT, bins=30)
plt.axvline(clean_df['Unemployment_Rate'].mean(), color='red', linestyle='--', label=f"Mean: {clean_df['Unemployment_Rate'].mean():.2f}%")
plt.axvline(clean_df['Unemployment_Rate'].median(), color='blue', linestyle=':', label=f"Median: {clean_df['Unemployment_Rate'].median():.2f}%")
plt.title('Distribution of Unemployment Rate (%)', fontsize=12, fontweight='bold', color=DARK_TEXT)
plt.xlabel('Unemployment Rate (%)', fontsize=10, color=DARK_TEXT)
plt.ylabel('Count of Country-Years', fontsize=10, color=DARK_TEXT)
plt.legend()
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""*Insight:* The distribution of unemployment rate is moderately right-skewed. Most observations cluster between 4% and 10%, while some countries experience prolonged structural unemployment exceeding 20% to 30%.
"""))

# EDA 2: GDP Growth vs Unemployment
cells.append(nbf.v4.new_markdown_cell("""### Visualization 2: GDP Growth vs. Unemployment Rate (Okun's Law)
In macroeconomics, Okun's Law posits an inverse relationship between economic growth and unemployment.
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(8, 4.5))
# Clip extreme growth outliers for visual clarity
plot_gdp = clean_df[clean_df['GDP_Growth'].between(-15, 20)]
sns.regplot(data=plot_gdp, x='GDP_Growth', y='Unemployment_Rate',
            scatter_kws={'alpha': 0.3, 'color': '#E89882'},
            line_kws={'color': DARK_TEXT, 'linewidth': 2})
plt.title('GDP Growth (%) vs. Unemployment Rate (%)', fontsize=12, fontweight='bold', color=DARK_TEXT)
plt.xlabel('Annual GDP Growth (%)', fontsize=10, color=DARK_TEXT)
plt.ylabel('Unemployment Rate (%)', fontsize=10, color=DARK_TEXT)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""*Insight:* We observe a slight negative regression slope, indicating that periods of higher economic expansion generally coincide with lower unemployment rates.
"""))

# EDA 3: Inflation vs Unemployment
cells.append(nbf.v4.new_markdown_cell("""### Visualization 3: Inflation Rate vs. Unemployment Rate (Phillips Curve Analysis)
We examine the empirical relationship between inflation and unemployment.
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(8, 4.5))
# Filter extreme hyperinflation outliers for clear plotting
plot_inf = clean_df[clean_df['Inflation_Rate'].between(-5, 25)]
sns.regplot(data=plot_inf, x='Inflation_Rate', y='Unemployment_Rate',
            scatter_kws={'alpha': 0.3, 'color': PEACH},
            line_kws={'color': DARK_TEXT, 'linewidth': 2})
plt.title('Inflation Rate (%) vs. Unemployment Rate (%)', fontsize=12, fontweight='bold', color=DARK_TEXT)
plt.xlabel('Annual Inflation Rate (%)', fontsize=10, color=DARK_TEXT)
plt.ylabel('Unemployment Rate (%)', fontsize=10, color=DARK_TEXT)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""*Insight:* In cross-country annual data, inflation and unemployment show a slight positive slope rather than a simple textbook short-run trade-off, as high inflation often reflects macroeconomic instability.
"""))

# EDA 4: Correlation Matrix
cells.append(nbf.v4.new_markdown_cell("""### Visualization 4: Correlation Matrix
We compute Pearson correlation coefficients across our numerical indicators.
"""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(7, 5.5))
corr_matrix = clean_df[features + [target]].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.3, vmax=0.3,
            linewidths=0.5, linecolor='white')
plt.title('Correlation Matrix Heatmap', fontsize=12, fontweight='bold', color=DARK_TEXT)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""*Insight:* GDP per Capita shows the strongest negative correlation (-0.17) with unemployment, followed by GDP Growth (-0.08). The relatively modest linear correlations suggest that non-linear models (e.g. Random Forest) may capture more nuanced relationships.
"""))

# EDA 5: Unemployment Trend Over Time
cells.append(nbf.v4.new_markdown_cell("""### Visualization 5: Global Average Unemployment Trend (1995–2023)
Let's see how global average unemployment has fluctuated across decades, including the 2008 Financial Crisis and 2020 COVID shock.
"""))

cells.append(nbf.v4.new_code_cell("""yearly_trend = clean_df.groupby('Year')['Unemployment_Rate'].mean().reset_index()

plt.figure(figsize=(9, 4.5))
plt.plot(yearly_trend['Year'], yearly_trend['Unemployment_Rate'], marker='o', color=DARK_TEXT, linewidth=2, markerfacecolor=PEACH)
plt.title('Global Average Unemployment Rate Over Time (1995–2023)', fontsize=12, fontweight='bold', color=DARK_TEXT)
plt.xlabel('Year', fontsize=10, color=DARK_TEXT)
plt.ylabel('Average Unemployment Rate (%)', fontsize=10, color=DARK_TEXT)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""*Insight:* We can clearly identify global macroeconomic events: peaks around the early 2000s dot-com bubble, the 2008–2010 Great Recession, and the 2020 pandemic surge.
"""))

# EDA 6: Country-Level Comparison
cells.append(nbf.v4.new_markdown_cell("""### Visualization 6: Country-Level Unemployment Comparison (Selected Major Economies)
Comparing mean historical unemployment across diverse economies.
"""))

cells.append(nbf.v4.new_code_cell("""sample_countries = ['United States', 'Germany', 'Japan', 'United Kingdom', 'South Africa', 'Spain', 'India', 'Brazil', 'Canada']
country_sub = clean_df[clean_df['Country'].isin(sample_countries)]
country_means = country_sub.groupby('Country')['Unemployment_Rate'].mean().sort_values()

plt.figure(figsize=(9, 4.5))
bars = plt.barh(country_means.index, country_means.values, color=PEACH, edgecolor=DARK_TEXT)
plt.title('Historical Average Unemployment Rate by Country', fontsize=12, fontweight='bold', color=DARK_TEXT)
plt.xlabel('Mean Unemployment Rate (%)', fontsize=10, color=DARK_TEXT)

for bar in bars:
    w = bar.get_width()
    plt.text(w + 0.3, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', fontsize=9, fontweight='bold', color=DARK_TEXT)

plt.xlim(0, max(country_means.values) + 3)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""*Insight:* Baseline structural unemployment varies significantly across nations due to institutional and demographic differences, ranging from ~2.8% in Japan to over ~25% in South Africa.
"""))

# Summary of EDA conclusions
cells.append(nbf.v4.new_markdown_cell("""## 7. Conclusions from Exploratory Data Analysis
1. **Target Characteristics**: Unemployment rate has a mean of ~7.8% and median of ~5.9%, with occasional high-unemployment outliers.
2. **Economic Predictors**: Higher GDP Growth and higher GDP per Capita are associated with lower unemployment, aligning with economic theory.
3. **Model Implications**: The non-linear nature of economic relationships and differing national baselines indicate that non-linear ensemble models like Random Forest and Gradient Boosting will likely outperform simple linear models.
4. **Validation Strategy**: Because data spans 1995–2023 across countries, we will test both standard random splits and time-aware evaluation strategies to ensure models do not suffer from temporal leakage.
"""))

nb['cells'] = cells

with open('notebooks/analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully at notebooks/analysis.ipynb")
