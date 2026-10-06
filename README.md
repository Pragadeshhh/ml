# Economic Data Analysis: Predicting Unemployment Rate Using Economic Indicators

A beginner-friendly Machine Learning regression project built with Python, Scikit-Learn, and Streamlit. This project investigates the empirical relationship between fundamental macroeconomic indicators and the national unemployment rate across countries using real-world data from the World Bank.

---

## 1. Problem Statement
Unemployment is one of the most critical indicators of macroeconomic health, social welfare, and economic stability. Understanding how factors like economic expansion, price inflation, real borrowing costs, and national income per capita relate to national unemployment rates is a central question in macroeconomics. 

This project applies basic Machine Learning regression workflows to evaluate how accurately these core indicators predict the unemployment rate across world economies, while testing classical concepts such as **Okun's Law** (growth vs. unemployment) and the **Phillips Curve** (inflation vs. unemployment).

---

## 2. Objective
1. Explore and visualize real historical macroeconomic indicators from 1995 to 2023.
2. Build an end-to-end, reproducible data cleaning and preprocessing pipeline.
3. Train, tune, and evaluate multiple regression models (Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, and Gradient Boosting).
4. Evaluate models using standard regression metrics: **MAE**, **MSE**, **RMSE**, and **R²**.
5. Interpret model behavior through standardized coefficients and tree-based feature importances.
6. Deploy a clean, interactive Streamlit web application with a peach-and-cream aesthetic.

---

## 3. Real-World Dataset
* **Dataset Name:** World Bank World Development Indicators (WDI)
* **Source:** The World Bank Group (DataBank)
* **Official URL:** [https://databank.worldbank.org/source/world-development-indicators](https://databank.worldbank.org/source/world-development-indicators)
* **License:** Creative Commons Attribution 4.0 International (CC-BY 4.0) — Public Open Data
* **Time Period:** 1995 – 2023 (29 years)
* **Observations:** 7,540 country-year rows across 260 countries and territories
* **Local Storage:** `data/economic_data.csv`

### Column Descriptions
| Column Name | Type | Description | World Bank Series Code |
| :--- | :---: | :--- | :--- |
| `Country` | Categorical | Country or territory name | — |
| `Country_Code` | Categorical | 3-letter ISO country code (`USA`, `DEU`, `IND`, etc.) | — |
| `Year` | Integer | Calendar observation year (1995–2023) | — |
| `GDP_Growth` | Float | Annual percentage growth rate of real GDP (%) | `NY.GDP.MKTP.KD.ZG` |
| `GDP_Per_Capita` | Float | Gross domestic product per capita in current US Dollars ($) | `NY.GDP.PCAP.CD` |
| `Inflation_Rate` | Float | Annual inflation measured by Consumer Price Index (%) | `FP.CPI.TOTL.ZG` |
| `Population` | Integer | Total national population count | `SP.POP.TOTL` |
| `Real_Interest_Rate` | Float | Lending interest rate adjusted for GDP deflator inflation (%) | `FR.INR.RINR` |
| **`Unemployment_Rate`** | Float | **Target Variable:** Total unemployment as % of labor force (ILO) | `SL.UEM.TOTL.ZS` |

> **Missing Data Handling:** `Real_Interest_Rate` has missing observations for emerging nations without formal reporting. Rather than generating synthetic or imputed numbers, rows with missing modeling features were dropped, leaving **3,127 authentic country-year observations** across 137 countries.

---

## 4. Technologies Used
* **Programming Language:** Python 3.10+
* **Data Processing & Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning & Modeling:** Scikit-Learn (`linear_model`, `tree`, `ensemble`, `model_selection`, `metrics`)
* **Model Serialization:** Joblib
* **Interactive Web Application:** Streamlit
* **Interactive Notebook:** Jupyter Notebook (`analysis.ipynb`)

---

## 5. Machine Learning Methodology

```
Real World Bank Dataset (data/economic_data.csv)
                     │
                     ▼
       Data Cleaning & Deduplication
                     │
                     ▼
   Feature Engineering (Log GDP Per Capita)
                     │
                     ▼
  Train/Test Split (80% Train, 20% Held-Out Test)
                     │
                     ▼
  Feature Scaling (StandardScaler on Train Only)
                     │
                     ▼
  Train & Tune 6 Regression Algorithms
  (Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting)
                     │
                     ▼
  Evaluation on Unseen Test Data (MAE, MSE, RMSE, R²)
                     │
                     ▼
  Serialization (models/final_model.pkl, scaler.pkl)
                     │
                     ▼
  Streamlit Web Application (Inference & Visualization)
```

### Models Evaluated
1. **Linear Regression (Baseline):** Highly interpretable parametric model providing standardized coefficients.
2. **Ridge Regression (L2 Regularization):** Penalizes large weights to mitigate multicollinearity.
3. **Lasso Regression (L1 Regularization):** Performs automatic feature selection through sparse weights.
4. **Decision Tree Regressor:** Single tree capturing non-linear feature splits.
5. **Random Forest Regressor (Ensemble):** Bagging ensemble of randomized decision trees.
6. **Gradient Boosting Regressor (Ensemble):** Sequential boosting optimizing pseudo-residuals.

---

## 6. Actual Experimental Results

All metrics below are **actual values calculated directly on the held-out test set (626 samples)** from `data/economic_data.csv`:

| Model | Model Type | MAE (% pts) | MSE | RMSE (% pts) | R² Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Tuned)** | **Tuned** | **4.2029** | **31.7564** | **5.6353** | **0.2020** |
| Random Forest (Baseline) | Baseline | 4.2039 | 31.8140 | 5.6404 | 0.2006 |
| Gradient Boosting (Tuned) | Tuned | 4.3255 | 32.8719 | 5.7334 | 0.1740 |
| Gradient Boosting (Baseline) | Baseline | 4.3222 | 33.0692 | 5.7506 | 0.1690 |
| Decision Tree | Baseline | 4.4031 | 34.0472 | 5.8350 | 0.1444 |
| Linear Regression | Baseline | 4.7961 | 38.8037 | 6.2293 | 0.0249 |
| Ridge Regression | Baseline | 4.7961 | 38.8040 | 6.2293 | 0.0249 |
| Ridge (Tuned: alpha=50.0) | Tuned | 4.7957 | 38.8155 | 6.2302 | 0.0246 |
| Lasso Regression | Baseline | 4.7882 | 38.8624 | 6.2340 | 0.0234 |

### Final Selected Model
* **Model:** **Random Forest Regressor (Tuned: `n_estimators=150`, `max_depth=10`, `random_state=42`)**
* **Why Selected:** Achieves the highest test R² (0.2020) and lowest prediction error (MAE: 4.20%, RMSE: 5.64%). Tree ensembles effectively partition development tiers and capture non-linear thresholds that linear lines cannot fit.

---

## 7. Model Interpretability & Economic Insights

### Random Forest Feature Importance (MDI)
| Economic Feature | Importance Score | Share |
| :--- | :---: | :---: |
| **GDP Per Capita** | **0.4482** | **44.8%** |
| **Inflation Rate** | **0.2011** | **20.1%** |
| **Real Interest Rate** | **0.1908** | **19.1%** |
| **GDP Growth** | **0.1599** | **16.0%** |

### Linear Regression Standardized Coefficients
* **Intercept:** `7.6477%` (baseline global average unemployment)
* **GDP Growth:** `-0.5296` — Higher GDP growth is associated with lower unemployment (empirically supporting **Okun's Law**).
* **GDP Per Capita:** `-1.0146` — Wealthier economies with capital accumulation average lower structural unemployment.
* **Inflation Rate:** `+0.2104` — Across multi-country annual panels, high inflation often correlates with macroeconomic distress.
* **Real Interest Rate:** `-0.1235` — Moderate lending rates correlate with developed credit markets.

---

## 8. Project Structure

```
economic-data-analysis/
│
├── data/
│   └── economic_data.csv               # Raw World Bank dataset (7,540 rows)
│
├── notebooks/
│   └── analysis.ipynb                  # Fully executed exploratory analysis notebook
│
├── src/
│   ├── data_preprocessing.py           # Cleaning, feature engineering, split & scaling
│   ├── train.py                        # Model training, hyperparameter tuning & saving
│   ├── evaluate.py                     # Visualizations & metrics export
│   └── predict.py                      # Standalone prediction module & CLI
│
├── models/
│   ├── final_model.pkl                 # Trained Random Forest model
│   ├── linear_model.pkl                # Trained Linear Regression model
│   └── scaler.pkl                      # Fitted StandardScaler
│
├── outputs/
│   ├── figures/                        # Generated evaluation charts (R2, RMSE, EDA, Heatmap)
│   └── metrics/                        # Exported model comparison CSV & JSON summaries
│
├── app/
│   └── app.py                          # 5-Page Streamlit Web Application (Peach Theme)
│
├── app.py                              # Root entry point wrapper
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Git ignore rules
└── README.md                           # Complete project documentation
```

---

## 9. How to Run the Project

### 1. Clone & Install Dependencies
```bash
git clone <repository-url>
cd economic-data-analysis
pip install -r requirements.txt
```

### 2. Run Data Preprocessing & Training
```bash
python src/train.py
```
*Trains all 6 models, performs GridSearchCV hyperparameter tuning, prints comparison table, and saves serialized models into `models/`.*

### 3. Generate Evaluation Figures
```bash
python src/evaluate.py
```
*Generates all plots in `outputs/figures/` (Residuals, Actual vs. Predicted, Feature Importance, Correlation Heatmap).*

### 4. Test Single Predictions from Command Line
```bash
python src/predict.py --gdp_growth 3.5 --inflation 2.8 --interest 2.0 --gdp_per_capita 32000
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
*Opens the web application in your browser at `http://localhost:8501` featuring the Peach + Cream design theme.*

### 6. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## 10. Methodological Limitations & Notice
1. **Correlation vs. Causation:** This machine learning system identifies empirical patterns and associations present in historical data. The reported feature importances and coefficients must **not** be interpreted as demonstrating macroeconomic causality.
2. **Omitted Variable Bias:** National unemployment is significantly influenced by domestic labor regulations, minimum wage policies, tax structures, and demographics (age distribution, education), which are not present in four aggregate macroeconomic numbers.
3. **Data Availability:** Developing nations occasionally exhibit reporting gaps in lending rates (`Real_Interest_Rate`), which reduces the clean sample size to 3,127 observations.
4. **Academic Demonstration:** This project is intended as an educational demonstration of the machine learning regression lifecycle and should not be used as a commercial financial forecasting tool.

---

## 11. Author
* **Project:** Economic Data Analysis: Predicting Unemployment Rate Using Economic Indicators
* **Purpose:** Beginner-friendly College Machine Learning Project
