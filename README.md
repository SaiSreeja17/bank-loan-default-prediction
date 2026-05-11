# Bank Loan Default Prediction

> Machine learning classification model to predict whether a borrower will default on a loan, built on 255,347 real-world loan records.

---

## Results at a Glance

| Metric | Score |
|--------|-------|
| Recall (Default class) | 68.70% |
| ROC AUC | 0.75 |
| Accuracy | 68.70% |
| Best Model | Logistic Regression |
| Best Parameters | C=0.01, solver=liblinear |
| Dataset Size | 255,347 records |

---

## Overview

Banks and financial institutions face significant losses when borrowers default on loans. This project builds a binary classification model to flag high-risk borrowers before loan approval, using borrower demographics, financial history, and loan characteristics as input features.

**Why Recall?** Missing an actual defaulter (false negative) is far more costly than a false alarm. Recall is the primary metric — it measures what percentage of real defaulters the model successfully catches.

---

## Dataset

**Source:** [Kaggle — Loan Default Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default)

| Feature | Description |
|---------|-------------|
| Age | Borrower age |
| Income | Annual income ($) |
| LoanAmount | Total loan amount ($) |
| CreditScore | Credit score (300–850) |
| MonthsEmployed | Months at current employer |
| NumCreditLines | Number of open credit lines |
| InterestRate | Loan interest rate (%) |
| LoanTerm | Loan term in months |
| DTIRatio | Debt-to-income ratio |
| Education | High School / Bachelor's / Master's / PhD |
| EmploymentType | Full-time / Part-time / Self-employed / Unemployed |
| MaritalStatus | Single / Married / Divorced |
| HasMortgage | Yes / No |
| HasDependents | Yes / No |
| LoanPurpose | Auto / Business / Education / Home / Other |
| HasCoSigner | Yes / No |
| **Default** | **Target — 1 = Defaulted, 0 = No Default** |

- **255,347** total records
- **11.6%** default rate (class imbalance handled with `class_weight='balanced'`)

---

## Project Structure

```
├── Bank_Loan_Default_Prediction.ipynb   # Main notebook
├── Loan_default.csv                     # Dataset (download from Kaggle)
├── loan_predictions.csv                 # Model output — predictions + risk scores
├── feature_importance.csv               # Feature coefficients for Power BI
├── loan_default_model.pkl               # Saved model
├── loan_default_scaler.pkl              # Saved scaler
└── README.md
```

---

## Notebook Walkthrough

| Step | Description |
|------|-------------|
| 1 | Import libraries |
| 2 | Load & inspect data — shape, dtypes, missing values, class balance |
| 3 | EDA — 6 business questions answered visually |
| 4 | Data cleaning & preprocessing — encoding, label encoding |
| 5 | Train/test split (80/20 stratified) + StandardScaler |
| 6 | Model comparison — Logistic Regression vs Random Forest vs XGBoost (cross-validated Recall) |
| 7 | Hyperparameter tuning — GridSearchCV across 10 combinations |
| 8 | Final evaluation — Accuracy, Recall, ROC AUC, Confusion Matrix, ROC Curve |
| 9 | Feature importance — coefficient magnitude analysis |
| 10 | Export predictions and feature importance for Power BI |
| 11 | Save model and scaler with pickle |

---

## Key Findings

- **Interest Rate** is the strongest predictor of default — high rates are assigned to already-risky borrowers, creating a compounding signal
- **DTI Ratio** and **Credit Score** are the next most impactful features
- Borrowers with **no co-signer** default at a meaningfully higher rate
- **Employment type** matters — unemployed and part-time borrowers carry higher default risk
- The model achieves **0.75 AUC**, significantly outperforming random guessing (0.50)

---

## How to Run

1. Clone the repo
2. Download `Loan_default.csv` from [Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default) and place it in the same folder as the notebook
3. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
4. Open and run `Bank_Loan_Default_Prediction.ipynb` top to bottom

---

## Tools & Skills

`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Matplotlib` `Seaborn` `Logistic Regression` `Random Forest` `GridSearchCV` `ROC AUC` `Machine Learning` `Financial Analytics`

---

## Background

Built as a portfolio project demonstrating end-to-end ML workflow in a financial services context — from raw data to a deployable model with exported outputs ready for Power BI reporting.
