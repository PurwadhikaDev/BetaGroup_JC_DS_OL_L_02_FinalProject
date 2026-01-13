# Final Project - Purwadhika Bootcamp
## E-Commerce Customer Churn Analysis & Prediction

**Created By:** Beta Team JCDSOLL02

## Project Overview

This project focuses on analyzing and predicting customer churn in an e-commerce platform using Machine Learning techniques. Customer churn is one of the main challenges in e-commerce business as it significantly impacts revenue and growth.

## Business Problem

### Context
An e-commerce company wants to improve customer retention and reduce the number of customers who stop using their platform. By predicting which customers are likely to churn, the company can take proactive measures to retain them.

### Problem Statement
Customer churn prediction is crucial for:
- Identifying at-risk customers before they leave
- Implementing targeted retention strategies
- Reducing customer acquisition costs
- Improving overall business profitability

### Metrics Evaluation

The project uses confusion matrix metrics for model evaluation:

- **Type 1 Error (False Positive):** Model predicts customer will churn, but they actually don't
- **Type 2 Error (False Negative):** Model predicts customer won't churn, but they actually do

**Focus:** Minimizing False Negatives to catch as many potential churn cases as possible.

## Dataset

**Source:** [E-Commerce Customer Churn Dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)

### Dataset Characteristics
- **Total Rows:** 5,630 customers
- **Total Columns:** 20 features
- **Missing Values:** 1,856 values (concentrated in numerical features)
- **Data Types:** Integer, Float, and categorical data

### Key Features
- **CustomerID:** Unique identifier (removed before modeling)
- **Demographics:** PreferredLoginDevice, CityTier, PreferredPaymentMode, Gender, HourSpendOnApp, NumberOfDeviceRegistered, SatisfactionScore
- **Behavioral:** Tenure, WarehouseToHome, OrderAmountHikeFromlastYear, CouponUsed, OrderCount, DaySinceLastOrder
- **Target:** Churn (0 = Not Churned, 1 = Churned)

### Churn Distribution
- **Not Churned:** 4,682 customers (83.16%)
- **Churned:** 948 customers (16.84%)

## Methodology

### 1. Data Understanding & Exploration
- Comprehensive exploratory data analysis (EDA)
- Statistical summary and visualization
- Identification of data quality issues

### 2. Data Cleaning & Analysis
- **Missing Value Handling:** Imputation strategies for numerical features
- **Data Type Validation:** Ensuring consistent data types
- **Outlier Detection:** Identifying and handling outliers
- **Categorical Consistency:** Fixing inconsistent category naming

### 3. Feature Engineering
- **Label Encoding:** Converting categorical labels to numerical format
- **Feature Selection:** Using SHAP (SHapley Additive exPlanations) values
- **Normalization:** Scaling numerical features for model compatibility

### 4. Modeling
- Multiple machine learning algorithms tested
- SHAP values for feature importance and interpretation
- Model evaluation using confusion matrix and classification metrics

## Key Insights

### Device Preferences
- **70.98%** of customers use **Mobile Phone** to access the platform
- Device preference may correlate with churn behavior

### Customer Behavior Analysis
- Correlation analysis between features and churn
- Comparative analysis of churned vs non-churned customers
- Identification of key risk factors

### Feature Importance (SHAP)
- SHAP analysis reveals top contributing factors to churn
- Model-agnostic feature importance ranking

## Technologies & Libraries

```python
# Core Data Analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Feature Importance
import shap

# Data Loading
import kagglehub
```

## Notebook Structure

1. **Business Problem Understanding**
   - Context and problem statement
   - Metrics evaluation framework

2. **Load the Dataset**
   - Data import from Kaggle
   - Initial data loading

3. **Data Understanding & Exploration**
   - Dataset overview function
   - Statistical summaries
   - Data quality assessment

4. **Data Cleaning & Analysis**
   - Missing value treatment
   - Outlier detection and handling
   - Categorical data cleaning

5. **Feature Engineering**
   - Label encoding implementation
   - SHAP-based feature selection
   - Data normalization

6. **Modeling & Evaluation**
   - Model training
   - Performance evaluation
   - Results interpretation

## Key Findings

1. **Data Quality Issues Identified:**
   - Inconsistent category naming in categorical features
   - Missing values concentrated in numerical columns
   - Presence of outliers in key features

2. **Customer Patterns:**
   - Clear behavioral differences between churned and non-churned customers
   - Specific device and payment method preferences
   - Varying satisfaction scores impacting churn

3. **Predictive Modeling:**
   - SHAP values provide interpretable feature importance
   - Multiple features contribute to churn prediction
   - Model performance evaluated on multiple metrics

## Usage

To run this notebook:

```python
# Download dataset
import kagglehub
path = kagglehub.dataset_download("ankitverma2010/ecommerce-customer-churn-analysis-and-prediction")

# Load data
import pandas as pd
filename = f'{path}/E Commerce Dataset.xlsx'
df = pd.read_excel(filename, sheet_name='E Comm')

# Run subsequent cells for analysis and modeling
```

## Learning Objectives

This notebook demonstrates:
- End-to-end data science workflow
- Real-world customer churn prediction
- Feature engineering and selection techniques
- SHAP for model interpretability
- Handling imbalanced datasets

## Business Impact

By accurately predicting customer churn:
- **Proactive Retention:** Target at-risk customers with personalized offers
- **Cost Reduction:** Lower customer acquisition costs by retaining existing customers
- **Revenue Growth:** Improve customer lifetime value
- **Strategic Decision Making:** Data-driven business decisions

## References

- Dataset: [Kaggle - E-Commerce Customer Churn](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
- SHAP Documentation: [SHAP GitHub](https://github.com/slundberg/shap)

---

**Note:** This is a final project for Purwadhika Bootcamp JCDSOL02 by Beta Team for educational and analytical purposes.
