# Credit-card-fraud-detection

import os

# Content for the README.md file
readme_content = """# Credit Card Fraud Detection using Machine Learning

This project implements a robust machine learning pipeline to detect fraudulent credit card transactions. It addresses the significant challenge of **class imbalance** in financial datasets using SMOTE (Synthetic Minority Over-sampling Technique) and compares multiple classification models.

## 🚀 Overview

Fraud detection is a classic needle-in-a-haystack problem. In this dataset, fraudulent transactions represent a tiny fraction of the total. This project demonstrates how to:
1.  **Analyze and Visualize** highly imbalanced data.
2.  **Resample** data using SMOTE to balance the training set.
3.  **Evaluate** three different models: Logistic Regression, Random Forest, and XGBoost.
4.  **Visualize Performance** using ROC Curves and Risk Score distributions.

## 📊 Dataset
The project uses the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset. 
- **Features:** PCA-transformed features (V1-V28), 'Time', and 'Amount'.
- **Target:** 'Class' (1 for fraud, 0 for normal).

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** - `Pandas`, `NumPy` (Data manipulation)
    - `Matplotlib`, `Seaborn` (Visualization)
    - `Scikit-Learn` (Modeling & Metrics)
    - `Imbalanced-learn` (SMOTE)
    - `XGBoost` (Gradient Boosting)

## 📈 Methodology

### 1. Data Preprocessing & Balancing
The original dataset is extremely skewed. We use **SMOTE** on the training set to generate synthetic examples of the minority class, ensuring the models learn the characteristics of fraud rather than just predicting "Normal" for everything.

### 2. Model Training
We compare three algorithms with optimized settings for performance:
- **Logistic Regression:** The baseline statistical model.
- **Random Forest:** Ensemble method using 50 trees with parallel processing.
- **XGBoost:** High-performance gradient boosting using the `hist` tree method for speed.

### 3. Evaluation
Models are evaluated using:
- **Precision-Recall/Classification Reports:** Vital for understanding the trade-off between catching fraud and annoying customers with false alarms.
- **ROC-AUC Score:** Measuring the model's ability to distinguish between classes.
- **Risk Score Distribution:** Analyzing the probability outputs of the XGBoost model.

## 🖼️ Visualizations

The script generates several key insights:
- **Class Distribution:** Visualizing the rarity of fraud.
- **ROC Curve Comparison:** Comparing the True Positive Rate vs. False Positive Rate across all models.
- **Risk Score Histogram:** A look at how the XGBoost model assigns probabilities to transactions.
