# Credit Card Fraud Detection

A machine learning classification project that detects whether a credit card transaction is fraudulent or legitimate.

This project was built to practise working with tabular classification data, particularly imbalanced datasets where fraudulent transactions are significantly less common than legitimate transactions.

## Project Overview

The project compares three machine learning models:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

The objective is to evaluate how different models perform when detecting fraudulent transactions while focusing on metrics that are more meaningful for imbalanced datasets, including precision, recall, F1-score, and Precision-Recall AUC.

## Dataset

The dataset used in this project was obtained from Kaggle:

**Credit Card Fraud Detection**  
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains credit card transaction records, where each transaction is labelled as either:

- `0` = Legitimate transaction
- `1` = Fraudulent transaction

Due to the small number of fraudulent transactions, the dataset is highly imbalanced.

## Features

- Loads and prepares transaction data
- Splits the dataset into training and testing sets
- Uses `Pipeline` to streamline preprocessing and model training
- Applies SMOTE oversampling to address class imbalance
- Trains and compares:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
- Evaluates model performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Precision-Recall AUC
- Generates Precision-Recall curves for model comparison
- Compares model performance on an imbalanced classification problem

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- XGBoost
- matplotlib