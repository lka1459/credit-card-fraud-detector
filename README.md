# Credit Card Fraud Detection

A machine learning classification project that detects whether a credit card transaction is fraudulent or legitimate.

This project was built to practise working with tabular classification data, especially imbalanced datasets where fraudulent transactions are much less common than normal transactions.

## Project Overview

The project currently uses three machine learning models:

- Logistic Regression
- Random Forest Classifier
- XGB Booster Classifier

The goal is to compare how different models perform when detecting fraud, while looking beyond accuracy and focusing on more useful metrics such as precision, recall, F1-score, and Precision-Recall AUC.

## Dataset

The dataset used in this project was taken from Kaggle:

**Credit Card Fraud Detection**  
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains credit card transaction data, where each transaction is labelled as either:

- `0` = legitimate transaction
- `1` = fraudulent transaction

Because fraud cases are rare, the dataset is highly imbalanced.

## Features

- Loads and prepares transaction data
- Splits the dataset into training and testing sets
- Trains Logistic Regression and Random Forest models
- Evaluates model performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Precision-Recall AUC
- Displays a Precision-Recall curve to better understand model performance on fraud detection.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- imbalanced-learn