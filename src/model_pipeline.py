import pandas as pd
import sys

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def model_selection():
    while True:
        print("\nSelect a model: \n [1] Logistical Regression \n [2] Random Forest (Best performance) \n [3] XGB Booster (Fastest) \n  \n [4] Exit \n")
        userInput: str = input("Enter choice: ")
        if userInput == "1":
            chosen_model: LogisticRegression =  LogisticRegression(max_iter=10000, random_state=0)
            return chosen_model
        elif userInput == "2":
            chosen_model: RandomForestClassifier = RandomForestClassifier(min_samples_leaf=5, n_estimators=100, max_features="sqrt")
            return chosen_model
        elif userInput == "3":
            chosen_model: XGBClassifier = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42) 
            return chosen_model
        elif userInput == "4":
            sys.exit()
        else:
            print("\nInvalid choice, please try again.")

def fit_model(chosen_model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    smt: SMOTE = SMOTE(random_state=42)

    pipeline: Pipeline= Pipeline([
        ('smt', smt),
        ('model', chosen_model)
    ])

    print("\nWaiting for model to train...")

    trained_model: Pipeline = pipeline.fit(X_train, y_train)

    print("\nModel finished training!")

    return trained_model




