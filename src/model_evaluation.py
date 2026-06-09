import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_curve, auc

def model_evaluation(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred: np.ndarray = model.predict(X_test)
    y_scores: np.ndarray = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))

    while True:
        userChoice: str = input("Would you like to see the Precision-Recall evaluation? (y/n)")

        if userChoice.lower() == "y":
            precision, recall, threshold = precision_recall_curve(y_test, y_scores)
            auc_score: float = auc(recall, precision)

            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {auc_score:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title("Precision-Recall Curve")
            plt.legend()
            plt.show()
            print("\nEvaluation complete.")
            return 
        
        elif userChoice.lower() == "n":
            print("\nEvaluation complete.")
            return 
        
        else:
            print("Invalid selection. Please try again.")
    
