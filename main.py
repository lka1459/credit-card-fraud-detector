import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.model_pipeline import model_selection, fit_model
from src.model_evaluation import model_evaluation

def main() -> None:
    df: pd.DataFrame = pd.read_csv(r'data\creditcard.csv')
    df.drop_duplicates(inplace=True)

    X: pd.DataFrame = df.drop(columns=["Class"])
    y: pd.Series = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Credit Card Fraud Detection")
    print("=" * 30)

    chosen_model = model_selection()

    print("\nModel Training")
    print("=" * 30)

    model: Pipeline = fit_model(chosen_model, X_train, y_train)

    print("\nModel Evaluation")
    print("=" * 30)
    model_evaluation(model, X_test, y_test)

if __name__ == "__main__":
    main()