
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "processed" / "dry_bean_clean.csv"

TABLE_DIR = ROOT / "outputs" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

def load_data():

    print("=" * 60)
    print("BEEJATLAS - REPEATED CROSS-VALIDATION")
    print("=" * 60)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE)

    if data.duplicated().any():
        raise ValueError("Duplicate records detected.")

    if data.isnull().any().any():
        raise ValueError("Missing values detected.")

    X = data.drop(columns=["Class"])
    y = data["Class"]

    X = X.apply(pd.to_numeric, errors="raise")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite numerical values detected.")

    print("\nObservations:", len(data))
    print("Features:", X.shape[1])
    print("Categories:", y.nunique())

    return X, y


# --------------------------------------------------
# 4. DEFINE MODELS
# --------------------------------------------------

def create_models():

    return {

        "Dummy Classifier": DummyClassifier(
            strategy="most_frequent"
        ),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1
        )
    }


# --------------------------------------------------
# 5. REPEATED STRATIFIED CROSS-VALIDATION
# --------------------------------------------------

def evaluate_models(X, y):

    models = create_models()

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )

    # Generate the folds once so every model
    # receives exactly the same train/test partitions.
    splits = list(cv.split(X, y))

    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "balanced_accuracy": "balanced_accuracy"
    }

    fold_records = []
    summary_records = []

    for model_name, model in models.items():

        print("\n" + "-" * 60)
        print("MODEL:", model_name)
        print("Evaluating 15 folds...")

        results = cross_validate(
            model,
            X,
            y,
            cv=splits,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
            return_train_score=False
        )

        accuracy = results["test_accuracy"]
        macro_f1 = results["test_macro_f1"]
        balanced = results["test_balanced_accuracy"]

        print(
            f"Accuracy: {accuracy.mean():.4f} "
            f"+/- {accuracy.std(ddof=1):.4f}"
        )

        print(
            f"Macro F1: {macro_f1.mean():.4f} "
            f"+/- {macro_f1.std(ddof=1):.4f}"
        )

        print(
            f"Balanced Accuracy: {balanced.mean():.4f} "
            f"+/- {balanced.std(ddof=1):.4f}"
        )

        summary_records.append({
            "Model": model_name,
            "Accuracy Mean": accuracy.mean(),
            "Accuracy Std": accuracy.std(ddof=1),
            "Macro F1 Mean": macro_f1.mean(),
            "Macro F1 Std": macro_f1.std(ddof=1),
            "Balanced Accuracy Mean": balanced.mean(),
            "Balanced Accuracy Std": balanced.std(ddof=1)
        })

        for fold in range(len(splits)):

            fold_records.append({
                "Model": model_name,
                "Fold": fold + 1,
                "Accuracy": accuracy[fold],
                "Macro F1": macro_f1[fold],
                "Balanced Accuracy": balanced[fold]
            })

    return (
        pd.DataFrame(summary_records),
        pd.DataFrame(fold_records)
    )


# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

def main():

    X, y = load_data()

    summary, fold_results = evaluate_models(X, y)

    summary_path = (
        TABLE_DIR / "repeated_cv_summary.csv"
    )

    fold_path = (
        TABLE_DIR / "repeated_cv_fold_results.csv"
    )

    summary.to_csv(summary_path, index=False)
    fold_results.to_csv(fold_path, index=False)

    print("\n" + "=" * 60)
    print("FINAL REPEATED CROSS-VALIDATION COMPARISON")
    print("=" * 60)

    print(summary.round(4).to_string(index=False))

    print("\nSummary saved:", summary_path)
    print("Fold results saved:", fold_path)

    print("\nREPEATED CROSS-VALIDATION COMPLETED!")


if __name__ == "__main__":
    main()
