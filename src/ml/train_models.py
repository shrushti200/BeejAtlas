
from pathlib import Path

import joblib
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "processed" / "dry_bean_clean.csv"

OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

MODEL_DIR = ROOT / "models"

for folder in [FIGURES_DIR, TABLES_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

SCORING = {
    "macro_f1": "f1_macro",
    "balanced_accuracy": "balanced_accuracy"
}


# --------------------------------------------------
# 3. LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_data():

    print("\n1. LOADING CLEANED DATASET")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE)

    if "Class" not in data.columns:
        raise ValueError("Target column 'Class' is missing.")

    if data.isnull().any().any():
        raise ValueError(
            "Missing values detected. Investigate before training."
        )

    if data.duplicated().any():
        raise ValueError(
            "Duplicate observations detected. Clean the dataset first."
        )

    X = data.drop(columns=["Class"])
    y = data["Class"]

    X = X.apply(pd.to_numeric, errors="raise")

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite numerical values detected.")

    print("Observations:", len(data))
    print("Features:", X.shape[1])
    print("Categories:", y.nunique())

    return X, y


# --------------------------------------------------
# 4. STRATIFIED TRAIN-TEST SPLIT
# --------------------------------------------------

def split_data(X, y):

    print("\n2. CREATING TRAIN-TEST SPLIT")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training observations:", len(X_train))
    print("Testing observations:", len(X_test))

    print("\nTraining class distribution:")
    print(y_train.value_counts())

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# 5. DEFINE MODELS
# --------------------------------------------------

def create_models():

    models = {

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

    return models


# --------------------------------------------------
# 6. CROSS-VALIDATION AND TEST EVALUATION
# --------------------------------------------------

def train_and_evaluate(models, X_train, X_test, y_train, y_test):

    print("\n3. TRAINING AND EVALUATING MODELS")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_records = []
    test_records = []

    classes = sorted(y_train.unique())

    for name, model in models.items():

        print("\n" + "-" * 55)
        print("MODEL:", name)

        # Cross-validation uses training data only.
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            return_train_score=False,
            n_jobs=1
        )

        macro_mean = cv_results["test_macro_f1"].mean()
        macro_std = cv_results["test_macro_f1"].std()

        balanced_mean = cv_results[
            "test_balanced_accuracy"
        ].mean()

        balanced_std = cv_results[
            "test_balanced_accuracy"
        ].std()

        print(
            f"Cross-validation Macro F1: "
            f"{macro_mean:.4f} +/- {macro_std:.4f}"
        )

        print(
            f"Cross-validation Balanced Accuracy: "
            f"{balanced_mean:.4f} +/- {balanced_std:.4f}"
        )

        cv_records.append({
            "Model": name,
            "CV Macro F1 Mean": macro_mean,
            "CV Macro F1 Std": macro_std,
            "CV Balanced Accuracy Mean": balanced_mean,
            "CV Balanced Accuracy Std": balanced_std
        })

        # Fit on the complete training set.
        model.fit(X_train, y_train)

        # Evaluate on the untouched test set.
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        balanced_accuracy = balanced_accuracy_score(
            y_test, predictions
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        print("\nTEST SET RESULTS")
        print("Accuracy:", round(accuracy, 4))
        print("Balanced Accuracy:", round(balanced_accuracy, 4))
        print("Macro F1:", round(macro_f1, 4))

        print("\nCLASSIFICATION REPORT")

        report = classification_report(
            y_test,
            predictions,
            labels=classes,
            target_names=classes,
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        safe_name = name.lower().replace(" ", "_")

        report_df.to_csv(
            TABLES_DIR / f"{safe_name}_classification_report.csv"
        )

        print(classification_report(
            y_test,
            predictions,
            labels=classes,
            target_names=classes,
            zero_division=0
        ))

        test_records.append({
            "Model": name,
            "Test Accuracy": accuracy,
            "Test Balanced Accuracy": balanced_accuracy,
            "Test Macro F1": macro_f1
        })

        # Confusion matrix
        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=classes
        )

        fig, ax = plt.subplots(figsize=(9, 7))

        image = ax.imshow(
            matrix,
            cmap="Blues"
        )

        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))

        ax.set_xticklabels(classes, rotation=35, ha="right")
        ax.set_yticklabels(classes)

        ax.set_xlabel("Predicted Category")
        ax.set_ylabel("Actual Category")
        ax.set_title(f"{name} - Confusion Matrix")

        for row in range(len(classes)):
            for col in range(len(classes)):
                ax.text(
                    col,
                    row,
                    str(matrix[row, col]),
                    ha="center",
                    va="center",
                    fontsize=8
                )

        fig.colorbar(image, ax=ax)
        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / f"{safe_name}_confusion_matrix.png",
            dpi=300
        )

        plt.close()

        # Save fitted model
        joblib.dump(
            model,
            MODEL_DIR / f"{safe_name}.joblib"
        )

        # Random Forest feature importance
        if name == "Random Forest":

            importance = pd.DataFrame({
                "Feature": X_train.columns,
                "Importance": model.feature_importances_
            }).sort_values(
                "Importance",
                ascending=False
            )

            importance.to_csv(
                TABLES_DIR / "random_forest_feature_importance.csv",
                index=False
            )

            print("\nRANDOM FOREST FEATURE IMPORTANCE")
            print(importance.to_string(index=False))

    # Save comparison tables
    pd.DataFrame(cv_records).to_csv(
        TABLES_DIR / "cross_validation_results.csv",
        index=False
    )

    comparison = pd.DataFrame(test_records)

    comparison.to_csv(
        TABLES_DIR / "model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(comparison.round(4).to_string(index=False))


# --------------------------------------------------
# 7. MAIN PROGRAM
# --------------------------------------------------

def main():

    print("=" * 60)
    print("BEEJATLAS - MACHINE LEARNING EXPERIMENT")
    print("=" * 60)

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    models = create_models()

    train_and_evaluate(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\n" + "=" * 60)
    print("BEEJATLAS ML EXPERIMENT COMPLETED!")
    print("=" * 60)

    print("\nResults saved in:", TABLES_DIR)
    print("Charts saved in:", FIGURES_DIR)
    print("Models saved in:", MODEL_DIR)


if __name__ == "__main__":
    main()
