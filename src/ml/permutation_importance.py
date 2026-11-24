
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "processed" / "dry_bean_clean.csv"

TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_REPEATS = 20


# --------------------------------------------------
# 3. LOAD AND SPLIT DATA
# --------------------------------------------------

def load_and_split_data():

    print("=" * 60)
    print("BEEJATLAS - PERMUTATION FEATURE IMPORTANCE")
    print("=" * 60)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE)

    if data.duplicated().any():
        raise ValueError("Duplicate records detected.")

    X = data.drop(columns=["Class"])
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nObservations:", len(data))
    print("Training observations:", len(X_train))
    print("Testing observations:", len(X_test))
    print("Features:", len(X.columns))

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# 4. DEFINE MODELS
# --------------------------------------------------

def create_models():

    models = {

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
# 5. CALCULATE PERMUTATION IMPORTANCE
# --------------------------------------------------

def calculate_importance(
    models,
    X_train,
    X_test,
    y_train,
    y_test
):

    all_results = []

    for name, model in models.items():

        print("\n" + "-" * 60)
        print("MODEL:", name)

        print("Training model...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        baseline_f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        print(
            "Baseline test Macro F1:",
            round(baseline_f1, 4)
        )

        print("Calculating permutation importance...")
        print("Repeats per feature:", N_REPEATS)

        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="f1_macro",
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=1
        )

        model_results = pd.DataFrame({
            "Model": name,
            "Feature": X_test.columns,
            "Importance Mean": result.importances_mean,
            "Importance Std": result.importances_std,
            "Baseline Macro F1": baseline_f1
        })

        model_results = model_results.sort_values(
            "Importance Mean",
            ascending=False
        )

        print("\nFEATURE IMPORTANCE:")
        print(
            model_results[
                ["Feature", "Importance Mean", "Importance Std"]
            ].round(5).to_string(index=False)
        )

        all_results.append(model_results)

    return pd.concat(all_results, ignore_index=True)


# --------------------------------------------------
# 6. SAVE COMBINED RESULTS
# --------------------------------------------------

def save_results(results):

    output_file = TABLE_DIR / "permutation_importance.csv"

    results.to_csv(output_file, index=False)

    print("\nCombined results saved:")
    print(output_file)


# --------------------------------------------------
# 7. CREATE COMPARISON CHART
# --------------------------------------------------

def create_comparison_chart(results):

    means = results.pivot(
        index="Feature",
        columns="Model",
        values="Importance Mean"
    )

    deviations = results.pivot(
        index="Feature",
        columns="Model",
        values="Importance Std"
    )

    # Show the twelve features with the largest
    # positive importance estimate in either model.
    selected = (
        means.max(axis=1)
        .sort_values(ascending=False)
        .head(12)
        .index
    )

    selected = list(reversed(selected))

    positions = np.arange(len(selected))
    height = 0.36

    fig, ax = plt.subplots(figsize=(12, 9))

    ax.barh(
        positions - height / 2,
        means.loc[selected, "Logistic Regression"],
        height,
        xerr=deviations.loc[selected, "Logistic Regression"],
        label="Logistic Regression"
    )

    ax.barh(
        positions + height / 2,
        means.loc[selected, "Random Forest"],
        height,
        xerr=deviations.loc[selected, "Random Forest"],
        label="Random Forest"
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(selected)

    ax.set_xlabel("Decrease in Macro F1 after feature permutation")
    ax.set_ylabel("Feature")

    ax.set_title(
        "Permutation Feature Importance Comparison"
    )

    ax.axvline(0, linewidth=0.8)
    ax.legend()
    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()

    chart_file = (
        FIGURE_DIR /
        "permutation_importance_comparison.png"
    )

    plt.savefig(
        chart_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nComparison chart saved:")
    print(chart_file)


# --------------------------------------------------
# 8. MAIN PROGRAM
# --------------------------------------------------

def main():

    X_train, X_test, y_train, y_test = load_and_split_data()

    models = create_models()

    results = calculate_importance(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    save_results(results)

    create_comparison_chart(results)

    print("\n" + "=" * 60)
    print("PERMUTATION IMPORTANCE COMPLETED!")
    print("=" * 60)

    print("\nResults:", TABLE_DIR)
    print("Charts:", FIGURE_DIR)


if __name__ == "__main__":
    main()
