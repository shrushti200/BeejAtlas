
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

CORRELATION_THRESHOLD = 0.90
N_REPEATS = 30


# --------------------------------------------------
# 3. IDENTIFY CORRELATED FEATURE GROUPS
# --------------------------------------------------

def identify_feature_groups(X_train):

    print("\n1. IDENTIFYING FEATURE GROUPS")
    print("-" * 55)

    # Correlations are calculated using training data only.
    correlation = X_train.corr().abs()

    features = list(X_train.columns)

    parent = {
        feature: feature
        for feature in features
    }

    def find(feature):
        while parent[feature] != feature:
            parent[feature] = parent[parent[feature]]
            feature = parent[feature]

        return feature

    def union(feature_a, feature_b):
        root_a = find(feature_a)
        root_b = find(feature_b)

        if root_a != root_b:
            parent[root_b] = root_a

    # Connect strongly correlated features.
    for i, feature_a in enumerate(features):

        for feature_b in features[i + 1:]:

            if (
                correlation.loc[feature_a, feature_b]
                >= CORRELATION_THRESHOLD
            ):
                union(feature_a, feature_b)

    # Collect connected components.
    components = {}

    for feature in features:
        root = find(feature)
        components.setdefault(root, []).append(feature)

    groups = list(components.values())

    groups.sort(
        key=lambda group: (-len(group), group[0])
    )

    records = []

    for index, group in enumerate(groups, start=1):

        group_name = f"Group {index}"

        print(f"\n{group_name}:")
        print(", ".join(group))

        records.append({
            "Group": group_name,
            "Feature Count": len(group),
            "Features": ", ".join(group)
        })

    groups_df = pd.DataFrame(records)

    groups_df.to_csv(
        TABLE_DIR / "correlated_feature_groups.csv",
        index=False
    )

    print("\nNumber of groups:", len(groups))

    return groups


# --------------------------------------------------
# 4. CREATE MODELS
# --------------------------------------------------

def create_models():

    return {

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
# 5. GROUPED PERMUTATION IMPORTANCE
# --------------------------------------------------

def calculate_group_importance(
    models,
    groups,
    X_train,
    X_test,
    y_train,
    y_test
):

    print("\n2. CALCULATING GROUPED IMPORTANCE")
    print("-" * 55)

    all_results = []

    for model_index, (model_name, model) in enumerate(
        models.items()
    ):

        print("\nMODEL:", model_name)

        model.fit(X_train, y_train)

        baseline_predictions = model.predict(X_test)

        baseline_f1 = f1_score(
            y_test,
            baseline_predictions,
            average="macro"
        )

        print(
            "Baseline test Macro F1:",
            round(baseline_f1, 5)
        )

        for group_index, features in enumerate(groups):

            importance_values = []

            for repeat in range(N_REPEATS):

                # Use a reproducible, different permutation
                # for each model, group and repetition.
                seed = (
                    RANDOM_STATE
                    + model_index * 100000
                    + group_index * 1000
                    + repeat
                )

                rng = np.random.default_rng(seed)

                permutation = rng.permutation(
                    len(X_test)
                )

                shuffled_data = X_test.copy()

                # Jointly shuffle all columns in this group.
                # One shared row permutation preserves their
                # internal relationships.
                shuffled_data.loc[:, features] = (
                    X_test.iloc[permutation][features].to_numpy()
                )

                predictions = model.predict(shuffled_data)

                shuffled_f1 = f1_score(
                    y_test,
                    predictions,
                    average="macro"
                )

                importance = baseline_f1 - shuffled_f1

                importance_values.append(importance)

            importance_mean = np.mean(importance_values)
            importance_std = np.std(
                importance_values,
                ddof=1
            )

            group_name = f"Group {group_index + 1}"

            print(
                f"{group_name}: {importance_mean:.5f}"
                f" +/- {importance_std:.5f}"
            )

            all_results.append({
                "Model": model_name,
                "Group": group_name,
                "Feature Count": len(features),
                "Features": ", ".join(features),
                "Importance Mean": importance_mean,
                "Importance Std": importance_std,
                "Baseline Macro F1": baseline_f1
            })

    return pd.DataFrame(all_results)


# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

def save_results(results):

    output_file = (
        TABLE_DIR / "grouped_permutation_importance.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print("\nResults saved to:")
    print(output_file)


# --------------------------------------------------
# 7. CREATE COMPARISON CHART
# --------------------------------------------------

def create_chart(results):

    means = results.pivot(
        index="Group",
        columns="Model",
        values="Importance Mean"
    )

    deviations = results.pivot(
        index="Group",
        columns="Model",
        values="Importance Std"
    )

    group_names = list(means.index)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, max(6, len(group_names) * 0.65)),
        sharey=True
    )

    models = [
        "Logistic Regression",
        "Random Forest"
    ]

    for ax, model_name in zip(axes, models):

        values = means[model_name].sort_values()

        names = list(values.index)
        positions = np.arange(len(names))

        errors = deviations.loc[
            names,
            model_name
        ].to_numpy()

        ax.barh(
            positions,
            values.to_numpy(),
            xerr=errors
        )

        ax.set_yticks(positions)
        ax.set_yticklabels(names)

        ax.axvline(0, linewidth=0.8)

        ax.set_title(model_name)

        ax.set_xlabel(
            "Decrease in test Macro F1"
        )

        ax.grid(
            axis="x",
            alpha=0.2
        )

    axes[0].set_ylabel("Correlated feature group")

    fig.suptitle(
        "BeejAtlas - Grouped Permutation Importance",
        fontsize=15
    )

    plt.tight_layout()

    chart_file = (
        FIGURE_DIR /
        "grouped_permutation_importance.png"
    )

    plt.savefig(
        chart_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nChart saved to:")
    print(chart_file)


# --------------------------------------------------
# 8. MAIN PROGRAM
# --------------------------------------------------

def main():

    print("=" * 60)
    print("BEEJATLAS - GROUPED PERMUTATION IMPORTANCE")
    print("=" * 60)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE)

    if data.duplicated().any():
        raise ValueError(
            "Duplicate observations detected."
        )

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

    groups = identify_feature_groups(X_train)

    models = create_models()

    results = calculate_group_importance(
        models,
        groups,
        X_train,
        X_test,
        y_train,
        y_test
    )

    save_results(results)

    create_chart(results)

    print("\n" + "=" * 60)
    print("GROUPED PERMUTATION ANALYSIS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
