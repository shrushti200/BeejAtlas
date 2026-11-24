
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = ROOT / "data" / "raw" / "dry_bean.csv"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"

FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 55)
    print("BEEJATLAS - FEATURE COMPARISON")
    print("=" * 55)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE)

    features = [
        "Area",
        "MajorAxisLength",
        "Eccentricity",
        "Compactness",
        "ShapeFactor3"
    ]

    for feature in features:
        if feature not in data.columns:
            raise ValueError(f"Missing feature: {feature}")

    classes = sorted(data["Class"].unique())

    print("\nNumber of observations:", len(data))
    print("Number of categories:", len(classes))

    summary = data.groupby("Class")[features].agg(
        ["mean", "median", "std"]
    )

    print("\nFEATURE STATISTICS BY CATEGORY")
    print(summary.round(3))

    summary.to_csv(
        TABLES / "feature_statistics_by_class.csv"
    )

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        values = [
            data.loc[data["Class"] == category, feature].dropna()
            for category in classes
        ]

        axes[i].boxplot(
            values,
            tick_labels=classes,
            showfliers=False
        )

        axes[i].set_title(f"{feature} by Bean Category")
        axes[i].set_xlabel("Bean Category")
        axes[i].set_ylabel(feature)
        axes[i].tick_params(axis="x", rotation=30)

    axes[-1].set_visible(False)

    fig.suptitle(
        "Physical Feature Comparison Across Bean Categories"
    )

    plt.tight_layout()

    chart = FIGURES / "feature_comparison_by_class.png"
    plt.savefig(chart, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nAnalysis completed successfully!")
    print("Chart:", chart)
    print("Statistics:", TABLES / "feature_statistics_by_class.csv")


if __name__ == "__main__":
    main()
