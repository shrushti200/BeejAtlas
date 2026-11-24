
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "dry_bean.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

def load_data():

    print("=" * 60)
    print("BEEJATLAS - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE)

    print("\nDataset loaded successfully!")

    return data


# --------------------------------------------------
# 3. BASIC DATASET INFORMATION
# --------------------------------------------------

def inspect_data(data):

    print("\n1. DATASET DIMENSIONS")
    print("-" * 40)

    print("Rows:", data.shape[0])
    print("Columns:", data.shape[1])

    print("\n2. COLUMN NAMES")
    print("-" * 40)

    for column in data.columns:
        print(column)

    print("\n3. DATA TYPES")
    print("-" * 40)

    print(data.dtypes)


# --------------------------------------------------
# 4. MISSING VALUES AND DUPLICATES
# --------------------------------------------------

def check_data_quality(data):

    print("\n4. MISSING VALUES")
    print("-" * 40)

    missing_values = data.isnull().sum()

    print(missing_values)

    print("\nTotal missing values:", missing_values.sum())

    duplicate_count = data.duplicated().sum()

    print("\nDuplicate rows:", duplicate_count)

    quality_report = pd.DataFrame({
        "Metric": [
            "Total rows",
            "Total columns",
            "Total missing values",
            "Duplicate rows",
            "Number of classes"
        ],
        "Value": [
            len(data),
            len(data.columns),
            int(missing_values.sum()),
            int(duplicate_count),
            data["Class"].nunique()
        ]
    })

    quality_report.to_csv(
        TABLE_DIR / "data_quality_report.csv",
        index=False
    )


# --------------------------------------------------
# 5. CLASS DISTRIBUTION
# --------------------------------------------------

def analyze_classes(data):

    print("\n5. BEAN CLASS DISTRIBUTION")
    print("-" * 40)

    class_counts = data["Class"].value_counts()

    print(class_counts)

    class_counts.to_csv(
        TABLE_DIR / "class_distribution.csv"
    )

    plt.figure(figsize=(10, 6))

    class_counts.plot(
        kind="bar"
    )

    plt.title("Distribution of Dry Bean Categories")
    plt.xlabel("Bean Category")
    plt.ylabel("Number of Observations")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "class_distribution.png",
        dpi=300
    )

    plt.close()

    return class_counts


# --------------------------------------------------
# 6. DESCRIPTIVE STATISTICS
# --------------------------------------------------

def analyze_statistics(data):

    numeric_data = data.select_dtypes(
        include="number"
    )

    statistics = numeric_data.describe().T

    print("\n6. DESCRIPTIVE STATISTICS")
    print("-" * 40)

    print(statistics.round(2))

    statistics.to_csv(
        TABLE_DIR / "descriptive_statistics.csv"
    )

    return numeric_data


# --------------------------------------------------
# 7. FEATURE DISTRIBUTIONS
# --------------------------------------------------

def analyze_feature_distributions(numeric_data):

    selected_features = numeric_data.columns[:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes = axes.flatten()

    for index, feature in enumerate(selected_features):

        axes[index].hist(
            numeric_data[feature],
            bins=30,
            edgecolor="black"
        )

        axes[index].set_title(
            f"Distribution of {feature}"
        )

        axes[index].set_xlabel(feature)
        axes[index].set_ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "feature_distributions.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 8. CORRELATION ANALYSIS
# --------------------------------------------------

def analyze_correlations(numeric_data):

    correlation = numeric_data.corr()

    print("\n7. FEATURE CORRELATIONS")
    print("-" * 40)

    print(correlation.round(2))

    correlation.to_csv(
        TABLE_DIR / "correlation_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(13, 10))

    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))

    ax.set_xticklabels(
        correlation.columns,
        rotation=90,
        fontsize=8
    )

    ax.set_yticklabels(
        correlation.columns,
        fontsize=8
    )

    ax.set_title("Dry Bean Feature Correlation Matrix")

    fig.colorbar(image, ax=ax, label="Correlation")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "correlation_heatmap.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 9. AREA COMPARISON
# --------------------------------------------------

def compare_area_by_class(data):

    if "Area" not in data.columns:
        print("\nArea column not found. Skipping comparison.")
        return

    classes = sorted(data["Class"].unique())

    grouped_data = [
        data.loc[data["Class"] == category, "Area"]
        for category in classes
    ]

    plt.figure(figsize=(11, 6))

    plt.boxplot(
        grouped_data,
        tick_labels=classes,
        showfliers=False
    )

    plt.title("Area Distribution Across Bean Categories")
    plt.xlabel("Bean Category")
    plt.ylabel("Area")
    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "area_by_class.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 10. MAIN PROGRAM
# --------------------------------------------------

def main():

    data = load_data()

    if "Class" not in data.columns:
        raise ValueError("The target column 'Class' is missing.")

    inspect_data(data)

    check_data_quality(data)

    class_counts = analyze_classes(data)

    numeric_data = analyze_statistics(data)

    analyze_feature_distributions(numeric_data)

    analyze_correlations(numeric_data)

    compare_area_by_class(data)

    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS COMPLETED!")
    print("=" * 60)

    print("\nClasses found:", len(class_counts))

    print("\nCharts saved in:")
    print(FIGURE_DIR)

    print("\nTables saved in:")
    print(TABLE_DIR)


if __name__ == "__main__":
    main()
