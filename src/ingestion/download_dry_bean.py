
from pathlib import Path

from ucimlrepo import fetch_ucirepo
import pandas as pd


# Locate the main BeejAtlas project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Create the raw data folder if it does not exist
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset():
    print("=" * 55)
    print("BEEJATLAS - DRY BEAN DATASET")
    print("=" * 55)

    print("\nConnecting to the UCI Machine Learning Repository...")

    # Download dataset number 602
    dry_bean = fetch_ucirepo(id=602)

    # Separate features and target
    features = dry_bean.data.features
    targets = dry_bean.data.targets

    # Combine them into one DataFrame
    data = features.copy()
    data["Class"] = targets.iloc[:, 0].to_numpy()

    # Save the original downloaded observations
    output_file = RAW_DATA_DIR / "dry_bean.csv"
    data.to_csv(output_file, index=False)

    print("\nDataset downloaded successfully!")

    print("\nDATASET INFORMATION")
    print("-" * 55)

    print("Number of observations:", data.shape[0])
    print("Number of columns:", data.shape[1])
    print("Number of features:", features.shape[1])

    print("\nFirst five observations:")
    print(data.head())

    print("\nClass distribution:")
    print(data["Class"].value_counts())

    print("\nMissing values:")
    print(data.isnull().sum())

    print("\nDuplicate rows:", data.duplicated().sum())

    print("\nSaved to:", output_file)

    print("\nUCI DATASET REFERENCE")
    print("https://doi.org/10.24432/C50S4B")

    print("\nDownload and inspection complete!")


if __name__ == "__main__":
    download_dataset()
