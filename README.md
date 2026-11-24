# BeejAtlas

**Exploring seed diversity through data, databases and machine learning**

BeejAtlas is an independent student project inspired by seed conservation, agricultural biodiversity and community-led knowledge preservation. Its planned long-term direction is to make responsibly reusable information about seed diversity easier to explore through data engineering and visualisation.

> **Important scope note:** The machine-learning experiment currently implemented in this repository uses the UCI Dry Bean benchmark. It classifies seven registered dry-bean categories from pre-computed numerical measurements. It does **not** identify Indian traditional seed varieties, assess climate resilience, or measure the diversity of India's community seed banks. The Indian seed-conservation explorer is a planned component and is not yet implemented with CSE data.

## Project status

### Implemented

- Downloaded the UCI Dry Bean dataset using `ucimlrepo`.
- Explored class frequencies, descriptive statistics, missing values, duplicates and feature correlations.
- Removed 68 exact duplicate records, retaining 13,543 unique observations.
- Compared selected physical measurements across the seven classes.
- Trained and evaluated a Dummy Classifier, Logistic Regression and Random Forest.
- Used a stratified 80:20 train-test split and five-fold stratified cross-validation on the training split.
- Examined confusion matrices, individual permutation importance and grouped permutation importance.
- Conducted repeated stratified 5-fold cross-validation with three repetitions (15 fold scores per model).
- Built an interactive Streamlit dashboard using Plotly for data exploration, model evaluation and feature-importance analysis.
- Prepared a research report documenting methodology, experimental findings and limitations.

### Planned

- Integrate appropriately licensed Indian seed-conservation information, subject to written permission.
- Conduct independent external validation of the classification experiment.
- Add automated tests and strengthen reproducibility.
- Deploy the Streamlit application publicly.

## Why this project?

BeejAtlas is inspired by seed-saving and biodiversity-conservation work, including the broader public discussion around traditional seeds and community seed banks. The aim is to use data tools responsibly while keeping community knowledge, data permissions and limitations visible.

The current UCI experiment is a separate technical benchmark used to practise and demonstrate a complete tabular machine-learning workflow. It should not be interpreted as evidence about the seed varieties documented by Indian community seed banks.

## Dataset currently used

### UCI Dry Bean

The UCI Dry Bean dataset contains 13,611 observations of seven registered dry-bean types. Each observation has 16 numerical features derived from high-resolution bean images, including area, perimeter, axis lengths, aspect ratio, eccentricity, convex area, equivalent diameter, extent, solidity, roundness, compactness and four shape factors.

- **Source:** [UCI Machine Learning Repository — Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean)
- **DOI:** [10.24432/C50S4B](https://doi.org/10.24432/C50S4B)
- **Licence:** CC BY 4.0, subject to attribution to the dataset source and its creators.
- **Original dataset size:** 13,611 rows, 16 features and one target column.
- **Target classes:** BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER and SIRA.

The downloaded working file is `data/raw/dry_bean.csv`. After exact duplicate removal, the cleaned file is `data/processed/dry_bean_clean.csv` with 13,543 rows. The original raw file is retained unchanged.

### Indian community seed-bank data — planned, permission pending

The Centre for Science and Environment (CSE) publishes an interactive Community Seed Banks resource. BeejAtlas's author has contacted CSE to request written permission for analysis, transformation and public display of data or derived outputs. Until permission and any applicable conditions are confirmed, **CSE's underlying dataset must not be copied into this repository, redistributed, or used to power a public BeejAtlas deployment**.

- **Public resource:** [CSE Community Seed Banks](https://www.cseindia.org/page/seedbank)
- **CSE terms:** [CSE Terms and Conditions](https://www.cseindia.org/terms-1234)

The CSE resource describes a snapshot of information collected from NGOs, community seed banks and individual seed savers. Its coverage should not be presented as a complete census of Indian seed diversity.

## Repository structure

```text
BeejAtlas/
├── README.md
├── requirements.txt
├── .gitignore
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── docs/
│   └── RESEARCH_REPORT.md
├── notebooks/
├── src/
│   ├── ingestion/
│   │   └── download_dry_bean.py
│   ├── processing/
│   │   ├── explore_dry_bean.py
│   │   └── analyze_class_features.py
│   └── ml/
│       ├── train_models.py
│       ├── permutation_importance.py
│       ├── grouped_permutation_importance.py
│       └── repeated_cross_validation.py
├── tests/
├── models/
└── outputs/
    ├── figures/
    └── tables/

Generated data, model files and outputs may be excluded from Git using `.gitignore`; keep the scripts and documentation so the workflow remains reproducible. Do not commit CSE source data unless written permission explicitly allows it.

## Technology stack

- Python
- pandas and NumPy
- Matplotlib
- scikit-learn
- ucimlrepo
- joblib
- Streamlit
- Plotly
- SQLite — planned for future structured data storage.

## Setup

Open Command Prompt in the project directory:

```cmd
cd /d C:\Users\saaks\BeejAtlas
```

Install the packages used by the current scripts:

```cmd
py -m pip install -r requirements.txt
```

### 1. Download the UCI Dry Bean dataset

```cmd
py src\ingestion\download_dry_bean.py
```

### 2. Explore the raw dataset

```cmd
py src\processing\explore_dry_bean.py
```

### 3. Remove exact duplicate records

The current cleaning operation keeps the first copy of each exact row and writes a separate processed file. It does not overwrite the raw dataset:

```cmd
py -c "import pandas as pd; from pathlib import Path; d=pd.read_csv('data/raw/dry_bean.csv'); before=len(d); d=d.drop_duplicates(keep='first').reset_index(drop=True); out=Path('data/processed/dry_bean_clean.csv'); out.parent.mkdir(parents=True,exist_ok=True); d.to_csv(out,index=False); print('Original:',before,'Removed:',before-len(d),'Cleaned:',len(d))"
```

For the recorded run, 68 exact duplicate rows were removed, leaving 13,543 rows. Re-run the duplicate check if you download a new copy of the source data.

### 4. Compare features by class

```cmd
py src\processing\analyze_class_features.py
```

### 5. Train and evaluate the baseline models

```cmd
py src\ml\train_models.py
```

### 6. Individual permutation importance

```cmd
py src\ml\permutation_importance.py
```

### 7. Grouped permutation importance

```cmd
py src\ml\grouped_permutation_importance.py
```

The scripts generate tables under `outputs/tables/`, figures under `outputs/figures/`, and fitted model files under `models/`.

### 8. Repeated stratified cross-validation

```cmd
py src\ml\repeated_cross_validation.py
```

### 9. Launch the Streamlit dashboard

```cmd
py -m streamlit run app\streamlit_app.py
```

## Selected research visualisations

The following figures summarise the exploratory analysis and model interpretation performed on the UCI Dry Bean benchmark.

### 1. Dry-bean category distribution

![Dry Bean Category Distribution](outputs/figures/class_distribution.png)

### 2. Feature correlation heatmap

![Feature Correlation Heatmap](outputs/figures/correlation_heatmap.png)

### 3. Random Forest confusion matrix

![Random Forest Confusion Matrix](outputs/figures/random_forest_confusion_matrix.png)

### 4. Grouped permutation importance

![Grouped Permutation Importance](outputs/figures/grouped_permutation_importance.png)

## Machine-learning methodology

### Data preparation

- Target column: `Class`.
- Predictors: all 16 numerical feature columns.
- Exact duplicate rows: 68 removed before splitting.
- Split: 80% training / 20% test, stratified by class, `random_state=42`.
- Training rows: 10,834.
- Test rows: 2,709.

### Models

1. **Dummy Classifier** — `most_frequent` strategy; reference baseline.
2. **Logistic Regression** — `StandardScaler` and `LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)` inside a scikit-learn pipeline.
3. **Random Forest** — 200 trees, `class_weight="balanced_subsample"`, `random_state=42`, `n_jobs=1`.

Five-fold `StratifiedKFold` cross-validation (`shuffle=True`, `random_state=42`) was run on the training partition. Macro F1 and balanced accuracy were recorded. Each model was then fitted on the full training partition and evaluated on the held-out test partition.

### Recorded model results

The following results are from the project's recorded run. Percentages are rounded to two decimal places.

| Model | CV macro F1 (mean ± fold SD) | CV balanced accuracy (mean ± fold SD) | Test accuracy | Test balanced accuracy | Test macro F1 |
|---|---:|---:|---:|---:|---:|
| Dummy Classifier | 5.93% ± 0.00 pp | 14.29% ± 0.00 pp | 26.17% | 14.29% | 5.93% |
| Logistic Regression | 93.39% ± 0.55 pp | 93.48% ± 0.54 pp | 91.55% | 93.00% | 92.79% |
| Random Forest | 93.66% ± 0.44 pp | 93.51% ± 0.49 pp | 91.92% | 92.97% | 93.08% |

Random Forest has a small lead in test accuracy and macro F1; Logistic Regression has a marginally higher test balanced accuracy. The differences are small. These results do not establish that one model is statistically superior.

### Repeated stratified 5-fold cross-validation

A separate experiment evaluated all 13,543 cleaned observations using RepeatedStratifiedKFold with five folds and three repetitions, producing 15 fold-level scores per model. All three models used identical partitions.

Results are reported as mean and standard deviation across the 15 fold scores. Scores are on a 0-1 scale.

| Model | Accuracy mean (SD) | Macro F1 mean (SD) | Balanced accuracy mean (SD) |
|---|---:|---:|---:|
| Dummy Classifier | 0.2618 (0.0001) | 0.0593 (0.0000) | 0.1429 (0.0000) |
| Logistic Regression | 0.9189 (0.0058) | 0.9324 (0.0048) | 0.9336 (0.0045) |
| Random Forest | 0.9231 (0.0044) | 0.9345 (0.0039) | 0.9328 (0.0039) |

Random Forest achieved the highest mean accuracy and Macro F1. Logistic Regression achieved slightly higher mean balanced accuracy. The differences are small and do not establish statistical superiority.

The standard deviations describe fold-to-fold variation, not confidence intervals. The 15 fold scores are dependent because observations are reused across repetitions. This experiment provides internal validation, not independent external validation.

### Random Forest class-level test results

| Class | Precision | Recall | F1-score | Test support |
|---|---:|---:|---:|---:|
| BARBUNYA | 92.58% | 89.43% | 90.98% | 265 |
| BOMBAY | 100.00% | 100.00% | 100.00% | 104 |
| CALI | 91.87% | 93.56% | 92.71% | 326 |
| DERMASON | 90.71% | 92.24% | 91.47% | 709 |
| HOROZ | 96.44% | 94.62% | 95.52% | 372 |
| SEKER | 93.95% | 95.57% | 94.75% | 406 |
| SIRA | 86.87% | 85.39% | 86.12% | 527 |

SIRA had the lowest Random Forest recall in this test split. BOMBAY achieved perfect test metrics on 104 test observations; this should not be interpreted as a guarantee of perfect performance on new data.

### Confusion-matrix observations

The largest reciprocal confusion between two categories was between DERMASON and SIRA:

- Logistic Regression: 71 DERMASON observations were predicted as SIRA; 30 SIRA observations were predicted as DERMASON (101 combined errors).
- Random Forest: 43 DERMASON observations were predicted as SIRA; 58 SIRA observations were predicted as DERMASON (101 combined errors).

The two models therefore make different directional errors, although their combined errors for this pair are equal in this test split.

### Feature importance

Random Forest's impurity-based feature importance ranked Perimeter, ShapeFactor1, Compactness, MinorAxisLength and ShapeFactor3 among the top five features in the recorded run. Impurity-based importance can distribute importance across correlated predictors and should not be interpreted as causal influence.

Individual permutation importance was calculated on the held-out test set with 20 repeats using macro F1 as the scoring metric. Grouped permutation importance used 30 repeats. Feature groups were formed from absolute Pearson correlations of at least 0.90, calculated on the training partition, using connected components.

| Correlated group | Features | Logistic Regression importance (mean ± SD) | Random Forest importance (mean ± SD) |
|---|---|---:|---:|
| Group 1 | Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea, EquivDiameter, ShapeFactor1 | 0.5494 ± 0.0106 | 0.5706 ± 0.0072 |
| Group 2 | AspectRatio, Eccentricity, Compactness, ShapeFactor3 | 0.3078 ± 0.0093 | 0.3222 ± 0.0078 |
| Group 3 | Extent | 0.0014 ± 0.0011 | 0.0009 ± 0.0012 |
| Group 4 | Roundness | 0.0750 ± 0.0052 | 0.0763 ± 0.0053 |
| Group 5 | ShapeFactor2 | 0.0906 ± 0.0053 | 0.0008 ± 0.0012 |
| Group 6 | ShapeFactor4 | 0.0842 ± 0.0044 | 0.0420 ± 0.0028 |
| Group 7 | Solidity | 0.0101 ± 0.0019 | 0.0038 ± 0.0014 |

Grouped importance is the mean decrease in test macro F1 after jointly permuting all columns in a group. These values are not percentages and do not sum to 100%. Joint permutation preserves relationships within the permuted group but breaks its relationships with the other predictors; this can create combinations that may not occur naturally. Treat the results as exploratory evidence of model reliance, not causal feature effects.

## Limitations

- The UCI benchmark describes seven registered dry-bean categories; it is not a representative dataset of India's traditional seed varieties or community seed banks.
- The current experiment uses pre-extracted numerical measurements, not raw seed images. It is not an image-recognition application.
- Class frequencies are uneven. The BOMBAY class has 522 observations in the cleaned dataset, compared with 3,546 DERMASON observations.
- The held-out test set was reused for model comparison and permutation-based interpretation. Consequently, test metrics should be treated as exploratory, not as a final independent estimate. A new external test set or a carefully designed nested evaluation would be needed for stronger generalisation claims.
- Grouped permutation may create unrealistic combinations between the permuted group and the remaining predictors.
- The project has not demonstrated field performance, seed viability, climate resilience, farmer outcomes or conservation impact.
- The CSE community seed-bank data is not included. Public use is pending written permission and any conditions CSE specifies.

## Responsible data use

BeejAtlas is an educational, non-commercial student project. It will distinguish public benchmark data from community-held knowledge. Personal, sensitive or community-restricted information should not be published without an appropriate lawful basis and explicit permission. Any future CSE integration must follow CSE's written permission and attribution requirements.

## References

1. M. Koklu and I. A. Özkan (2020). **Dry Bean** [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50S4B. Dataset page: https://archive.ics.uci.edu/dataset/602/dry+bean
2. Centre for Science and Environment. **Community Seed Banks**. https://www.cseindia.org/page/seedbank
3. Centre for Science and Environment. **Terms and Conditions**. https://www.cseindia.org/terms-1234

## Author

Shrushti Bhoyar — B.Sc. Data Science student, Savitribai Phule Pune University, India.

---

*BeejAtlas is an independent student project. It is not affiliated with, endorsed by, or an official product of UCI, CSE, Navdanya or Vandana Shiva.*
