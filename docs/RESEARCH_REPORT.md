# BeejAtlas — Preliminary Machine-Learning Research Report

**Author:** Shrushti Bhoyar  
**Project:** BeejAtlas  
**Experiment:** UCI Dry Bean multiclass classification  
**Status:** Exploratory student benchmark; not a deployed seed-identification system

## Abstract

BeejAtlas is an independent student project inspired by seed conservation and agricultural biodiversity. This report documents its implemented machine-learning benchmark using the UCI Dry Bean dataset. The benchmark contains 13,611 image-derived observations from seven registered dry-bean categories and 16 numerical features. After removing 68 exact duplicate rows, 13,543 observations remained. A stratified 80:20 split was used to compare a majority-class Dummy Classifier, Logistic Regression and Random Forest. Five-fold stratified cross-validation was conducted on the training partition. Random Forest achieved a test macro F1 of 0.9308 and Logistic Regression achieved 0.9279. The small difference does not demonstrate statistical superiority. Error analysis showed reciprocal confusion between DERMASON and SIRA. Individual and grouped permutation analyses suggested that size- and shape-related feature groups contain substantial predictive information for both models. A separate repeated stratified 5-fold cross-validation experiment with three repetitions produced mean Macro F1 scores of 0.9324 for Logistic Regression and 0.9345 for Random Forest. These are internal validation estimates, and the standard deviations describe fold-to-fold variation rather than confidence intervals. The original held-out test results remain exploratory because that set was reused for model interpretation.

## 1. Project motivation and scope

BeejAtlas is inspired by seed-saving, biodiversity conservation and community-led preservation of agricultural knowledge. The intended longer-term direction is to support responsible exploration of appropriately reusable information about seed diversity.

The implemented experiment in this report is deliberately narrower: it uses the UCI Dry Bean benchmark to practise and demonstrate data ingestion, cleaning, exploratory analysis, classification and model interpretation. The UCI categories are not Indian traditional seed varieties. The experiment does not assess traditional seed conservation, climate resilience, seed viability or agricultural outcomes.

CSE's Community Seed Banks resource is a potential future source for an Indian seed-conservation module. Written permission was requested for reuse and public display. Its underlying data is not part of this experiment or repository unless and until permission is confirmed.

## 2. Dataset

The UCI Dry Bean dataset contains 13,611 observations, 16 numerical features and a categorical target with seven classes: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER and SIRA. The measurements were derived from high-resolution images of dry beans. The dataset is distributed by UCI under CC BY 4.0; attribution is required.

**Source:** M. Koklu and I. A. Özkan (2020), Dry Bean, UCI Machine Learning Repository, https://doi.org/10.24432/C50S4B.

## 3. Data preparation

### 3.1 Duplicate audit

The downloaded dataset contained 13,611 rows. The audit found:

- Exact duplicate rows: 68
- Duplicate feature vectors: 68
- Feature vectors with conflicting class labels: 0

One copy of each exact duplicate was retained. The original downloaded file was preserved, and a separate cleaned file containing 13,543 rows was written.

### 3.2 Class distribution after cleaning

| Class | Observations |
|---|---:|
| DERMASON | 3,546 |
| SIRA | 2,636 |
| SEKER | 2,027 |
| HOROZ | 1,860 |
| CALI | 1,630 |
| BARBUNYA | 1,322 |
| BOMBAY | 522 |
| **Total** | **13,543** |

The classes are imbalanced. Models were therefore assessed with macro F1 and balanced accuracy in addition to accuracy. Class weighting was enabled for Logistic Regression and Random Forest.

### 3.3 Exploratory analysis

Exploratory analysis included:

- Dataset dimensions, feature types and missing-value checks.
- Class-frequency visualisation.
- Descriptive statistics and feature distributions.
- Pearson correlation heatmap.
- Boxplots for Area, MajorAxisLength, Eccentricity, Compactness and ShapeFactor3 across classes.

Visual inspection suggested that some category medians differ substantially for certain features. These observations were treated as exploratory; feature usefulness was assessed further through model evaluation and permutation analysis.

## 4. Experimental design

### 4.1 Train-test split

- Split: 80% training and 20% test.
- Stratification: by target class.
- Random state: 42.
- Training observations: 10,834.
- Test observations: 2,709.

### 4.2 Models

**Dummy Classifier:** `strategy="most_frequent"`, used as a simple majority-class baseline.

**Logistic Regression:** `StandardScaler` followed by `LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)` in a scikit-learn pipeline.

**Random Forest:** `RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=42, n_jobs=1)`.

### 4.3 Cross-validation

Five-fold `StratifiedKFold` cross-validation was run on the training partition only, with shuffling enabled and random state 42. Macro F1 and balanced accuracy were recorded as mean and fold standard deviation. Each model was then fitted on the complete training partition and evaluated on the held-out test partition.

## 5. Results

### 5.1 Model comparison

| Model | CV macro F1 mean | CV macro F1 SD | CV balanced accuracy mean | CV balanced accuracy SD | Test accuracy | Test balanced accuracy | Test macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy Classifier | 0.059291 | 0.000046 | 0.142857 | 0.000000 | 0.261720 | 0.142857 | 0.059266 |
| Logistic Regression | 0.933887 | 0.005469 | 0.934837 | 0.005355 | 0.915467 | 0.929975 | 0.927917 |
| Random Forest | 0.936578 | 0.004441 | 0.935077 | 0.004900 | 0.919158 | 0.929734 | 0.930784 |

Random Forest has a small numerical advantage in test accuracy and macro F1. Logistic Regression has a marginally higher test balanced accuracy. The differences are small and no statistical significance test was conducted; it is not appropriate to claim that Random Forest is definitively superior.

### 5.2 Random Forest class-level results

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| BARBUNYA | 0.9258 | 0.8943 | 0.9098 | 265 |
| BOMBAY | 1.0000 | 1.0000 | 1.0000 | 104 |
| CALI | 0.9187 | 0.9356 | 0.9271 | 326 |
| DERMASON | 0.9071 | 0.9224 | 0.9147 | 709 |
| HOROZ | 0.9644 | 0.9462 | 0.9552 | 372 |
| SEKER | 0.9395 | 0.9557 | 0.9475 | 406 |
| SIRA | 0.8687 | 0.8539 | 0.8612 | 527 |

SIRA has the lowest Random Forest recall on this test split. BOMBAY has perfect metrics for 104 test observations; this is a result for this split, not a guarantee of perfect performance on future observations.

### 5.3 Confusion-matrix observations

For the DERMASON/SIRA pair:

- Logistic Regression classified 71 DERMASON observations as SIRA and 30 SIRA observations as DERMASON.
- Random Forest classified 43 DERMASON observations as SIRA and 58 SIRA observations as DERMASON.

Both models made 101 combined errors in these two directions, but the direction of the errors differed. This suggests different decision patterns, not that either model resolved the pair better overall.

## 6. Feature interpretation

### 6.1 Random Forest impurity-based importance

The leading features in the recorded run were Perimeter (0.1135), ShapeFactor1 (0.0983), Compactness (0.0972), MinorAxisLength (0.0918) and ShapeFactor3 (0.0912). Impurity-based importance is model-specific and can distribute importance across correlated predictors. It should not be interpreted causally.

### 6.2 Individual permutation importance

Individual permutation importance used test macro F1 and 20 repeats. The analysis suggested that Logistic Regression was particularly sensitive to ShapeFactor1, AspectRatio, ShapeFactor2, ShapeFactor4 and Roundness. Random Forest's largest individual permutation effects were associated with Roundness and ShapeFactor4 in the recorded chart. Some Random Forest features had near-zero or slightly negative estimates. A negative estimate means the score happened to improve slightly after shuffling that feature in these repetitions; it does not establish that the feature is inherently harmful.

### 6.3 Grouped permutation importance

Absolute Pearson correlations were calculated on the training partition. Features connected by an absolute correlation of at least 0.90 were placed in the same connected component. Because this is a connected-component grouping, not every pair of features within a group must itself exceed the threshold.

| Group | Features | Logistic Regression mean decrease (SD) | Random Forest mean decrease (SD) |
|---|---|---:|---:|
| 1 | Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea, EquivDiameter, ShapeFactor1 | 0.549399 (0.010627) | 0.570623 (0.007208) |
| 2 | AspectRatio, Eccentricity, Compactness, ShapeFactor3 | 0.307841 (0.009325) | 0.322182 (0.007843) |
| 3 | Extent | 0.001417 (0.001125) | 0.000948 (0.001174) |
| 4 | Roundness | 0.075028 (0.005210) | 0.076268 (0.005330) |
| 5 | ShapeFactor2 | 0.090583 (0.005281) | 0.000842 (0.001249) |
| 6 | ShapeFactor4 | 0.084229 (0.004352) | 0.042023 (0.002817) |
| 7 | Solidity | 0.010142 (0.001938) | 0.003839 (0.001446) |

The two largest groups for both models were the size-related group (Group 1) and shape-related group (Group 2). These values are absolute macro-F1 decreases, not percentages and not additive contributions. Joint permutation preserves relationships among features within the group but breaks their relationships with the remaining predictors. It can create feature combinations that may not occur naturally, so the results are exploratory evidence of model reliance rather than causal effects.

## 7. Discussion

The results show that both Logistic Regression and Random Forest classify this benchmark well relative to the majority-class baseline. Their overall test metrics are close. Random Forest is slightly higher on test macro F1 and accuracy, while Logistic Regression is slightly higher on balanced accuracy. The confusion matrices show that both models struggle most with some DERMASON/SIRA distinctions, although their error directions differ.

The feature analyses suggest that groups of size- and shape-related measurements carry substantial predictive information in this benchmark. This is consistent with the fact that the UCI features describe dimensions and shape measurements extracted from bean images. Correlation and permutation importance do not establish causal relationships or biological mechanisms.

## 8. Limitations and next steps

1. **Benchmark scope:** This dataset contains seven registered dry-bean categories and is not an Indian traditional-seed dataset.
2. **Feature representation:** The experiment uses 16 pre-extracted numerical measurements, not raw images.
3. **Class imbalance:** Class counts vary considerably, especially between DERMASON and BOMBAY.
4. **Test-set reuse:** The held-out test set was used for model comparison and subsequent feature-importance interpretation. The reported test metrics should therefore be treated as exploratory. A new external test set or nested evaluation is needed for stronger claims.
5. **Correlated variables:** Grouped permutation can create unrealistic cross-group feature combinations. Importance estimates should be interpreted cautiously.
6. **No field validation:** No claims are made about real-world seed identification, viability, climate adaptation or conservation impact.
7. **CSE data permission:** The CSE dataset is not included. Written permission was requested before any reuse or public display.

Recommended next steps:

- Document the full workflow and exact package versions.
- Add reproducibility checks and automated tests.
- Build an interactive dashboard for the UCI benchmark without implying Indian-variety coverage.
- If CSE grants permission, design a separate, appropriately scoped community seed-bank data explorer and follow all conditions imposed.
- Before making strong performance claims, evaluate the final model on a fresh independent dataset or use a carefully designed nested validation procedure.

## References

- Koklu, M. and Özkan, I. A. (2020). *Multiclass classification of dry beans using computer vision and machine learning techniques*. Computers and Electronics in Agriculture. Dataset record: https://archive.ics.uci.edu/dataset/602/dry+bean. DOI: https://doi.org/10.24432/C50S4B.
- Centre for Science and Environment. *Community Seed Banks*. https://www.cseindia.org/page/seedbank.
- Centre for Science and Environment. *Terms and Conditions*. https://www.cseindia.org/terms-1234.


# Repeated Stratified Cross-Validation

## 1. Objective

The objective of this experiment was to evaluate the consistency of the classification models across multiple stratified training and validation partitions.

Repeated Stratified K-Fold Cross-Validation was used to obtain a more comprehensive estimate of model performance.

## 2. Experimental Configuration

The cleaned UCI Dry Bean dataset containing 13,543 observations and 16 numerical features was used.

The experimental configuration was:

- Cross-validation method: Repeated Stratified K-Fold
- Number of folds: 5
- Number of repetitions: 3
- Total evaluation folds per model: 15
- Random state: 42
- Evaluation metrics: Accuracy, Macro F1 and Balanced Accuracy

Three classifiers were evaluated:

1. Dummy Classifier
2. Logistic Regression
3. Random Forest

The same cross-validation partitions were used for all three models.

StandardScaler was incorporated into the Logistic Regression pipeline to ensure that feature scaling was fitted independently within each training fold.

## 3. Experimental Results

The results below represent the mean and standard deviation across the 15 fold-level scores.

| Model | Accuracy | Macro F1 | Balanced Accuracy |
|---|---:|---:|---:|
| Dummy Classifier | 26.18% ± 0.01 pp | 5.93% ± 0.00 pp | 14.29% ± 0.00 pp |
| Logistic Regression | 91.89% ± 0.58 pp | 93.24% ± 0.48 pp | 93.36% ± 0.45 pp |
| Random Forest | 92.31% ± 0.44 pp | 93.45% ± 0.39 pp | 93.28% ± 0.39 pp |

Note: pp represents percentage points. Standard deviations are calculated across the 15 fold-level scores.

## 4. Interpretation of Results

Random Forest achieved the highest mean Accuracy and Macro F1 among the three evaluated models.

Logistic Regression achieved a slightly higher mean Balanced Accuracy.

The difference in mean Macro F1 between Random Forest and Logistic Regression was approximately 0.21 percentage points.

Both models showed relatively low variation across the evaluated folds.

These results suggest that both classifiers provide consistent performance on the UCI Dry Bean benchmark.

However, the small differences do not establish statistically significant superiority of one model over the other.

## 5. Research Limitations

- Repeated cross-validation provides internal validation rather than independent external validation.
- The 15 fold scores are correlated because observations are reused across repeated partitions.
- The reported standard deviations are descriptive measures of fold-to-fold variation, not confidence intervals.
- The dataset contains an uneven distribution of seven dry-bean categories.
- The experiment does not establish real-world identification accuracy for traditional Indian seed varieties.
- The benchmark does not assess germination, nutritional value, seed quality or climate resilience.

## 6. Conclusion

The repeated cross-validation experiment supports the earlier findings that Logistic Regression and Random Forest can classify the seven dry-bean categories effectively.

Random Forest demonstrated a marginal advantage in mean Accuracy and Macro F1, while Logistic Regression achieved slightly higher Balanced Accuracy.

Both models are retained for further interpretation and documentation.

The findings contribute to understanding the relationship between physical seed characteristics and classification performance within the UCI Dry Bean benchmark.

