# RNA-seq Machine Learning Classification Pipeline

A modular Python workflow for **binary classification of raw RNA-seq gene counts**. It compares seven supervised learning algorithms using **nested stratified cross-validation**, with gene filtering, log2-CPM normalization, ANOVA feature selection and standardization fitted inside cross-validation.

## Workflow

```text
Raw gene count TSV + sample metadata CSV
                 |
         Match sample identifiers
                 |
     5-fold outer stratified CV
                 |
     Training-only preprocessing
      low-count filter + log2-CPM
                 |
     ANOVA SelectKBest (20/50/100)
                 |
          StandardScaler
                 |
   4-fold inner CV / GridSearchCV
                 |
      Train selected classifier
                 |
      Held-out fold predictions
                 |
      CSV summaries and plots
```

The exploratory filtering and normalization calculations in the original notebook/script are **not** used as precomputed inputs for cross-validation. The training pipeline starts with the **original raw counts**.

## Algorithms

- Logistic Regression (L2, class-weight balanced)
- Linear Support Vector Machine
- Random Forest
- XGBoost
- K-Nearest Neighbors
- Decision Tree
- Gaussian Naive Bayes

The feature-count grid (`k=20, 50, 100`) and model-specific tuning grids follow the original project implementation. Inner cross-validation optimizes balanced accuracy. The outer folds provide held-out predictions.

## Repository layout

```text
RNAseq-ML-Classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── example_gene_counts.tsv
│   └── example_metadata.csv
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── plot_results.py
│   ├── refit_best_model.py
│   └── predict.py
├── results/                  # synthetic demonstration results
├── figures/                  # synthetic demonstration figures
└── models/                   # synthetic demo model and metadata
```

## Input format

**`data/example_gene_counts.tsv`**: tab-separated **raw integer** counts, with gene identifiers in the first column and samples in subsequent columns:

```tsv
Geneid	Sample_001	Sample_002	Sample_003
GENE_0001	105	121	86
GENE_0002	44	52	47
```

**`data/example_metadata.csv`**: sample IDs and binary labels:

```csv
Patient_ID,ResponseGroup
Sample_001,Responder
Sample_002,NonResponder
Sample_003,Responder
```

The actual included example has **60 synthetic samples and 150 genes**. These values and labels are invented for testing the workflow, **not** from the melanoma cohort. The short snippets above illustrate column layout only.

Sample IDs must match between the two files. Metadata may contain extra columns; only `Patient_ID` and `ResponseGroup` are required. Raw counts must be finite, nonnegative integers. With the original 5-fold outer and 4-fold inner setup, each class needs enough samples to be represented in all folds. At least 100 genes are needed for the original feature-selection grid, and at least 100 genes must survive training-fold filtering.

## Installation and usage

Run commands from the repository root. Python 3.10+ is recommended, using package versions compatible with your Python installation.

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train and export CSV reports:

```bash
python src/train_models.py --counts data/example_gene_counts.tsv --metadata data/example_metadata.csv --output results
```

Generate plots **without rerunning model training**:

```bash
python src/plot_results.py --results results --figures figures
```

To run on your own dataset, replace the `--counts` and `--metadata` paths. Model fitting can be computationally expensive on large gene matrices; the original code uses `n_jobs=1`.

## Outputs

| File | Description |
|---|---|
| `results/seven_models_comparison.csv` | Accuracy, balanced accuracy, ROC-AUC, precision, sensitivity, specificity, F1, and TN/FP/FN/TP |
| `results/seven_models_patient_predictions.csv` | One out-of-fold prediction per sample per classifier, including predicted probability |
| `results/outer_fold_performance.csv` | Outer-fold balanced accuracy and fold sizes |
| `results/best_hyperparameters_per_fold.csv` | Selected hyperparameters and inner-CV balanced accuracy for each outer fold |
| `figures/model_performance_comparison.png` | Side-by-side comparison of five evaluation metrics |
| `figures/roc_curves_all_models.png` | Pooled out-of-fold ROC curves |
| `figures/confusion_matrices_all_models.png` | Confusion matrices for seven models |
| `figures/outer_fold_stability.png` | Distribution of outer-fold balanced accuracy |
| `figures/model_metrics_heatmap.png` | Heatmap of model evaluation metrics |

Note: pooled out-of-fold ROC-AUC summarizes predictions made by different fold-trained models and is not an independent external validation result. Estimated probabilities are not necessarily calibrated.

## Example research application

The original workflow was evaluated on **46 pretreatment melanoma tumor RNA-seq samples**, with **19 responders (CR/PR)** and **27 nonresponders (PD)**. The eight stable-disease samples were excluded from this binary task. Seven models yielded **balanced accuracy between 0.473 and 0.549**, near chance-level performance. The example demonstrates **reproducible benchmarking**, **not** an effective or clinically validated immunotherapy-response predictor.

## Limitations

- Small cohorts and high-dimensional gene-expression data can produce unstable feature selection and noisy performance estimates.
- Classification performance depends on labels, treatment heterogeneity and possible technical confounding.
- `SelectKBest(f_classif)` is a univariate ANOVA-based ML feature selector, **not** DESeq2 differential expression analysis.
- The included synthetic example is for functional demonstration and does not establish biological validity.
- The current implementation does not perform independent external validation or probability calibration.

## Intended use

Educational and research benchmarking only. **Not for clinical diagnosis or treatment decisions.**


## Refit and save the best-performing algorithm

Nested cross-validation evaluates each model using held-out outer folds. After comparison, the following command **selects the algorithm with the highest pooled out-of-fold balanced accuracy** in `results/seven_models_comparison.csv`, tunes it again on all supplied labeled samples with 4-fold stratified CV, and saves the fitted preprocessing-and-classification pipeline. Ties are resolved alphabetically.

```bash
python src/refit_best_model.py --counts data/example_gene_counts.tsv --metadata data/example_metadata.csv
```

Outputs: `models/best_model.joblib` and `models/model_info.json`. The model includes training-derived gene filtering, log2-CPM transformation, ANOVA selection, scaling and classifier. It requires the same gene identifiers as training. **Refitting is not independent validation**, and choosing the best algorithm from the same CV results may introduce selection optimism. Do not report the refitted model's training score as an unbiased estimate of generalization.

To make predictions with the saved model on a compatible raw-count matrix:

```bash
python src/predict.py --counts data/example_gene_counts.tsv --model models/best_model.joblib --output results/new_sample_predictions.csv
```

The command above predicts on the example training samples purely to demonstrate loading and inference. **These are in-sample predictions, not an external test or performance evaluation.** For real use, pass a separate raw-count matrix generated with compatible gene definitions and counting conventions. Never load an untrusted `.joblib` file.

## Included synthetic demonstration results

The committed `results/`, `figures/`, and `models/` are generated from **60 artificial samples and 150 artificial genes**, with an intentionally strong synthetic class signal. They do **not** represent melanoma patients, biological discoveries, or clinical predictive performance.

To generate these demonstration outputs quickly, the repository supports a reduced search grid while retaining 5 outer and 4 inner folds:

```bash
python src/train_models.py --counts data/example_gene_counts.tsv --metadata data/example_metadata.csv --quick-demo
python src/plot_results.py
python src/refit_best_model.py --counts data/example_gene_counts.tsv --metadata data/example_metadata.csv --quick-demo
python src/predict.py --counts data/example_gene_counts.tsv
```

`--quick-demo` uses fewer hyperparameter candidates (and 20 trees for forest/XGBoost), **so its results are not directly comparable to the original full-grid analysis**. Omit the flag to reproduce the full model search. The example output currently selects XGBoost; other datasets may select another algorithm. For a real dataset, use the same dataset in both training and refitting commands and regenerate all outputs.
