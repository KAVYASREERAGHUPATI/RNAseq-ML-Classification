## Running the Complete Pipeline

The project includes `run_pipeline.py`, a main execution script that automates the complete RNA-seq machine learning classification workflow.

Instead of executing individual Python scripts manually, users can run the entire workflow with a single command.

### 1. Run with Example Data

```bash
python run_pipeline.py
```

### 2. Run with Your Own RNA-seq Data

```bash
python run_pipeline.py --counts data/my_gene_counts.tsv --metadata data/my_metadata.csv
```

### 3. Automated Workflow

The main script executes the following steps sequentially:

1. **Model Training and Evaluation (`train_models.py`):** Performs RNA-seq preprocessing, feature selection, nested cross-validation, hyperparameter optimization, and evaluation of seven machine learning algorithms.
2. **Best Model Selection and Retraining (`refit_best_model.py`):** Identifies the best-performing algorithm based on cross-validation balanced accuracy, performs final hyperparameter tuning, and refits the complete preprocessing and classification pipeline on the available labeled dataset.
3. **Results Visualization (`plot_results.py`):** Generates model performance comparisons, ROC curves, confusion matrices, and other evaluation plots.

### 4. Generated Outputs

| Directory | Description |
|---|---|
| `results/` | Model comparison metrics, cross-validation performance, hyperparameters, and predictions |
| `figures/` | Model evaluation and visualization plots |
| `models/` | Saved final trained model (`.joblib`) and model information (`.json`) |

The saved model can subsequently be loaded using `predict.py` to generate predictions on compatible RNA-seq datasets.

**Note:** The example dataset and its generated outputs are synthetic demonstrations. Results from real biological datasets require separate interpretation and validation.
