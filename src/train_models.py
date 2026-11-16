"""Nested-CV benchmarking of seven RNA-seq binary classifiers.

Run from repository root:
  python src/train_models.py --counts data/example_gene_counts.tsv --metadata data/example_metadata.csv
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from preprocessing import RNASeqPreprocessor

SEED = 42


def load_inputs(counts_path, metadata_path):
    counts = pd.read_csv(counts_path, sep="\t", index_col=0)
    metadata = pd.read_csv(metadata_path)
    required = {"Patient_ID", "ResponseGroup"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata must include {sorted(required)}")
    counts.columns = counts.columns.astype(str).str.strip()
    metadata["Patient_ID"] = metadata["Patient_ID"].astype(str).str.strip()
    if counts.columns.duplicated().any() or metadata["Patient_ID"].duplicated().any():
        raise ValueError("Duplicate sample identifiers found.")
    if not metadata["ResponseGroup"].isin(["Responder", "NonResponder"]).all():
        raise ValueError("ResponseGroup must be Responder or NonResponder.")
    missing = set(metadata["Patient_ID"]) - set(counts.columns)
    if missing:
        raise ValueError(f"Metadata samples missing from counts: {sorted(missing)}")
    metadata = metadata.set_index("Patient_ID")
    X = counts.loc[:, metadata.index].T.apply(pd.to_numeric, errors="raise")
    y = metadata["ResponseGroup"].map({"NonResponder": 0, "Responder": 1})
    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all() or (X < 0).any().any():
        raise ValueError("Raw counts must be finite, nonnegative, and nonmissing.")
    if not np.all(np.equal(np.mod(X.to_numpy(), 1), 0)):
        raise ValueError("Input must contain raw integer gene counts, not normalized values.")
    if X.columns.duplicated().any():
        raise ValueError("Duplicate gene identifiers found.")
    return X, y


def model_definitions():
    # Same seven models and hyperparameter grids as the uploaded script.
    return {
        "Logistic Regression": (LogisticRegression(solver="liblinear", C=1.0,
            class_weight="balanced", max_iter=5000, random_state=SEED),
            {"model__C": [0.01, 0.1, 1, 10]}),
        "Linear SVM": (SVC(kernel="linear", probability=True,
            class_weight="balanced", random_state=SEED),
            {"model__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (RandomForestClassifier(class_weight="balanced",
            n_jobs=1, random_state=SEED),
            {"model__n_estimators": [100, 200], "model__max_depth": [3, None]}),
        "XGBoost": (XGBClassifier(objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=1, random_state=SEED),
            {"model__n_estimators": [100, 200], "model__max_depth": [2, 3]}),
        "KNN": (KNeighborsClassifier(), {"model__n_neighbors": [3, 5, 7]}),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
            {"model__max_depth": [2, 3, 5], "model__min_samples_leaf": [2, 4]}),
        "Gaussian Naive Bayes": (GaussianNB(),
            {"model__var_smoothing": [1e-9, 1e-8, 1e-7]}),
    }


def evaluate_nested_model(name, estimator, model_grid, X, y, outer_cv, inner_cv, quick_demo=False):
    pipeline = Pipeline([
        ("rna", RNASeqPreprocessor()),
        ("select", SelectKBest(score_func=f_classif)),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])
    # Avoid selecting more genes than survive expression filtering in an inner fold.
    # This is checked using TRAINING-ONLY data and never outer test labels.
    k_candidates = [20, 50, 100]
    predictions, folds, params = [], [], []
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # The uploaded implementation uses k=[20,50,100]. Keep it unchanged.
        grid = {"select__k": ([20] if quick_demo else k_candidates), **({k: [v[0]] for k, v in model_grid.items()} if quick_demo else model_grid)}
        search = GridSearchCV(clone(pipeline), grid, scoring="balanced_accuracy",
                              cv=inner_cv, n_jobs=1, refit=True, error_score="raise")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Features .* are constant")
            warnings.filterwarnings("ignore", message="invalid value encountered in divide")
            search.fit(X_train, y_train)
        predicted = search.predict(X_test)
        probabilities = search.predict_proba(X_test)[:, 1]
        score = balanced_accuracy_score(y_test, predicted)
        folds.append({"Model": name, "Fold": fold, "Train_N": len(train_idx),
                      "Test_N": len(test_idx), "Balanced_Accuracy": score})
        params.append({"Model": name, "Fold": fold,
                       "Best_Parameters": str(search.best_params_),
                       "Inner_CV_Balanced_Accuracy": search.best_score_})
        for patient, actual, pred, prob in zip(X_test.index, y_test, predicted, probabilities):
            predictions.append({"Model": name, "Fold": fold, "Patient_ID": str(patient),
                                "Actual": int(actual), "Predicted": int(pred),
                                "Responder_Probability": float(prob)})
        print(f"{name}: outer fold {fold}/{outer_cv.n_splits}, balanced accuracy {score:.3f}", flush=True)
    pred_df = pd.DataFrame(predictions)
    order = {patient: i for i, patient in enumerate(X.index)}
    pred_df = pred_df.sort_values("Patient_ID", key=lambda s: s.map(order)).reset_index(drop=True)
    actual, predicted = pred_df["Actual"], pred_df["Predicted"]
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    metrics = {"Model": name,
               "Accuracy": accuracy_score(actual, predicted),
               "Balanced_Accuracy": balanced_accuracy_score(actual, predicted),
               "ROC_AUC": roc_auc_score(actual, pred_df["Responder_Probability"]),
               "Precision": precision_score(actual, predicted, zero_division=0),
               "Sensitivity": recall_score(actual, predicted, zero_division=0),
               "Specificity": tn / (tn + fp),
               "F1_Score": f1_score(actual, predicted, zero_division=0),
               "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    return metrics, pred_df, pd.DataFrame(folds), pd.DataFrame(params)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True, help="Raw counts TSV (genes x samples)")
    parser.add_argument("--metadata", type=Path, required=True, help="CSV with Patient_ID and ResponseGroup")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--quick-demo", action="store_true", help="Fast illustrative run on synthetic data only; reduced hyperparameter search")
    args = parser.parse_args()
    X, y = load_inputs(args.counts, args.metadata)
    if y.value_counts().min() < 5:
        raise ValueError("At least five samples in each class are required for 5-fold outer CV.")
    if X.shape[1] < 100:
        raise ValueError("At least 100 genes are required for the original k=[20,50,100] grid.")
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} genes; class counts: {y.value_counts().to_dict()}")
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    all_metrics, all_preds, all_folds, all_params = [], [], [], []
    for name, (estimator, grid) in model_definitions().items():
        if args.quick_demo and name in ("Random Forest", "XGBoost"):
            estimator.set_params(n_estimators=20)
            grid = {k: ([20] if k == "model__n_estimators" else v) for k, v in grid.items()}
        metrics, preds, folds, params = evaluate_nested_model(name, estimator, grid, X, y, outer_cv, inner_cv, quick_demo=args.quick_demo)
        all_metrics.append(metrics)
        all_preds.append(preds)
        all_folds.append(folds)
        all_params.append(params)
    args.output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(all_metrics).sort_values("Balanced_Accuracy", ascending=False)
    result.to_csv(args.output / "seven_models_comparison.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(args.output / "seven_models_patient_predictions.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(args.output / "outer_fold_performance.csv", index=False)
    pd.concat(all_params, ignore_index=True).to_csv(args.output / "best_hyperparameters_per_fold.csv", index=False)
    print("\n" + result.round(3).to_string(index=False))
    print(f"\nSaved CSV results in {args.output.resolve()}")


if __name__ == "__main__":
    main()
