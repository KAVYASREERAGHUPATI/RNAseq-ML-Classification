"""Predict from a saved model using raw count data with matching gene IDs."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts', type=Path, required=True, help='Raw gene-count TSV, genes x samples')
    p.add_argument('--model', type=Path, default=Path('models/best_model.joblib'))
    p.add_argument('--output', type=Path, default=Path('results/new_sample_predictions.csv'))
    a = p.parse_args()
    # Only load joblib artifacts you created or trust: pickle-based formats can execute code.
    artifact = joblib.load(a.model)
    counts = pd.read_csv(a.counts, sep='\t', index_col=0)
    if counts.index.duplicated().any() or counts.columns.duplicated().any():
        raise ValueError('Duplicate gene or sample IDs')
    expected = artifact['gene_ids']
    missing = set(expected) - set(counts.index)
    if missing:
        raise ValueError(f'Missing {len(missing)} genes expected by saved model')
    X = counts.loc[expected].T.apply(pd.to_numeric, errors='raise')
    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all() or (X < 0).any().any():
        raise ValueError('Counts must be finite and nonnegative')
    if not np.all(np.equal(np.mod(X.to_numpy(), 1), 0)):
        raise ValueError('Counts must be raw integers')
    model = artifact['pipeline']
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, list(model.classes_).index(1)]
    out = pd.DataFrame({'Sample_ID': X.index, 'Predicted_Label':
                        np.where(pred == 1, 'Responder', 'NonResponder'),
                        'Responder_Probability': prob})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(f'Saved predictions: {a.output}')


if __name__ == '__main__':
    main()
