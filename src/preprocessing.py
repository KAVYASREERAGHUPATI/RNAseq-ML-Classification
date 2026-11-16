"""Fold-safe preprocessing for raw RNA-seq count matrices."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RNASeqPreprocessor(BaseEstimator, TransformerMixin):
    """Training-fold low-count filter followed by per-sample log2(CPM + 1).

    Library sizes are calculated from all original genes, before filtering.
    """

    def __init__(self, min_count=10, min_samples=5):
        self.min_count = min_count
        self.min_samples = min_samples

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.keep_genes_ = (X >= self.min_count).sum(axis=0) >= self.min_samples
        if not self.keep_genes_.any():
            raise ValueError("No genes passed expression filtering.")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        library_sizes = X.sum(axis=1)
        if np.any(library_sizes <= 0):
            raise ValueError("A sample has zero library size.")
        cpm = X[:, self.keep_genes_] / library_sizes[:, None] * 1_000_000
        return np.log2(cpm + 1)
