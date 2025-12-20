"""Utility functions."""

import random
import numpy as np
import torch


def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def drop_allnan_cols(X, name=None):
    """Remove columns that are entirely NaN."""
    keep = ~np.all(np.isnan(X), axis=0)
    if not np.all(keep) and name:
        print(f'  dropping {(~keep).sum()} all-NaN columns from {name}')
    return X[:, keep], keep


def aggregate_seeds(results, metrics=None):
    """Aggregate results over seeds, returns mean/std."""
    if metrics is None:
        metrics = ['accuracy', 'balanced_accuracy', 'f1_macro',
                   'f1_weighted', 'mcc', 'auc_roc']
    agg = {}
    for m in metrics:
        vals = [r.get(m, np.nan) for r in results]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            agg[f'{m}_mean'] = np.mean(vals)
            agg[f'{m}_std'] = np.std(vals)
    return agg
