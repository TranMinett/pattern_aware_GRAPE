"""
Training and evaluation utilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             matthews_corrcoef, roc_auc_score, f1_score)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.utils.class_weight import compute_class_weight

from .data import BipartiteGraphDataset, collate_graphs
from .models import GRAPEBase, GRAPEPatternAware


class MICEImputer:
    """Wrapper for MICE."""
    def __init__(self, max_iter=10, random_state=42):
        self.imp = IterativeImputer(max_iter=max_iter, random_state=random_state)

    def fit_transform(self, X):
        return self.imp.fit_transform(X)

    def transform(self, X):
        return self.imp.transform(X)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * batch.num_graphs
    return total / len(loader.dataset)


def evaluate(model, loader, device):
    """
    Evaluate model, returns dict of metrics.

    Primary metric is balanced_accuracy for imbalanced data.
    """
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            probs.append(F.softmax(out, dim=1).cpu())
            preds.append(out.argmax(dim=1).cpu())
            labels.append(batch.y.cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    probs = torch.cat(probs).numpy()

    res = {
        'accuracy': accuracy_score(labels, preds),
        'balanced_accuracy': balanced_accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(labels, preds),
    }
    try:
        if len(np.unique(labels)) > 1:
            if probs.shape[1] == 2:
                res['auc_roc'] = roc_auc_score(labels, probs[:, 1])
            else:
                res['auc_roc'] = roc_auc_score(labels, probs,
                                               multi_class='ovr', average='macro')
        else:
            res['auc_roc'] = np.nan
    except Exception:
        res['auc_roc'] = np.nan
    return res


def _get_imputer(method, seed):
    if 'Mean' in method:
        return SimpleImputer(strategy='mean')
    elif 'Median' in method:
        return SimpleImputer(strategy='median')
    elif 'KNN' in method:
        return KNNImputer(n_neighbors=5)
    elif 'MICE' in method:
        return MICEImputer(max_iter=10, random_state=seed)
    return None


def _get_model(method, n_feat, h_dim, n_cls, dropout, edge_drop):
    if method == 'GRAPE-NoPattern' or 'Impute' in method:
        return GRAPEBase(n_feat, h_dim, n_cls, dropout=dropout,
                         edge_dropout=edge_drop)
    enc_map = {
        'GRAPE-RandomPattern': 'random',
        'GRAPE-LearnedPattern': 'learned',
        'GRAPE-StatisticalPattern': 'statistical',
        'GRAPE-HierarchicalPattern': 'hierarchical',
    }
    if method in enc_map:
        return GRAPEPatternAware(n_feat, h_dim, n_cls,
                                 pattern_encoder=enc_map[method],
                                 dropout=dropout, edge_dropout=edge_drop)
    raise ValueError(f'unknown method: {method}')


def run_experiment(X_train, X_val, X_test, y_train, y_val, y_test,
                   method, device, cfg):
    """
    Run single experiment.

    Parameters
    ----------
    X_train, X_val, X_test : ndarray
        Feature matrices, may contain NaN.
    y_train, y_val, y_test : ndarray
        Labels.
    method : str
        Method name, e.g. 'GRAPE-LearnedPattern'.
    device : str
        'cuda' or 'cpu'.
    cfg : dict
        Config with hidden_dim, batch_size, patience, learning_rate,
        weight_decay, dropout, edge_dropout, epochs, seed.

    Returns
    -------
    dict : test metrics
    """
    n_cls = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    n_feat = X_train.shape[1]

    # store original masks
    mask_tr = np.isnan(X_train).astype(float)
    mask_va = np.isnan(X_val).astype(float)
    mask_te = np.isnan(X_test).astype(float)

    # imputation if needed
    if 'Impute' in method:
        imp = _get_imputer(method, cfg.get('seed', 42))
        X_tr = np.nan_to_num(imp.fit_transform(X_train), nan=0.0)
        X_va = np.nan_to_num(imp.transform(X_val), nan=0.0)
        X_te = np.nan_to_num(imp.transform(X_test), nan=0.0)
        mask_tr = np.zeros_like(X_tr)
        mask_va = np.zeros_like(X_va)
        mask_te = np.zeros_like(X_te)
    else:
        X_tr, X_va, X_te = X_train, X_val, X_test

    model = _get_model(method, n_feat, cfg['hidden_dim'], n_cls,
                       cfg['dropout'], cfg['edge_dropout'])

    ds_tr = BipartiteGraphDataset(X_tr, y_train, mask_tr)
    ds_va = BipartiteGraphDataset(X_va, y_val, mask_va)
    ds_te = BipartiteGraphDataset(X_te, y_test, mask_te)

    ld_tr = DataLoader(ds_tr, batch_size=cfg['batch_size'], shuffle=True,
                       collate_fn=collate_graphs)
    ld_va = DataLoader(ds_va, batch_size=cfg['batch_size'], shuffle=False,
                       collate_fn=collate_graphs)
    ld_te = DataLoader(ds_te, batch_size=cfg['batch_size'], shuffle=False,
                       collate_fn=collate_graphs)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'],
                           weight_decay=cfg['weight_decay'])

    # class weights
    try:
        uniq = np.unique(y_train)
        cw = compute_class_weight('balanced', classes=uniq, y=y_train)
        cw_full = np.ones(n_cls)
        for i, c in enumerate(uniq):
            cw_full[c] = cw[i]
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(cw_full).to(device))
    except Exception:
        criterion = nn.CrossEntropyLoss()

    best_metric, best_state, patience_ctr = 0, None, 0
    for _ in range(cfg['epochs']):
        train_epoch(model, ld_tr, opt, criterion, device)
        val_res = evaluate(model, ld_va, device)
        if val_res['balanced_accuracy'] > best_metric:
            best_metric = val_res['balanced_accuracy']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= cfg['patience']:
            break

    if best_state:
        model.load_state_dict(best_state)
        model.to(device)

    return evaluate(model, ld_te, device)


def prepare_splits(X, y, seed, test_size=0.2, val_size=0.2):
    """Stratified train/val/test split."""
    X_tv, X_te, y_tv, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=val_size / (1 - test_size),
        random_state=seed, stratify=y_tv)
    return X_tr, X_va, X_te, y_tr, y_va, y_te
