"""Shared leakage-safe evaluation utilities for remaining tasks."""
import numpy as np
from src.readout.preprocessing import fit_standardizer, transform_standardizer
from src.readout.ridge import train_ridge_regression, predict_ridge_regression


def history_features(u, order):
    u = np.asarray(u, float).reshape(-1)
    X = np.zeros((len(u), order))
    for lag in range(order):
        X[lag:, lag] = u[:len(u)-lag]
    return X


def evaluate(X, y, train_idx, test_idx, task_type, ridge=1e-6):
    X = np.asarray(X, float)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) != len(y) or not np.isfinite(X).all():
        raise ValueError('Invalid feature matrix')
    mu, std = fit_standardizer(X[train_idx])
    A = transform_standardizer(X[train_idx], mu, std)
    B = transform_standardizer(X[test_idx], mu, std)
    if task_type == 'classification':
        classes = np.unique(y[train_idx])
        if len(classes)<2: raise ValueError('Training data must contain multiple classes')
        targets = (y[train_idx, None] == classes[None, :]).astype(float)
        W = train_ridge_regression(A, targets, ridge=ridge)
        pred_train = classes[np.argmax(predict_ridge_regression(A, W), axis=1)]
        pred_test = classes[np.argmax(predict_ridge_regression(B, W), axis=1)]
        from sklearn.metrics import confusion_matrix
        return {'train': float(np.mean(pred_train == y[train_idx])),
                'test': float(np.mean(pred_test == y[test_idx])),
                'confusion': confusion_matrix(y[test_idx], pred_test, labels=classes),
                'labels': classes}
    if task_type == 'regression':
        W = train_ridge_regression(A, y[train_idx], ridge=ridge)
        pred_train = np.asarray(predict_ridge_regression(A, W)).reshape(-1)
        pred_test = np.asarray(predict_ridge_regression(B, W)).reshape(-1)
        def score(target, pred):
            scale = np.std(target)
            return float(np.sqrt(np.mean((target-pred)**2))/scale) if scale>0 else float('nan')
        return {'train': score(y[train_idx], pred_train),
                'test': score(y[test_idx], pred_test)}
    raise ValueError('Unknown task type')


def sequential_indices(task):
    valid = np.arange(task.washout, len(task.u))
    n = int(len(valid)*task.train_ratio)
    return valid[:n], valid[n:]


def stratified_indices(y, train_ratio=0.8, seed=123):
    rng=np.random.default_rng(seed)
    tr=[]; te=[]
    for c in np.unique(y):
        ix=rng.permutation(np.flatnonzero(y==c))
        n=int(len(ix)*train_ratio)
        tr.extend(ix[:n]); te.extend(ix[n:])
    return np.array(tr,dtype=int),np.array(te,dtype=int)
