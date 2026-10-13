import numpy as np


def fit_standardizer(X_train, eps=1e-12):

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std[std < eps] = 1.0

    return mean, std


def transform_standardizer(X, mean, std):

    return (X - mean) / std