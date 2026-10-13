import numpy as np


def one_hot_encode(y, n_classes):
    return np.eye(n_classes)[y]


def predict_classes(scores):
    return np.argmax(scores, axis=1)