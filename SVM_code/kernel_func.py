
import numpy as np


def linear():
    def _linear(x_j, y_j):
        return np.dot(x_j, y_j)
    return _linear


def poly(coef = 1.0, degree = 3):
    def _poly(x_j, y_j):
        return np.power(np.dot(x_j, y_j)+coef, degree)
    return _poly

def rbf(gamma = 1.0):
    def _rbf(x, y):
        if len(x.shape) == 1 and len(y.shape) == 1:
            return np.exp(-np.dot(x - y, x - y) / (2 * gamma ** 2))
        return np.exp(- np.sum((x - y) ** 2, axis=1) / (2 * gamma ** 2))
    return _rbf