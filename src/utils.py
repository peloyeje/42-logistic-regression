import csv
import pathlib

import numpy as np

class OneHotEncoder:
    def __init__(self):
        self.categories = []

    def fit(self, X):
        """"""
        self.categories = np.unique(X)
        return X

    def transform(self, X):
        """"""
        return np.array(
            [(X == k).astype('int') for k in self.categories]
        ).T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
