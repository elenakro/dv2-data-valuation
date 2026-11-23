import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


class SafeLogisticRegression(BaseEstimator, ClassifierMixin):


    def __init__(self, n_classes=2, max_iter=1000):
        self.n_classes = n_classes
        self.max_iter = max_iter

    def fit(self, X, y):
        self.classes_ = np.arange(self.n_classes)  # needed by sklearn scorers
        if len(y) == 0:
            self.model_ = DummyClassifier(strategy="constant", constant=0).fit([[0]], [0])
        elif len(np.unique(y)) < self.n_classes:
            self.model_ = DummyClassifier(strategy="most_frequent").fit(X, y)
        else:
            self.model_ = LogisticRegression(max_iter=self.max_iter).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)