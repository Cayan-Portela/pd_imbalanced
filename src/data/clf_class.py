from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression

class ClfSwitcher(ClassifierMixin, BaseEstimator):

    def __init__(
        self, 
        estimator = LogisticRegression(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 

        self.estimator = estimator


    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self


    def predict(self, X, y=None):
        return self.estimator.predict(X)


    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


    def score(self, X, y):
        return self.estimator.score(X, y)