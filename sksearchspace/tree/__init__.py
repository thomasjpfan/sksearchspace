from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from .._config import EstimatorSpace


class DecisionTreeClassifierSpace(EstimatorSpace):
    estimator_cls = DecisionTreeClassifier


class DecisionTreeRegressorSpace(EstimatorSpace):
    estimator_cls = DecisionTreeRegressor
