"""Test supported for supported models"""
from pathlib import Path
import inspect
import numpy as np

import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import pairwise_distances
from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sksearchspace import SearchSpace

SUPPORTED_ESTIMATORS = [DecisionTreeRegressor, DecisionTreeClassifier]


def _construct_instance(Estimator):
    """Construct Estimator instance if possible"""
    required_parameters = getattr(Estimator, "_required_parameters", [])
    if len(required_parameters):
        if required_parameters in (["estimator"], ["base_estimator"]):
            if issubclass(Estimator, RegressorMixin):
                estimator = Estimator(Ridge())
            else:
                estimator = Estimator(LinearDiscriminantAnalysis())
        else:
            pytest.skip("Can't instantiate estimator "
                        f"{Estimator.__name__} which requires "
                        f"parameters {required_parameters}")
    else:
        estimator = Estimator()
    return estimator


def _pairwise_estimator_convert_X(X, estimator, kernel=linear_kernel):
    if bool(getattr(estimator, "metric", None) == 'precomputed'):
        return pairwise_distances(X, metric='euclidean')
    if bool(getattr(estimator, "_pairwise", False)):
        return kernel(X, X)

    return X


def _enforce_estimator_tags_y(estimator, y):
    # Estimators with a `requires_positive_y` tag only accept strictly positive
    # data
    if estimator._get_tags()["requires_positive_y"]:
        # Create strictly positive y. The minimal increment above 0 is 1, as
        # y could be of integer dtype.
        y += 1 + abs(y.min())
    # Estimators with a `binary_only` tag only accept up to two unique y values
    if estimator._get_tags()["binary_only"] and y.size > 0:
        y = np.where(y == y.flat[0], y, y.flat[0] + 1)
    # Estimators in mono_output_task_error raise ValueError if y is of 1-D
    # Convert into a 2-D y for those estimators.
    if estimator._get_tags()["multioutput_only"]:
        return np.reshape(y, (-1, 1))
    return y


@pytest.mark.parametrize("Estimator", SUPPORTED_ESTIMATORS)
def test_for_sklearn_estimator(Estimator):
    estimator_space = SearchSpace.for_sklearn_estimator(Estimator, seed=10)
    assert isinstance(estimator_space, SearchSpace)
    config = estimator_space.configuration

    # check that configuration parameters are a valid hyperparameter in
    # Estimator
    config_parameters = set(config.get_hyperparameter_names())
    est_parameters = set(inspect.signature(Estimator).parameters)
    assert config_parameters <= est_parameters

    estimator = _construct_instance(Estimator)
    rng = np.random.RandomState(0)
    X = 3 * rng.uniform(size=(20, 5))
    X = _pairwise_estimator_convert_X(X, estimator)
    y = X[:, 0].astype(int)

    est_parameters = set(inspect.signature(Estimator).parameters)
    for _ in range(10):
        # make sure the sampled paramters is a subset of the estimator
        # parameters
        sample_parameters = estimator_space.sample()
        assert set(sample_parameters) <= est_parameters

        estimator.set_params(**sample_parameters)
        # Does not fail
        estimator.fit(X, y)


class CheckingEstimator:
    def __init__(self, a=None, b=True, c=False):
        self.a = a
        self.b = b
        self.c = c


def test_checking_estimator_space():
    pcs_path = Path(__file__).parent / "CheckingEstimator.pcs_new"
    with pcs_path.open('r') as f:
        estimator_space = SearchSpace(f.read(), seed=42)
    assert isinstance(estimator_space, SearchSpace)
    config = estimator_space.configuration

    # check that configuration parameters are a valid hyperparameter in
    # Estimator
    config_parameters = set(config.get_hyperparameter_names())
    est_parameters = set(inspect.signature(CheckingEstimator).parameters)
    assert config_parameters <= est_parameters

    for _ in range(20):
        # make sure the sampled paramters is a subset of the estimator
        # parameters
        sample_parameters = estimator_space.sample()
        est_parameters = set(inspect.signature(CheckingEstimator).parameters)
        assert set(sample_parameters) <= est_parameters

        assert sample_parameters['a'] in (None, 'cat')
        assert sample_parameters['b'] in (True, False)
        assert sample_parameters['c'] in (True, False, None)
