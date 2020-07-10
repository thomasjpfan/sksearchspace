"""Test supported for supported models"""
from pathlib import Path
import inspect

import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sksearchspace import EstimatorSpace

SUPPORTED_ESTIMATORS = [DecisionTreeRegressor, DecisionTreeClassifier]


@pytest.mark.parametrize("Estimator", SUPPORTED_ESTIMATORS)
def test_for_sklearn_estimator(Estimator):
    estimator_space = EstimatorSpace.for_sklearn_estimator(Estimator, seed=10)
    assert isinstance(estimator_space, EstimatorSpace)
    config = estimator_space.configuration

    # check that configuration parameters are a valid hyperparameter in
    # Estimator
    config_parameters = set(config.get_hyperparameter_names())
    est_parameters = set(inspect.signature(Estimator).parameters)
    assert config_parameters <= est_parameters

    for _ in range(10):
        # make sure the sampled paramters is a subset of the estimator
        # parameters
        sample_parameters = set(estimator_space.sample())
        est_parameters = set(inspect.signature(Estimator).parameters)
        assert sample_parameters <= est_parameters


class CheckingEstimator:
    def __init__(self, a=None, b=True, c=False):
        self.a = a
        self.b = b
        self.c = c


def test_checking_estimator_space():
    pcs_path = Path(__file__).parent / "CheckingEstimator.pcs_new"
    with pcs_path.open('r') as f:
        estimator_space = EstimatorSpace(f.read(), seed=42)
    assert isinstance(estimator_space, EstimatorSpace)
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
