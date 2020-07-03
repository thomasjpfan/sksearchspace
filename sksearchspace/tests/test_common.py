"""Test supported for supported models"""
import inspect

import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sksearchspace import EstimatorSpace

from sksearchspace import get_estimator_space

SUPPORTED_ESTIMATORS = [DecisionTreeClassifier, DecisionTreeRegressor]


@pytest.mark.parametrize("Estimator", SUPPORTED_ESTIMATORS)
def test_get_estimator_space(Estimator):
    estimator_space = get_estimator_space(Estimator, seed=10)
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
