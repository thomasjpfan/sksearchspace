"""Test supported for supported models"""
import random
import string

import numpy as np
import pytest
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import pairwise_distances
from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sksearchspace import SearchSpace
from sksearchspace._paths import ESTIMATOR_TO_PCS_PATH


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


def get_random_string(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def _enforce_estimator_tags_x(X, estimator, kernel=linear_kernel):
    if bool(getattr(estimator, "metric", None) == "precomputed"):
        return pairwise_distances(X, metric="euclidean")
    if bool(getattr(estimator, "_pairwise", False)):
        return kernel(X, X)
    if "dict" in estimator._get_tags()["X_types"]:
        names = [f"feat_{i}" for i in range(X.shape[1])]
        return [dict(zip(names, row)) for row in X]
    if "string" in estimator._get_tags()["X_types"]:
        return [get_random_string(8) for i in range(X.shape[0])]
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


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("Estimator", ESTIMATOR_TO_PCS_PATH)
def test_for_sklearn_estimator(Estimator):
    estimator_space = SearchSpace.for_sklearn_estimator(Estimator, seed=10)
    assert isinstance(estimator_space, SearchSpace)

    estimator = _construct_instance(Estimator)
    est_parameters = estimator.get_params()
    est_parameters_set = set(est_parameters)

    rng = np.random.RandomState(0)
    X = 3 * rng.uniform(size=(200, 15))
    y = X[:, 0].astype(int)
    y = _enforce_estimator_tags_y(estimator, y)
    X = _enforce_estimator_tags_x(X, estimator)

    iterations = 10
    sample_parameters_orig = estimator_space.sample()

    estimator.set_params(**sample_parameters_orig)
    # Does not fail
    try:
        estimator.fit(X, y)
    except Exception as e:
        raise AssertionError(
            f"failed with parameters {sample_parameters_orig}") from e

    for _ in range(iterations):
        # make sure the sampled paramters is a subset of the estimator
        # parameters
        sample_parameters = estimator_space.sample()

        # If the next sample is the sample then continue
        if sample_parameters == sample_parameters_orig:
            continue

        assert set(sample_parameters) <= est_parameters_set, \
               set(sample_parameters) - est_parameters_set

        estimator.set_params(**sample_parameters)
        # Does not fail
        try:
            estimator.fit(X, y)
        except Exception as e:
            raise AssertionError(
                f"failed with parameters {sample_parameters}") from e
