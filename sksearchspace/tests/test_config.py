import pytest

from sksearchspace import SearchSpace
from sksearchspace._config import check_none


def test_SearchSpace_raises_unrecognized():
    class BadEstimator:
        pass

    msg = "BadEstimator is not recognized"
    with pytest.raises(ValueError, match=msg):
        SearchSpace.for_sklearn_estimator(BadEstimator)


def test_check_none():
    assert check_none('None') is None
    assert check_none('cat') == 'cat'
