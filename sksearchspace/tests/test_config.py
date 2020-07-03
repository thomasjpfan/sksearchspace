import pytest

from sksearchspace import get_estimator_space
from sksearchspace._config import check_bool
from sksearchspace._config import check_none


def test_get_estimator_space_raises_unrecognized():
    class BadEstimator:
        pass

    msg = "BadEstimator is not recognized"
    with pytest.raises(ValueError, match=msg):
        get_estimator_space(BadEstimator)


def test_get_estimator_space_raises_instances():
    class BadEstimator:
        pass

    msg = "estimator must be a class and not an instance"
    with pytest.raises(ValueError, match=msg):
        get_estimator_space(BadEstimator())


def test_check_bool():
    assert check_bool('True') is True
    assert check_bool('False') is False
    assert check_bool('cat') == 'cat'


def test_check_none():
    assert check_none('None') is None
    assert check_none('cat') == 'cat'
