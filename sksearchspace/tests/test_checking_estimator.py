import inspect
from pathlib import Path

from sksearchspace import SearchSpace


class CheckingEstimator:
    def __init__(self, a=None, b=True, c=False, random_state=None):
        self.a = a
        self.b = b
        self.c = c
        self.random_state = random_state


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

    for _ in range(40):
        # make sure the sampled paramters is a subset of the estimator
        # parameters
        sample_parameters = estimator_space.sample()
        assert set(sample_parameters) <= est_parameters

        assert sample_parameters['a'] in (None, 'cat')
        assert sample_parameters['b'] in (True, False)
        assert sample_parameters['c'] in (True, False, None)
        assert isinstance(sample_parameters['random_state'], int)
