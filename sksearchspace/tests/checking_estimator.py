from pathlib import Path
from .._config import EstimatorSpace
from .._config import Conversion


class CheckingEstimator:
    def __init__(self, a=None, b=True, c=False):
        self.a = a
        self.b = b
        self.c = c


class CheckingEstimatorSpace(EstimatorSpace):
    estimator_cls = CheckingEstimator

    @property
    def pcs_path(self):
        return Path(__file__).parent / "CheckingEstimator.pcs_new"

    @property
    def parameter_conversion(self):
        return {
            'a': Conversion.NONE,
            'b': Conversion.BOOL,
            'c': Conversion.BOOL | Conversion.NONE,
        }
