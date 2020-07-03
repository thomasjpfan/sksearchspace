from abc import ABC, abstractmethod
import inspect
from pathlib import Path
import pkgutil
from importlib import import_module
from enum import IntFlag
import warnings

with warnings.catch_warnings():
    # ignore warning from pyparsing
    warnings.filterwarnings("ignore", category=FutureWarning)
    from ConfigSpace.read_and_write import pcs_new


class Conversion(IntFlag):
    NONE = 1
    BOOL = 2


def check_bool(value):
    if value == 'True':
        return True
    elif value == 'False':
        return False
    return value


def check_none(value):
    if value == 'None':
        return None
    return value


class EstimatorSpace(ABC):
    def __init__(self, seed=None):
        with self.pcs_path.open('r') as f:
            self.configuration = pcs_new.read(f)
        self.configuration.seed(seed=seed)

    def sample(self):
        """Sample configuration."""
        sample = self.configuration.sample_configuration()
        sample = sample.get_dictionary()
        if not self.parameter_conversion:
            return sample

        # there are parameters to convert
        for param, conversion in self.parameter_conversion.items():
            value = sample[param]
            if Conversion.BOOL in conversion:
                value = check_bool(value)
            if Conversion.NONE in conversion:
                value = check_none(value)
            value = sample[param]

        return sample

    @property
    def parameter_conversion(self):
        """Dictionary mapping for parametres that need to be converted to a
        python builtin type such as a bool or None.

        For example: {'max_depth': Conversion.NONE,
                      'copy': Conversion.BOOL,
                      'both': Conversion.BOOL | Converstion.NONE}
        """
        return {}

    @property
    @abstractmethod
    def estimator_cls(self):
        """Class of estimator"""

    @property
    def pcs_path(self):
        """Path to pcs file"""
        submodule = self.__module__.split('.')[-1]
        estimator_name = self.estimator_cls.__name__ + ".pcs_new"
        return Path(__file__).parent / submodule / estimator_name


# Go through packages to find estimator spaces
_estimator_spaces = {}

root = str(Path(__file__).parent)
for importer, modname, ispkg in pkgutil.walk_packages(path=[root],
                                                      prefix='sksearchspace.'):
    module = import_module(modname)
    classes = inspect.getmembers(module, inspect.isclass)

    for name, cur_class in classes:
        if name != 'EstimatorSpace' and issubclass(cur_class, EstimatorSpace):
            _estimator_spaces[cur_class.estimator_cls] = cur_class


def get_estimator_space(Estimator, seed=None):
    """Get configspace representation of configuration."""
    if not inspect.isclass(Estimator):
        raise ValueError("estimator must be a class and not an instance")
    try:
        return _estimator_spaces[Estimator](seed=seed)
    except KeyError:
        raise ValueError(f"{Estimator.__name__} is not recognized")
