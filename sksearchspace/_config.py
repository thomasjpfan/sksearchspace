from abc import ABC, abstractmethod
import inspect
from pathlib import Path
import pkgutil
from importlib import import_module
from enum import IntFlag
from io import StringIO
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
            file_str = f.read()
        with StringIO(file_str) as f:
            self.configuration = pcs_new.read(f)

        self.configuration.seed(seed=seed)
        # get comments to check for parameter conversions
        paramters = set(self.configuration.get_hyperparameter_names())
        self.parameter_conversion = {}
        for line in file_str.split("\n"):
            if '|' in line:
                # conditional
                continue
            if line.startswith("{") and line.endswith("}"):
                # forbidden
                continue
            if "}" not in line and "]" not in line:
                continue
            if len(line.strip()) == 0:
                continue

            parameter_name = line.split(maxsplit=1)[0]
            if parameter_name not in paramters:
                continue

            # find comment
            pos = line.find("#")
            if pos == -1:
                # no comment
                continue

            comment = line[pos:].lower()
            none_in_comment = 'none' in comment
            bool_in_comment = 'bool' in comment

            if none_in_comment and bool_in_comment:
                self.parameter_conversion[parameter_name] = (Conversion.BOOL
                                                             | Conversion.NONE)
            elif none_in_comment:
                self.parameter_conversion[parameter_name] = Conversion.NONE
            elif bool_in_comment:
                self.parameter_conversion[parameter_name] = Conversion.BOOL

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
            sample[param] = value

        return sample

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
