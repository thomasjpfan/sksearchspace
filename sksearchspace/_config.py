import inspect
from enum import IntFlag
from io import StringIO
import warnings

from ._paths import ESTIMATOR_TO_PCS_PATH

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


class SearchSpace:
    def __init__(self, file_str, seed=None):
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

    @classmethod
    def for_sklearn_estimator(cls, Estimator, seed=None):
        if not inspect.isclass(Estimator):
            raise ValueError("estimator must be a class and not an instance")
        try:
            pcs_path = ESTIMATOR_TO_PCS_PATH[Estimator]
        except KeyError:
            raise ValueError(f"{Estimator.__name__} is not recognized")

        with pcs_path.open('r') as f:
            return cls(f.read(), seed=seed)
