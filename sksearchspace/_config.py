from importlib import import_module
import inspect
from enum import IntFlag

from ._paths import ESTIMATOR_TO_PCS_PATH

from ConfigSpace.read_and_write import json as configspace_json
import json


class Conversion(IntFlag):
    NULL = 0
    NONE = 1
    IMPORT = 2


def check_none(value):
    if value == "None":
        return None
    return value


def check_import(value):
    try:
        *module_str, name = value.split(".")
        module = import_module(".".join(module_str))
        return getattr(module, name)
    except Exception:
        return value


class SearchSpace:
    def __init__(self, file_str, seed=None):
        self.configuration = configspace_json.read(file_str)

        self.configuration.seed(seed=seed)
        # get comments to check for parameter conversions
        # paramters = set(self.configuration.get_hyperparameter_names())
        self.parameter_conversion = {}
        json_items = json.loads(file_str)
        hyperparameters = json_items["hyperparameters"]
        for hyperparameter in hyperparameters:
            name = hyperparameter["name"]
            convert_mask = Conversion.NULL
            converts = set(hyperparameter.get("converts", []))

            if "None" in converts:
                convert_mask |= Conversion.NONE
            if "import" in converts:
                convert_mask |= Conversion.IMPORT

            self.parameter_conversion[name] = convert_mask

    def sample(self):
        """Sample configuration."""
        sample = self.configuration.sample_configuration()
        sample = sample.get_dictionary()
        if not self.parameter_conversion:
            return sample

        # there are parameters to convert
        for param, conversion in self.parameter_conversion.items():
            value = sample[param]
            if Conversion.NONE in conversion:
                value = check_none(value)
            if Conversion.IMPORT in conversion:
                value = check_import(value)
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

        with pcs_path.open("r") as f:
            return cls(f.read(), seed=seed)
