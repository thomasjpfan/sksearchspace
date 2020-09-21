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
    CHOICE = 3


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
            if "choice" in converts:
                convert_mask |= Conversion.CHOICE

            self.parameter_conversion[name] = convert_mask

    def sample(self):
        """Sample configuration."""
        sample = self.configuration.sample_configuration()
        sample = sample.get_dictionary()
        return self.normalize_config_space_dict(sample)

    def normalize_config_space_dict(self, config_dict):
        if not self.parameter_conversion:
            return config_dict

        config_dict = config_dict.copy()
        config_dict_keys = list(config_dict.keys())
        # there are parameters to convert
        for param in config_dict_keys:
            value = config_dict[param]
            conversion = self.parameter_conversion[param]
            if Conversion.NONE in conversion:
                value = check_none(value)
            if Conversion.IMPORT in conversion:
                value = check_import(value)
            config_dict[param] = value

        keys_to_remove = []
        for param in config_dict_keys:
            value = config_dict[param]
            conversion = self.parameter_conversion[param]
            if Conversion.CHOICE in conversion:
                choice = config_dict[param]
                config_dict[param] = config_dict[choice]
                keys_to_remove.append(choice)

        for keys in keys_to_remove:
            del config_dict[keys]

        return config_dict

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
