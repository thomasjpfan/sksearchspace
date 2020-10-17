from contextlib import suppress
from sklearn.base import BaseEstimator
from ._config import SearchSpace


class _MetaSearchSpace:
    def sample(self):
        output = {}
        for name, space in self.search_spaces.items():
            sampled_params = space.sample()
            output.update({
                f"{name}__{inner_name}": value
                for inner_name, value in sampled_params.items()
            })
        return output


class ColumnTransformerSearchSpace(_MetaSearchSpace):
    def __init__(self, column_transformer, seed=None):
        self.search_spaces = {}
        for name, est, _ in column_transformer.transformers:
            if not isinstance(est, BaseEstimator):
                continue

            with suppress(ValueError):
                self.search_spaces[name] = \
                    SearchSpace.for_sklearn_estimator(est)


class PipelineSearchSpace(_MetaSearchSpace):
    def __init__(self, pipeline, seed=None):
        self.search_spaces = {}
        for name, est in pipeline.steps:
            if not isinstance(est, BaseEstimator):
                continue
            with suppress(ValueError):
                self.search_spaces[name] = \
                    SearchSpace.for_sklearn_estimator(est)
