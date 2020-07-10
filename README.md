# sksearchspace

![.github/workflows/ci.yml](https://github.com/thomasjpfan/sksearchspace/workflows/.github/workflows/ci.yml/badge.svg) [![codecov](https://codecov.io/gh/thomasjpfan/sksearchspace/branch/master/graph/badge.svg)](https://codecov.io/gh/thomasjpfan/sksearchspace)

Scikit-learn Search Space Configurations with curated search spaces for [scikit-learn estimators](http://github.com/scikit-learn/scikit-learn).

## Usage

```py
from sksearchspace import SearchSpace
from sklearn.tree import DecisionTreeClassifier

estimator_space = SearchSpace.for_sklearn_estimator(DecisionTreeClassifier, seed=42)
estimator_space.sample()
# {'criterion': 'entropy','min_samples_leaf': 15, 'min_samples_split': 11}

estimator_space.sample()
# {'criterion': 'entropy', 'min_samples_leaf': 12, 'min_samples_split': 4}
```

`sksearchspace` uses [ConfigSpace](https://automl.github.io/ConfigSpace/master/) for sampling. The `ConfigSpace` configuration can be accessed through an attribute:

```py
estimator_space.configuration
# Configuration space object:
# Hyperparameters:
#   criterion, Type: Categorical, Choices: {gini, entropy}, Default: gini
#   min_samples_leaf, Type: UniformInteger, Range: [1, 20], Default: 1
#   min_samples_split, Type: UniformInteger, Range: [2, 20], Default: 2
```

# License

Copyright (c) 2020 Thomas J. Fan

Distributed under the terms of the MIT license, pytest is free and open source software.
