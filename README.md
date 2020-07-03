# sksearchspace

![.github/workflows/ci.yml](https://github.com/thomasjpfan/sksearchspace/workflows/.github/workflows/ci.yml/badge.svg) [![codecov](https://codecov.io/gh/thomasjpfan/sksearchspace/branch/master/graph/badge.svg)](https://codecov.io/gh/thomasjpfan/sksearchspace)

Scikit-learn Search Space Configurations with curated search spaces for [scikit-learn estimators](http://github.com/scikit-learn/scikit-learn).

## Usage

```py
from sksearchspace import get_estimator_space
from sklearn.tree import DecisionTreeClassifier

estimator_space = get_estimator_space(DecisionTreeClassifier, seed=42)
estimator_space.sample()
# {'criterion': 'entropy','min_samples_leaf': 15, 'min_samples_split': 11}
```

# License

Copyright (c) 2020 Thomas J. Fan

Distributed under the terms of the MIT license, pytest is free and open source software.
