# sksearchspace

![.github/workflows/ci.yml](https://github.com/thomasjpfan/sksearchspace/workflows/.github/workflows/ci.yml/badge.svg) [![codecov](https://codecov.io/gh/thomasjpfan/sksearchspace/branch/master/graph/badge.svg)](https://codecov.io/gh/thomasjpfan/sksearchspace)

Scikit-learn Search Space Configurations with curated search spaces for [scikit-learn estimators](http://github.com/scikit-learn/scikit-learn).

## Usage

### Auto Halving Search

`AutoHalvingRandomSearchCV` automatically generates search spaces and uses
successive halving to train a model:

```py
from sksearchspace import AutoHalvingRandomSearchCV

cat_prep = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='sk_missing')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

ct = ColumnTransformer([
    ('num', 'passthrough', make_column_selector(dtype_include=['number'])),
    ('cat', cat_prep, make_column_selector(dtype_include=['object', 'category']))
])
pipe = Pipeline(
    [('preprocess', ct),
     ('clf', HistGradientBoostingClassifier())]
)

X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

auto_halving = AutoHalvingRandomSearchCV(pipe, verbose=1, scoring='f1_macro')
auto_halving.fit(X_train, y_train)
```

### SearchSpace

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

A json file can be loaded as follows:

```py
with open("search_space.json", "r") as f:
    estimator_space = SearchSpace(f.read())
```

# License

Copyright (c) 2020 Thomas J. Fan

Distributed under the terms of the MIT license, pytest is free and open source software.
