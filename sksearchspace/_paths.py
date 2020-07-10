from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

HERE = Path(__file__).parent

ESTIMATOR_TO_PCS_PATH = {
    DecisionTreeClassifier: HERE / "tree" / "DecisionTreeClassifier.pcs_new",
    DecisionTreeRegressor: HERE / "tree" / "DecisionTreeRegressor.pcs_new"
}
