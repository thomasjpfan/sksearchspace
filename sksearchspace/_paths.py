from pathlib import Path

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import ExtraTreeRegressor

HERE = Path(__file__).parent

ESTIMATOR_TO_PCS_PATH = {

    # cluster
    AffinityPropagation: HERE / "cluster" / "AffinityPropagation.pcs_new",
    AgglomerativeClustering:
    HERE / "cluster" / "AgglomerativeClustering.pcs_new",

    # tree
    DecisionTreeClassifier: HERE / "tree" / "DecisionTreeClassifier.pcs_new",
    DecisionTreeRegressor: HERE / "tree" / "DecisionTreeRegressor.pcs_new",
    ExtraTreeClassifier: HERE / "tree" / "ExtraTreeClassifier.pcs_new",
    ExtraTreeRegressor: HERE / "tree" / "ExtraTreeRegressor.pcs_new"
}
