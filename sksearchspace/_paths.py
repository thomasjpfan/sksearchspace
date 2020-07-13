from pathlib import Path

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import SpectralBiclustering
from sklearn.cluster import SpectralCoclustering

from sklearn.cross_decomposition import CCA
from sklearn.cross_decomposition import PLSCanonical
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_decomposition import PLSSVD

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
    Birch: HERE / "cluster" / "Birch.pcs_new",
    DBSCAN: HERE / "cluster" / "DBSCAN.pcs_new",
    FeatureAgglomeration: HERE / "cluster" / "FeatureAgglomeration.pcs_new",
    KMeans: HERE / "cluster" / "KMeans.pcs_new",
    MiniBatchKMeans: HERE / "cluster" / "MiniBatchKMeans.pcs_new",
    MeanShift: HERE / "cluster" / "MeanShift.pcs_new",
    SpectralClustering: HERE / "cluster" / "SpectralClustering.pcs_new",
    SpectralBiclustering: HERE / "cluster" / "SpectralBiclustering.pcs_new",
    SpectralCoclustering: HERE / "cluster" / "SpectralCoclustering.pcs_new",

    # cross_decomposition
    CCA: HERE / "cross_decomposition" / "CCA.pcs_new",
    PLSCanonical: HERE / "cross_decomposition" / "PLSCanonical.pcs_new",
    PLSRegression: HERE / "cross_decomposition" / "PLSRegression.pcs_new",
    PLSSVD: HERE / "cross_decomposition" / "PLSSVD.pcs_new",

    # tree
    DecisionTreeClassifier: HERE / "tree" / "DecisionTreeClassifier.pcs_new",
    DecisionTreeRegressor: HERE / "tree" / "DecisionTreeRegressor.pcs_new",
    ExtraTreeClassifier: HERE / "tree" / "ExtraTreeClassifier.pcs_new",
    ExtraTreeRegressor: HERE / "tree" / "ExtraTreeRegressor.pcs_new"
}
