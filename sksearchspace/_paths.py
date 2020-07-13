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

from sklearn.covariance import EmpiricalCovariance
from sklearn.covariance import EllipticEnvelope
from sklearn.covariance import GraphicalLasso
from sklearn.covariance import GraphicalLassoCV
from sklearn.covariance import LedoitWolf
from sklearn.covariance import MinCovDet
from sklearn.covariance import OAS
from sklearn.covariance import ShrunkCovariance

from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import MiniBatchSparsePCA
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.dummy import DummyClassifier
from sklearn.dummy import DummyRegressor

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import ExtraTreeRegressor

HERE = Path(__file__).parent

MODULE_TO_ESTIMATORS = {
    "cluster": [
        AffinityPropagation, AgglomerativeClustering, Birch, DBSCAN,
        FeatureAgglomeration, KMeans, MiniBatchKMeans, MeanShift,
        SpectralClustering, SpectralBiclustering, SpectralCoclustering
    ],
    "cross_decomposition": [CCA, PLSCanonical, PLSRegression, PLSSVD],
    "covariance": [
        EmpiricalCovariance, EllipticEnvelope, GraphicalLasso,
        GraphicalLassoCV, LedoitWolf, MinCovDet, OAS, ShrunkCovariance
    ],
    "decomposition": [
        DictionaryLearning, FactorAnalysis, FastICA, IncrementalPCA, KernelPCA,
        LatentDirichletAllocation, MiniBatchDictionaryLearning,
        MiniBatchDictionaryLearning, MiniBatchSparsePCA, NMF, PCA, SparsePCA,
        TruncatedSVD
    ],
    "discriminant_analysis":
    [LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis],
    "dummy": [DummyClassifier, DummyRegressor],
    "tree": [
        DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier,
        ExtraTreeRegressor
    ],
}

ESTIMATOR_TO_PCS_PATH = {}
for module, estimators in MODULE_TO_ESTIMATORS.items():
    for estimator in estimators:
        name = estimator.__name__
        ESTIMATOR_TO_PCS_PATH[estimator] = HERE / module / f"{name}.pcs_new"
