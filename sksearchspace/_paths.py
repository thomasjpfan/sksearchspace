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

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomTreesEmbedding

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import Lars
from sklearn.linear_model import LarsCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.linear_model import GammaRegressor
from sklearn.linear_model import PassiveAggressiveRegressor

from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_approximation import SkewedChi2Sampler

from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE

from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsTransformer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsTransformer
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NeighborhoodComponentsAnalysis

from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading

from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVC
from sklearn.svm import NuSVR
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
from sklearn.svm import SVR

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import ExtraTreeRegressor

HERE = Path(__file__).parent

MODULE_TO_ESTIMATORS = {
    "cluster": [
        AffinityPropagation,
        AgglomerativeClustering,
        Birch,
        DBSCAN,
        FeatureAgglomeration,
        KMeans,
        MiniBatchKMeans,
        MeanShift,
        SpectralClustering,
        SpectralBiclustering,
        SpectralCoclustering,
    ],
    "cross_decomposition": [CCA, PLSCanonical, PLSRegression, PLSSVD],
    "covariance": [
        EmpiricalCovariance,
        EllipticEnvelope,
        GraphicalLasso,
        GraphicalLassoCV,
        LedoitWolf,
        MinCovDet,
        OAS,
        ShrunkCovariance,
    ],
    "decomposition": [
        DictionaryLearning,
        FactorAnalysis,
        FastICA,
        IncrementalPCA,
        KernelPCA,
        LatentDirichletAllocation,
        MiniBatchDictionaryLearning,
        MiniBatchDictionaryLearning,
        MiniBatchSparsePCA,
        NMF,
        PCA,
        SparsePCA,
        TruncatedSVD,
    ],
    "discriminant_analysis": [
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    ],
    "dummy": [DummyClassifier, DummyRegressor],
    "ensemble": [
        AdaBoostClassifier,
        AdaBoostRegressor,
        BaggingClassifier,
        BaggingRegressor,
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        IsolationForest,
        RandomForestClassifier,
        RandomForestRegressor,
        RandomTreesEmbedding,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    ],
    "feature_extraction": [DictVectorizer, FeatureHasher],
    "feature_extraction.text": [
        CountVectorizer,
        HashingVectorizer,
        TfidfTransformer,
        TfidfVectorizer,
    ],
    "feature_selection": [
        GenericUnivariateSelect,
        SelectPercentile,
        SelectFpr,
        SelectFdr,
        SelectFromModel,
        SelectFwe,
        RFE,
        RFECV,
        VarianceThreshold,
    ],
    "gaussian_process": [GaussianProcessClassifier, GaussianProcessRegressor],
    "impute": [SimpleImputer, IterativeImputer, KNNImputer],
    "kernel_ridge": [KernelRidge],
    "linear_model": [
        LogisticRegression,
        LogisticRegressionCV,
        PassiveAggressiveClassifier,
        Perceptron,
        RidgeClassifier,
        RidgeClassifierCV,
        SGDClassifier,
        LinearRegression,
        Ridge,
        RidgeCV,
        SGDRegressor,
        ElasticNetCV,
        Lars,
        LarsCV,
        Lasso,
        LassoCV,
        LassoLars,
        LassoLarsCV,
        LassoLarsIC,
        OrthogonalMatchingPursuit,
        OrthogonalMatchingPursuitCV,
        ARDRegression,
        BayesianRidge,
        HuberRegressor,
        RANSACRegressor,
        TheilSenRegressor,
        PoissonRegressor,
        TweedieRegressor,
        GammaRegressor,
        PassiveAggressiveRegressor,
    ],
    "kernel_approximation": [
        AdditiveChi2Sampler,
        Nystroem,
        RBFSampler,
        SkewedChi2Sampler,
    ],
    "manifold": [Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE],
    "mixture": [BayesianGaussianMixture, GaussianMixture],
    "naive_bayes": [
        BernoulliNB,
        CategoricalNB,
        ComplementNB,
        GaussianNB,
        MultinomialNB,
    ],
    "neighbors": [
        KNeighborsClassifier,
        KNeighborsRegressor,
        KNeighborsTransformer,
        LocalOutlierFactor,
        RadiusNeighborsClassifier,
        RadiusNeighborsClassifier,
        RadiusNeighborsRegressor,
        RadiusNeighborsTransformer,
        NearestCentroid,
        NearestNeighbors,
        NeighborhoodComponentsAnalysis,
    ],
    "neural_network": [BernoulliRBM, MLPClassifier, MLPRegressor],
    "preprocessing": [
        Binarizer,
        KBinsDiscretizer,
        MaxAbsScaler,
        MaxAbsScaler,
        MinMaxScaler,
        Normalizer,
        OneHotEncoder,
        OrdinalEncoder,
        PolynomialFeatures,
        PowerTransformer,
        QuantileTransformer,
        RobustScaler,
        StandardScaler,
    ],
    "semi_supervised": [LabelPropagation, LabelSpreading],
    "svm": [LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR],
    "tree": [
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        ExtraTreeClassifier,
        ExtraTreeRegressor,
    ],
}

ESTIMATOR_TO_PCS_PATH = {}
for module, estimators in MODULE_TO_ESTIMATORS.items():
    module_folder = HERE.joinpath(*module.split("."))
    for estimator in estimators:
        name = estimator.__name__
        ESTIMATOR_TO_PCS_PATH[estimator] = module_folder / f"{name}.json"
