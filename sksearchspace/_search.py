from sklearn.model_selection._search_successive_halving import \
    BaseSuccessiveHalving
import numpy as np
from sklearn.utils.validation import check_random_state
from ._config import SearchSpace


class AutoHalvingRandomSearchCV(BaseSuccessiveHalving):
    def __init__(self,
                 estimator,
                 *,
                 n_candidates='exhaust',
                 factor=3,
                 resource='n_samples',
                 max_resources='auto',
                 min_resources='smallest',
                 aggressive_elimination=False,
                 cv=5,
                 scoring=None,
                 refit=True,
                 error_score=np.nan,
                 return_train_score=True,
                 random_state=None,
                 n_jobs=None,
                 verbose=0):
        super().__init__(estimator,
                         scoring=scoring,
                         n_jobs=n_jobs,
                         refit=refit,
                         verbose=verbose,
                         cv=cv,
                         random_state=random_state,
                         error_score=error_score,
                         return_train_score=return_train_score,
                         max_resources=max_resources,
                         resource=resource,
                         factor=factor,
                         min_resources=min_resources,
                         aggressive_elimination=aggressive_elimination)
        self.n_candidates = n_candidates

    def _generate_candidate_params(self):
        n_candidates_first_iter = self.n_candidates
        if n_candidates_first_iter == 'exhaust':
            # This will generate enough candidate so that the last iteration
            # uses as much resources as possible
            n_candidates_first_iter = (self.max_resources_ //
                                       self.min_resources_)
        rng = check_random_state(self.random_state)
        seed = rng.randint(2048)
        search_space = SearchSpace.for_sklearn_estimator(self.estimator,
                                                         seed=seed)
        return [search_space.sample() for _ in range(n_candidates_first_iter)]
