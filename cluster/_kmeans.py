from sklearn.cluster._kmeans import KMeans, k_means, kmeans_plusplus, MiniBatchKMeans
from ..base import ModelParameterProxy


class KMeansParameterProxy(ModelParameterProxy):
    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4, verbose=0, random_state=None, copy_x=True, algorithm='auto'):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm
        
    def _make_estimator(self):
        estimator = KMeans(n_clusters=self.n_clusters, init=self.init, n_init=self.n_init, max_iter=self.max_iter, tol=self.tol, verbose=self.verbose, 
                           random_state=self.random_state, copy_x=self.copy_x, algorithm=self.algorithm)
        self.estimator = estimator
