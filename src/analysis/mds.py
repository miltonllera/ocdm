# Author: Milton Montero  -- <lleramontero@gmail.com>

# This is an implementation of the technique described in:
# Sparse multidimensional scaling using landmark points
# http://graphics.stanford.edu/courses/cs468-05-winter/Papers/Landmarks/Silva_landmarks5.pdf

from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist
from numpy.random import choice
from sklearn.manifold import MDS


class LandmarkMDS(MDS):
    def __init__(
        self,
        n_components:int = 2,
        n_landmarks:int = 100,
        *,
        metric:bool = True,
        n_init:int = 4,
        max_iter: int = 300,
        verbose: int = 0,
        eps: float = 0.001,
        n_jobs: Optional[int] = None,
        random_state = None,
        dissimilarity:str = "euclidean"
    ):
        super().__init__(
            n_components=n_components,
            metric=metric,
            n_init=n_init,
            max_iter=max_iter,
            verbose=verbose,
            eps=eps,
            n_jobs=n_jobs,
            random_state=random_state,
            dissimilarity=dissimilarity
        )

        self.n_landmarks = n_landmarks

    def fit(self, X):
        X_land = X[choice(len(X), size=self.n_landmarks)]
        return super().fit(X_land)

    def fit_transform(self, X):
        X_land = X[choice(len(X), size=self.n_landmarks)]
        return super().fit_transform(X_land)

    def transform(self, X):
        return self.fit_transform(X)


class OOSLandmarkMDS:
    def __init__(
        self,
        n_components: int = 2,
        n_landmarks: int = 1000,
    ) -> None:
        """
        Initialise the Space and calculate active samples' positions.
        The DM must be strictly finite, real and symmetric. It's index must be
        equal to its keys.
        :param dm: a metric distance matrix
        """
        self.n_components = n_components
        self.n_landmarks = n_landmarks

    def fit(self, X):
        X = X[choice(len(X), size=self.n_landmarks)]
        distances = cdist(X, X)

        if not np.isfinite(distances).all():
            raise ValueError("all values in the distance matrix must be finite")

        if distances.diagonal().any():
            raise ValueError("a distance matrix can't have nonzero diagonal "
                             "elements")
        if not (distances == distances.T).all():
            raise ValueError("a distance matrix must be absolutely symmetric")

        # calculate coordinates for active samples
        M = X.shape[1]
        d = distances ** 2
        masses = np.identity(M) - np.full([M]*2, 1/M)
        s = -0.5 * (masses @ d @ masses.T)

        # decompose
        eigen = np.linalg.eigh(s)

        # drop negative eigenvalues and sort by eigenvalues in descending order
        ordering = np.where(eigen[0] > 0)[0][::-1]
        values = eigen[0][ordering]
        vectors = eigen[1][:, ordering]

        # compute and format coordinates
        coord = np.diag(np.repeat((1/M)**(-0.5), M)) @ vectors @ np.diag(values**0.5)

        # keep items required to project supplementary values
        self._active = coord
        self._d_act = d
        self._masses_act = masses
        self._values = values
        self._vectors = vectors
        self._explained = (values * 100 / values.sum()).round(3)

    def transform(self, X: np.ndarray, distances: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Project X into the space already fitted given some landmark points
        """
        if distances is not None:
            if len(distances) != len(X):
                raise ValueError(
                    "Distance matrix must have same number of elements as design matrix"
                )
        else:
            distances = cdist(X, X)

        if not np.isfinite(distances).all():
            raise ValueError("all values in the distance matrix must be finite")

        n_act = len(self.active)
        n_sup = distances.shape[0]
        d_sup = distances ** 2
        masses_sup = np.full((n_act, n_sup), (1 / n_act))
        s_sup = -0.5 * self._masses_act @ (d_sup.T - (self._d_act @ masses_sup))
        f_sup = s_sup.T @ self.active @ np.diag(self._values**-1)
        return f_sup[:,:self.n_components]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @property
    def explained(self) -> np.ndarray:
        """
        The fraction of variance explained by each dimension.
        :return: an array of floating point numbers
        """
        return self._explained.copy()

    @property
    def ndim(self) -> int:
        """
        The number of dimensions. There can be at most `len(keys)` dimensions,
        though it is not uncommon to get fewer dimensions. For example, due to
        numerical instability or poor metric estimations some dimensions might
        end up with near-zero negative eigenvalues and are consequently dropped.
        :return: an integer
        """
        return self._active.shape[1]

    @property
    def active(self) -> np.ndarray:
        """
        Active samples's coordinates.
        :return: a pandas data frame with coordinates of active samples,
        one sample per row; each column encodes a dimension in the Space;
        columns are sorted with respect to the fraction of variance explained
        by the corresponding dimensions in descending order
        """
        return self._active.copy()
