from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


class GaussianMixtureInfo:
    def __init__(
        self,
        n_components: int,
        means: np.ndarray,
        covariances: np.ndarray,
        weights: np.ndarray
    ) -> None:
        super().__init__()
        self._n_components = n_components
        self._means_ = means
        self._sqrt_covariances_ = covariances
        self._weights_ = weights

    @property
    def n_components(self) -> int:
        return self._n_components

    @property
    def means_(self) -> np.ndarray:
        return self._means_

    @property
    def sqrt_covariances_(self) -> np.ndarray:
        return self._sqrt_covariances_

    @property
    def weights_(self) -> np.ndarray:
        return self._weights_

    def set_params(self, means: np.ndarray, covariances: np.ndarray, weights: np.ndarray) -> None:
        self._means_ = means
        self._sqrt_covariances_ = covariances
        self._weights_ = weights

    @staticmethod
    def init(gmm: GaussianMixture) -> GaussianMixtureInfo:
        sqrt_covariances = np.zeros(gmm.covariances_.shape)
        for i in range(gmm.covariances_.shape[0]):
            sqrt_cov = np.linalg.cholesky(gmm.covariances_[i, :, :])
            sqrt_covariances[i, :, :] = sqrt_cov.T

        return GaussianMixtureInfo(gmm.n_components, gmm.means_, sqrt_covariances, gmm.weights_)
