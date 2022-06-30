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
        self._covariances_ = covariances
        self._weights_ = weights

    @property
    def n_components(self) -> int:
        return self._n_components

    @property
    def means_(self) -> np.ndarray:
        return self._means_

    @property
    def covariances_(self) -> np.ndarray:
        return self._covariances_

    @property
    def weights_(self) -> np.ndarray:
        return self._weights_

    def set_params(self, means: np.ndarray, covariances: np.ndarray, weights: np.ndarray) -> None:
        self._means_ = means
        self._covariances_ = covariances
        self._weights_ = weights

    @staticmethod
    def init(gmm: GaussianMixture) -> GaussianMixtureInfo:
        return GaussianMixtureInfo(gmm.n_components, gmm.means_, gmm.covariances_, gmm.weights_)
