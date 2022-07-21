from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from bayesian_framework.inference.stochastic_models.covariance_type import CovarianceType
from bayesian_framework.shared.covariance_utils import to_covariance_with_type


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

    def set_params(
            self,
            means: np.ndarray,
            covariances: np.ndarray,
            weights: np.ndarray,
            cov_type: CovarianceType
    ) -> None:
        self._means_ = means
        self._sqrt_covariances_ = to_covariance_with_type(covariances, cov_type, CovarianceType.sqrt)
        self._weights_ = weights

    @staticmethod
    def init(gmm: GaussianMixture) -> GaussianMixtureInfo:
        sqrt_covariances = np.linalg.cholesky(gmm.covariances_, )
        return GaussianMixtureInfo(gmm.n_components, gmm.means_, sqrt_covariances, gmm.weights_)
