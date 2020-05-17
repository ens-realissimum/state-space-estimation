import numpy as np
from numpy.random.mtrand import uniform, random

from bayesian_framework.inference.stochastic_models import stochastic_models as sm
from bayesian_framework.inference.stochastic_models.covariance_type import CovarianceType


def get_shape_scale_pairs():
    return zip(uniform(low=0, high=100, size=(200,)), uniform(low=0, high=100, size=(200,)))


def build_gmm(dim: int = 2) -> sm.GaussianMixtureStochasticModel:
    shape = (dim, dim)
    cov_sqrt_1 = np.random.random(shape)
    cov_sqrt_2 = np.random.random(shape)
    cov_sqrt_3 = np.random.random(shape)
    mean_1 = np.random.random(dim)
    mean_2 = np.random.random(dim)
    mean_3 = np.random.random(dim)

    return sm.GaussianMixtureStochasticModel(
        mixture_size=3,
        mean=[mean_1, mean_2, mean_3],
        covariance=[cov_sqrt_1, cov_sqrt_2, cov_sqrt_3],
        covariance_type=sm.CovarianceType.sqrt
    )


def build_gamma() -> sm.GammaStochasticModel:
    return sm.GammaStochasticModel(shape=random(), scale=random())


def build_gauss(dim: int = 3, covariance_type: CovarianceType = CovarianceType.sqrt_diag) -> sm.GaussianStochasticModel:
    return sm.GaussianStochasticModel(
        mean=np.random.random(dim),
        covariance=np.random.random((dim, dim)),
        covariance_type=covariance_type
    )