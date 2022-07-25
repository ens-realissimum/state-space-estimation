from typing import Union

import bayesian_framework.inference.stochastic_models.stochastic_models as sm
from bayesian_framework.inference.stochastic_models.covariance_type import CovarianceType
from bayesian_framework.core.covariance_utils import to_covariance_with_type

GaussNoise = Union[sm.GaussianStochasticModel, sm.GaussianMixtureStochasticModel]


def convert_to_gaussian(stochastic_model: sm.GeneralStochasticModel, target_cov_type: CovarianceType) -> GaussNoise:
    """
    Convert stochastic process (stochastic_model) to Gaussian stochastic process (single or mixture) with
    specified covariance type (target_cov_type). Target process has same mean and covariance
    (but covariance is converted to requested type) as source process.
    Always creates new instance of converted stochastic process.
    :param stochastic_model: model of source stochastic process
    :param target_cov_type: covariance type of target Gaussian process
    :return: Gaussian process (single or mixture) created based on mean and covariance of source process.
    """
    cov = to_covariance_with_type(stochastic_model.covariance, stochastic_model.covariance_type, target_cov_type)

    if isinstance(stochastic_model, sm.GaussianMixtureStochasticModel):
        return sm.GaussianMixtureStochasticModel(
            mixture_size=stochastic_model.n_components,
            mean=stochastic_model.mean,
            covariance=cov,
            weights=stochastic_model.weights,
            covariance_type=target_cov_type
        )

    return sm.GaussianStochasticModel(mean=stochastic_model.mean, covariance=cov, covariance_type=target_cov_type)


def convert_to_gmm(stochastic_model: sm.GeneralStochasticModel, target_cov_type: CovarianceType) -> sm.GaussianMixtureStochasticModel:
    if isinstance(stochastic_model, sm.GaussianMixtureStochasticModel):
        return convert_to_gaussian(stochastic_model, target_cov_type)

    return sm.GaussianMixtureStochasticModel.from_gaussian(convert_to_gaussian(stochastic_model, target_cov_type))
