from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import NoReturn, Optional

import numpy as np

import bayesian_framework.inference.stochastic_models.stochastic_models as sm
from bayesian_framework.inference.stochastic_models.stochastic_models import GaussianStochasticModel, GeneralStochasticModel
from bayesian_framework.core.covariance_adapdation import CovarianceMatrixAdaptation
from bayesian_framework.core.linearization_type import LinearizationType


class StateSpaceModel(ABC):
    """
    Model which allows to formulate filtration issue in bayesian estimation terminology.
    """

    def __init__(
            self,
            state_noise: GeneralStochasticModel,
            observation_noise: GeneralStochasticModel,
            reconcile_strategy: CovarianceMatrixAdaptation = None
    ):
        self._x_noise = deepcopy(state_noise)
        self._z_noise = deepcopy(observation_noise)
        self._reconcile_strategy = reconcile_strategy

    @property
    @abstractmethod
    def type(self) -> str:
        return 'General state space model'

    @property
    @abstractmethod
    def tag(self) -> str:
        return ''

    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def control_state_dim(self) -> Optional[int]:
        return None

    @property
    @abstractmethod
    def control_observation_dim(self) -> Optional[int]:
        return None

    @property
    def state_noise_dimension(self) -> int:
        return self._x_noise.dim

    @property
    def observation_noise_dimension(self) -> int:
        return self._z_noise.dim

    @property
    def state_noise(self) -> GeneralStochasticModel:
        return self._x_noise

    @property
    def observation_noise(self) -> GeneralStochasticModel:
        return self._z_noise

    @property
    def reconcile_strategy(self) -> CovarianceMatrixAdaptation:
        return self._reconcile_strategy

    @abstractmethod
    def transition_func(
            self,
            state: np.ndarray,
            state_noise: Optional[np.ndarray] = None,
            control_vect: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        State transition function. Calculate (predict) value of state space vector at time = k based on value of
        state space vector at time = k-1.
        :return: value of state space vector
        """
        pass

    @abstractmethod
    def observation_func(
            self,
            state: np.ndarray,
            observation_noise: Optional[np.ndarray] = None,
            control_vect: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Observation function (mapping state to observation)
        :return: observation calculated based on value of state space vector at time = k.
        """
        pass

    @abstractmethod
    def prior(
            self,
            next_state: np.ndarray,
            state: np.ndarray,
            control_x: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Function to calculate that calculates P(x(k)|x(k-1)), ie probability that value of state space vector
        at time = k equal to 'x(k)' if at previous time the value has been equal to x(k-1).
        :return: probability that value of state space vector at time = k equal to 'x(k)' if at previous time
        the value has been equal to x(k-1).
        """
        pass

    @abstractmethod
    def likelihood(
            self,
            observation: np.ndarray,
            state: np.ndarray,
            control_z: Optional[np.ndarray] = None
    ) -> np.matrix:
        """
        Function to calculate the observation likelihood function that calculates p(z(k)|x(k))
        :return: calculated likelihood.
        """
        pass

    @staticmethod
    def innovation(observation: np.ndarray, observation_predicted: np.ndarray) -> np.ndarray:
        """
        Function to calculate the innovation model function that calculates the difference between the output
        of the observation function (observation_func) and the actual 'real-world' measurement/observation
        of that signal.
        :param observation: 'real-world' measurement/observation of that signal at time = k
        :param observation_predicted: the output of the observation function (observation_func) at time = k.
        :return: difference between the output of the observation function and the actual 'real-world' observation.
        """
        return observation - observation_predicted

    @abstractmethod
    def linearize(
            self,
            lin_type: LinearizationType,
            state: Optional[np.ndarray],
            state_noise: Optional[np.ndarray],
            observation_noise: Optional[np.ndarray],
            control_x: Optional[np.ndarray],
            control_z: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        The linearization function that calculates Jacobians e.t.c.
        :return: value of the function calculated in the linear approximation.
        """
        pass

    @abstractmethod
    def set_params(self, **kwargs) -> NoReturn:
        """
        Allow to update specific model parameters. Do nothing by default
        :return:
        """
        pass

    @abstractmethod
    def clone(
            self,
            state_noise: GeneralStochasticModel,
            observation_noise: GeneralStochasticModel,
            reconcile_strategy: CovarianceMatrixAdaptation
    ) -> StateSpaceModel:
        pass

    def reconcile(self, filter_cov: np.ndarray, **kwargs) -> NoReturn:
        if self._reconcile_strategy is None:
            return

        if not isinstance(self._x_noise, GaussianStochasticModel):
            raise Exception("Adaptation is allowed only for Gaussian (not mixture) stochastic process")

        cov = self._reconcile_strategy.reconcile(self._x_noise, filter_cov, **kwargs)

        self._x_noise = GaussianStochasticModel(mean=self._x_noise.mean, covariance=cov, covariance_type=self._x_noise.covariance_type)

    @property
    def observation_noise_dim(self):
        return self.observation_noise.dim

    @property
    def state_noise_dim(self):
        return self.state_noise.dim

    def set_state_noise(self, state_noise: sm.GeneralStochasticModel) -> StateSpaceModel:
        return self.clone(state_noise, self.observation_noise, self.reconcile_strategy)

    def set_observation_noise(self, observation_noise: sm.GeneralStochasticModel) -> StateSpaceModel:
        return self.clone(self.state_noise, observation_noise, self.reconcile_strategy)
