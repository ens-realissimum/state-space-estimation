from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np
from numpy.random import random
from sklearn.mixture import GaussianMixture

import bayesian_framework.shared.numerical_computations as num_computations
import utils.matrix_utils as m_utils
from bayesian_framework.inference.gssm import StateSpaceModel
from bayesian_framework.inference.stochastic_models.covariance_type import CovarianceType
from bayesian_framework.shared.gaussian_mixture_info import GaussianMixtureInfo
from bayesian_framework.shared.linearization_type import LinearizationType


def init_gmi(
        x_init: np.ndarray,
        x_dim: int,
        particles_count: int,
        n_components: int
) -> GaussianMixtureInfo:
    particles = np.squeeze(
        np.random.randn(x_dim, particles_count) + np.tile(x_init, (1, particles_count))
    )

    if particles.ndim == 1:
        particles = np.transpose(np.atleast_2d(particles))

    gmm = GaussianMixture(n_components)
    gmm.fit(particles)
    return GaussianMixtureInfo.init(gmm)


class KalmanFilterInternalInfo:
    def __init__(
            self,
            state_mean_predicted: np.ndarray,
            state_cov_predicted: np.ndarray,
            observation_mean_predicted: np.ndarray,
            observation_cov_predicted: np.ndarray,
            innovation: np.ndarray,
            filter_gain: np.ndarray
    ):
        self._x_mean_predicted = state_mean_predicted
        self._x_cov_predicted = state_cov_predicted
        self._z_mean_predicted = observation_mean_predicted
        self._z_cov_predicted = observation_cov_predicted
        self._innovation = innovation
        self._filter_gain = filter_gain

    @property
    def state_mean_predicted(self) -> np.ndarray:
        return self._x_mean_predicted

    @property
    def state_cov_predicted(self) -> np.ndarray:
        return self._x_cov_predicted

    @property
    def observation_mean_predicted(self) -> np.ndarray:
        return self._z_mean_predicted

    @property
    def observation_cov_predicted(self) -> np.ndarray:
        return self._z_cov_predicted

    @property
    def innovation(self) -> np.ndarray:
        return self._innovation

    @property
    def filter_gain(self) -> np.ndarray:
        return self._filter_gain


KalmanFilterResult = Tuple[np.ndarray, np.ndarray, KalmanFilterInternalInfo]


class LocalApproximationKalmanFilter(ABC):
    """
    Base class (interface) for linear form of Kalman filter and all filters
    which are based on approximation of non-linear transformations like sigma-point filters, extended Kalman filter, e.t.c.
    Another way to estimate is approximate probability density function (pdf). This is used in particle filters and their variations.
    """

    @abstractmethod
    def estimate(
            self,
            state: np.ndarray,
            cov_state: np.ndarray,
            observation: np.ndarray,
            model: StateSpaceModel,
            ctrl_x: Optional[np.ndarray],
            ctrl_z: Optional[np.ndarray]
    ) -> KalmanFilterResult:
        pass


class PdfApproximationKalmanFilter(ABC):
    """
    Base class (interface) for Kalman filter (and all other filters, e.g. particle filters family) which are based on approximation of
    predictive distributions (probability density function (pdf)) after a  non-linear transformations, e.t.c.
    """

    def __init__(self, estimate_type: Union[EstimateType, None] = None):
        self._estimate_type = estimate_type if estimate_type is not None else EstimateType.mean

    @abstractmethod
    def estimate(
            self,
            data_set: BootstrapDataSet,
            observation: np.ndarray,
            model: StateSpaceModel,
            ctrl_x: Optional[np.ndarray],
            ctrl_z: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, BootstrapDataSet]:
        """
        Estimates state mean (x(k)) of dynamic system at time k based on noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)].

        :param data_set: BootstrapDataSet, particle filter data structure (contains set of particles as well as their corresponding weights).
        :param observation:
        :param observation: noisy observations starting at time k, i.e. z(k).
        :param model: inference state space model, which fully describes filtration issue (evolution of state, relation between state and
            observation, state noise, observation noise, etc).
        :param ctrl_x: exogenous input to state transition function starting at time k-1, i.e. u1(k-1).
        :param ctrl_z: exogenous input to state observation function starting at time k, i.e. u2(k).
        :return: tuple of following elements:
            estimate: np array, estimates of state at time k, i.e. E[x(t)|z(1), z(2), ..., z(t)] for t = k;
            data_set: BootstrapDataSet, updated Particle filter data structure. Contains set of particles as well as their corresponding weights.
        """
        pass

    def eval_final_state_estimate(self, data_set: BootstrapDataSet, model: StateSpaceModel) -> np.ndarray:
        if self._estimate_type is EstimateType.mean:
            return np.sum(np.tile(data_set.weights, (model.state_dim, 1)) * data_set.particles, axis=1)
        elif self._estimate_type is EstimateType.median:
            return np.median(np.tile(data_set.weights, (model.state_dim, 1)) * data_set.particles, axis=1)
        else:
            raise Exception(f"Unknown estimate type ('{self._estimate_type}')")


class LinearKf(LocalApproximationKalmanFilter):
    @abstractmethod
    def eval_h_matrix(
            self: Kf,
            model: StateSpaceModel,
            x_predicted: np.ndarray,
            ctrl_x: Union[np.ndarray, None],
            ctrl_z: Union[np.ndarray, None]
    ) -> np.ndarray:
        pass

    @abstractmethod
    def eval_c_matrix(
            self: Kf,
            model: StateSpaceModel,
            x_predicted: np.ndarray,
            ctrl_x: Union[np.ndarray, None],
            ctrl_z: Union[np.ndarray, None]
    ) -> np.ndarray:
        pass

    @abstractmethod
    def eval_g_matrix(
            self: Kf,
            model: StateSpaceModel,
            state: np.ndarray,
            ctrl_x: Union[np.ndarray, None],
            ctrl_z: Union[np.ndarray, None]
    ) -> np.ndarray:
        pass

    @abstractmethod
    def eval_f_matrix(
            self: Kf,
            model: StateSpaceModel,
            state: np.ndarray,
            ctrl_x: Union[np.ndarray, None],
            ctrl_z: Union[np.ndarray, None]
    ) -> np.ndarray:
        pass


class Kf(LinearKf):
    """
    KF - Kalman Filter (linear).

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)],

    The filter is a linear estimator, i.e. the filter works under assumption that the dynamic state space system is a linear.
    The filter is used linearize function of dynamic state space system to reduce corresponding model to linear model.
    Therefore, the filter can be used with nonlinear dynamic systems.
    In this case the filter will work as linear estimator of nonlinear dynamic system.

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.
    """

    def __str__(self):
        return "Kalman Filter"

    def estimate(
            self,
            state: np.ndarray,
            cov_state: np.ndarray,
            observation: np.ndarray,
            model: StateSpaceModel,
            ctrl_x: Optional[np.ndarray],
            ctrl_z: Optional[np.ndarray]
    ) -> KalmanFilterResult:
        """
        Estimates state mean (x(k)) and covariance (cov_x(k)) of dynamic system at time k
        based on noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)],

        :param state: state mean at time k-1, i.e. x(k-1).
        :param cov_state: state covariance at time k-1, i.e. Px(k-1).
        :param observation: noisy observations starting at time k, i.e. z(k).
        :param model: inference state space model, which fully describes filtration issue (evolution of state, relation between state and
        observation, state noise, observation noise, e.t.c.); state noise must be of type 'gaussian' or 'combo-gaussian';
        observation noise must be of type 'gaussian' or 'combo-gaussian'.
        :param ctrl_x: exogenous input to state transition function starting at time k-1, i.e. u1(k-1).
        :param ctrl_z: exogenous input to state observation function starting at time k, i.e. u2(k).
        :return: tuple of following elements:
            state_new: estimates of state starting at time k, i.e. E[x(t)|z(1), z(2), ..., z(t)] for t = k;
            state_cov_new: state covariance at time k;
            filter_internal_data: internal variables. Provide information about filtration process.
            This is an instance of :class:`KalmanFilterInternalInfo`:
                x_mean_predicted: predicted state mean, i.e. E[x(t)|z(1), z(2), ..., z(t-1)] for t = k.
                x_cov_predicted: predicted state covariance.
                z_mean_predicted: predicted observation, i.e. E[z(k)|Z(k-1)].
                z_cov_predicted: innovation covariance.
                innovation: innovation signal.
                filter_gain: filter gain.
        """
        # Time update (prediction)
        f = self.eval_f_matrix(model, state, ctrl_x, ctrl_z)
        g = self.eval_g_matrix(model, state, ctrl_x, ctrl_z)

        x_predicted = model.transition_func(state, model.state_noise.mean, ctrl_x)
        x_cov_predicted = f @ cov_state @ f.T + g @ model.state_noise.covariance @ g.T

        # Observation / measurement update (correction)
        c = self.eval_c_matrix(model, x_predicted, ctrl_x, ctrl_z)
        h = self.eval_h_matrix(model, x_predicted, ctrl_x, ctrl_z)

        z_cov_predicted = c @ x_cov_predicted @ c.T + h @ model.observation_noise.covariance @ h.T
        filter_gain = m_utils.divide_inv(x_cov_predicted @ c.T, z_cov_predicted)

        z_predicted = model.observation_func(x_predicted, model.observation_noise.mean, ctrl_z)
        innovation = model.innovation(observation, z_predicted)

        state_new = x_predicted + filter_gain @ innovation
        state_cov_new = x_cov_predicted - filter_gain @ z_cov_predicted @ filter_gain.T

        internal_vars = KalmanFilterInternalInfo(
            x_predicted,
            x_cov_predicted,
            z_predicted,
            z_cov_predicted,
            innovation,
            filter_gain
        )

        return state_new, state_cov_new, internal_vars

    def eval_h_matrix(
            self: Kf,
            model: StateSpaceModel,
            x_predicted: np.ndarray,
            ctrl_x: Union[np.ndarray, None],
            ctrl_z: Union[np.ndarray, None]
    ) -> np.ndarray:
        return model.linearize(LinearizationType.H, x_predicted, None, model.observation_noise.mean, None, ctrl_z)

    def eval_c_matrix(
            self: Kf,
            model: StateSpaceModel,
            x_predicted: np.ndarray,
            ctrl_x: Union[np.ndarray, None],
            ctrl_z: Union[np.ndarray, None]
    ) -> np.ndarray:
        return model.linearize(LinearizationType.C, x_predicted, None, model.observation_noise.mean, None, ctrl_z)

    def eval_g_matrix(
            self: Kf,
            model: StateSpaceModel,
            state: np.ndarray,
            ctrl_x: Union[np.ndarray, None],
            ctrl_z: Union[np.ndarray, None]
    ) -> np.ndarray:
        return model.linearize(LinearizationType.G, state, model.state_noise.mean, None, ctrl_x, None)

    def eval_f_matrix(
            self: Kf,
            model: StateSpaceModel,
            state: np.ndarray,
            ctrl_x: Union[np.ndarray, None],
            ctrl_z: Union[np.ndarray, None]
    ) -> np.ndarray:
        return model.linearize(LinearizationType.F, state, model.state_noise.mean, None, ctrl_x, None)


class Ekf(Kf):
    """
    EKF - Extended Kalman Filter.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)],

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.
    """

    def __str__(self):
        return "Extended Kalman Filter"

    def eval_h_matrix(
            self: Kf,
            model: StateSpaceModel,
            x_predicted: np.ndarray,
            ctrl_x: Union[np.ndarray, None],
            ctrl_z: Union[np.ndarray, None]
    ) -> np.ndarray:
        return model.linearize(LinearizationType.H, x_predicted, model.state_noise.mean, model.observation_noise.mean,
                               ctrl_x, ctrl_z)

    def eval_c_matrix(
            self: Kf,
            model: StateSpaceModel,
            x_predicted: np.ndarray,
            ctrl_x: Union[np.ndarray, None],
            ctrl_z: Union[np.ndarray, None]
    ) -> np.ndarray:
        return model.linearize(LinearizationType.C, x_predicted, model.state_noise.mean, model.observation_noise.mean,
                               ctrl_x, ctrl_z)

    def eval_g_matrix(
            self: Kf,
            model: StateSpaceModel,
            state: np.ndarray,
            ctrl_x: Union[np.ndarray, None],
            ctrl_z: Union[np.ndarray, None]
    ) -> np.ndarray:
        return model.linearize(LinearizationType.G, state, model.state_noise.mean, model.observation_noise.mean, ctrl_x,
                               ctrl_z)

    def eval_f_matrix(
            self: Kf,
            model: StateSpaceModel,
            state: np.ndarray,
            ctrl_x: Union[np.ndarray, None],
            ctrl_z: Union[np.ndarray, None]
    ) -> np.ndarray:
        return model.linearize(LinearizationType.F, state, model.state_noise.mean, model.observation_noise.mean, ctrl_x,
                               ctrl_z)


class UnscentedTransformFilter(LocalApproximationKalmanFilter):
    @abstractmethod
    def estimate_state_cov(
            self, filter_gain: np.ndarray,
            x_cov_predicted: np.ndarray,
            z_cov_predicted: np.ndarray,
            model: StateSpaceModel
    ) -> np.ndarray:
        pass

    @abstractmethod
    def eval_filter_gain(self, cross_cov: np.ndarray, z_cov_predicted: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def eval_z_cov_predicted(self, model: StateSpaceModel, w: np.ndarray, z_dev: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def eval_x_cov_predicted(self, w: np.ndarray, x_dev: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def eval_sigma_points_offset(self, augment_dim: int, cov_state: np.ndarray, model: StateSpaceModel) -> np.ndarray:
        pass


class Ukf(UnscentedTransformFilter):
    """
    UKF - Unscented Kalman Filter. Sigma-Point Kalman Filter variant.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)],

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        alpha - UKF scale factor;
        beta - UKF covariance correction factor;
        kappa - UKF secondary scaling parameter.
    """

    def __init__(self, alpha, beta, kappa):
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa

    def __str__(self):
        return f"UKF: alpha: {self._alpha}; beta: {self._beta}; kappa: {self._kappa}"

    def estimate(
            self,
            state: np.ndarray,
            cov_state: np.ndarray,
            observation: np.ndarray,
            model: StateSpaceModel,
            ctrl_x: Optional[np.ndarray],
            ctrl_z: Optional[np.ndarray]
    ) -> KalmanFilterResult:
        """
        Estimates state mean (x(k)) and covariance (cov_x(k)) of dynamic system at time k
        based on noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)],

        :param state: state mean at time k-1, i.e. x(k-1).
        :param cov_state: state covariance at time k-1, i.e. Px(k-1).
        :param observation: noisy observations starting at time k, i.e. z(k).
        :param model: inference state space model, which fully describes filtration issue (evolution of state, relation between state and
        observation, state noise, observation noise, etc).
        :param ctrl_x: exogenous input to state transition function starting at time k-1, i.e. u1(k-1).
        :param ctrl_z: exogenous input to state observation function starting at time k, i.e. u2(k).
        :return: tuple of following elements:
            state_new: estimates of state starting at time k, i.e. E[x(t)|z(1), z(2), ..., z(t)] for t = k;
            state_cov_new: state covariance at time k;
            filter_internal_data: internal variables. Provide information about filtration process.
            This is an instance of :class:`KalmanFilterInternalInfo`:
                x_mean_predicted: predicted state mean, i.e. E[x(t)|z(1), z(2), ..., z(t-1)] for t = k.
                x_cov_predicted: predicted state covariance.
                z_mean_predicted: predicted observation, i.e. E[z(k)|Z(k-1)].
                z_cov_predicted: innovation covariance.
                innovation: innovation signal.
                filter_gain: filter gain.
        """
        x_dim = model.state_dim
        v_dim = model.state_noise.dim

        augment_dim = x_dim + v_dim + model.observation_noise.dim
        sigma_set_size = 2 * augment_dim + 1
        _lambda = self._alpha ** 2 * (augment_dim + self._kappa) - augment_dim

        w = np.asarray([_lambda, 0.5, 0]) / (augment_dim + _lambda)
        w[2] = w[0] + (1 - self._alpha ** 2) + self._beta

        # Generate sigma points
        sigma_points_offset = self.eval_sigma_points_offset(augment_dim, cov_state, model)

        offset = ((augment_dim + _lambda) ** 0.5) * np.hstack(
            (np.zeros((augment_dim, 1)), sigma_points_offset, -sigma_points_offset))
        sigma_points = np.tile(
            np.vstack((state, model.state_noise.mean, model.observation_noise.mean)),
            (1, sigma_set_size)
        ) + offset

        # Propagate sigma points through state model
        x_predicted = model.transition_func(sigma_points[:x_dim, :], sigma_points[x_dim:x_dim + v_dim, :], ctrl_x)
        x_mean_predicted = w[0] * x_predicted[:, 0] + w[1] * np.sum(x_predicted[:, 1:], axis=1)
        x_dev = x_predicted - np.tile(x_mean_predicted, sigma_set_size)
        x_cov_predicted = self.eval_x_cov_predicted(w, x_dev)

        # Propagate sigma points through observation model
        z_predicted = model.observation_func(x_predicted, sigma_points[x_dim + v_dim:, :], ctrl_z)
        z_mean_predicted = w[0] * z_predicted[:, 0] + w[1] * np.sum(z_predicted[:, 1:], axis=1)
        z_dev = z_predicted - np.tile(z_mean_predicted, sigma_set_size)
        z_cov_predicted = self.eval_z_cov_predicted(model, w, z_dev)

        # Correction (measurement update)
        cross_cov = w[2] * x_dev[:, 0] @ z_dev[:, 0].T + w[1] * x_dev[:, 1:] @ z_dev[:, 1:].T
        filter_gain = self.eval_filter_gain(cross_cov, z_cov_predicted)
        innovation = model.innovation(observation, z_mean_predicted)

        state_new = x_mean_predicted + filter_gain @ innovation
        state_cov_new = self.estimate_state_cov(filter_gain, x_cov_predicted, z_cov_predicted, model)

        internal_vars = KalmanFilterInternalInfo(
            x_mean_predicted,
            x_cov_predicted,
            z_mean_predicted,
            z_cov_predicted,
            innovation,
            filter_gain
        )

        return state_new, state_cov_new, internal_vars

    def estimate_state_cov(
            self, filter_gain: np.ndarray,
            x_cov_predicted: np.ndarray,
            z_cov_predicted: np.ndarray,
            model: StateSpaceModel
    ) -> np.ndarray:
        return x_cov_predicted - filter_gain @ z_cov_predicted @ filter_gain.T

    def eval_filter_gain(self, cross_cov: np.ndarray, z_cov_predicted: np.ndarray) -> np.ndarray:
        return m_utils.divide_inv(cross_cov, z_cov_predicted)

    def eval_z_cov_predicted(self, model: StateSpaceModel, w: np.ndarray, z_dev: np.ndarray) -> np.ndarray:
        return w[2] * z_dev[:, 0] @ z_dev[:, 0].T + w[1] * z_dev[:, 1:] @ z_dev[:,
                                                                                         1:].T + model.observation_noise.covariance

    def eval_x_cov_predicted(self, w: np.ndarray, x_dev: np.ndarray) -> np.ndarray:
        return w[2] * x_dev[:, 0] @ x_dev[:, 0].T + w[1] * x_dev[:, 1:] @ x_dev[:, 1:].T

    def eval_sigma_points_offset(self, augment_dim: int, cov_state: np.ndarray, model: StateSpaceModel) -> np.ndarray:
        return m_utils.put_matrices_into_zero_matrix_one_by_one(
            augment_dim, [
                np.linalg.cholesky(cov_state),
                np.linalg.cholesky(model.state_noise.covariance),
                np.linalg.cholesky(model.observation_noise.covariance)
            ]
        )


class SrUkf(Ukf):
    """
    SRUKF - Square Root Unscented Kalman Filter. Sigma-Point Kalman Filter variant.

    This is a square root form of UKF. The square-root Unscented Kalman filter which is also O(L ** 3) for general state estimation
    and O(L ** 2) for parameter estimation (note the original formulation of the UKF for parameter-estimation was O(L ** 3)).
    In addition, the square-root forms have the added benefit of numerical stability and
    guaranteed positive semi-definiteness of the state covariances.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        alpha - UKF scale factor;
        beta - UKF covariance correction factor;
        kappa - UKF secondary scaling parameter.
    """

    def __init__(self, alpha, beta, kappa):
        super().__init__(alpha, beta, kappa)

    def __str__(self):
        return f"SRUKF: alpha: {self._alpha}; beta: {self._beta}; kappa: {self._kappa}"

    def estimate_state_cov(self, filter_gain: np.ndarray, x_cov_predicted: np.ndarray, z_cov_predicted: np.ndarray,
                           model: StateSpaceModel) -> np.ndarray:
        cov_updated = filter_gain @ z_cov_predicted

        sqrt_cov_state_new = x_cov_predicted
        for j in range(model.observation_dim):
            sqrt_cov_state_new = m_utils.cholesky_update(sqrt_cov_state_new, cov_updated[:, j], "-")

        return sqrt_cov_state_new.T  # should be lowered triangle

    def eval_filter_gain(self, cross_cov: np.ndarray, z_cov_predicted: np.ndarray) -> np.ndarray:
        cross_cov_on_sqrt_z_cov = m_utils.divide_inv(cross_cov, z_cov_predicted.T)
        return m_utils.divide_inv(cross_cov_on_sqrt_z_cov, z_cov_predicted)

    def eval_z_cov_predicted(self, model: StateSpaceModel, w: np.ndarray, z_dev: np.ndarray) -> np.ndarray:
        sqrt_w = np.sqrt(np.abs(w))
        _, z_cov_predicted = np.linalg.qr(
            np.vstack(
                (
                    sqrt_w[1] * z_dev[:, 1:].T, model.observation_noise.covariance
                )
            )
        )
        z_cov_predicted = m_utils.cholesky_update(z_cov_predicted, sqrt_w[2] * z_dev[:, 0],
                                                  "+" if w[2] > 0 else "-")
        return z_cov_predicted.T  # should be lowered triangle

    def eval_x_cov_predicted(self, w: np.ndarray, x_dev: np.ndarray) -> np.ndarray:
        sqrt_w = np.sqrt(np.abs(w))
        _, x_cov_predicted = np.linalg.qr((sqrt_w[1] * x_dev[:, 1:]).T)
        return m_utils.cholesky_update(x_cov_predicted, sqrt_w[2] * x_dev[:, 0], "+" if w[2] > 0 else "-")

    def eval_sigma_points_offset(self, augment_dim: int, cov_state: np.ndarray, model: StateSpaceModel) -> np.ndarray:
        return m_utils.put_matrices_into_zero_matrix_one_by_one(
            augment_dim, [
                cov_state,
                model.state_noise.covariance,
                model.observation_noise.covariance
            ]
        )


class CentralDiffTransformFilter(LocalApproximationKalmanFilter):
    @abstractmethod
    def repair_covariance(self, sqrt_state_cov_new: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def eval_unweighted_z_offset(self, model: StateSpaceModel, x_cov_predicted_sqrt: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def eval_unweighted_x_offset(self, cov_state: np.ndarray, model: StateSpaceModel) -> np.ndarray:
        pass


class Cdkf(CentralDiffTransformFilter):
    """
    CDKF - Central Difference Kalman Filter. Sigma-Point Kalman Filter variant.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        scale_factor - scale factor / difference step size.
    """

    def __init__(self, scale_factor):
        self._scale_factor = scale_factor

    def __str__(self):
        return f"CDKF. scale factor: {self._scale_factor}"

    def estimate(
            self,
            state: np.ndarray,
            cov_state: np.ndarray,
            observation: np.ndarray,
            model: StateSpaceModel,
            ctrl_x: Optional[np.ndarray],
            ctrl_z: Optional[np.ndarray]
    ) -> KalmanFilterResult:
        """
        Estimates state mean (x(k)) and covariance (cov_x(k)) of dynamic system at time k
        based on noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)],

        :param state: state mean at time k-1, i.e. x(k-1).
        :param cov_state: state covariance at time k-1, i.e. Px(k-1).
        :param observation: noisy observations starting at time k, i.e. z(k).
        :param model: inference state space model, which fully describes filtration issue (evolution of state, relation between state and
        observation, state noise, observation noise, etc).
        :param ctrl_x: exogenous input to state transition function starting at time k-1, i.e. u1(k-1).
        :param ctrl_z: exogenous input to state observation function starting at time k, i.e. u2(k).
        :return: tuple of following elements:
            state_new: estimates of state starting at time k, i.e. E[x(t)|z(1), z(2), ..., z(t)] for t = k;
            state_cov_new: state covariance at time k;
            filter_internal_data: internal variables. Provide information about filtration process.
            This is an instance of :class:`KalmanFilterInternalInfo`:
                x_mean_predicted: predicted state mean, i.e. E[x(t)|z(1), z(2), ..., z(t-1)] for t = k.
                x_cov_predicted: sqrt version of predicted state covariance.
                z_mean_predicted: predicted observation, i.e. E[z(k)|Z(k-1)].
                z_cov_predicted: sqrt version of innovation covariance.
                innovation: innovation signal.
                filter_gain: filter gain.
        """
        x_dim = model.state_dim
        v_dim = model.state_noise_dim
        n_dim = model.observation_noise_dim
        scale_factor_square = self._scale_factor ** 2

        w_x = np.asarray(
            [
                [(scale_factor_square - x_dim - v_dim) / scale_factor_square, 1 / (2 * scale_factor_square)],
                [1 / (2 * self._scale_factor), np.sqrt(scale_factor_square - 1) / (2 * scale_factor_square)]
            ]
        )

        w_z = w_x.copy()
        w_z[0, 0] = (scale_factor_square - x_dim - n_dim) / scale_factor_square

        sigma_set_size_z = 2 * (x_dim + n_dim) + 1

        # Generate sigma points for the state
        offset_x = self.eval_unweighted_x_offset(cov_state, model)
        offset_x_weighted = self._scale_factor * np.hstack((np.zeros((x_dim + v_dim, 1)), offset_x, -offset_x))
        sigma_set_x = np.tile(np.vstack((state, model.state_noise.mean)), (1, sigma_set_size_z)) + offset_x_weighted

        # Propagate state sigma points through state model
        x_predicted = model.transition_func(sigma_set_x[:x_dim, :], sigma_set_x[x_dim:, :], ctrl_x)
        x_mean_predicted, a, b = self.eval_prediction(x_predicted, v_dim, model, w_x)

        _, x_cov_predicted_sqrt = np.linalg.qr(np.hstack((a, b)).T)
        x_cov_predicted_sqrt = x_cov_predicted_sqrt.T

        # Generate sigma points for the observations
        offset_z = self.eval_unweighted_z_offset(model, x_cov_predicted_sqrt)
        offset_z_weighted = self._scale_factor * np.hstack((np.zeros((x_dim + n_dim, 1)), offset_z, -offset_z))
        sigma_set_z = np.tile([x_mean_predicted, model.observation_noise.mean],
                              (1, sigma_set_size_z)) + offset_z_weighted

        # Propagate observation sigma points through observation model
        z_predicted = model.observation_func(sigma_set_z[:x_dim, :], sigma_set_z[x_dim:, :], ctrl_z)
        z_mean_predicted, c, d = self.eval_prediction(z_predicted, n_dim, model, w_z)

        _, z_cov_predicted_sqrt = np.linalg.qr((np.hstack((c, d, model.observation_noise.covariance))).T)
        z_cov_predicted_sqrt = z_cov_predicted_sqrt.T

        # Correction (measurement update)
        cross_cov = x_cov_predicted_sqrt @ c[:, :x_dim].T

        cross_cov_on_sqrt_z_cov = m_utils.divide_inv(cross_cov, z_cov_predicted_sqrt.T)
        filter_gain = m_utils.divide_inv(cross_cov_on_sqrt_z_cov, z_cov_predicted_sqrt)

        innovation = model.innovation(observation, z_mean_predicted)

        state_new = x_mean_predicted + filter_gain @ innovation

        _, sqrt_state_cov_new = np.linalg.qr(
            np.hstack(
                (
                    x_cov_predicted_sqrt - filter_gain @ c[:, :x_dim],
                    filter_gain @ c[:, x_dim:],
                    filter_gain @ d
                )
            ).T
        )
        sqrt_state_cov_new = sqrt_state_cov_new.T
        state_cov_new = self.repair_covariance(sqrt_state_cov_new)

        internal_vars = KalmanFilterInternalInfo(
            x_mean_predicted,
            x_cov_predicted_sqrt,
            z_mean_predicted,
            z_cov_predicted_sqrt,
            innovation,
            filter_gain
        )

        return state_new, state_cov_new, internal_vars

    @staticmethod
    def eval_prediction(
            predicted_points: np.ndarray,
            noise_dim: int,
            model: StateSpaceModel,
            weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_dim = model.state_dim
        mean_predicted = weights[0, 0] * predicted_points[:, 0] + weights[0, 1] * np.sum(predicted_points[:, 1:],
                                                                                         axis=1)
        x1 = weights[1, 0] * (
                    predicted_points[:, 1:x_dim + noise_dim + 1] - predicted_points[:, x_dim + noise_dim + 1:])
        x2 = weights[1, 1] * (predicted_points[:, 1:x_dim + noise_dim + 1]
                              + predicted_points[:, x_dim + noise_dim + 1:]
                              - np.tile(2 * predicted_points[:, 0], (1, x_dim + noise_dim)))
        return mean_predicted, x1, x2

    def repair_covariance(self, sqrt_state_cov_new: np.ndarray) -> np.ndarray:
        return sqrt_state_cov_new @ sqrt_state_cov_new.T

    def eval_unweighted_z_offset(self, model: StateSpaceModel, x_cov_predicted_sqrt: np.ndarray) -> np.ndarray:
        sqrt_n_cov = np.linalg.cholesky(model.observation_noise.covariance)
        return m_utils.put_matrices_into_zero_matrix_one_by_one(model.state_dim + model.observation_noise_dim,
                                                                [x_cov_predicted_sqrt, sqrt_n_cov])

    def eval_unweighted_x_offset(self, cov_state: np.ndarray, model: StateSpaceModel) -> np.ndarray:
        sqrt_v_cov = np.linalg.cholesky(model.state_noise.covariance)
        sqrt_x_cov = np.linalg.cholesky(cov_state)
        return m_utils.put_matrices_into_zero_matrix_one_by_one(model.state_dim + model.state_noise_dim,
                                                                [sqrt_x_cov, sqrt_v_cov])


class SrCdkf(Cdkf):
    """
    SR CDKF - Square Root Central Difference Kalman Filter. Sigma-Point Kalman Filter variant.

    This is a square root form of CDKF. The square-root Central Difference Kalman filter which is also O(L ** 3) for general state
    estimation and O(L ** 2) for parameter estimation. In addition, the square-root forms have the added benefit of numerical stability
    and guaranteed positive semi-definiteness of the state covariances.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        scale_factor - scale factor / difference step size.
    """

    def __init__(self, scale_factor):
        super().__init__(scale_factor)

    def __str__(self):
        return f"SR CDKF. Scale factor: {self._scale_factor}"

    def repair_covariance(self, sqrt_state_cov_new: np.ndarray) -> np.ndarray:
        return sqrt_state_cov_new

    def eval_unweighted_z_offset(self, model: StateSpaceModel, x_cov_predicted_sqrt: np.ndarray) -> np.ndarray:
        return m_utils.put_matrices_into_zero_matrix_one_by_one(
            model.state_dim + model.observation_noise_dim, [x_cov_predicted_sqrt, model.observation_noise.covariance]
        )

    def eval_unweighted_x_offset(self, cov_state: np.ndarray, model: StateSpaceModel) -> np.ndarray:
        return m_utils.put_matrices_into_zero_matrix_one_by_one(
            model.state_dim + model.state_noise_dim, [cov_state, model.state_noise.covariance]
        )


class CubatureTransformFilter(LocalApproximationKalmanFilter):
    @abstractmethod
    def estimate_state_cov(
            self,
            filter_gain: np.ndarray,
            x_cov_predicted: np.ndarray,
            z_cov_predicted: np.ndarray,
            model: StateSpaceModel,
            x: np.ndarray,
            z: np.ndarray
    ) -> np.ndarray:
        pass

    @abstractmethod
    def eval_filter_gain(self, cross_cov: np.ndarray, z_cov_predicted: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def eval_z_cov_predicted(self, model: StateSpaceModel, z: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def eval_offset_z(self, cubature_set_size: int, x_cov_predicted: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def eval_x_cov_predicted(
            self, cubature_set_size: int,
            model: StateSpaceModel,
            x_mean_predicted: np.ndarray,
            x_predicted: np.ndarray
    ) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate_x_offset(self, cov_state: np.ndarray, cubature_set_size: int) -> np.ndarray:
        pass


class Ckf(CubatureTransformFilter):
    """
    CKF - Cubature Kalman Filter. Sigma-Point Kalman Filter variant.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.
    """

    def __init__(self):
        pass

    def __str__(self):
        return "Cubature Kalman Filter"

    def estimate(
            self,
            state: np.ndarray,
            cov_state: np.ndarray,
            observation: np.ndarray,
            model: StateSpaceModel,
            ctrl_x: Optional[np.ndarray],
            ctrl_z: Optional[np.ndarray]
    ) -> KalmanFilterResult:
        """
        Estimates state mean (x(k)) and covariance (cov_x(k)) of dynamic system at time k
        based on noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)].

        :param state: state mean at time k-1, i.e. x(k-1).
        :param cov_state: state covariance at time k-1, i.e. Px(k-1).
        :param observation: noisy observations starting at time k, i.e. z(k).
        :param model: inference state space model, which fully describes filtration issue (evolution of state, relation between state and
        observation, state noise, observation noise, etc).
        :param ctrl_x: exogenous input to state transition function starting at time k-1, i.e. u1(k-1).
        :param ctrl_z: exogenous input to state observation function starting at time k, i.e. u2(k).
        :return: tuple of following elements:
            state_new: estimates of state starting at time k, i.e. E[x(t)|z(1), z(2), ..., z(t)] for t = k;
            state_cov_new: state covariance at time k;
            filter_internal_data: internal variables. Provide information about filtration process.
            This is an instance of :class:`KalmanFilterInternalInfo`:
                x_mean_predicted: predicted state mean, i.e. E[x(t)|z(1), z(2), ..., z(t-1)] for t = k.
                x_cov_predicted: predicted state covariance.
                z_mean_predicted: predicted observation, i.e. E[z(k)|Z(k-1)].
                z_cov_predicted: innovation covariance.
                innovation: innovation signal.
                filter_gain: filter gain.
        """
        x_dim = model.state_dim
        cubature_set_size = 2 * x_dim

        # Generate cubature points for the state
        offset_x = self.evaluate_x_offset(cov_state, cubature_set_size)
        cubature_set_x = np.tile(state, (1, cubature_set_size)) + offset_x @ np.hstack((np.eye(x_dim), -np.eye(x_dim)))

        # Propagate state cubature points through state model
        x_predicted = model.transition_func(cubature_set_x, np.tile(model.state_noise.mean, (1, cubature_set_size)),
                                            ctrl_x)
        x_mean_predicted = np.sum(x_predicted, axis=1) / cubature_set_size
        x_cov_predicted = self.eval_x_cov_predicted(cubature_set_size, model, x_mean_predicted, x_predicted)

        # Generate cubature points for the observations
        offset_z = self.eval_offset_z(cubature_set_size, x_cov_predicted)
        cubature_set_z = np.tile(x_mean_predicted, (1, cubature_set_size)) + offset_z @ np.hstack(
            (np.eye(x_dim), -np.eye(x_dim)))

        # Propagate observation cubature points through observation model
        z_predicted = model.observation_func(cubature_set_z,
                                             np.tile(model.observation_noise.mean, (1, cubature_set_size)), ctrl_z)
        z_mean_predicted = np.sum(z_predicted, axis=1) / cubature_set_size

        # Correction (measurement update)
        x = (cubature_set_z - np.tile(x_mean_predicted, (1, cubature_set_size))) / np.sqrt(cubature_set_size)
        z = (z_predicted - np.tile(z_mean_predicted, (1, cubature_set_size))) / np.sqrt(cubature_set_size)

        z_cov_predicted = self.eval_z_cov_predicted(model, z)
        cross_cov = x @ z.T
        filter_gain = self.eval_filter_gain(cross_cov, z_cov_predicted)

        innovation = model.innovation(observation, z_mean_predicted)

        state_new = x_mean_predicted + filter_gain @ innovation
        state_cov_new = self.estimate_state_cov(filter_gain, x_cov_predicted, z_cov_predicted, model, x, z)

        internal_vars = KalmanFilterInternalInfo(
            x_mean_predicted,
            x_cov_predicted,
            z_mean_predicted,
            z_cov_predicted,
            innovation,
            filter_gain
        )

        return state_new, state_cov_new, internal_vars

    def estimate_state_cov(
            self,
            filter_gain: np.ndarray,
            x_cov_predicted: np.ndarray,
            z_cov_predicted: np.ndarray,
            model: StateSpaceModel,
            x: np.ndarray,
            z: np.ndarray
    ) -> np.ndarray:
        return x_cov_predicted - filter_gain @ z_cov_predicted @ filter_gain.T

    def eval_filter_gain(self, cross_cov: np.ndarray, z_cov_predicted: np.ndarray) -> np.ndarray:
        return cross_cov @ np.linalg.pinv(z_cov_predicted)

    def eval_z_cov_predicted(self, model: StateSpaceModel, z: np.ndarray) -> np.ndarray:
        z_cov_predicted = z @ z.T + model.observation_noise.covariance
        return z_cov_predicted

    def eval_offset_z(self, cubature_set_size: int, x_cov_predicted: np.ndarray) -> np.ndarray:
        return m_utils.svd_sqrt(x_cov_predicted) * np.sqrt(cubature_set_size / 2)

    def eval_x_cov_predicted(
            self,
            cubature_set_size: int,
            model: StateSpaceModel,
            x_mean_predicted: np.ndarray,
            x_predicted: np.ndarray
    ) -> np.ndarray:
        x_predicted_dev = (x_predicted - np.tile(x_mean_predicted, (1, cubature_set_size))) / np.sqrt(cubature_set_size)
        x_cov_predicted = x_predicted_dev @ x_predicted_dev.T + model.state_noise.covariance
        return x_cov_predicted

    def evaluate_x_offset(self, cov_state: np.ndarray, cubature_set_size: int) -> np.ndarray:
        return m_utils.svd_sqrt(cov_state) * np.sqrt(cubature_set_size / 2)


class SrCkf(Ckf):
    """
    SR CKF - Square Root Cubature Kalman Filter. Sigma-Point Kalman Filter variant.

    This is a square root form of CKF. The square-root Cubature Kalman filter which is also O(L ** 3) for general state
    estimation and O(L ** 2) for parameter estimation. In addition, the square-root forms have the added benefit of numerical stability
    and guaranteed positive semi-definiteness of the state covariances.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "SR Cubature Kalman Filter"

    def estimate_state_cov(
            self,
            filter_gain: np.ndarray,
            x_cov_predicted: np.ndarray,
            z_cov_predicted: np.ndarray,
            model: StateSpaceModel,
            x: np.ndarray,
            z: np.ndarray
    ) -> np.ndarray:
        _, sqrt_state_cov_new = np.linalg.qr(
            np.hstack((x - filter_gain @ z, filter_gain @ model.observation_noise.covariance)).T)
        return sqrt_state_cov_new.T

    def eval_filter_gain(self, cross_cov: np.ndarray, z_cov_predicted: np.ndarray) -> np.ndarray:
        cross_cov_on_sqrt_z_cov = m_utils.divide_inv(cross_cov, z_cov_predicted.T)
        return m_utils.divide_inv(cross_cov_on_sqrt_z_cov, z_cov_predicted)

    def eval_z_cov_predicted(self, model: StateSpaceModel, z: np.ndarray) -> np.ndarray:
        _, z_cov_predicted_sqrt = np.linalg.qr(np.hstack((z, model.observation_noise.covariance)).T)
        return z_cov_predicted_sqrt.T

    def eval_offset_z(self, cubature_set_size: np.ndarray, x_cov_predicted: np.ndarray) -> np.ndarray:
        return x_cov_predicted * np.sqrt(cubature_set_size / 2)

    def eval_x_cov_predicted(self, cubature_set_size: int, model: StateSpaceModel, x_mean_predicted: np.ndarray,
                             x_predicted: np.ndarray) -> np.ndarray:
        x_weighted_dev_set = (x_predicted - np.tile(x_mean_predicted, (1, cubature_set_size))) / np.sqrt(cubature_set_size)
        _, x_cov_predicted_sqrt = np.linalg.qr(np.hstack((x_weighted_dev_set, model.state_noise.covariance)).T)
        return x_cov_predicted_sqrt.T

    def evaluate_x_offset(self, cov_state: np.ndarray, cubature_set_size: int) -> np.ndarray:
        return cov_state * np.sqrt(cubature_set_size / 2)


class HdLkf(LocalApproximationKalmanFilter):
    """
    HDLKF - High Degree 'Local Approximation'  Kalman Filter.
    Base class that encapsulate filtration process for 'Local approximation' Kalman filters,
    e.g. Cubature filters with degree more than 3, Gauss-Hermite Quadrature Filter.
    The class specifies abstract methods for cubature points and weights calculation. These methods must be implemented in the inherited classes.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.
    """

    def estimate(
            self,
            state: np.ndarray,
            cov_state: np.ndarray,
            observation: np.ndarray,
            model: StateSpaceModel,
            ctrl_x: Optional[np.ndarray],
            ctrl_z: Optional[np.ndarray]
    ) -> KalmanFilterResult:
        """
        Estimates state mean (x(k)) and covariance (cov_x(k)) of dynamic system at time k
        based on noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)].

        :param state: state mean at time k-1, i.e. x(k-1).
        :param cov_state: state covariance at time k-1, i.e. Px(k-1).
        :param observation: noisy observations starting at time k, i.e. z(k).
        :param model: inference state space model, which fully describes filtration issue (evolution of state, relation between state and
        observation, state noise, observation noise, etc).
        :param ctrl_x: exogenous input to state transition function starting at time k-1, i.e. u1(k-1).
        :param ctrl_z: exogenous input to state observation function starting at time k, i.e. u2(k).
        :return: tuple of following elements:
            state_new: estimates of state starting at time k, i.e. E[x(t)|z(1), z(2), ..., z(t)] for t = k;
            state_cov_new: state covariance at time k;
            filter_internal_data: internal variables. Provide information about filtration process.
            This is an instance of :class:`KalmanFilterInternalInfo`:
                x_mean_predicted: predicted state mean, i.e. E[x(t)|z(1), z(2), ..., z(t-1)] for t = k.
                x_cov_predicted: predicted state covariance.
                z_mean_predicted: predicted observation, i.e. E[z(k)|Z(k-1)].
                z_cov_predicted: innovation covariance.
                innovation: innovation signal.
                filter_gain: filter gain.
        """
        set_size = self.eval_set_size(model)
        points, w = self.evaluate_points_and_weights(model)

        w_x = np.tile(np.atleast_2d(w), (model.state_dim, 1))
        w_z = np.tile(np.atleast_2d(w), (model.observation_dim, 1))

        # Generate points for the state
        offset_x = np.linalg.cholesky(cov_state) @ points
        set_x = np.tile(state, (1, set_size)) + offset_x

        # Propagate state points through state model
        x_predicted = model.transition_func(set_x, np.tile(model.state_noise.mean, (1, set_size)), ctrl_x)
        x_mean_predicted = x_predicted @ w
        x_predicted_dev = x_predicted - np.tile(x_mean_predicted, (1, set_size))
        x_cov_predicted = w_x * x_predicted_dev @ x_predicted_dev.T + model.state_noise.covariance

        # Generate points for the observations
        offset_z = np.linalg.cholesky(x_cov_predicted) @ points
        set_z = np.tile(x_mean_predicted, (1, set_size)) + offset_z

        # Propagate observation points through observation model
        z_predicted = model.observation_func(set_z, np.tile(model.observation_noise.mean, (1, set_size)), ctrl_z)
        z_mean_predicted = z_predicted @ w

        # Correction (measurement update)
        z_dev = z_predicted - np.tile(z_mean_predicted, (1, set_size))
        innovation_cov = w_z * z_dev @ z_dev.T + model.observation_noise.covariance
        cross_cov = w_x * (set_z - np.tile(x_mean_predicted, (1, set_size))) @ z_dev.T
        filter_gain = cross_cov @ np.linalg.pinv(innovation_cov)

        innovation = model.innovation(observation, z_mean_predicted)

        state_new = x_mean_predicted + filter_gain @ innovation
        state_cov_new = x_cov_predicted - filter_gain @ innovation_cov @ filter_gain.T

        internal_vars = KalmanFilterInternalInfo(
            x_mean_predicted, x_cov_predicted,
            z_mean_predicted, innovation_cov,
            innovation, filter_gain
        )

        return state_new, state_cov_new, internal_vars

    def evaluate_points_and_weights(self, model: StateSpaceModel) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def eval_set_size(self, model: StateSpaceModel) -> int:
        pass


class FdCkf(HdLkf):
    """
    FDCKF - Fifth Degree Cubature Kalman Filter. Fifth degree Spherical Simplex-Radial Cubature Kalman Filter.
    Sigma Point Kalman Filter variant. Approximation points (sigma-point) calculated via Spherical Simplex-Radial Rule.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.
    """

    def eval_set_size(self, model: StateSpaceModel) -> Tuple[np.ndarray, np.ndarray]:
        return 2 * model.state_dim ** 2 + 1

    def evaluate_points_and_weights(self, model: StateSpaceModel) -> Tuple[np.ndarray, np.ndarray]:
        return num_computations.eval_fifth_degree_cubature_rule(model.state_dim)


class Cqkf(HdLkf):
    """
    CQKF - Cubature Quadrature Kalman Filter (some kind of High-degree Cubature Kalman Filter).
    Sigma Point Kalman Filter variant.
    Quadrature points are calculated based on Gauss-Laguerre quadrature rule.
    Cubature points are calculated based on intersection with unit hypersphere.

    This filter assumes the following standard state-space model:

     x(k) = f[x(k-1), v(k-1), u1(k-1)];
     z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        order - order of Gauss-Laguerre rule.
    """

    def __init__(self, order):
        self._order = order

    def __str__(self):
        return f"Cubature Quadrature Kalman Filter. order: {self._order}"

    def evaluate_points_and_weights(self, model: StateSpaceModel) -> Tuple[np.ndarray, np.ndarray]:
        return num_computations.eval_cubature_quadrature_points(model.state_dim, self._order)

    def eval_set_size(self, model: StateSpaceModel) -> int:
        return 2 * model.state_dim * self._order


class Ghqf(HdLkf):
    """
    GHQF - Gauss-Hermite Quadrature Filter. Sigma Point Kalman Filter variant.
    Estimates state and state covariance for nonlinear systems with additive Gaussian noise by linearizing the process and observation functions using
    statistical linear regression (SLR) through a set of Gauss-Hermite quadrature points that parameterize the Gaussian density.

    For more details see "Discrete-Time Nonlinear Filtering Algorithms Using Gauss-Hermite Quadrature" by Ienkaran Arasaratnam, Simon Haykin
    and Robert J. Elliott. Vol. 95, No. 5, May 2007 | Proceedings of the IEEE

    This filter assumes the following standard state-space model:

    x(k) = f[x(k-1), v(k-1), u1(k-1)];
    z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        order - order of Gauss-Hermite rule.
    """

    def __init__(self, order):
        self._order = order

    def __str__(self):
        return f"Gauss-Hermite Quadrature Filter. order: {self._order}"

    def evaluate_points_and_weights(self, model: StateSpaceModel) -> Tuple[np.ndarray, np.ndarray]:
        return num_computations.eval_gauss_hermite_rule(self._order, model.state_dim)

    def eval_set_size(self, model: StateSpaceModel) -> int:
        return self._order ** model.state_dim


#  todo: not completed yet. there is a bug in points & weights calculation. fix it
class Sghqf(HdLkf):
    """
    SGHQF. Sparse Gauss-Hermite quadrature filter is proposed using a sparse-grid method for multidimensional
    numerical integration in the Bayesian estimation framework. The conventional Gauss-Hermite quadrature filter is
    computationally expensive for multidimensional problems, because the number of Gauss-Hermite quadrature
    points increases exponentially with the dimension. The number of sparse-grid points of the computationally efficient
    sparse Gauss-Hermite quadrature filter, however, increases only polynomially with the dimension. In addition, it is
    proven in this paper that the unscented Kalman filter using the suggested optimal parameter is a subset of the sparse
    Gauss-Hermite quadrature filter.

    For more details see "Sparse Gauss-Hermite Quadrature Filter with Application to Spacecraft Attitude Estimation"
    By Bin Jia, Ming Xin and Yang Cheng. JOURNAL OF GUIDANCE, CONTROL, AND DYNAMICS Vol. 34, No. 2, March-April 2011

    This filter assumes the following standard state-space model:

    x(k) = f[x(k-1), v(k-1), u1(k-1)];
    z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        order - order of Gauss-Hermite rule;
        manner - increase manner: L, 2*L-1, 2^L-1.
    """

    def __init__(self, order: int, manner: int):
        """

        Parameters
        ----------
        order : int
        manner : int
        """
        self._order = order
        self._manner = manner

    def __str__(self):
        return f"Sparse Gauss-Hermite Quadrature Filter. order: {self._order}; manner: {self._manner}"

    def eval_set_size(self, model: StateSpaceModel) -> int:
        raise Exception("Not implemented yet")

    def evaluate_points_and_weights(self, model: StateSpaceModel) -> Tuple[np.ndarray, np.ndarray]:
        raise Exception("Not implemented yet")


class EstimateType(Enum):
    unknown = 0
    mean = 1
    median = 2


@dataclass(init=True, repr=True, eq=True, order=False)
class BootstrapDataSet:
    particles: np.ndarray
    weights: np.ndarray

    @property
    def capacity(self) -> int:
        return self.particles.shape[1]


class ResampleType(Enum):
    unknown = 0,
    residual = 1,
    residual2 = 2,
    multinomial = 3,
    stratified = 4,
    systematic = 5


class ResampleStrategy(ABC):
    @abstractmethod
    def resample(self, weights: np.ndarray) -> np.ndarray:
        """
        Performs the resampling algorithm used by particle filters
        :param weights: list-like of float list of weights as floats
        :return: np array of ints, array of indexes into the weights defining the resample.
        """
        pass

    @staticmethod
    def resolve(resample_type: ResampleType) -> ResampleStrategy:
        if resample_type is ResampleType.unknown:
            raise Exception("Can't construct strategy for resample_type=unknown")
        elif resample_type is ResampleType.residual:
            return ResidualResampleStrategy()
        elif resample_type is ResampleType.residual2:
            return ResidualResample2Strategy()
        elif resample_type is ResampleType.multinomial:
            return MultinomialResampleStrategy()
        elif resample_type is ResampleType.stratified:
            return StratifiedResampleStrategy()
        elif resample_type is ResampleType.systematic:
            return SystematicResampleStrategy()
        else:
            raise Exception(f"Unknown resampling type, i.e. resampling type {type} is not supported")


class ResidualResampleStrategy(ResampleStrategy):
    def __str__(self):
        return "Residual resample algorithm"

    def resample(self, weights: np.ndarray) -> np.ndarray:
        particles_count = len(weights)
        input_index = range(particles_count)

        out_index = np.zeros(particles_count, dtype=int)

        # first integer part
        weights_residual = particles_count * weights
        integer_weights_kind = np.fix(weights_residual)

        residual_particles_count = int(particles_count - np.sum(integer_weights_kind))

        if residual_particles_count > 0:
            weights_residual = (weights_residual - integer_weights_kind) / residual_particles_count
            cum_dist = np.cumsum(weights_residual)

            # generate N (N = residual_particles_count) ordered random variables uniformly distributed in [0, 1]
            u = np.flip(
                np.cumprod(
                    np.power(
                        np.random.rand(residual_particles_count),
                        1 / np.asarray(list(range(1, residual_particles_count + 1))[::-1])
                    )
                )
            )

            j = 0
            for i in range(residual_particles_count):
                while u[i] > cum_dist[j]:
                    j += 1

                integer_weights_kind[j] += 1

        index = 0
        for i in range(particles_count):
            if integer_weights_kind[i] > 0:
                upper_index = int(index + integer_weights_kind[i])

                for j in range(index, upper_index):
                    out_index[j] = int(input_index[i])

            index = int(index + integer_weights_kind[i])

        return out_index


class ResidualResample2Strategy(ResampleStrategy):
    def __str__(self):
        return "Residual resample algorithm (2)"

    """
    Residual resample algorithm.
    J. S. Liu and R. Chen. Sequential Monte Carlo methods for dynamic systems.
    Journal of the American Statistical Association, 93(443):1032–1044, 1998.
    """

    def resample(self, weights: np.ndarray) -> np.ndarray:
        n = len(weights)

        # take bool(N*w) copies of each weight to find out what particle will survive after re-sampling,
        # particle will survive if weights is 'big enough'.
        # the procedure  ensures particles with the same weight are drawn uniformly
        renormalized_weights = np.floor(n * np.asarray(weights)).astype(int)
        indexes = np.zeros(n, dtype=int)
        k = 0
        for i in range(n):
            for _ in range(renormalized_weights[i]):  # make copies
                indexes[k] = i
                k += 1

        # use multinormal resample on the residual to fill up the rest (it maximizes the variance of the samples)
        residual = weights - renormalized_weights
        residual /= sum(residual)
        cumulative_sum = m_utils.cum_sum_with_last_equal_to_one(residual)
        indexes[k:n] = np.searchsorted(cumulative_sum, random(n - k))

        return indexes


class MultinomialResampleStrategy(ResampleStrategy):
    """
    The naive form of roulette sampling where we compute the cumulative sum of the weights and then use binary search
    to select the resampled point based on a uniformly distributed random number.
    """

    def __str__(self):
        return "Multinomial resample algorithm"

    def resample(self, weights: np.ndarray) -> np.ndarray:
        cumulative_sum = m_utils.cum_sum_with_last_equal_to_one(weights)
        return np.searchsorted(cumulative_sum, random(len(weights)))


class StratifiedResampleStrategy(ResampleStrategy):
    """
    Performs the stratified resampling algorithm used by particle filters.
    This algorithm makes selections relatively uniformly across the particles.
    It divides the cumulative sum of the weights into N equal divisions, and then selects one particle randomly from each division.
    This guarantees that each sample is between 0 and 2/N apart.
    """

    def __str__(self):
        return "Stratified resampling algorithm"

    def resample(self, weights: np.ndarray) -> np.ndarray:
        n = len(weights)
        # make N subdivisions, and chose a random position within each one
        positions = (random(n) + range(n)) / n
        return resample_from_sub_divisions(weights, positions)


class SystematicResampleStrategy(ResampleStrategy):
    """
    Performs the systematic resampling algorithm used by particle filters.
    This algorithm separates the sample space into N divisions.
    A single random offset is used to choose where to sample from for all divisions.
    This guarantees that every sample is exactly 1/N apart.
    """

    def __str__(self):
        return "Systemic resampling algorithm"

    def resample(self, weights: np.ndarray) -> np.ndarray:
        n = len(weights)
        # make n subdivisions, and choose positions with a consistent random offset
        positions = (random() + np.arange(n)) / n
        return resample_from_sub_divisions(weights, positions)


def resample_from_sub_divisions(weights: np.ndarray, positions: np.ndarray) -> np.ndarray:
    n = len(weights)
    indexes = np.zeros(n, dtype=int)
    cumulative_sum = np.cumsum(weights)

    i = 0
    j = 0
    while i < n:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    return indexes


class Pf(PdfApproximationKalmanFilter):
    """
    Generic Particle filter. This filter is also known as the 'Bootstrap Particle Filter' or the 'Condensation Algorithm'.
    Particle filters implement the prediction-updating updates in an approximate manner.
    The samples from the distribution are represented by a set of particles;
    each particle has a likelihood weight assigned to it that represents the probability of that particle being sampled from
    the probability density function.

    This filter assumes the following standard state-space model:

    x(k) = f[x(k-1), v(k-1), u1(k-1)];
    z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        estimate_type - EstimateType, estimate type : 'mean', 'median', etc;
        resample_threshold - float, if the ratio of the 'effective particle set size' to the total number of particles
            drop below this threshold  i.e.  (effective size / total number of particles) < resample threshold
            the particles will be re-sampled. ("effective size" is always less than or equal to the total number of particles).
        resample_strategy - ResampleStrategy, resampling strategy.
    """

    def __init__(
            self,
            resample_threshold: float,
            estimate_type: Union[EstimateType, None] = None,
            resample_strategy: Union[ResampleStrategy, None] = None
    ):
        super().__init__(estimate_type)
        self._resample_threshold = resample_threshold
        self._resample_strategy = resample_strategy if resample_strategy is not None else ResidualResample2Strategy()

    def __str__(self):
        return f"Resample threshold: {self._resample_threshold}; " \
               f"Estimate type: {self._estimate_type}; " \
               f"Resampling method: {self._resample_strategy}"

    def estimate(
            self,
            data_set: BootstrapDataSet,
            observation: np.ndarray,
            model: StateSpaceModel,
            ctrl_x: Optional[np.ndarray],
            ctrl_z: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, BootstrapDataSet]:
        """
        Estimates state mean (x(k))of dynamic system at time k based on noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)].

        :param data_set: BootstrapDataSet, particle filter data structure (contains set of particles as well as their corresponding weights).
        :param observation:
        :param observation: noisy observations starting at time k, i.e. z(k).
        :param model: inference state space model, which fully describes filtration issue (evolution of state, relation between state and
            observation, state noise, observation noise, etc).
        :param ctrl_x: exogenous input to state transition function starting at time k-1, i.e. u1(k-1).
        :param ctrl_z: exogenous input to state observation function starting at time k, i.e. u2(k).
        :return: tuple of following elements:
            estimate: np array, estimates of state at time k, i.e. E[x(t)|z(1), z(2), ..., z(t)] for t = k;
            data_set: BootstrapDataSet, updated Particle filter data structure. Contains set of particles as well as their corresponding weights.
        """
        state_noise = model.state_noise.sample(data_set.capacity)
        particles_predicted = model.transition_func(data_set.particles, state_noise, ctrl_x)

        # Evaluate importance weights
        likelihood = model.likelihood(np.tile(observation, (1, data_set.capacity)), particles_predicted, ctrl_z) + 1e-99
        weights = data_set.weights * likelihood
        weights /= np.sum(weights)

        data_set_updated = self._resample(BootstrapDataSet(particles_predicted, weights))
        estimate = self.eval_final_state_estimate(data_set_updated, model)

        return estimate, data_set_updated

    def _resample(self, data_set: BootstrapDataSet) -> BootstrapDataSet:
        resample_threshold = round(self._resample_threshold * data_set.capacity)
        effective_size = 1 / np.sum((data_set.weights ** 2))

        if effective_size >= resample_threshold:
            return BootstrapDataSet(data_set.particles, data_set.weights)

        out_index = self._resample_strategy.resample(data_set.weights)
        particles = data_set.particles[:, out_index]
        weights = np.tile(1 / data_set.capacity, data_set.capacity)
        return BootstrapDataSet(particles, weights)


class Gspf(PdfApproximationKalmanFilter):
    """
    Gaussian Sum Particle Filter.
    The filter approximate the filtering and predictive distributions by weighted Gaussian mixtures and are basically
    banks of Gaussian particle filters. Then, we extend the use of Gaussian particle filters and Gaussian sum particle filters to
    dynamic state space (DSS) models with non-Gaussian noise. With non-Gaussian noise approximated by Gaussian mixtures, the non-Gaussian
    noise models are approximated by banks of Gaussian noise models, and Gaussian mixture filters are developed using
    algorithms developed for Gaussian noise models.

    For more details please see:
        Jayesh H. Kotecha and Petar M. Djuric, "Gaussian Sum Particle Filtering for Dynamic State Space Models",
        Proceedings of ICASSP-2001, Salt Lake City, Utah, May 2001.

    This filter assumes the following standard state-space model:

    x(k) = f[x(k-1), v(k-1), u1(k-1)];
    z(k) = h[x(k), n(k), u2(k)].

    where:
       x  - is the system state;
       v  - the process noise;
       n  - the observation noise;
       u1 - the exogenous input to the state;
       f  - the transition function;
       u2 - the exogenous input to the state observation function;
       z  - the noisy observation of the system.

    Filter parameters:
        estimate_type - EstimateType, estimate type : 'mean', 'median', etc;
        resample_threshold - float, if the ratio of the 'effective particle set size' to the total number of particles
            drop below this threshold  i.e.  (effective size / total number of particles) < resample threshold
            the particles will be re-sampled. ("effective size" is always less than or equal to the total number of particles).
        resample_strategy - ResampleStrategy, resampling strategy.
    """

    def __init__(
            self,
            resample_threshold: float,
            n_samples: int,
            estimate_type: Union[EstimateType, None] = None,
            resample_strategy: Union[ResampleStrategy, None] = None
    ):
        super().__init__(estimate_type)
        self._resample_threshold = resample_threshold
        self._n_samples = n_samples
        self._resample_strategy = resample_strategy if resample_strategy is not None else ResidualResampleStrategy()

    def __str__(self):
        return f"Resample threshold: {self._resample_threshold}; " \
               f"Estimate type: {self._estimate_type}; " \
               f"Resampling method: {self._resample_strategy}; " \
               f"Particles count: {self._n_samples}"

    def estimate(
            self,
            gmi: GaussianMixtureInfo,
            observation: np.ndarray,
            model: StateSpaceModel,
            ctrl_x: Optional[np.ndarray],
            ctrl_z: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, GaussianMixtureInfo]:
        """
        Estimates state mean (x(k))of dynamic system at time k based on noisy observations starting at time k (z(k)).
        The filter assumes the following standard state-space model:

        x(k) = f[x(k-1), v(k-1), u1(k-1)];
        z(k) = h[x(k), n(k), u2(k)].

        :param gmi: GaussianMixture, model of gaussian mixture probability distribution.
        :param observation:
        :param observation: noisy observations starting at time k, i.e. z(k).
        :param model: inference state space model, which fully describes filtration issue (evolution of state, relation between state and
            observation, state noise, observation noise, etc).
        :param ctrl_x: exogenous input to state transition function starting at time k-1, i.e. u1(k-1).
        :param ctrl_z: exogenous input to state observation function starting at time k, i.e. u2(k).
        :return: tuple of following elements:
            estimate: np array, estimates of state at time k, i.e. E[x(t)|z(1), z(2), ..., z(t)] for t = k;
            data_set: BootstrapDataSet, updated Particle filter data structure. Contains set of particles
             as well as their corresponding weights.
        """
        n_components = gmi.n_components * model.state_noise.n_components

        x_weights_new = np.zeros(n_components)
        state_mean_new = np.zeros((n_components, model.state_dim))
        state_sqrt_cov_new = np.zeros((n_components, model.state_dim, model.state_dim))

        # Time update (prediction)
        x_current_buf = np.random.randn(gmi.n_components, model.state_dim, self._n_samples)
        for g in range(gmi.n_components):
            mean_g = np.tile(gmi.means[g, :], (model.state_dim, self._n_samples))
            x_current_buf[g, :, :] = gmi.sqrt_covariances[g, :, :] @ x_current_buf[g, :, :] + mean_g

        x_predicted_buf = np.zeros((n_components, model.state_dim, self._n_samples))
        noise_buf = np.random.randn(n_components, model.state_dim, self._n_samples)
        x_current_buf_m = np.zeros(n_components)  # TODO
        x_current_buf_c = np.zeros(n_components)  # TODO
        for k in range(model.state_noise.n_components):
            mean_k = np.tile(model.state_noise.mean[k, :], (model.state_dim, self._n_samples))
            for g in range(gmi.n_components):
                gk = g + k * gmi.n_components
                x_noise_buf = model.state_noise.covariance[k, :, :] @ noise_buf[gk, :, :] + mean_k
                x_predicted_buf[gk, :, :] = model.transition_func(x_current_buf[g, :, :], x_noise_buf, ctrl_x)
                x_weights_new[gk] = gmi.weights[g] * model.state_noise.weights[k]
                x_current_buf_m[gk] = np.mean(x_predicted_buf[gk, :, :])  # TODO
                x_current_buf_c[gk] = np.cov(x_predicted_buf[gk, :, :])  # TODO

        x_weights_new /= np.sum(x_weights_new)

        for g in range(n_components):
            mean_component_g = np.sum(x_predicted_buf[g, :, :], 1) / self._n_samples
            state_mean_new[g, :] = mean_component_g
            x_diff = x_predicted_buf[g, :, :] - np.tile(mean_component_g, (1, self._n_samples))
            _, cov_sqrt_g = np.linalg.qr(x_diff.T)
            state_sqrt_cov_new[g, :, :] = np.transpose(cov_sqrt_g) / np.sqrt(self._n_samples - 1)

        # Observation / measurement update (correction)
        z_expected_buf = np.zeros((n_components, model.state_dim, self._n_samples))
        importance_w = np.zeros((n_components, self._n_samples))
        obs = np.tile(observation, (1, self._n_samples))
        noise_buf = np.random.randn(n_components, model.state_dim, self._n_samples)
        importance_w_max = np.zeros((16,))  # TODO:
        for g in range(n_components):
            mean_component_g = np.tile(state_mean_new[g, :], (1, self._n_samples))
            z_expected_buf[g, :, :] = state_sqrt_cov_new[g, :, :] @ noise_buf[g, :, :] + mean_component_g
            importance_w[g, :] = model.likelihood(obs, z_expected_buf[g, :, :], ctrl_z)
            importance_w_max[g] = np.max(importance_w[g, :])  # TODO:

        weight_norm = 0
        for g in range(n_components):
            tmp_w = importance_w[g, :]
            tmp_weights_cum = np.sum(tmp_w)
            mean_component_g = np.sum(
                np.tile(tmp_w, (model.state_dim, 1)) * z_expected_buf[g, :, :], 1
            ) / tmp_weights_cum

            state_mean_new[g, :] = mean_component_g

            w = np.tile(np.sqrt(tmp_w), (model.state_dim, 1))
            x_diff = np.transpose(
                (w * (z_expected_buf[g, :, :] - np.tile(mean_component_g, (1, self._n_samples))))
            )
            _, cov_sqrt_g = np.linalg.qr(x_diff)
            state_sqrt_cov_new[g, :, :] = cov_sqrt_g.T / np.sqrt(tmp_weights_cum)

            x_weights_new[g] *= tmp_weights_cum
            weight_norm += tmp_weights_cum

        x_weights_new /= weight_norm
        x_weights_new /= np.sum(x_weights_new)

        if self._estimate_type is EstimateType.mean:
            estimate = np.sum(
                np.multiply(
                    np.repeat(np.matrix(x_weights_new), model.state_dim, 0),
                    np.matrix(state_mean_new).T
                ),
                1
            )
        else:
            raise Exception(f"Estimate type '{self._estimate_type}' is not supported")

        idx = self.resample_mixture_components(n_components, gmi.n_components, x_weights_new)
        gmi.set_params(
            state_mean_new[idx, :],
            state_sqrt_cov_new[idx, :, :],
            np.ones((gmi.n_components,)) / gmi.n_components,
            CovarianceType.sqrt
        )

        return np.array(estimate).flatten(), gmi

    def resample_mixture_components(self, n_components: int, n_x_components: int, weights: np.ndarray) -> np.ndarray:
        resample_idx = self._resample_strategy.resample(weights)
        r_idx = np.argsort(np.random.rand(n_components))[0:n_x_components]

        return resample_idx[r_idx]
