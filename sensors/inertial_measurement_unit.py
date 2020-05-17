from __future__ import annotations

from dataclasses import dataclass
from os import linesep

import numpy as np
from scipy.constants import g
from scipy.stats import norm

from bayesian_framework.inference.stochastic_models.stochastic_processes import WienerProcessIterative
from motions.angular_velocity_models import AngularVelocityProvider
from motions.non_gravity_acceleration import NonGravityAccelerationProvider
from utils.matrix_utils import get_locked_copy


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=True, frozen=True)
class GyroParams:
    g_sensitive_bias: np.ndarray
    scale_factor: np.ndarray
    noise_std_var: float
    bias_mu: np.ndarray
    bias_sigma: float

    def __str__(self):
        param_names = [
            "g sensitive bias={0}".format(self.g_sensitive_bias),
            "scale factor = {0}".format(self.scale_factor),
            "noise std var = {0} rad".format(self.noise_std_var),
            "bias mean = {0} rad".format(self.bias_mu),
            "bias sigma = {0] rad".format(self.bias_sigma)
        ]

        params_str = ";{0}".format(linesep).join(param_names)
        return "Gyro params:{0}{1}".format(linesep, params_str)

    def __post_init__(self):
        object.__setattr__(self, 'g_sensitive_bias', get_locked_copy(self.g_sensitive_bias))
        object.__setattr__(self, 'scale_factor', get_locked_copy(self.scale_factor))
        object.__setattr__(self, 'noise_var', self.noise_std_var)
        object.__setattr__(self, 'bias_mu', get_locked_copy(self.bias_mu))
        object.__setattr__(self, 'bias_sigma', self.bias_sigma)


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=True, frozen=True)
class AccelerometerParams:
    level_arm: np.ndarray
    scale_factor: np.ndarray
    noise_std_var: float
    bias_mu: np.ndarray
    bias_sigma: float

    def __str__(self):
        param_names = [
            "level arm = {0}".format(self.level_arm),
            "scale factor = {0}".format(self.scale_factor),
            "noise std var = {0} km/sec**2".format(self.noise_std_var),
            "bias mean = {0} km/sec**2".format(self.bias_mu),
            "bias sigma = {0} km/sec**2".format(self.bias_sigma)
        ]

        param_str = ";{0}".format(linesep).join(param_names)
        return "Accelerometer params: {0}{1}".format(linesep, param_str)

    def __post_init__(self):
        object.__setattr__(self, 'level_arm', get_locked_copy(self.level_arm))
        object.__setattr__(self, 'scale_factor', get_locked_copy(self.scale_factor))
        object.__setattr__(self, 'noise_std_var', self.noise_std_var)
        object.__setattr__(self, 'bias_mu', get_locked_copy(self.bias_mu))
        object.__setattr__(self, 'bias_sigma', self.bias_sigma)


class InertialMeasurementUnit:
    """
    Describe inertial measurement unit (accelerometer and gyro).
    Provides a measurement of acceleration (km / sec**2) from three body axes accelerometer in body fixed frame
    and a measurement of angular velocity (radian per second) from three body axes gyro in body fixed frame.
    """

    def __init__(
            self,
            gyro_params: GyroParams,
            accelerometer_params: AccelerometerParams,
            angular_velocity_provider: AngularVelocityProvider,
            acceleration_provider: NonGravityAccelerationProvider,
            dt: float
    ):
        """

        :param gyro_params: Gyro parameters.
        :param accelerometer_params: Accelerometer parameters.
        :param angular_velocity_provider: Provider of angular velocity.
        :param acceleration_provider: Provider of acceleration.
        :param dt: float, The time step.
        """
        self._gyro_params = gyro_params
        self._accelerometer_params = accelerometer_params
        self._angular_velocity_provider = angular_velocity_provider
        self._acceleration_provider = acceleration_provider
        self._gyro_bias_process = WienerProcessIterative(shift=gyro_params.bias_mu, delta=gyro_params.bias_sigma, dt=dt)
        self._acc_bias_process = WienerProcessIterative(shift=accelerometer_params.bias_mu, delta=accelerometer_params.bias_sigma, dt=dt)

    def __str__(self):
        return "Accelerometer params:{0}{1}Gyro params:{2}{3}".format(linesep, self._accelerometer_params, linesep, self._gyro_params)

    def eval_angular_velocity(self, k: int) -> np.ndarray:
        """
        Provide angular velocity (radian per second) in 3D space in body fixed frame at time t(k).
        :param k: int, digit time k, t(k), [-].
        :return: np array, Angular velocity in 3D space in body fixed frame, [rad / sec].
        """
        w = self._angular_velocity_provider.eval(k)
        b = self._gyro_bias_process.eval(k)
        s = self._gyro_params.scale_factor
        ga = self._gyro_params.g_sensitive_bias @ self._acceleration_provider.eval(k) / g
        n = norm.rvs(size=(3,), scale=self._gyro_params.noise_std_var)

        return s @ w + b + ga + n

    def eval_acceleration(self, k: int) -> np.ndarray:
        """
        Provide non gravity acceleration (km / sec ** 2) in 3D space in body fixed frame at time t(k).
        :param k: int, digit time k, k for t(k), [-].
        :return: np array, Non gravity acceleration in 3D space in body fixed frame, [km / sec**2].
        """
        n = norm.rvs(size=(3,), scale=self._accelerometer_params.noise_std_var)
        s = self._accelerometer_params.scale_factor
        b = self._acc_bias_process.eval(k)
        a = self._acceleration_provider.eval(k)
        w = self._angular_velocity_provider.eval(k)
        w_la = np.cross(self._accelerometer_params.level_arm, w)
        w_w_la = np.cross(w, np.cross(w, self._accelerometer_params.level_arm))

        return s @ (w_la + w_w_la + a) + b + n
