# todo: consider sun and moon influence, should be part of non-gravity acceleration
# todo: consider non-spherical part of gravity earth acceleration, should be part of gravity acceleration

import numpy as np
import quaternion as npq
from scipy.integrate import RK45, solve_ivp

from motions.models import KinematicState
from utils.matrix_utils import vect_to_quaternion


def uniform_acceleration_motion_equation(x: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    """
    Equation of motion (differential equation) with constant acceleration and without angular motion.
    NB: All vectors must be in the same frame.
    NB: All vectors must be in the same measurements units, e.g. x is [km, km/sec] and acceleration is [km/sec^2].
    :param x: position (vector elements from 0 till 3) and velocity (vector elements from 3 till 6).
    :param acceleration: acceleration vector (in 3D space)
    :return: Derivative of position and velocity vectors.
    """
    d_x = np.zeros(6)
    d_x[:3] = x[3:]
    d_x[3:] = acceleration

    return d_x


def uniform_angular_motion_equation(quaternion: np.ndarray, angular_velocity: np.ndarray) -> npq.quaternion:
    """
    Equation of uniform angular motion (with constant angular velocity).
    :param quaternion: quaternion that represent orientation of body frame according to inertial frame.
    :param angular_velocity: angular velocity of body frame according to inertial frame in inertial frame, [rad/sec].
    :return: Derivative of quaternion
    """
    q_new = -0.5 * npq.from_float_array(quaternion) * vect_to_quaternion(angular_velocity)
    return npq.as_float_array(q_new)


def solve_kinematic_motion_equation(
        time_span: np.ndarray,
        state: KinematicState,
        acceleration: np.ndarray,
        gravity_acceleration: np.ndarray,
        angular_velocity: np.ndarray) -> KinematicState:
    """
    Solve kinematic differential equation of motion.
    Describe motion in phase space that represent spatial position and angular position of body.
    'Body' is a rigid body (physical model).
    Phase space is:
        - position vector (cartesian coordinates),
        - velocity vector (cartesian coordinates),
        - quaternion that represent angular position according to inertial frame.
    NB: All vectors related to uniform acceleration motion must be in the same measurements units, e.g. x is [km, km/sec],
         acceleration and gravity_acceleration are [km/sec^2] .
    NB: angular_velocity must be in [rad, sec].
    :param time_span: time span.
    :param state: value of state space vector of body in phase space.
    :param acceleration: acceleration of body that caused by all non-gravitational forces in body frame.
    :param gravity_acceleration: acceleration of body that caused by gravitational force.
    :param angular_velocity: angular velocity of rotation of body frame respect to inertial frame, [rad, sec].
    :return: State  space vector at current time, i.e.
        - position vector (cartesian coordinates),
        - velocity vector (cartesian coordinates),
        - quaternion that represent angular position according to inertial frame.
    """
    acc = npq.rotate_vectors(state.quaternion, acceleration) + gravity_acceleration
    x0 = np.hstack((state.position, state.velocity))
    x_new = solve_ivp(lambda t, x: uniform_acceleration_motion_equation(x, acc), time_span, x0, method=RK45)
    q_new = solve_ivp(lambda t, q: uniform_angular_motion_equation(q, angular_velocity), time_span, state.quaternion.components, method=RK45)

    return KinematicState(x_new.y[:3, -1], x_new.y[3:, -1], npq.from_float_array(q_new.y[:, -1]).normalized())
