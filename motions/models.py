from copy import deepcopy
from typing import Iterable, List

import numpy as np
import quaternion as npq


class KinematicState:
    """
    Represent phase state of rigid body during kinematic motion:
        Position vector (cartesian coordinates), [km];
        Velocity vector (cartesian coordinates), [km/sec];
        Quaternion that represent angular position according to inertial frame.
    """

    def __init__(self, position: Iterable[float], velocity: Iterable[float], quaternion: npq.quaternion):
        """
        Creates new instance of KinematicState.
        :param position: position vector (cartesian coordinates), [km].
        :param velocity: velocity vector (cartesian coordinates), [km/sec].
        :param quaternion: quaternion that represent angular position according to inertial frame.
        """
        self._position = np.asarray(position).copy()
        self._velocity = np.asarray(velocity).copy()
        self._quaternion = deepcopy(quaternion)

    def __str__(self):
        pattern = "Position: x={0} km, y={1} km, z={2} km; Velocity: vx={3} km/sec, vy={4} km/sec, vz={5} km/sec; Quaternion: qw={6}, qx={7}, qy={8}, qz={9}"
        return pattern.format(
            self._position[0], self._position[1], self._position[2],
            self._velocity[0], self._velocity[1], self._velocity[2],
            self._quaternion.w, self._quaternion.x, self._quaternion.y, self._quaternion.z
        )

    @property
    def position(self) -> np.ndarray:
        return self._position.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity.copy()

    @property
    def quaternion(self) -> npq.quaternion:
        return deepcopy(self._quaternion)

    @property
    def r_x(self) -> float:
        return self._position[0]

    @property
    def r_y(self) -> float:
        return self._position[1]

    @property
    def r_z(self) -> float:
        return self._position[2]

    @property
    def v_x(self) -> float:
        return self._velocity[0]

    @property
    def v_y(self) -> float:
        return self._velocity[1]

    @property
    def v_z(self) -> float:
        return self._velocity[2]

    @property
    def q_w(self) -> float:
        return self._quaternion.w

    @property
    def q_x(self) -> float:
        return self._quaternion.x

    @property
    def q_y(self) -> float:
        return self._quaternion.y

    @property
    def q_z(self) -> float:
        return self._quaternion.z
