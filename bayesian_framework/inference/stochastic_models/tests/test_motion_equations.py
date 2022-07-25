import quaternion as npq
import numpy as np
from numpy.testing import assert_array_almost_equal

from kinematics.models import KinematicState
from kinematics.motion_equations import solve_kinematic_motion_equation


def test_solve_kinematic_motion_equation():
    x0 = KinematicState([1, 1, 1], [2, 2, 2], npq.from_float_array([1, 0, 0, 0]))
    x1 = solve_kinematic_motion_equation((0, 100), x0, np.asarray([1, 1, 1]), np.asarray[0., 9.8, 0], np.asarray([0.25, 0.5, 0.75]))

    assert_array_almost_equal(x1.position, [5201.0, 54201.0, 5201.0], decimal=1)
    assert_array_almost_equal(x1.velocity, [102.0, 1082.0, 102.0], decimal=1)
    assert_array_almost_equal(x1.quaternion.components, [-0.94307022, -0.0888894,  -0.1777788,  -0.2666682], decimal=8)
