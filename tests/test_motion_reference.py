import numpy as np

from scripts.motion_reference import G1Reference, resample_linear


def test_resample_linear_preserves_endpoints_and_target_rate():
    source = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    result = resample_linear(source, source_fps=2.0, target_fps=4.0)

    np.testing.assert_allclose(result[:, 0], [0.0, 0.5, 1.0, 1.5])


def test_reference_observation_contains_ten_future_frames_in_isaac_order():
    reference = G1Reference(
        joint_positions_mujoco=np.arange(30 * 29, dtype=np.float64).reshape(30, 29),
        joint_velocities_mujoco=np.ones((30, 29), dtype=np.float64),
        root_orientation_6d=np.tile(np.array([1, 0, 0, 0, 1, 0]), (30, 1)),
        root_positions=np.zeros((30, 3), dtype=np.float64),
        fps=50.0,
    )

    positions, velocities, orientations = reference.future_window(3, 10)

    assert positions.shape == (10, 29)
    assert velocities.shape == (10, 29)
    assert orientations.shape == (10, 6)
    np.testing.assert_array_equal(positions[0], reference.joint_positions_isaac[3])
    np.testing.assert_array_equal(positions[-1], reference.joint_positions_isaac[12])
