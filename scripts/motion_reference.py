"""Standalone G1 reference-motion loader for the Sonic low-latency contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
MUJOCO_TO_ISAACLAB = np.argsort(ISAACLAB_TO_MUJOCO)


def resample_linear(values: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("values must have shape [frames, features] with at least two frames")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("fps must be positive")
    duration = (values.shape[0] - 1) / source_fps
    target_times = np.arange(0.0, duration, 1.0 / target_fps)
    source_times = np.arange(values.shape[0], dtype=np.float64) / source_fps
    result = np.empty((len(target_times), values.shape[1]), dtype=np.float64)
    for column in range(values.shape[1]):
        result[:, column] = np.interp(target_times, source_times, values[:, column])
    return result


def _resample_quaternion_xyzw(quaternions: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    duration = (quaternions.shape[0] - 1) / source_fps
    target_times = np.arange(0.0, duration, 1.0 / target_fps)
    source_times = np.arange(quaternions.shape[0], dtype=np.float64) / source_fps
    result = np.empty((len(target_times), 4), dtype=np.float64)
    for output_index, target_time in enumerate(target_times):
        right = min(int(np.searchsorted(source_times, target_time, side="right")), len(source_times) - 1)
        left = max(0, right - 1)
        if right == left:
            fraction = 0.0
        else:
            fraction = (target_time - source_times[left]) / (source_times[right] - source_times[left])
        q0 = quaternions[left] / np.linalg.norm(quaternions[left])
        q1 = quaternions[right] / np.linalg.norm(quaternions[right])
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            interpolated = q0 + fraction * (q1 - q0)
        else:
            theta = np.arccos(np.clip(dot, -1.0, 1.0))
            sin_theta = np.sin(theta)
            interpolated = (np.sin((1.0 - fraction) * theta) * q0 + np.sin(fraction * theta) * q1) / sin_theta
        result[output_index] = interpolated / np.linalg.norm(interpolated)
    return result


def _quaternion_xyzw_to_rotation_6d(quaternions: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternions.T
    result = np.empty((len(quaternions), 6), dtype=np.float64)
    result[:, 0] = 1 - 2 * (y * y + z * z)
    result[:, 1] = 2 * (x * y - z * w)
    result[:, 2] = 2 * (x * y + z * w)
    result[:, 3] = 1 - 2 * (x * x + z * z)
    result[:, 4] = 2 * (y * z - x * w)
    result[:, 5] = 1 - 2 * (x * x + y * y)
    # The deployment observation flattens the first two columns row-wise.
    return result.reshape(-1, 3, 2).reshape(-1, 6)


@dataclass
class G1Reference:
    joint_positions_mujoco: np.ndarray
    joint_velocities_mujoco: np.ndarray
    root_orientation_6d: np.ndarray
    root_positions: np.ndarray
    fps: float
    root_quaternions_xyzw: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.joint_positions_mujoco = np.asarray(self.joint_positions_mujoco, dtype=np.float64)
        self.joint_velocities_mujoco = np.asarray(self.joint_velocities_mujoco, dtype=np.float64)
        self.root_orientation_6d = np.asarray(self.root_orientation_6d, dtype=np.float64)
        self.root_positions = np.asarray(self.root_positions, dtype=np.float64)
        if self.root_quaternions_xyzw is None:
            self.root_quaternions_xyzw = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(self.joint_positions_mujoco), 1))
        else:
            self.root_quaternions_xyzw = np.asarray(self.root_quaternions_xyzw, dtype=np.float64)
        if self.joint_positions_mujoco.ndim != 2 or self.joint_positions_mujoco.shape[1] != 29:
            raise ValueError("joint positions must have shape [frames, 29]")
        if self.joint_velocities_mujoco.shape != self.joint_positions_mujoco.shape:
            raise ValueError("joint velocities must match joint positions")
        if self.root_orientation_6d.shape != (len(self.joint_positions_mujoco), 6):
            raise ValueError("root orientation must have shape [frames, 6]")
        if self.root_quaternions_xyzw.shape != (len(self.joint_positions_mujoco), 4):
            raise ValueError("root quaternions must have shape [frames, 4]")

    @classmethod
    def from_joblib(cls, path: Path, target_fps: float = 50.0) -> "G1Reference":
        import joblib

        loaded = joblib.load(path)
        data = next(iter(loaded.values())) if len(loaded) == 1 and isinstance(next(iter(loaded.values())), dict) else loaded
        source_fps = float(data.get("fps", 30.0))
        positions = resample_linear(data["dof"], source_fps, target_fps)
        root_positions = resample_linear(data["root_trans_offset"], source_fps, target_fps)
        root_quaternions = _resample_quaternion_xyzw(data["root_rot"], source_fps, target_fps)
        velocities = np.gradient(positions, 1.0 / target_fps, axis=0, edge_order=1)
        return cls(
            positions,
            velocities,
            _quaternion_xyzw_to_rotation_6d(root_quaternions),
            root_positions,
            target_fps,
            root_quaternions,
        )

    @property
    def num_frames(self) -> int:
        return len(self.joint_positions_mujoco)

    @property
    def joint_positions_isaac(self) -> np.ndarray:
        return self.joint_positions_mujoco[:, MUJOCO_TO_ISAACLAB]

    @property
    def joint_velocities_isaac(self) -> np.ndarray:
        return self.joint_velocities_mujoco[:, MUJOCO_TO_ISAACLAB]

    def future_window(self, frame: int, count: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.clip(np.arange(frame, frame + count), 0, self.num_frames - 1)
        return self.joint_positions_isaac[indices], self.joint_velocities_isaac[indices], self.root_orientation_6d[indices]
