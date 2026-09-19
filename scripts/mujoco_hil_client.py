#!/usr/bin/env python3
"""Headless/interactive MuJoCo client for the RK3576 Sonic HIL server."""

from __future__ import annotations

import argparse
import socket
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from scripts.hil_protocol import ACTION_ELEMENTS, pack_request, unpack_response
from scripts.motion_reference import G1Reference
from scripts.render_mailbox import LatestState

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "assets/g1_mocap_29dof/g1_mocap_29dof.xml"
DEFAULT_REFERENCE = ROOT / "assets/reference/walk_forward_amateur_001__A001.pkl"
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
MUJOCO_TO_ISAACLAB = np.argsort(ISAACLAB_TO_MUJOCO)
DEFAULT_ANGLES = np.array(
    [-0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669, -0.363, 0,
     0, 0, 0, 0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0],
    dtype=np.float64,
)
ARMATURE = np.array([0.025101925, 0.025101925, 0.010177520, 0.025101925, 0.003609725, 0.003609725,
                     0.025101925, 0.025101925, 0.010177520, 0.025101925, 0.003609725, 0.003609725,
                     0.010177520, 0.003609725, 0.003609725, 0.003609725, 0.003609725, 0.003609725,
                     0.003609725, 0.003609725, 0.00425, 0.00425, 0.003609725, 0.003609725,
                     0.003609725, 0.003609725, 0.003609725, 0.00425, 0.00425])
EFFORT = np.array([139, 139, 88, 139, 25, 25, 139, 139, 88, 139, 25, 25, 88, 25, 25,
                   25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5], dtype=np.float64)
ACTION_SCALE = 0.25 * EFFORT / (ARMATURE * (10 * 2 * np.pi) ** 2)


def recv_full(sock: socket.socket, size: int) -> bytes:
    chunks = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise ConnectionError("RK3576 server closed the connection")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def isaac_order(values_mujoco: np.ndarray) -> np.ndarray:
    return values_mujoco[MUJOCO_TO_ISAACLAB]


def gravity_body(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    rotation = data.xmat[pelvis_id].reshape(3, 3)
    return rotation.T @ np.array([0.0, 0.0, -1.0])


def make_observations(model: mujoco.MjModel, data: mujoco.MjData, history: list[np.ndarray], reference: G1Reference, reference_frame: int) -> tuple[np.ndarray, np.ndarray]:
    encoder = np.zeros(1247, dtype=np.float32)
    encoder[0] = 0.0  # low-latency G1 reference mode
    reference_q, reference_dq, reference_orientation = reference.future_window(reference_frame, 10)
    encoder[4:294] = reference_q.reshape(-1)
    encoder[294:584] = reference_dq.reshape(-1)
    encoder[584:644] = reference_orientation.reshape(-1)

    decoder = np.zeros(994, dtype=np.float32)
    for frame, state in enumerate(history):
        decoder[64 + frame * 3:67 + frame * 3] = state[0:3]
        decoder[94 + frame * 29:123 + frame * 29] = state[3:32]
        decoder[384 + frame * 29:413 + frame * 29] = state[32:61]
        decoder[674 + frame * 29:703 + frame * 29] = state[61:90]
        decoder[964 + frame * 3:967 + frame * 3] = state[90:93]
    return encoder, decoder


class RenderWorker:
    """Own the GUI and a separate MuJoCo data object outside the control loop."""

    def __init__(self, xml: Path, mailbox: LatestState, rate: float) -> None:
        self.xml = xml
        self.mailbox = mailbox
        self.rate = max(rate, 1.0)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="mujoco-render", daemon=True)
        self.error: BaseException | None = None

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        if self.error is not None:
            raise RuntimeError("render worker failed") from self.error

    def _run(self) -> None:
        try:
            render_model = mujoco.MjModel.from_xml_path(str(self.xml))
            render_data = mujoco.MjData(render_model)
            with mujoco.viewer.launch_passive(render_model, render_data) as viewer:
                while not self.stop_event.is_set() and viewer.is_running():
                    snapshot = self.mailbox.latest()
                    if snapshot is not None:
                        render_data.qpos[:] = snapshot
                        mujoco.mj_forward(render_model, render_data)
                    viewer.sync()
                    self.stop_event.wait(1.0 / self.rate)
        except BaseException as error:  # report after control loop has stopped
            self.error = error


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default="192.168.3.40")
    parser.add_argument("--port", type=int, default=39001)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-rate", type=float, default=60.0)
    parser.add_argument("--period", type=float, default=0.02)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    reference = G1Reference.from_joblib(args.reference, target_fps=50.0)
    data = mujoco.MjData(model)
    if model.nq != 36 or model.nu != ACTION_ELEMENTS:
        raise RuntimeError(f"expected G1 nq=36, nu=29; got nq={model.nq}, nu={model.nu}")
    data.qpos[7:36] = reference.joint_positions_mujoco[0]
    mujoco.mj_forward(model, data)
    mailbox = LatestState()
    renderer = RenderWorker(args.xml, mailbox, args.render_rate) if args.render else None
    if renderer is not None:
        renderer.start()
    history: list[np.ndarray] = []
    last_action = np.zeros(29, dtype=np.float32)
    for _ in range(10):
        history.append(np.zeros(93, dtype=np.float32))
    rtts = []
    control_ticks = []
    sequence = 0
    with socket.create_connection((args.host, args.port), timeout=10.0) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"connected to RK3576 {args.host}:{args.port}; xml={args.xml}")
        try:
            for step in range(args.steps):
                tick_started = time.perf_counter()
                q = isaac_order(data.qpos[7:36])
                dq = isaac_order(data.qvel[6:35])
                state = np.concatenate((data.qvel[3:6], q, dq, last_action, gravity_body(model, data)))
                history.pop(0)
                history.append(state.astype(np.float32))
                encoder, decoder = make_observations(model, data, history, reference, step)
                sequence += 1
                started = time.perf_counter()
                sock.sendall(pack_request(sequence, encoder, decoder))
                response = recv_full(sock, 16 + ACTION_ELEMENTS * 4)
                received_sequence, status, action = unpack_response(response)
                if received_sequence != sequence or status != 0:
                    raise RuntimeError(f"bad response sequence/status: {received_sequence}/{status}")
                rtts.append((time.perf_counter() - started) * 1000.0)
                last_action = action
                action_mujoco = action[ISAACLAB_TO_MUJOCO]
                target = DEFAULT_ANGLES + ACTION_SCALE * action_mujoco
                for _ in range(max(1, round(args.period / model.opt.timestep))):
                    torque = 8.0 * (target - data.qpos[7:36]) - 0.2 * data.qvel[6:35]
                    data.ctrl[:] = np.clip(torque, -1.0, 1.0)
                    mujoco.mj_step(model, data)
                if renderer is not None:
                    mailbox.publish(data.qpos)
                if step == 0 or (step + 1) % 100 == 0:
                    print(f"step={step + 1} rtt_ms={rtts[-1]:.3f} q0={data.qpos[7]:.3f} action0={action[0]:.3f}")
                control_ticks.append((time.perf_counter() - tick_started) * 1000.0)
        finally:
            if renderer is not None:
                renderer.stop()
    rtts_np = np.asarray(rtts)
    ticks_np = np.asarray(control_ticks)
    print(f"hil_steps={len(rtts)} avg_rtt_ms={rtts_np.mean():.3f} p99_rtt_ms={np.percentile(rtts_np, 99):.3f} "
          f"avg_control_tick_ms={ticks_np.mean():.3f} control_hz={1000.0 / ticks_np.mean():.2f}")


if __name__ == "__main__":
    main()
