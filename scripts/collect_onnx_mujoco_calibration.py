#!/usr/bin/env python3
"""Collect representative Sonic INT8 calibration inputs from ONNX + MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

from scripts.motion_reference import G1Reference
from scripts.mujoco_hil_client import (
    ACTION_SCALE_MUJOCO,
    DEFAULT_ANGLES,
    ISAACLAB_TO_MUJOCO,
    KD_MUJOCO,
    KP_MUJOCO,
    gravity_body,
    isaac_order,
    make_observations,
)
from scripts.render_mailbox import LatestState
from scripts.mujoco_hil_client import RenderWorker


def reset_episode(model: mujoco.MjModel, data: mujoco.MjData, reference: G1Reference) -> None:
    data.qpos[:] = 0.0
    data.qpos[0:3] = reference.root_positions[0]
    root_xyzw = reference.root_quaternions_xyzw[0]
    data.qpos[3:7] = [root_xyzw[3], root_xyzw[0], root_xyzw[1], root_xyzw[2]]
    data.qpos[7:36] = reference.joint_positions_mujoco[0]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--xml", type=Path, default=Path("assets/g1_mocap_29dof/g1_mocap_29dof.xml"))
    parser.add_argument("--reference", type=Path, default=Path("assets/reference/walk_forward_amateur_001__A001.pkl"))
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--decoder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--episode-length", type=int, default=256)
    parser.add_argument("--period", type=float, default=0.02)
    parser.add_argument("--pd-scale", type=float, default=0.1)
    parser.add_argument("--action-limit", type=float, default=1.0)
    parser.add_argument("--validate-only", action="store_true", help="run ONNX+MuJoCo without writing calibration samples")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-rate", type=float, default=60.0)
    args = parser.parse_args()
    if args.count <= 0 or args.episode_length <= 0:
        raise SystemExit("count and episode-length must be positive")

    reference = G1Reference.from_joblib(args.reference, target_fps=50.0)
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    if model.nq != 36 or model.nu != 29:
        raise RuntimeError(f"expected G1 nq=36, nu=29; got nq={model.nq}, nu={model.nu}")
    reset_episode(model, data, reference)
    mailbox = LatestState()
    renderer = RenderWorker(args.xml, mailbox, args.render_rate) if args.render else None
    if renderer is not None:
        renderer.start()

    encoder_session = ort.InferenceSession(str(args.encoder), providers=["CPUExecutionProvider"])
    decoder_session = ort.InferenceSession(str(args.decoder), providers=["CPUExecutionProvider"])
    encoder_input = encoder_session.get_inputs()[0].name
    decoder_input = decoder_session.get_inputs()[0].name

    encoder_samples: list[np.ndarray] = []
    decoder_samples: list[np.ndarray] = []
    history = [np.zeros(93, dtype=np.float32) for _ in range(10)]
    last_action = np.zeros(29, dtype=np.float32)
    reference_frame = 0
    resets = 0
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    min_root_z = float("inf")
    min_upright = float("inf")
    max_speed = 0.0

    while len(encoder_samples) < args.count:
        if not args.validate_only and len(encoder_samples) > 0 and len(encoder_samples) % args.episode_length == 0:
            reset_episode(model, data, reference)
            history = [np.zeros(93, dtype=np.float32) for _ in range(10)]
            last_action = np.zeros(29, dtype=np.float32)
            resets += 1

        q = isaac_order(data.qpos[7:36])
        dq = isaac_order(data.qvel[6:35])
        state = np.concatenate((data.qvel[3:6], q, dq, last_action, gravity_body(model, data)))
        history.pop(0)
        history.append(state.astype(np.float32))
        encoder_obs, decoder_obs = make_observations(model, data, history, reference, reference_frame)

        token = np.asarray(
            encoder_session.run(None, {encoder_input: encoder_obs[None, :]})[0], dtype=np.float32
        ).reshape(-1)
        decoder_calibration = decoder_obs.copy()
        decoder_calibration[:64] = token
        action = np.asarray(
            decoder_session.run(None, {decoder_input: decoder_calibration[None, :]})[0], dtype=np.float32
        ).reshape(-1)
        if not np.isfinite(encoder_obs).all() or not np.isfinite(decoder_calibration).all() or not np.isfinite(action).all():
            raise RuntimeError(f"non-finite ONNX data at sample {len(encoder_samples)}")

        encoder_samples.append(encoder_obs.copy())
        decoder_samples.append(decoder_calibration)

        action = np.clip(action, -args.action_limit, args.action_limit)
        last_action = action
        target = DEFAULT_ANGLES + ACTION_SCALE_MUJOCO * action[ISAACLAB_TO_MUJOCO]
        for _ in range(max(1, round(args.period / model.opt.timestep))):
            torque = args.pd_scale * (KP_MUJOCO * (target - data.qpos[7:36]) - KD_MUJOCO * data.qvel[6:35])
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
        if renderer is not None:
            mailbox.publish(data.qpos)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            if args.validate_only:
                raise RuntimeError(f"ONNX+MuJoCo state became non-finite at step {len(encoder_samples)}")
            reset_episode(model, data, reference)
        pelvis_up = data.xmat[pelvis_id].reshape(3, 3)[:, 2]
        min_root_z = min(min_root_z, float(data.qpos[2]))
        min_upright = min(min_upright, float(pelvis_up[2]))
        max_speed = max(max_speed, float(np.max(np.abs(data.qvel))))
        reference_frame = (reference_frame + 1) % max(1, reference.num_frames)

    metrics = {
        "min_root_z": min_root_z,
        "min_pelvis_up_z": min_upright,
        "max_abs_qvel": max_speed,
    }
    print(json.dumps(metrics, indent=2))
    if args.validate_only:
        if renderer is not None:
            renderer.stop()
        if min_root_z < 0.35 or min_upright < 0.2 or max_speed > 50.0:
            raise RuntimeError(f"ONNX+MuJoCo validation failed: {json.dumps(metrics)}")
        return

    if renderer is not None:
        renderer.stop()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder_array = np.asarray(encoder_samples, dtype=np.float32)
    decoder_array = np.asarray(decoder_samples, dtype=np.float32)
    np.save(args.output_dir / "encoder_inputs.npy", encoder_array)
    np.save(args.output_dir / "decoder_inputs.npy", decoder_array)
    metadata = {
        "count": len(encoder_samples),
        "encoder_shape": list(encoder_array.shape),
        "decoder_shape": list(decoder_array.shape),
        "source": "ONNX policy closed loop with MuJoCo G1 29DoF and walk_forward reference",
        "onnx_encoder": str(args.encoder),
        "onnx_decoder": str(args.decoder),
        "period": args.period,
        "pd_scale": args.pd_scale,
        "action_limit": args.action_limit,
        "episode_length": args.episode_length,
        "resets": resets,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
