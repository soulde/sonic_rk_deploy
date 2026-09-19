#!/usr/bin/env python3
"""Prepare and score a low-latency Sonic encoder/decoder validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.rknn_pipeline import MODEL_SPECS


def _normal(rng: np.random.Generator, shape: tuple[int, ...], scale: float) -> np.ndarray:
    return rng.normal(0.0, scale, size=shape).astype(np.float32)


def make_validation_inputs(count: int, seed: int = 20260919) -> tuple[np.ndarray, np.ndarray]:
    """Create physically bounded, correctly laid-out low-latency smoke inputs.

    This is deliberately not claimed to be robot data. It exercises both the
    G1 and SMPL encoder layouts and the decoder's ten-sample history layout.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    encoder = np.zeros((count, 1247), dtype=np.float32)
    decoder = np.zeros((count, 994), dtype=np.float32)

    for index in range(count):
        if index % 2 == 0:  # G1 reference mode
            encoder[index, 0] = 0.0
            encoder[index, 4:294] = _normal(rng, (290,), 0.25)
            encoder[index, 294:584] = _normal(rng, (290,), 0.5)
            encoder[index, 584:644] = _normal(rng, (60,), 0.15)
        else:  # SMPL low-latency mode: current + 3 future frames
            encoder[index, 0] = 2.0
            encoder[index, 911:1199] = _normal(rng, (288,), 0.2)
            encoder[index, 1199:1223] = _normal(rng, (24,), 0.15)
            encoder[index, 1223:1247] = _normal(rng, (24,), 0.15)

        for frame in range(10):
            decoder[index, 64 + frame * 3:67 + frame * 3] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            decoder[index, 94 + frame * 29:123 + frame * 29] = _normal(rng, (29,), 0.25)
            decoder[index, 384 + frame * 29:413 + frame * 29] = _normal(rng, (29,), 0.5)
            decoder[index, 674 + frame * 29:703 + frame * 29] = _normal(rng, (29,), 0.15)
            decoder[index, 64 + frame * 3:67 + frame * 3] += _normal(rng, (3,), 0.03)
            decoder[index, 964 + frame * 3:967 + frame * 3] = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    return encoder, decoder


def summarize_error(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: reference={reference.shape}, actual={actual.shape}")
    delta = np.abs(actual - reference)
    valid_relative = np.abs(reference) > 1e-6
    relative = np.zeros_like(delta)
    relative[valid_relative] = delta[valid_relative] / np.abs(reference[valid_relative])
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(actual - reference)))),
        "max_relative": float(np.max(relative)),
        "reference_max_abs": float(np.max(np.abs(reference))),
    }


def _run_onnx(model_path: Path, inputs: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = [session.run(None, {input_name: sample.reshape(1, -1)})[0].reshape(-1) for sample in inputs]
    return np.asarray(outputs, dtype=np.float32)


def prepare(output_dir: Path, count: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_inputs, decoder_inputs = make_validation_inputs(count, seed)
    encoder_expected = _run_onnx(MODEL_SPECS["encoder"].path, encoder_inputs)
    decoder_expected = []
    for token, context in zip(encoder_expected, decoder_inputs):
        decoder_input = context.copy()
        decoder_input[:64] = token
        decoder_expected.append(_run_onnx(MODEL_SPECS["decoder"].path, decoder_input[None])[0])
    decoder_expected = np.asarray(decoder_expected, dtype=np.float32)

    encoder_inputs.tofile(output_dir / "encoder_input.f32")
    decoder_inputs.tofile(output_dir / "decoder_context.f32")
    encoder_expected.tofile(output_dir / "encoder_expected.f32")
    decoder_expected.tofile(output_dir / "action_expected.f32")
    metadata = {
        "count": count,
        "seed": seed,
        "encoder_input_elements": 1247,
        "decoder_input_elements": 994,
        "encoder_output_elements": 64,
        "decoder_output_elements": 29,
        "source": "deterministic bounded synthetic low-latency layout; not robot logs",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def compare(output_dir: Path) -> None:
    metadata = json.loads((output_dir / "metadata.json").read_text())
    count = int(metadata["count"])
    encoder_expected = np.fromfile(output_dir / "encoder_expected.f32", dtype=np.float32).reshape(count, 64)
    action_expected = np.fromfile(output_dir / "action_expected.f32", dtype=np.float32).reshape(count, 29)
    encoder_actual = np.fromfile(output_dir / "encoder_rknn.f32", dtype=np.float32).reshape(count, 64)
    action_actual = np.fromfile(output_dir / "action_rknn.f32", dtype=np.float32).reshape(count, 29)
    result = {"encoder": summarize_error(encoder_expected, encoder_actual),
              "decoder_action": summarize_error(action_expected, action_actual)}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--count", type=int, default=128)
    prepare_parser.add_argument("--seed", type=int, default=20260919)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output_dir, args.count, args.seed)
    else:
        compare(args.output_dir)


if __name__ == "__main__":
    main()
