#!/usr/bin/env python3
"""Inspect, calibrate, and convert Sonic encoder/decoder ONNX models to RKNN.

The models are exported with one fixed float32 vector input each.  RKNN's
calibration dataset is therefore written as one .npy tensor per sample and a
text manifest containing absolute paths to those tensors.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SOURCE_ROOT = Path("/home/jvwei/GR00T-WholeBodyControl/gear_sonic_deploy")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    onnx_path: Path
    input_name: str
    input_shape: tuple[int, ...]
    output_name: str
    output_shape: tuple[int, ...]


MODEL_SPECS = {
    "encoder": ModelSpec(
        "encoder",
        SOURCE_ROOT / "policy/low_latency/model_encoder.onnx",
        "obs_dict",
        (1, 1247),
        "encoded_tokens",
        (1, 64),
    ),
    "decoder": ModelSpec(
        "decoder",
        SOURCE_ROOT / "policy/low_latency/model_decoder.onnx",
        "obs_dict",
        (1, 994),
        "action",
        (1, 29),
    ),
}


def validate_input_array(name: str, array: np.ndarray) -> np.ndarray:
    """Validate and normalize calibration data to [N, 1, feature_dim]."""
    if name not in MODEL_SPECS:
        raise ValueError(f"unknown model {name!r}; expected encoder or decoder")
    spec = MODEL_SPECS[name]
    if array.dtype != np.float32:
        raise ValueError(f"{name} calibration data must be float32, got {array.dtype}")
    expected_dim = spec.input_shape[-1]
    if array.ndim == 2 and array.shape[1] == expected_dim:
        array = array[:, None, :]
    if array.ndim != 3 or array.shape[1:] != spec.input_shape:
        raise ValueError(
            f"{name} calibration data has shape {array.shape}; "
            f"expected [N, {spec.input_shape[0]}, {expected_dim}]"
        )
    return np.ascontiguousarray(array)


def build_calibration_samples(name: str, count: int, seed: int = 0) -> np.ndarray:
    """Create deterministic smoke-test inputs, not production calibration data."""
    if count <= 0:
        raise ValueError("count must be positive")
    if name not in MODEL_SPECS:
        raise ValueError(f"unknown model {name!r}")
    rng = np.random.default_rng(seed)
    feature_dim = MODEL_SPECS[name].input_shape[-1]
    # Keep the fallback bounded: encoder inputs contain discrete mode/index
    # fields embedded in the vector, and values outside the exported training
    # range can become invalid ScatterND indices during RKNN calibration.
    values = rng.normal(0.0, 1.0, size=(count, 1, feature_dim)).astype(np.float32)
    values[::3] *= np.float32(0.25)
    return np.clip(values, np.float32(-1.0), np.float32(1.0))


def load_calibration_samples(name: str, path: Path) -> np.ndarray:
    """Load real calibration samples from .npy or .npz and validate them."""
    if path.suffix == ".npz":
        with np.load(path) as archive:
            if name in archive:
                array = archive[name]
            elif "samples" in archive:
                array = archive["samples"]
            else:
                raise ValueError(f"{path} must contain {name!r} or 'samples'")
    else:
        array = np.load(path)
    return validate_input_array(name, np.asarray(array))


def write_calibration_dataset(name: str, samples: np.ndarray, output_dir: Path) -> Path:
    """Write per-sample .npy files and the RKNN dataset manifest."""
    samples = validate_input_array(name, samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "dataset.txt"
    paths: list[str] = []
    for index, sample in enumerate(samples):
        sample_path = output_dir / f"{name}_{index:05d}.npy"
        np.save(sample_path, sample)
        paths.append(str(sample_path.resolve()))
    manifest.write_text("\n".join(paths) + "\n", encoding="utf-8")
    return manifest


def inspect_model(spec: ModelSpec) -> None:
    import onnx

    model = onnx.load(str(spec.onnx_path), load_external_data=False)
    print(f"{spec.name}: {spec.onnx_path}")
    for value in model.graph.input:
        tensor = value.type.tensor_type
        shape = [d.dim_value or d.dim_param for d in tensor.shape.dim]
        print(f"  input  {value.name}: {shape}")
    for value in model.graph.output:
        tensor = value.type.tensor_type
        shape = [d.dim_value or d.dim_param for d in tensor.shape.dim]
        print(f"  output {value.name}: {shape}")


def convert_model(name: str, output: Path, dataset: Path | None, target: str) -> None:
    """Convert one model using RKNN Toolkit2; import remains optional on PC."""
    from rknn.api import RKNN

    spec = MODEL_SPECS[name]
    if not spec.onnx_path.is_file():
        raise FileNotFoundError(spec.onnx_path)
    if dataset is None:
        raise ValueError("INT8 conversion requires --dataset with real calibration samples")

    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(target_platform=target)
        if ret != 0:
            raise RuntimeError(f"RKNN config failed: {ret}")
        ret = rknn.load_onnx(
            model=str(spec.onnx_path),
            inputs=[spec.input_name],
            outputs=[spec.output_name],
            input_size_list=[list(spec.input_shape)],
        )
        if ret != 0:
            raise RuntimeError(f"RKNN load_onnx failed: {ret}")
        ret = rknn.build(do_quantization=True, dataset=str(dataset))
        if ret != 0:
            raise RuntimeError(f"RKNN build failed: {ret}")
        output.parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(str(output))
        if ret != 0:
            raise RuntimeError(f"RKNN export failed: {ret}")
    finally:
        rknn.release()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "make-calibration", "convert"))
    parser.add_argument("model", choices=tuple(MODEL_SPECS))
    parser.add_argument("--samples", type=Path, help="real .npy/.npz calibration samples")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/calibration"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", default="rk3576")
    parser.add_argument("--dataset", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    spec = MODEL_SPECS[args.model]
    if args.action == "inspect":
        inspect_model(spec)
    elif args.action == "make-calibration":
        samples = (
            load_calibration_samples(args.model, args.samples)
            if args.samples
            else build_calibration_samples(args.model, args.count, args.seed)
        )
        manifest = write_calibration_dataset(args.model, samples, args.output_dir / args.model)
        print(manifest)
    else:
        if args.output is None or args.dataset is None:
            raise SystemExit("convert requires --output and --dataset")
        convert_model(args.model, args.output, args.dataset, args.target)


if __name__ == "__main__":
    main()
