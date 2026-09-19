# Sonic RK3576 encoder/decoder deployment

This repository contains the host-side conversion utilities for the two
fixed-shape MLPs used by Sonic:

| model | input | output |
| --- | ---: | ---: |
| encoder | `obs_dict`, `[1, 1247]` | `encoded_tokens`, `[1, 64]` |
| decoder | `obs_dict`, `[1, 994]` | `action`, `[1, 29]` |

The ONNX files remain in the source checkout at
`/home/jvwei/GR00T-WholeBodyControl/gear_sonic_deploy`; they are not copied
into this repository.

## Inspect and prepare calibration

```bash
python3 scripts/rknn_pipeline.py inspect encoder
python3 scripts/rknn_pipeline.py inspect decoder

# Smoke-only data; do not use this for a production INT8 build.
python3 scripts/rknn_pipeline.py make-calibration encoder --count 256
python3 scripts/rknn_pipeline.py make-calibration decoder --count 256

# Preferred: samples collected from the real observation path.
python3 scripts/rknn_pipeline.py make-calibration encoder \
  --samples /path/to/encoder_samples.npy --output-dir artifacts/calibration
python3 scripts/rknn_pipeline.py make-calibration decoder \
  --samples /path/to/decoder_samples.npy --output-dir artifacts/calibration
```

Input samples must be `float32` and shaped `[N, 1247]` or `[N, 1, 1247]` for
the encoder, and `[N, 994]` or `[N, 1, 994]` for the decoder. The calibration
manifest contains one absolute `.npy` path per line.

## Convert with RKNN Toolkit2

Install a RKNN Toolkit2 wheel that supports `target_platform='rk3576'` in the
host Python environment, then run:

```bash
python3 scripts/rknn_pipeline.py convert encoder \
  --dataset artifacts/calibration/encoder/dataset.txt \
  --output artifacts/rknn/sonic_encoder_int8.rknn --target rk3576
python3 scripts/rknn_pipeline.py convert decoder \
  --dataset artifacts/calibration/decoder/dataset.txt \
  --output artifacts/rknn/sonic_decoder_int8.rknn --target rk3576
```

The RKNN Toolkit is intentionally optional here. The target board currently
has `/usr/bin/rknn_server`, but no Python `rknnlite` module was found; board
runtime integration therefore needs the vendor C/C++ runtime library and the
final RKNN files from the host conversion.

## Board inference benchmark

The standalone C++ runner is in [inference/](inference/). It was compiled on
the RK3576 against `/usr/lib/librknnrt.so` and runs the complete random path:
encoder output is injected into the first 64 elements of the decoder input.

For the low-latency observation contract, the decoder input contains 10
history samples for angular velocity, joint position/velocity, last action,
and gravity direction. At the 50 Hz control period this is the current sample
plus the previous 9 samples, spanning 180 ms. The encoder reference horizon is
mode-dependent: the SMPL branch uses 4 future frames (about 80 ms), while the
G1/teleoperation reference branches use their configured 10-frame fields.
This is reference lookahead, not total system latency.

The 5,000-iteration board benchmark (200 warmup iterations) measured with the
correct low-latency temporal field layout:

```text
encoder:    307.67 Hz average, p99 3.98 ms
decoder:    243.42 Hz average, p99 4.48 ms
end_to_end: 135.88 Hz average, p99 8.34 ms
```

This is a throughput smoke test with random inputs, not an accuracy result.
The deployed files are in the isolated board directory
`/home/soulde/sonic_rk_infer/`.

Do not treat the synthetic calibration command as an accuracy result. Before
deploying to a robot, compare ONNX Runtime and RKNN outputs on recorded,
representative encoder/decoder observations and check action error against the
control-policy tolerance.

## MuJoCo hardware-in-the-loop

The MuJoCo G1 29-DoF model and mesh files are copied into
`assets/g1_mocap_29dof/`; this project does not import or depend on the source
`pico_dds_bridge` project at runtime. The RK3576 server uses a fixed-size TCP
protocol with TCP_NODELAY. Each request contains the low-latency encoder
`[1247]` and decoder `[994]` observations; the response contains one `[29]`
action vector.

On the board:

```bash
./sonic_rk3576_hil_server sonic_encoder_int8.rknn sonic_decoder_int8.rknn 39001
```

On the host:

```bash
python3 -m scripts.mujoco_hil_client \
  --host 192.168.3.40 --port 39001 --steps 500
```

Add `--render` for visualization. Rendering runs in a dedicated thread with a
latest-only `qpos` mailbox; the control thread never calls `viewer.sync()` and
does not wait for the GUI. Use `--render-rate` to limit the visualization rate.

The first independent HIL run measured 7.86 ms average NPU/network RTT,
11.75 ms p99 RTT, and 109.27 Hz including the MuJoCo physics tick. This is a
headless control-loop smoke test; it does not yet claim stable walking.

Using the included real reference sample
`assets/reference/walk_forward_amateur_001__A001.pkl` (1202 frames, 29 DoF,
30 Hz source data), a 500-step headless HIL run completed successfully on the
RK3576:

```text
server throughput: 116.76 Hz
average RTT:          7.671 ms
p99 RTT:             11.494 ms
average control tick: 8.628 ms
control loop:        115.90 Hz
```

The run used the real reference motion for the 10-frame G1 encoder lookahead;
the MuJoCo physics and RK3576 encoder/decoder path completed all 500 steps
without a disconnect or non-finite output. This is still a deployment/HIL
check and not a claim of stable robot walking.

For step-by-step FP32 ONNX versus RK3576 INT8 comparison on the same
observations, run with `--realtime`; if host ONNX plus HIL exceeds 20 ms, the
simulation advances at the slower achieved rate instead of dropping frames:

```bash
python3 -m scripts.mujoco_hil_client \
  --host 192.168.3.40 --port 39001 --steps 300 --realtime \
  --pd-scale 0.1 \
  --onnx-encoder /home/jvwei/GR00T-WholeBodyControl/gear_sonic_deploy/policy/low_latency/model_encoder.onnx \
  --onnx-decoder /home/jvwei/GR00T-WholeBodyControl/gear_sonic_deploy/policy/low_latency/model_decoder.onnx
```

The first real-sample comparison completed 300 steps at an achieved compute
rate of 35.63 Hz (average control computation 28.063 ms). The raw RKNN and
ONNX decoder actions had mean per-step max-absolute error 1.178426, p99
2.817320, and maximum 4.477791. This is a clear quantization/conversion
validation failure for the current RKNN artifacts and must be investigated
before treating the INT8 policy as accuracy-equivalent.
