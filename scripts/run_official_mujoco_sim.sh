#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# The copied official package is intentionally isolated from the source project.
# `--interface lo` keeps DDS on the local loopback interface and cannot select
# the real robot NIC by accident.
export PYTHONPATH="$repo_root/official_sim${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# `--no-project` is important here: the official simulator only needs the
# already-installed local environment and must not re-resolve the deployment
# project's ONNX dependencies just to start MuJoCo.
exec uv run --no-project --python .venv/bin/python official_sim/run_sim_loop.py \
  --interface lo \
  --no-enable-onscreen \
  "$@"
