# Official MuJoCo simulator snapshot

This directory is a self-contained copy of the official GEAR SONIC MuJoCo
simulator used by the deployment project. It is deliberately kept outside
the host project package and does not import from `/home/jvwei/GR00T-WholeBodyControl`.

It exposes the same Unitree SDK2 DDS interface as the official deployment:

- the simulator publishes simulated `LowState` on local DDS;
- a controller publishes `LowCmd` on local DDS;
- the simulator applies the command with the official MuJoCo PD path.

Run headless on loopback:

```bash
./scripts/run_official_mujoco_sim.sh
```

For a local viewer, run the entry point directly:

```bash
PYTHONPATH="$PWD/official_sim" uv run --no-project --python .venv/bin/python \
  official_sim/run_sim_loop.py --interface lo
```

The copied model is the official `scene_43dof.xml`: 29 body joints plus two
7-DoF hands. This is intentional because it is the official Sonic simulation
configuration; the Sonic encoder/decoder still consumes the 29-DoF body state.
