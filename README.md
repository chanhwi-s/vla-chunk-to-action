# OFT Post-Inference Latency Measurement (LIBERO)

Measures the delay between **action-chunk-ready** and **actuator-command-ready** in
an OpenVLA-OFT pipeline on LIBERO. Reports mean / variance / std of that interval.

## What is being timed

```
VLA forward + parallel decode  ──►  [ T_start ]  unnormalize ─► chunk slice ─►
manipulation (desired EEF pose) ─► control (OSC → joint torques / actuator cmd)
──►  [ T_end ]  ──►  env.step() / sim.step()  (dispatch)
```

- **T_start**: action chunk fully decoded and in memory (after forward pass).
- **T_end**: actuator command computed, just before `env.step()` dispatches it.
- **Excluded**: VLA forward/decode, and `env.step()` / actuator dispatch.
- **Included**: unnormalization, chunk slicing, and the OSC controller
  computation that produces the gripper/actuator input
  (`MEASURE_CONTROLLER_COMPUTE = True`).

## Configure

All knobs are in the `Config` dataclass at the top of
`measure_latency.py`:

| field | default | note |
|---|---|---|
| `MODEL_PATH` | OFT libero-spatial ckpt | use `openvla/openvla-7b` + chunk=1 for base model |
| `LIBERO_TASK_SUITE` | `libero_spatial` | `libero_spatial`/`libero_object`/`libero_goal`/`libero_100` |
| `NUM_EPISODES` | 50 | |
| `ACTION_CHUNK_SIZE` | 8 | OFT chunk; set 1 for base openvla-7b |
| `MEASURE_CONTROLLER_COMPUTE` | True | include OSC compute in timed region |
| `SAVE_CSV` / `CSV_PATH` | True / `latency_results.csv` | raw per-step samples |

CLI overrides (no file edit needed):
```bash
python measure_latency.py --suite libero_object --episodes 20 --chunk 8
python measure_latency.py --chunk 1 --model openvla/openvla-7b   # base 7b
python measure_latency.py --no-ctrl   # exclude controller compute
```

## Install (Orin AGX / 5090 server)

The script must run where **OpenVLA-OFT** and **LIBERO** are installed — not in a
generic environment.

```bash
# 1) OpenVLA-OFT
git clone https://github.com/moojink/openvla-oft.git && cd openvla-oft
pip install -e .

# 2) LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO && pip install -r experiments/robot/libero/libero_requirements.txt

# 3) place this script at the openvla-oft repo root (so `experiments.robot...` imports resolve)
```

- **Orin AGX**: use the NVIDIA L4T / JetPack PyTorch build. FP16 recommended;
  if memory is tight, set `load_in_4bit`/`load_in_8bit` in `_VLACfg`.
- **5090 server**: standard CUDA PyTorch; good for quick iteration before the
  Orin run. Hardware is auto-detected and recorded in the printout + CSV.

## Verify on your stack (one spot)

`compute_actuator_command()` calls robosuite's `robot.control(action, policy_step=True)`
to run the OSC controller without advancing physics. Controller internals vary by
robosuite version — if you hit an API mismatch, **this is the only function to
adapt**. The timing boundaries (`T_start` / `T_end`) stay correct regardless.

## Output

Console prints n, mean (ms), variance (ms²), std (ms), plus min/median/max and
p95/p99. With `SAVE_CSV`, all raw per-step samples + a config/hardware metadata
header are written to `CSV_PATH`.

> First few steps are discarded as warmup (`WARMUP_STEPS`) to avoid CUDA/JIT/cache
> startup skew.
