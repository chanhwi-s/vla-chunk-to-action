#!/usr/bin/env python3
# =============================================================================
#  measure_oft_post_inference_latency.py
# -----------------------------------------------------------------------------
#  Post-inference latency measurement for an OpenVLA-OFT pipeline on LIBERO.
#
#  WHAT THIS MEASURES
#  ------------------
#  The delay between:
#    T_start : the moment the action CHUNK is fully decoded and sitting in
#              memory as a (normalized) numeric array
#              -> i.e. AFTER the VLA forward pass + parallel decode, but BEFORE
#                 any unnormalization / control computation.
#    T_end   : the moment just BEFORE the computed actuator command is handed
#              to the simulator / robot driver (env.step / sim.step).
#
#  The measured interval therefore contains:
#    (1) unnormalization + scaling of the chunk            (action post-proc)
#    (2) slicing the current step out of the chunk
#    (3) manipulation-level computation (desired EEF pose)
#    (4) control-level computation (OSC controller -> joint torques /
#        actuator command = "the input that goes into the gripper actuation")
#  ...and stops right before the real dispatch (env.step / sim.step).
#
#  It does NOT measure:
#    - the VLA model forward pass / token decode  (excluded by construction)
#    - env.step() / sim.step() / actuator dispatch (excluded by construction)
#
#  WHY OFT (vs. base openvla-7b)
#  -----------------------------
#  OpenVLA-OFT predicts a CHUNK of N action steps per inference (parallel
#  decode, continuous action head). Base openvla-7b predicts 1 step per
#  inference (autoregressive token decode). This script defaults to OFT but
#  works for both: set ACTION_CHUNK_SIZE = 1 and point MODEL_PATH at the base
#  model to reproduce the base-7b behavior.
#
#  HARDWARE
#  --------
#  Target: NVIDIA Jetson AGX Orin. Also runnable on the RTX 5090 test bench.
#  The actual hardware is auto-detected and recorded in the CSV / printout.
# =============================================================================

import argparse
import csv
import platform
import statistics
import sys
import time
from dataclasses import dataclass, asdict

import numpy as np

# =============================================================================
#  ██  CONFIG  ██  — edit everything you need right here, at the top.
# =============================================================================
@dataclass
class Config:
    # ── Model ────────────────────────────────────────────────────────────────
    # OFT checkpoint fine-tuned on the matching LIBERO suite. To use the base
    # (non-chunking) model instead, set this to "openvla/openvla-7b" AND set
    # ACTION_CHUNK_SIZE = 1 below.
    MODEL_PATH: str = "moojink/openvla-7b-oft-finetuned-libero-spatial"
    DEVICE: str = "cuda"          # "cuda" | "cuda:0" | "cpu"

    # ── Benchmark ────────────────────────────────────────────────────────────
    #   libero_spatial : same objects, DIFFERENT spatial layouts (spatial reasoning)
    #   libero_object  : same layout, DIFFERENT objects        (object generalization)
    #   libero_goal    : same objects+layout, DIFFERENT goals  (goal/skill generalization)
    #   libero_100     : 100-task long-horizon suite           (broad, harder)
    LIBERO_TASK_SUITE: str = "libero_spatial"
    NUM_EPISODES: int = 50

    # ── Action chunk ─────────────────────────────────────────────────────────
    # Number of action steps the model emits per inference call. With a chunk of
    # N, the model runs once and the controller consumes the N steps one by one
    # before the next inference. OFT-LIBERO default is 8; base openvla-7b = 1.
    ACTION_CHUNK_SIZE: int = 8

    # ── Controller (control-level) computation ───────────────────────────────
    # If True, the OSC controller computation (desired EEF pose -> joint
    # torques / actuator command) is executed INSIDE the timed region, so the
    # measured latency includes manipulation + control compute up to the
    # gripper/actuator input. If your env API differs, see compute_actuator_command().
    MEASURE_CONTROLLER_COMPUTE: bool = True

    # ── Output ───────────────────────────────────────────────────────────────
    SAVE_CSV: bool = True
    CSV_PATH: str = "latency_results.csv"

    # ── Run control ──────────────────────────────────────────────────────────
    MAX_STEPS_PER_EPISODE: int = 600   # safety cap so a stuck episode can't hang the run
    WARMUP_STEPS: int = 5              # discard first K samples (CUDA/JIT/cache warmup)
    SEED: int = 7


CONFIG = Config()

# Allowed task suites — validated at startup.
VALID_TASK_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_100")


# =============================================================================
#  Validation
# =============================================================================
def validate_config(cfg: Config) -> None:
    if cfg.LIBERO_TASK_SUITE not in VALID_TASK_SUITES:
        raise ValueError(
            f"Invalid LIBERO_TASK_SUITE={cfg.LIBERO_TASK_SUITE!r}. "
            f"Must be one of {VALID_TASK_SUITES}."
        )
    if cfg.NUM_EPISODES < 1:
        raise ValueError(f"NUM_EPISODES must be >= 1, got {cfg.NUM_EPISODES}.")
    if cfg.ACTION_CHUNK_SIZE < 1:
        raise ValueError(f"ACTION_CHUNK_SIZE must be >= 1, got {cfg.ACTION_CHUNK_SIZE}.")


# =============================================================================
#  Model / benchmark setup
#  -----------------------------------------------------------------------------
#  These wrap the official `openvla-oft` repo helpers
#  (experiments/robot/...). They are isolated here so that, if your installed
#  version exposes slightly different names, you only edit this section — the
#  timing logic below stays untouched.
# =============================================================================
def load_model_and_env(cfg: Config):
    """
    Returns (runner, env, suite_meta).

    `runner` is a small adapter exposing two methods used by the timing loop:
        runner.reset(env, init_state) -> obs
        runner.predict_normalized_chunk(obs, instruction) -> np.ndarray[N, A]
            (the VLA forward + parallel decode; this is what we EXCLUDE from
             the measured interval. It returns the NORMALIZED action chunk.)
        runner.unnorm_stats -> dict with "q01","q99","mask" for unnormalization
    """
    import torch  # noqa: F401  (imported lazily so config-only runs don't need it)

    # ---- OpenVLA-OFT model loading (adapt import paths to your install) ------
    # Reference: openvla-oft  experiments/robot/openvla_utils.py
    from experiments.robot.openvla_utils import (
        get_vla,
        get_processor,
        get_action_head,
        get_proprio_projector,
        get_vla_action,            # used as the forward+decode primitive
    )
    from experiments.robot.robot_utils import set_seed_everywhere
    from experiments.robot.libero.libero_utils import (
        get_libero_env,
        get_libero_dummy_action,   # noqa: F401  (handy if you need a no-op action)
    )
    from libero.libero import benchmark

    set_seed_everywhere(cfg.SEED)

    # Minimal config object the OFT helpers expect. Field names mirror the
    # repo's GenerateConfig; extend if your checkpoint needs more.
    class _VLACfg:
        pretrained_checkpoint = cfg.MODEL_PATH
        use_l1_regression = True
        use_diffusion = False
        use_film = False
        num_images_in_input = 1
        use_proprio = True
        center_crop = True
        num_open_loop_steps = cfg.ACTION_CHUNK_SIZE   # == chunk size
        unnorm_key = cfg.LIBERO_TASK_SUITE
        load_in_8bit = False
        load_in_4bit = False

    vla_cfg = _VLACfg()
    vla = get_vla(vla_cfg)
    processor = get_processor(vla_cfg)
    action_head = get_action_head(vla_cfg, vla.llm_dim)
    proprio_projector = (
        get_proprio_projector(vla_cfg, vla.llm_dim) if vla_cfg.use_proprio else None
    )

    # Normalization stats for the chosen suite (q01/q99/mask) live on the model.
    norm = vla.norm_stats[cfg.LIBERO_TASK_SUITE]["action"]
    unnorm_stats = {
        "q01": np.asarray(norm["q01"], dtype=np.float64),
        "q99": np.asarray(norm["q99"], dtype=np.float64),
        "mask": np.asarray(norm.get("mask", np.ones_like(norm["q01"], dtype=bool))),
    }

    # Build LIBERO benchmark + first task env.
    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[cfg.LIBERO_TASK_SUITE]()
    num_tasks = suite.n_tasks

    class _Runner:
        def __init__(self):
            self.vla = vla
            self.processor = processor
            self.action_head = action_head
            self.proprio_projector = proprio_projector
            self.cfg = vla_cfg
            self.unnorm_stats = unnorm_stats

        def predict_normalized_chunk(self, obs, instruction):
            # ----- VLA forward + parallel decode (EXCLUDED from timing) -------
            # We ask the OFT helper for the action chunk. We request the raw,
            # still-normalized chunk so that unnormalization happens INSIDE the
            # measured region below. If your get_vla_action only returns already-
            # unnormalized actions, set MEASURE_UNNORM_INSIDE=False handling in
            # the loop (the timing boundaries still hold; unnorm just becomes a
            # no-op inside the interval).
            chunk = get_vla_action(
                self.cfg, self.vla, self.processor, obs, instruction,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                normalized=True,            # <-- return normalized chunk
            )
            return np.asarray(chunk, dtype=np.float64).reshape(-1, 7)

    return _Runner(), suite, {"num_tasks": num_tasks, "get_libero_env": get_libero_env}


def make_env(suite, task_id, cfg, helpers):
    """Create the LIBERO env for a given task id and return (env, instruction, init_states)."""
    task = suite.get_task(task_id)
    env, _ = helpers["get_libero_env"](task, resolution=256)
    init_states = suite.get_task_init_states(task_id)
    instruction = task.language
    return env, instruction, init_states


# =============================================================================
#  Action post-processing  (THIS is the timed region's content)
# =============================================================================
def unnormalize_action(norm_action: np.ndarray, stats: dict) -> np.ndarray:
    """
    Standard OpenVLA unnormalization: map normalized [-1, 1] actions back to
    physical units using q01/q99, leaving masked dims (e.g. gripper) untouched.
    """
    q01, q99, mask = stats["q01"], stats["q99"], stats["mask"]
    unnorm = 0.5 * (norm_action + 1.0) * (q99 - q01) + q01
    return np.where(mask, unnorm, norm_action)


def compute_actuator_command(env, action: np.ndarray) -> None:
    """
    Manipulation + control computation: turn the 7-DoF action (delta EEF pose +
    gripper) into the low-level actuator command (joint torques written to
    sim.data.ctrl) WITHOUT advancing physics. This is the robosuite OSC
    controller's set_goal + run_controller, i.e. exactly the compute that
    produces "the input that goes into the gripper/actuator".

    NOTE (verify on your stack): LIBERO wraps robosuite. The robot's `control`
    method runs the controller for one policy step and writes the actuator
    command, but does NOT call sim.step(). We invoke it here purely to time the
    control computation; env.step() below will recompute+dispatch normally.
    If your robosuite version differs, this is the ONE function to adapt.
    """
    # env may be wrapped (OffScreenRenderEnv -> robosuite env). Unwrap to .env.
    base = getattr(env, "env", env)
    robot = base.robots[0]
    robot.control(action, policy_step=True)   # OSC compute -> sim.data.ctrl (no physics step)


# =============================================================================
#  Main measurement loop
# =============================================================================
def run(cfg: Config) -> None:
    validate_config(cfg)

    hw = detect_hardware(cfg)
    print("=" * 70)
    print("OpenVLA-OFT post-inference latency measurement")
    print(f"  model        : {cfg.MODEL_PATH}")
    print(f"  suite        : {cfg.LIBERO_TASK_SUITE}   episodes={cfg.NUM_EPISODES}")
    print(f"  chunk size   : {cfg.ACTION_CHUNK_SIZE}")
    print(f"  ctrl compute : {cfg.MEASURE_CONTROLLER_COMPUTE}")
    print(f"  hardware     : {hw}")
    print("=" * 70)

    runner, suite, helpers = load_model_and_env(cfg)
    num_tasks = helpers["num_tasks"]

    samples_ms = []   # one (T_end - T_start) sample per executed action step
    raw_rows = []     # for CSV: episode, step, latency_ms

    for ep in range(cfg.NUM_EPISODES):
        task_id = ep % num_tasks
        env, instruction, init_states = make_env(suite, task_id, cfg, helpers)
        init_state = init_states[ep % len(init_states)]

        env.reset()
        obs = env.set_init_state(init_state)

        chunk = None          # current normalized chunk in memory
        chunk_idx = 0         # which step of the chunk we're on
        done = False
        step = 0

        while not done and step < cfg.MAX_STEPS_PER_EPISODE:
            # --- (A) Get a fresh chunk only when the previous one is exhausted.
            #         The VLA forward + decode happens HERE and is NOT timed. ---
            if chunk is None or chunk_idx >= len(chunk):
                chunk = runner.predict_normalized_chunk(obs, instruction)  # [N,7] normalized
                chunk_idx = 0

            norm_action = chunk[chunk_idx]   # still-normalized current step

            # ================= T_start =====================================
            # Chunk is fully decoded and in memory; we are about to do the
            # post-processing + control computation. Forward pass already done.
            t_start = time.perf_counter()
            # ---------------------------------------------------------------

            # (1) unnormalize + scale this step
            action = unnormalize_action(norm_action, runner.unnorm_stats)
            # (2) (chunk slice already done above via chunk[chunk_idx])
            # (3)+(4) manipulation + control compute -> actuator command
            if cfg.MEASURE_CONTROLLER_COMPUTE:
                compute_actuator_command(env, action)

            # ================= T_end =======================================
            # Actuator command is computed and ready; we have NOT yet dispatched
            # it to env.step()/sim.step(). Stop the clock here.
            t_end = time.perf_counter()
            # ---------------------------------------------------------------

            latency_ms = (t_end - t_start) * 1e3

            # Warmup discard (CUDA/JIT/caches settling on the first few steps).
            if not (ep == 0 and step < cfg.WARMUP_STEPS):
                samples_ms.append(latency_ms)
                raw_rows.append((ep, step, latency_ms))

            # --- actual dispatch (EXCLUDED from the measurement) ---
            obs, reward, done, info = env.step(action)

            chunk_idx += 1
            step += 1

        try:
            env.close()
        except Exception:
            pass

        print(f"  [episode {ep + 1:>3}/{cfg.NUM_EPISODES}] "
              f"task_id={task_id} steps={step} samples={len(samples_ms)}")

    report(samples_ms, cfg, hw)
    if cfg.SAVE_CSV:
        write_csv(cfg.CSV_PATH, raw_rows, cfg, hw, samples_ms)
        print(f"\nRaw samples written to: {cfg.CSV_PATH}")


# =============================================================================
#  Reporting
# =============================================================================
def report(samples_ms, cfg: Config, hw: str) -> None:
    if not samples_ms:
        print("\n[!] No samples collected — check the run.")
        return
    mean = statistics.fmean(samples_ms)
    var = statistics.pvariance(samples_ms, mu=mean)   # population variance (ms^2)
    std = var ** 0.5
    arr = np.asarray(samples_ms)
    print("\n" + "-" * 70)
    print(f"Samples (n)        : {len(samples_ms)}")
    print(f"Mean latency       : {mean:.6f} ms")
    print(f"Variance           : {var:.6f} ms^2")
    print(f"Std deviation      : {std:.6f} ms")
    print(f"Min / Median / Max : {arr.min():.6f} / {np.median(arr):.6f} / {arr.max():.6f} ms")
    print(f"p95 / p99          : {np.percentile(arr, 95):.6f} / {np.percentile(arr, 99):.6f} ms")
    print("-" * 70)


def write_csv(path, raw_rows, cfg: Config, hw: str, samples_ms) -> None:
    mean = statistics.fmean(samples_ms) if samples_ms else float("nan")
    var = statistics.pvariance(samples_ms) if len(samples_ms) > 1 else float("nan")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # Metadata header block (commented) for reproducibility.
        for k, v in asdict(cfg).items():
            w.writerow([f"# {k}", v])
        w.writerow(["# hardware", hw])
        w.writerow(["# mean_ms", mean])
        w.writerow(["# variance_ms2", var])
        w.writerow([])
        w.writerow(["episode", "step", "latency_ms"])
        w.writerows(raw_rows)


# =============================================================================
#  Utilities
# =============================================================================
def detect_hardware(cfg: Config) -> str:
    try:
        import torch
        if cfg.DEVICE.startswith("cuda") and torch.cuda.is_available():
            idx = 0 if ":" not in cfg.DEVICE else int(cfg.DEVICE.split(":")[1])
            return f"{torch.cuda.get_device_name(idx)} (CUDA)"
    except Exception:
        pass
    return f"{platform.machine()} CPU ({platform.platform()})"


def parse_cli_overrides(cfg: Config) -> Config:
    """Optional CLI overrides so you can sweep without editing the file."""
    p = argparse.ArgumentParser(description="OFT post-inference latency measurement")
    p.add_argument("--suite", choices=VALID_TASK_SUITES)
    p.add_argument("--episodes", type=int)
    p.add_argument("--chunk", type=int)
    p.add_argument("--model")
    p.add_argument("--device")
    p.add_argument("--csv")
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--no-ctrl", action="store_true",
                   help="exclude controller compute from the timed region")
    a = p.parse_args()
    if a.suite:    cfg.LIBERO_TASK_SUITE = a.suite
    if a.episodes: cfg.NUM_EPISODES = a.episodes
    if a.chunk:    cfg.ACTION_CHUNK_SIZE = a.chunk
    if a.model:    cfg.MODEL_PATH = a.model
    if a.device:   cfg.DEVICE = a.device
    if a.csv:      cfg.CSV_PATH = a.csv
    if a.no_csv:   cfg.SAVE_CSV = False
    if a.no_ctrl:  cfg.MEASURE_CONTROLLER_COMPUTE = False
    return cfg


if __name__ == "__main__":
    cfg = parse_cli_overrides(CONFIG)
    try:
        run(cfg)
    except ModuleNotFoundError as e:
        print(f"\n[!] Missing dependency: {e}", file=sys.stderr)
        print("    This script must run where openvla-oft + libero are installed "
              "(Orin AGX / 5090 server). See README.", file=sys.stderr)
        sys.exit(1)
