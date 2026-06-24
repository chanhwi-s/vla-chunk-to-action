#!/usr/bin/env python3
# =============================================================================
#  measure_latency.py
# -----------------------------------------------------------------------------
#  Post-inference latency measurement for an OpenVLA-OFT pipeline on LIBERO.
#
#  Place this file at the ROOT of the openvla-oft repo (so that
#  `experiments.robot...` and `prismatic...` imports resolve), e.g. via symlink:
#     ln -s ~/workspace/vla-chunk-to-action/measure_latency.py \
#           ~/workspace/openvla-oft/measure_latency.py
#  then run it from the openvla-oft root.
#
#  WHAT THIS MEASURES  (per action chunk)
#  --------------------------------------
#    T_start : the moment the action CHUNK is fully decoded and in memory
#              -> right AFTER get_action() returns (VLA forward + parallel
#                 decode + un-normalization all happen inside that call and are
#                 EXCLUDED from timing).
#    T_end   : the moment the actuator command for the FIRST action of the
#              chunk is computed, just BEFORE env.step() dispatches it.
#
#  The timed interval contains:
#    (1) chunk assembly into the execution queue
#    (2) slicing the first step out of the chunk
#    (3) action post-processing (gripper normalize + sign-flip = process_action)
#    (4) manipulation + control computation: the robosuite OSC controller turns
#        the 7-DoF action (delta EEF pose + gripper) into the low-level actuator
#        command (joint torques written to sim.data.ctrl) -- "the input that goes
#        into the gripper/actuator". (MEASURE_CONTROLLER_COMPUTE = True)
#    ...and stops right before the real dispatch (env.step / sim.step).
#
#  IMPORTANT NOTE ON THE BOUNDARY (read me)
#  ----------------------------------------
#  In OpenVLA-OFT, de-tokenization / un-normalization is FUSED into the model's
#  predict_action() forward call and cannot be cleanly separated without editing
#  model internals. Therefore the measured region begins at "chunk ready"
#  (post-unnormalize) rather than at the raw token output. This is an unavoidable
#  consequence of the OFT architecture; the un-normalization cost itself is on
#  the order of microseconds. The region still captures the professor's target:
#  manipulation + control computation up to the actuator input.
#
#  Excluded: VLA forward/decode/unnormalize, and env.step()/actuator dispatch.
#
#  HARDWARE: target NVIDIA Jetson AGX Orin (JetPack 6 / CUDA 12.6). Also runs on
#  the RTX 5090 test bench. Actual hardware is auto-detected and recorded.
# =============================================================================

import os

# Headless offscreen rendering for LIBERO/MuJoCo on a server (needs EGL + the
# user's `render` group). Set BEFORE importing libero/robosuite.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import csv
import platform
import statistics
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

# =============================================================================
#  ██  CONFIG  ██  — edit everything you need right here, at the top.
# =============================================================================
@dataclass
class Config:
    # ── Model ────────────────────────────────────────────────────────────────
    # Leave MODEL_PATH = "" to auto-pick the OFT checkpoint matching the suite
    # (see SUITE_TO_CHECKPOINT below). Or set it explicitly.
    MODEL_PATH: str = ""
    DEVICE: str = "cuda"          # "cuda" | "cuda:0" | "cpu"

    # ── Benchmark ────────────────────────────────────────────────────────────
    #   libero_spatial : same objects, DIFFERENT spatial layouts (spatial reasoning)
    #   libero_object  : same layout, DIFFERENT objects        (object generalization)
    #   libero_goal    : same objects+layout, DIFFERENT goals  (goal generalization)
    #   libero_10      : 10 long-horizon tasks (a.k.a. LIBERO-Long; part of LIBERO-100)
    LIBERO_TASK_SUITE: str = "libero_spatial"
    NUM_EPISODES: int = 10        # claude.md suggested 50; latency stats stabilize
                                  # well before that and 10 runs much faster on Orin.
                                  # Each episode yields many per-chunk samples.

    # ── Action chunk ─────────────────────────────────────────────────────────
    # Steps the model emits per inference. MUST match the NUM_ACTIONS_CHUNK the
    # checkpoint was trained with (OFT-LIBERO = 8). Base openvla-7b would be 1.
    ACTION_CHUNK_SIZE: int = 8

    # ── Controller (control-level) computation ───────────────────────────────
    # True -> include the OSC controller computation (manipulation + control ->
    # actuator command) inside the timed region. If the robosuite control API on
    # your install differs, see compute_actuator_command() — it's the one spot
    # to adapt. Set False to time only chunk-slice + gripper post-processing.
    MEASURE_CONTROLLER_COMPUTE: bool = True

    # ── Output ───────────────────────────────────────────────────────────────
    SAVE_CSV: bool = True
    CSV_PATH: str = "latency_results.csv"

    # ── Run control ──────────────────────────────────────────────────────────
    NUM_STEPS_WAIT: int = 10      # sim settling steps (dummy action, not measured)
    WARMUP_CHUNKS: int = 2        # discard first K chunk-samples (CUDA/JIT warmup)
    SEED: int = 7


CONFIG = Config()

# OFT checkpoints that have matching action_head + proprio_projector on HF Hub.
SUITE_TO_CHECKPOINT = {
    "libero_spatial": "moojink/openvla-7b-oft-finetuned-libero-spatial",
    "libero_object": "moojink/openvla-7b-oft-finetuned-libero-object",
    "libero_goal": "moojink/openvla-7b-oft-finetuned-libero-goal",
    "libero_10": "moojink/openvla-7b-oft-finetuned-libero-10",
}
VALID_TASK_SUITES = tuple(SUITE_TO_CHECKPOINT.keys())


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
#  Build the openvla-oft GenerateConfig from our CONFIG
#  -----------------------------------------------------------------------------
#  We reuse the repo's real GenerateConfig + helper functions so the model /
#  benchmark wiring exactly matches the official LIBERO eval. This avoids API
#  drift between versions.
# =============================================================================
def build_oft_cfg():
    from experiments.robot.libero.run_libero_eval import GenerateConfig

    cfg = GenerateConfig()
    cfg.pretrained_checkpoint = CONFIG.MODEL_PATH or SUITE_TO_CHECKPOINT[CONFIG.LIBERO_TASK_SUITE]
    cfg.task_suite_name = CONFIG.LIBERO_TASK_SUITE
    cfg.num_open_loop_steps = CONFIG.ACTION_CHUNK_SIZE
    cfg.num_steps_wait = CONFIG.NUM_STEPS_WAIT
    cfg.seed = CONFIG.SEED
    # OFT-LIBERO model settings (match the official checkpoints):
    cfg.model_family = "openvla"
    cfg.use_l1_regression = True
    cfg.use_diffusion = False
    cfg.use_film = False
    cfg.use_proprio = True
    cfg.num_images_in_input = 2
    cfg.center_crop = True
    return cfg


# =============================================================================
#  Manipulation + control computation (the part we want inside the timed region)
# =============================================================================
def _find_robosuite_env(env):
    """Descend through LIBERO/robosuite wrappers to the env exposing `.robots`."""
    e = env
    for _ in range(12):
        if hasattr(e, "robots"):
            return e
        if hasattr(e, "env"):
            e = e.env
        else:
            break
    raise RuntimeError(
        "Could not locate the robosuite env (no `.robots` found). "
        "Adjust _find_robosuite_env() for your LIBERO/robosuite version."
    )


def compute_actuator_command(env, action: np.ndarray) -> None:
    """
    Run the OSC controller for ONE policy step WITHOUT advancing physics:
    turns the 7-DoF action (delta EEF pose + gripper) into the low-level actuator
    command (joint torques in sim.data.ctrl). This mirrors exactly what
    robosuite does inside env.step() before sim.step(), i.e. the manipulation +
    control computation that produces the gripper/actuator input.

    VERIFY ON YOUR STACK: robosuite's Robot.control(action, policy_step=True)
    computes + writes the actuator command but does not call sim.step(). If the
    signature differs in your robosuite version, this is the ONE function to
    adapt. The env.step() call afterwards re-runs control normally, so invoking
    it here only for timing is harmless.
    """
    rs_env = _find_robosuite_env(env)
    rs_env.robots[0].control(action, policy_step=True)


def _patch_torch_load() -> None:
    """
    PyTorch >= 2.6 defaults torch.load(weights_only=True), which rejects the
    numpy arrays stored in LIBERO's init-state files. All files we load are
    trusted (LIBERO package data + official checkpoints), so default to
    weights_only=False for calls that don't explicitly set it. Calls that pass
    weights_only=True (e.g. action_head/proprio loads) are left untouched.
    """
    import torch
    if getattr(torch.load, "_patched_wo", False):
        return
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _load._patched_wo = True
    torch.load = _load


# =============================================================================
#  Main measurement loop  (a timed variant of run_libero_eval.run_episode)
# =============================================================================
def run(cfg_user: Config) -> None:
    validate_config(cfg_user)
    _patch_torch_load()

    # Import here so a config error doesn't require the full stack.
    from experiments.robot.libero.run_libero_eval import (
        TaskSuite,
        TASK_MAX_STEPS,
        initialize_model,
        prepare_observation,
        process_action,
    )
    from experiments.robot.robot_utils import (
        get_action,
        get_image_resize_size,
        set_seed_everywhere,
    )
    from experiments.robot.libero.libero_utils import (
        get_libero_dummy_action,
        get_libero_env,
    )
    from libero.libero import benchmark

    hw = detect_hardware(cfg_user)
    cfg = build_oft_cfg()

    print("=" * 72)
    print("OpenVLA-OFT post-inference latency measurement")
    print(f"  checkpoint   : {cfg.pretrained_checkpoint}")
    print(f"  suite        : {cfg.task_suite_name}   episodes={cfg_user.NUM_EPISODES}")
    print(f"  chunk size   : {cfg.num_open_loop_steps}")
    print(f"  ctrl compute : {cfg_user.MEASURE_CONTROLLER_COMPUTE}")
    print(f"  hardware     : {hw}")
    print("=" * 72)

    set_seed_everywhere(cfg.seed)

    # initialize_model() also sets cfg.unnorm_key via check_unnorm_key().
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = TASK_MAX_STEPS[TaskSuite(cfg.task_suite_name)]

    samples_ms = []          # one (T_end - T_start) sample per action chunk
    rows = []                # CSV: episode, sim_step, latency_ms
    chunk_counter = 0

    for ep in range(cfg_user.NUM_EPISODES):
        task_id = ep % num_tasks
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        init_states = task_suite.get_task_init_states(task_id)

        env.reset()
        obs = env.set_init_state(init_states[ep % len(init_states)])

        action_queue = deque(maxlen=cfg.num_open_loop_steps)
        t = 0
        try:
            while t < max_steps + cfg.num_steps_wait:
                # Let objects settle first (dummy action, not measured).
                if t < cfg.num_steps_wait:
                    obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                observation, _ = prepare_observation(obs, resize_size)

                measure = False
                if len(action_queue) == 0:
                    # ---- VLA forward + parallel decode + unnormalize (EXCLUDED) ----
                    actions = get_action(
                        cfg, model, observation, task_description,
                        processor=processor,
                        action_head=action_head,
                        proprio_projector=proprio_projector,
                        noisy_action_projector=noisy_action_projector,
                        use_film=cfg.use_film,
                    )
                    # ===================== T_start ==============================
                    # Chunk fully decoded & in memory. Forward pass already done.
                    t_start = time.perf_counter()
                    action_queue.extend(actions)        # (1) chunk assembly
                    measure = True
                    # ------------------------------------------------------------

                action = action_queue.popleft()         # (2) slice current step

                if measure:
                    action = process_action(action, cfg.model_family)  # (3) gripper post-proc
                    if cfg_user.MEASURE_CONTROLLER_COMPUTE:
                        compute_actuator_command(env, action)          # (4) manip + control
                    # ===================== T_end ================================
                    # Actuator command for the first action is ready; not yet
                    # dispatched to env.step()/sim.step(). Stop the clock.
                    t_end = time.perf_counter()
                    # ------------------------------------------------------------
                    if chunk_counter >= cfg_user.WARMUP_CHUNKS:
                        ms = (t_end - t_start) * 1e3
                        samples_ms.append(ms)
                        rows.append((ep, t, ms))
                    chunk_counter += 1
                else:
                    action = process_action(action, cfg.model_family)

                # ---- actual dispatch (EXCLUDED from measurement) ----
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    break
                t += 1
        except Exception as e:  # keep going across episodes; report the issue
            print(f"  [episode {ep + 1}] error: {e}")
        finally:
            try:
                env.close()
            except Exception:
                pass

        print(f"  [episode {ep + 1:>3}/{cfg_user.NUM_EPISODES}] "
              f"task_id={task_id} samples={len(samples_ms)}")

    report(samples_ms)
    if cfg_user.SAVE_CSV:
        write_csv(cfg_user.CSV_PATH, rows, cfg_user, hw, samples_ms)
        print(f"\nRaw samples written to: {cfg_user.CSV_PATH}")


# =============================================================================
#  Reporting
# =============================================================================
def report(samples_ms) -> None:
    if not samples_ms:
        print("\n[!] No samples collected — check the run.")
        return
    mean = statistics.fmean(samples_ms)
    var = statistics.pvariance(samples_ms, mu=mean)   # population variance (ms^2)
    std = var ** 0.5
    arr = np.asarray(samples_ms)
    print("\n" + "-" * 72)
    print(f"Samples (n)        : {len(samples_ms)}")
    print(f"Mean latency       : {mean:.6f} ms")
    print(f"Variance           : {var:.6f} ms^2")
    print(f"Std deviation      : {std:.6f} ms")
    print(f"Min / Median / Max : {arr.min():.6f} / {np.median(arr):.6f} / {arr.max():.6f} ms")
    print(f"p95 / p99          : {np.percentile(arr, 95):.6f} / {np.percentile(arr, 99):.6f} ms")
    print("-" * 72)


def write_csv(path, rows, cfg_user: Config, hw: str, samples_ms) -> None:
    mean = statistics.fmean(samples_ms) if samples_ms else float("nan")
    var = statistics.pvariance(samples_ms) if len(samples_ms) > 1 else float("nan")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for k, v in asdict(cfg_user).items():
            w.writerow([f"# {k}", v])
        w.writerow(["# hardware", hw])
        w.writerow(["# mean_ms", mean])
        w.writerow(["# variance_ms2", var])
        w.writerow([])
        w.writerow(["episode", "sim_step", "latency_ms"])
        w.writerows(rows)


# =============================================================================
#  Utilities
# =============================================================================
def detect_hardware(cfg_user: Config) -> str:
    try:
        import torch
        if cfg_user.DEVICE.startswith("cuda") and torch.cuda.is_available():
            idx = 0 if ":" not in cfg_user.DEVICE else int(cfg_user.DEVICE.split(":")[1])
            return f"{torch.cuda.get_device_name(idx)} (CUDA)"
    except Exception:
        pass
    return f"{platform.machine()} CPU ({platform.platform()})"


if __name__ == "__main__":
    try:
        run(CONFIG)
    except ModuleNotFoundError as e:
        print(f"\n[!] Missing dependency / wrong working dir: {e}", file=sys.stderr)
        print("    Run from the openvla-oft repo root (so experiments.* / prismatic.* "
              "resolve), inside the conda env where torch+CUDA work.", file=sys.stderr)
        sys.exit(1)
