"""Measure every method at one sensor-degradation setting.

Replaces the hardware section. Sim-to-real is off the table, but the question
that section existed to answer -- what happens when the sensors are not perfect
-- is answerable here, and on locomotion it is the question JOSE most has to
face: JOSE reads joint encoders and nothing else, so encoder error is its worst
case by construction.

Two axes, run separately:

**IMU noise.** Only the joint+IMU methods read an IMU, so only they can move.
JOSE, the joint-only student and the teacher are flat lines here *by
construction*, which is a result to state rather than an experiment to run --
they are still measured once at 0x as the reference row.

**Encoder noise.** Every method reads the joints, so every method moves,
including the teacher -- whose curve is the ceiling for that axis.

One process per (axis, scale) because the policy-observation noise is baked into
the environment config at build time. Inside the process every method and seed is
measured against the *same* built environment, which is the point: a sweep whose
points came from separately constructed environments would confound the noise
setting with everything else that differs between two builds.

The metric is the 15-command grid -- the same instrument the main table uses. Its
survival rate saturates at 1.0 in clean conditions and is therefore uninformative
there, but that is exactly what stops being true as the sensors degrade, so on
this axis one evaluation yields both tracking error and survival.

Usage:

    python -m JOSE.eval_sensor_robustness --axis encoder --scale 1.0 \
        --study <method_comparison run dir> --set-study <set study>/locomotion \
        --teacher-checkpoint <walk teacher .pt> --headless
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Sensor-degradation sweep, one setting per process")
parser.add_argument("--axis", choices=("imu", "encoder"), required=True)
parser.add_argument(
    "--scale", type=float, required=True,
    help="Multiplier on the axis's 1x model. For encoders, 1x is the noise the "
    "walk teacher was trained under (joint angle +-0.01 rad, joint velocity "
    "+-1.5 rad/s). For the IMU, 1x is the model each student was trained with. "
    "0 is the clean condition Table I reports.",
)
parser.add_argument(
    "--methods", nargs="+", default=None,
    help="Subset to measure. Default: every method the axis can move, plus the "
    "teacher as the reference row.",
)
parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
parser.add_argument(
    "--study", required=True,
    help="method_comparison run directory holding methods/{jose,...}/window_25/joints_all/seed_N",
)
parser.add_argument(
    "--set-study", default=None,
    help="SET study directory for this task, e.g. <set output>/<run>/locomotion. "
    "Omit to skip SET.",
)
parser.add_argument("--teacher-checkpoint", required=True)
parser.add_argument("--task", default="Isaac-G1-PPO-Walk-Estimator-JOSE-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--adapter", choices=("amp", "ppo_walk"), default="ppo_walk")
parser.add_argument("--num-envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42, help="Environment construction seed")
parser.add_argument("--grid-settle-s", type=float, default=1.0)
parser.add_argument("--grid-measure-s", type=float, default=4.0)
parser.add_argument(
    "--encoder-bias-fraction", type=float, default=0.0,
    help="Per-episode constant encoder offset, as a fraction of the half-width. "
    "0 reproduces the environment's own noise model exactly, which is what keeps "
    "the 1x unit meaningful.",
)
parser.add_argument("--out", required=True, help="JSONL file to append rows to")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

from jose.teacher_setup import resolve_agent_entry_point  # noqa: E402

args_cli.agent = resolve_agent_entry_point(args_cli.adapter, args_cli.agent)

sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from datetime import datetime  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
import traceback  # noqa: E402

import torch  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402

from jose.distillation.command_eval import evaluate_student_command_grid  # noqa: E402
from jose.estimator.adapters import make_policy_adapter  # noqa: E402
from jose.robustness.methods import (  # noqa: E402
    load_jose, load_set, load_student, load_teacher,
)
from jose.robustness.registry import AXIS_METHODS, resolve  # noqa: E402
from jose.robustness.noise import (  # noqa: E402
    EncoderNoiseCfg, apply_encoder_noise_cfg, install_encoder_noise,
)
from jose.teacher_setup import build_env_and_teacher  # noqa: E402


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    study = Path(args_cli.study).resolve()
    set_study = Path(args_cli.set_study).resolve() if args_cli.set_study else None
    out = Path(args_cli.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    methods = tuple(args_cli.methods) if args_cli.methods else AXIS_METHODS[args_cli.axis]
    encoder_scale = args_cli.scale if args_cli.axis == "encoder" else 0.0
    imu_scale = args_cli.scale if args_cli.axis == "imu" else 0.0

    torch.manual_seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    # Before the build: the teacher acts on this vector, so degrading only the
    # estimator's view would leave the policy reading perfect encoders.
    observation_noise = apply_encoder_noise_cfg(env_cfg, encoder_scale)
    print(f"[robustness] policy observation: {observation_noise}", flush=True)

    env, teacher_agent = build_env_and_teacher(
        args_cli.task, args_cli.adapter, env_cfg, agent_cfg,
        args_cli.teacher_checkpoint, args_cli.device, seed=args_cli.seed,
    )
    teacher_agent.enable_training_mode(False, apply_to_models=True)
    base_adapter = make_policy_adapter(args_cli.adapter, env, "all")
    core = base_adapter.core_env
    device = torch.device(core.device)
    print(
        f"[robustness] axis={args_cli.axis} scale={args_cli.scale} "
        f"methods={list(methods)} seeds={args_cli.seeds}",
        flush=True,
    )

    encoder_cfg = EncoderNoiseCfg(
        scale=encoder_scale, bias_fraction=args_cli.encoder_bias_fraction
    )
    written = 0
    try:
        # After the build: installed on the environment, not on each adapter, so
        # no method can be handed clean joints by reading the other accessor.
        with install_encoder_noise(core, encoder_cfg) as encoder_corruptor:
            for method in methods:
                for seed in args_cli.seeds:
                    kind, checkpoint = resolve(method, study, set_study, seed)
                    if kind != "teacher" and (checkpoint is None or not checkpoint.is_file()):
                        print(f"  {method} seed {seed}: no checkpoint at {checkpoint}", flush=True)
                        continue
                    adapter = base_adapter
                    if kind in ("set", "set_enc"):
                        from jose.set_baseline.adapter import SETPolicyAdapter

                        adapter = SETPolicyAdapter(base_adapter)

                    if kind == "teacher":
                        act, on_step, info = load_teacher(adapter, teacher_agent)
                    elif kind == "estimator":
                        act, on_step, info = load_jose(
                            adapter, teacher_agent, checkpoint, device, args_cli.num_envs
                        )
                    elif kind in ("set", "set_enc"):
                        act, on_step, info = load_set(
                            adapter, teacher_agent, checkpoint, device, args_cli.num_envs,
                            imu_scale,
                        )
                    else:
                        act, on_step, info = load_student(
                            core, checkpoint, device, args_cli.num_envs, imu_scale
                        )

                    def stepped(dones, _on_step=on_step):
                        _on_step(dones)
                        encoder_corruptor.reset(dones)

                    metrics = evaluate_student_command_grid(
                        env, adapter, act, stepped,
                        settle_s=args_cli.grid_settle_s, measure_s=args_cli.grid_measure_s,
                        seed=seed,
                    )
                    row = {
                        "axis": args_cli.axis,
                        "scale": args_cli.scale,
                        "method": method,
                        "method_kind": kind,
                        "seed": seed,
                        "task": args_cli.task,
                        "measured_at": datetime.now().isoformat(),
                        "encoder_noise": encoder_cfg.describe(),
                        "observation_noise": observation_noise,
                        "imu_scale": imu_scale,
                        "method_info": info,
                        "metrics": metrics,
                    }
                    with out.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(row) + "\n")
                    written += 1
                    print(
                        f"  {method:26s} seed {seed}  "
                        f"track={metrics['track_error_norm']:.4f} "
                        f"survival={metrics['grid_survival_rate']:.3f}",
                        flush=True,
                    )
    finally:
        env.close()
    print(f"[robustness] wrote {written} rows to {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
