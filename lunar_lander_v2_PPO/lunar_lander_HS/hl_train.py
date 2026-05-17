#!/usr/bin/env python3
"""HL training loop for lunar_lander_HS: CEM + programmatic agent patches."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from heuristic_lunar_lander import PARAM_DIM, default_raw, save_params, load_raw
from hl_agent import apply_agent_round
from hl_env import backend_name
from hl_eval import evaluate, evaluate_batch, evaluation_to_dict
from hl_memory import append_jsonl, compress_trials, snapshot_policy

SCRIPT_DIR = Path(__file__).resolve().parent
HS_DIR = SCRIPT_DIR / "HS_checkpoints"
LOG_DIR = SCRIPT_DIR / "HL_logs"
MEMORY_DIR = SCRIPT_DIR / "HL_memory"

PPO_SOLVED_TARGET = 200.0


def log_print(msg: str) -> None:
    print(msg, flush=True)
    run_log = LOG_DIR / "hl_train_console.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    with run_log.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")


def cem_search(
    mean: np.ndarray,
    std: np.ndarray,
    best_raw: np.ndarray,
    best_score: float,
    args: argparse.Namespace,
    trials_path: Path,
    policy_path: Path,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(args.seed)
    for iteration in range(1, args.cem_iterations + 1):
        t0 = time.time()
        pop = mean + std * rng.standard_normal((args.population, PARAM_DIM))
        pop[0] = best_raw.copy()

        score, mean_ret, returns_obj = evaluate_batch(pop, args.repeats, args.eval_seed + iteration * 997)
        order = np.argsort(score)[::-1]
        top = int(order[0])
        batch_best_score = float(score[top])
        batch_best_mean = float(mean_ret[top])

        if batch_best_score > best_score:
            best_score = batch_best_score
            best_raw = pop[top].copy()
            save_params(
                best_raw,
                policy_path,
                extra={
                    "iteration": iteration,
                    "best_score": best_score,
                    "best_mean_return": batch_best_mean,
                },
            )
            snapshot_policy(policy_path, MEMORY_DIR / "policy_history", f"iter{iteration}")

        elites = pop[order[: args.elites]]
        new_mean = np.mean(elites, axis=0)
        new_std = np.std(elites, axis=0) + args.std_bonus
        mean = args.cem_alpha * new_mean + (1.0 - args.cem_alpha) * mean
        std = np.maximum(args.cem_alpha * new_std + (1.0 - args.cem_alpha) * std, args.min_std)

        record = {
            "phase": "CEM",
            "iteration": iteration,
            "batch_best_score": batch_best_score,
            "batch_best_mean_return": batch_best_mean,
            "batch_best_returns": [float(x) for x in returns_obj[top]],
            "global_best_score": best_score,
            "global_best_mean": float(evaluate(best_raw, args.repeats, args.eval_seed).mean_return),
            "population": args.population,
            "elites": args.elites,
            "elapsed_s": round(time.time() - t0, 2),
            "search_action": f"CEM top {args.elites}/{args.population}",
        }
        append_jsonl(trials_path, record)
        log_print(json.dumps(record, ensure_ascii=False))

        if batch_best_mean >= PPO_SOLVED_TARGET:
            log_print(f"[HL] Reached PPO-level mean return {batch_best_mean:.2f} >= {PPO_SOLVED_TARGET}")
            break

    return best_raw, best_score


def main() -> None:
    parser = argparse.ArgumentParser(description="HL train lunar_lander_HS")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--cem-iterations", type=int, default=25)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gait-std", type=float, default=0.35)
    parser.add_argument("--min-std", type=float, default=0.02)
    parser.add_argument("--std-bonus", type=float, default=0.05)
    parser.add_argument("--cem-alpha", type=float, default=0.7)
    parser.add_argument("--agent-rounds", type=int, default=3)
    args = parser.parse_args()

    HS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    policy_path = HS_DIR / "heuristic_lunar_lander_best.json"
    trials_path = LOG_DIR / "heuristic_lunar_lander_trials.jsonl"
    summary_path = LOG_DIR / "heuristic_lunar_lander_trials_summary.jsonl"
    diff_log = LOG_DIR / "hl_agent_diff.jsonl"

    if policy_path.exists():
        mean = load_raw(policy_path)
        log_print(f"[HL] Resume from {policy_path}")
    else:
        mean = default_raw()
        save_params(mean, policy_path, extra={"init": True})
        log_print("[HL] Initialized default heuristic policy")

    std = np.full(PARAM_DIM, args.gait_std, dtype=np.float64)
    best_raw = mean.copy()
    ev0 = evaluate(best_raw, args.repeats, args.eval_seed)
    best_score = ev0.mean_return - 0.15 * ev0.std_return
    log_print(f"[HL] Baseline mean={ev0.mean_return:.2f} std={ev0.std_return:.2f} success={ev0.success_rate:.2%}")

    log_print(f"[HL] Environment backend: {backend_name()}")
    log_print("=" * 80)
    log_print("[HL] Phase 1: CEM parameter search (automated HL update)")
    best_raw, best_score = cem_search(mean, std, best_raw, best_score, args, trials_path, policy_path)

    log_print("=" * 80)
    log_print("[HL] Phase 2: Agent rule patches (programmatic coding agent)")
    last_returns = ev0.returns
    for r in range(1, args.agent_rounds + 1):
        patched, note = apply_agent_round(best_raw, last_returns, policy_path, diff_log)
        log_print(note)
        ev = evaluate(patched, max(args.repeats, 8), args.eval_seed + r * 51)
        last_returns = ev.returns
        score = ev.mean_return - 0.15 * ev.std_return
        append_jsonl(
            trials_path,
            {
                "phase": "AgentPatch",
                "round": r,
                "mean_return": ev.mean_return,
                "success_rate": ev.success_rate,
                "returns": ev.returns,
                "patch_note": note,
            },
        )
        log_print(
            f"[HL-Agent round {r}] mean={ev.mean_return:.2f} "
            f"success={ev.success_rate:.2%} score={score:.2f}"
        )
        if score > best_score:
            best_score = score
            best_raw = patched.copy()
            save_params(best_raw, policy_path, extra={"agent_round": r, "score": score})

    compress_trials(trials_path, summary_path)

    final_ev = evaluate(best_raw, 30, args.eval_seed + 9999)
    final_path = HS_DIR / "final_evaluation.json"
    final_path.write_text(
        json.dumps(evaluation_to_dict(final_ev), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log_print("=" * 80)
    log_print(
        f"[HL DONE] final mean={final_ev.mean_return:.2f} "
        f"std={final_ev.std_return:.2f} "
        f"success={final_ev.success_rate:.2%} "
        f"min/max={final_ev.min_return:.1f}/{final_ev.max_return:.1f}"
    )
    log_print(f"[HL DONE] policy: {policy_path}")
    log_print(f"[HL DONE] trials: {trials_path}")
    log_print(f"[HL DONE] PPO target: {PPO_SOLVED_TARGET}")

    backend = backend_name()
    threshold = PPO_SOLVED_TARGET if backend == "gym+box2d" else 195.0
    solved = (
        final_ev.mean_return >= threshold
        and final_ev.success_rate >= 0.85
        and final_ev.min_return > 50.0
    )
    if not solved:
        log_print(
            f"[HL] Below target mean>={threshold} success>=85% (backend={backend}); "
            "consider more CEM iterations or install box2d for gym env."
        )
        sys.exit(1)
    log_print("[HL] Training criteria met (HS ready for deployment).")


if __name__ == "__main__":
    main()
