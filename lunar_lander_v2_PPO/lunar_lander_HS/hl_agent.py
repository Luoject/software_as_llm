"""Programmatic HL 'coding agent': rule-based HS patches from trial feedback."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from heuristic_lunar_lander import PARAM_DIM, default_raw, raw_to_params, save_params


def analyze_failure_returns(returns: list[float]) -> str:
    if not returns:
        return "no_returns"
    mean_r = float(np.mean(returns))
    if mean_r < -200:
        return "crash_early"
    if mean_r < 0:
        return "unstable_landing"
    if mean_r < 150:
        return "near_solved"
    return "good"


def suggest_patch(raw: np.ndarray, diagnosis: str, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Return patched raw vector and human-readable patch description."""
    patched = raw.copy()
    notes: list[str] = []

    if diagnosis == "crash_early":
        patched[0] += rng.normal(0, 0.4)
        patched[1] += rng.normal(0, 0.3)
        patched[16] += 0.25
        notes.append("increase angle PD + main thrust scale")
    elif diagnosis == "unstable_landing":
        patched[4] -= 0.15
        patched[10] -= 0.2
        patched[11] -= 0.15
        notes.append("tighter tilt threshold + landing vy/y zone")
    elif diagnosis == "near_solved":
        patched[5] += rng.normal(0, 0.2)
        patched[6] += rng.normal(0, 0.2)
        patched[13] += 0.1
        notes.append("fine-tune vertical PD + main bias")
    else:
        patched += rng.normal(0, 0.05, size=PARAM_DIM)
        notes.append("small exploration jitter")

    return patched, "; ".join(notes) or "noop patch"


def apply_agent_round(
    best_raw: np.ndarray,
    last_returns: list[float],
    policy_path: Path,
    diff_log: Path,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, str]:
    from hl_eval import evaluate

    diagnosis = analyze_failure_returns(last_returns)
    rng = rng or np.random.default_rng()
    patched, note = suggest_patch(best_raw, diagnosis, rng)

    old_params = raw_to_params(best_raw)
    new_params = raw_to_params(patched)
    diff = {
        "diagnosis": diagnosis,
        "patch_note": note,
        "param_delta": {
            k: round(getattr(new_params, k) - getattr(old_params, k), 6)
            for k in old_params.__dataclass_fields__
        },
    }
    diff_log.parent.mkdir(parents=True, exist_ok=True)
    with diff_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(diff, ensure_ascii=False) + "\n")

    trial_ev = evaluate(patched, 6, 777)
    base_ev = evaluate(best_raw, 6, 777)
    if trial_ev.mean_return >= base_ev.mean_return - 2.0:
        save_params(patched, policy_path, extra={"hl_patch": diff})
        return patched, f"[HL-Agent] {diagnosis}: {note} (accepted mean={trial_ev.mean_return:.1f})"
    return best_raw, f"[HL-Agent] {diagnosis}: {note} (rejected mean={trial_ev.mean_return:.1f} < {base_ev.mean_return:.1f})"
