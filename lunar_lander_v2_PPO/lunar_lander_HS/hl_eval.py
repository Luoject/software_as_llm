"""Evaluate HS policy in LunarLander-v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from heuristic_lunar_lander import MAX_EP_LEN, raw_to_params, select_action
from hl_env import make_env, backend_name


@dataclass
class Evaluation:
    episodes: int
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    returns: list[float]
    success_rate: float  # landed with both legs roughly


def rollout(raw: np.ndarray, seed: int, render: bool = False) -> tuple[float, bool]:
    env = make_env()
    if hasattr(env, "seed"):
        env.seed(seed)
    if render:
        env.render()
    np.random.seed(seed)
    state = env.reset()
    params = raw_to_params(raw)
    total = 0.0
    landed = False
    for _ in range(MAX_EP_LEN):
        action = select_action(np.asarray(state, dtype=np.float64), params)
        state, reward, done, _ = env.step(action)
        total += float(reward)
        if done:
            x, y, vx, vy, angle, ang_vel, leg1, leg2 = state
            landed = bool(leg1 and leg2) and abs(vx) < 0.5 and abs(vy) < 0.5
            break
    env.close()
    return total, landed


def evaluate(raw: np.ndarray, episodes: int, seed: int) -> Evaluation:
    returns: list[float] = []
    successes = 0
    for ep in range(episodes):
        ret, ok = rollout(raw, seed + ep * 17)
        returns.append(ret)
        successes += int(ok)
    arr = np.asarray(returns, dtype=np.float64)
    return Evaluation(
        episodes=episodes,
        mean_return=float(np.mean(arr)),
        std_return=float(np.std(arr)),
        min_return=float(np.min(arr)),
        max_return=float(np.max(arr)),
        returns=[float(x) for x in returns],
        success_rate=successes / max(episodes, 1),
    )


def evaluate_batch(raw_batch: np.ndarray, repeats: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized scoring: score = mean - 0.15*std (stability)."""
    candidates = np.asarray(raw_batch, dtype=np.float64)
    n = len(candidates)
    mean_ret = np.zeros(n, dtype=np.float64)
    std_ret = np.zeros(n, dtype=np.float64)
    all_returns = []
    for i, raw in enumerate(candidates):
        ev = evaluate(raw, repeats, seed + i * 101)
        mean_ret[i] = ev.mean_return
        std_ret[i] = ev.std_return
        all_returns.append(ev.returns)
    mins = np.array([float(np.min(r)) for r in all_returns], dtype=np.float64)
    score = mean_ret - 0.25 * std_ret + 0.08 * mins
    return score, mean_ret, np.asarray(all_returns, dtype=object)


def evaluation_to_dict(ev: Evaluation) -> dict:
    return asdict(ev)
