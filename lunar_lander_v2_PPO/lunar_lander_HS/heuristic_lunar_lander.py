#!/usr/bin/env python3
"""Lunar Lander heuristic policy (HS core). Pure NumPy, no neural network."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ENV_ID = "LunarLander-v2"
MAX_EP_LEN = 300

# CEM searches this many scalars; each maps to a bounded control gain/threshold.
PARAM_DIM = 18
PARAM_NAMES = [
    "kp_angle",
    "kd_angle",
    "kp_x",
    "kd_x",
    "tilt_threshold",
    "kp_y",
    "kd_y",
    "vy_main_thresh",
    "y_main_min",
    "y_main_max",
    "vy_land_thresh",
    "y_land_zone",
    "angle_land_max",
    "main_bias",
    "left_score_scale",
    "right_score_scale",
    "main_score_scale",
    "noop_penalty",
]


@dataclass(frozen=True)
class LunarParams:
    kp_angle: float
    kd_angle: float
    kp_x: float
    kd_x: float
    tilt_threshold: float
    kp_y: float
    kd_y: float
    vy_main_thresh: float
    y_main_min: float
    y_main_max: float
    vy_land_thresh: float
    y_land_zone: float
    angle_land_max: float
    main_bias: float
    left_score_scale: float
    right_score_scale: float
    main_score_scale: float
    noop_penalty: float


def _squash(raw: np.ndarray, lo: float, hi: float) -> float:
    x = float(np.tanh(raw))
    return lo + (hi - lo) * (x + 1.0) * 0.5


def raw_to_params(raw: np.ndarray) -> LunarParams:
    r = np.asarray(raw, dtype=np.float64).reshape(-1)
    if r.size < PARAM_DIM:
        r = np.pad(r, (0, PARAM_DIM - r.size))
    return LunarParams(
        kp_angle=_squash(r[0], 0.5, 4.0),
        kd_angle=_squash(r[1], 0.1, 2.0),
        kp_x=_squash(r[2], 0.0, 1.5),
        kd_x=_squash(r[3], 0.0, 1.0),
        tilt_threshold=_squash(r[4], 0.05, 0.8),
        kp_y=_squash(r[5], 0.0, 2.0),
        kd_y=_squash(r[6], 0.1, 2.5),
        vy_main_thresh=_squash(r[7], -2.0, 0.5),
        y_main_min=_squash(r[8], 0.05, 0.8),
        y_main_max=_squash(r[9], 0.5, 1.4),
        vy_land_thresh=_squash(r[10], -1.5, 0.0),
        y_land_zone=_squash(r[11], 0.02, 0.35),
        angle_land_max=_squash(r[12], 0.1, 0.6),
        main_bias=_squash(r[13], -0.5, 1.5),
        left_score_scale=_squash(r[14], 0.5, 3.0),
        right_score_scale=_squash(r[15], 0.5, 3.0),
        main_score_scale=_squash(r[16], 0.5, 3.0),
        noop_penalty=_squash(r[17], 0.0, 0.5),
    )


def default_raw() -> np.ndarray:
    """Hand-tuned seed (HL iteration 0)."""
    return np.array(
        [
            1.2,
            0.5,
            0.0,
            0.2,
            -0.5,
            0.8,
            0.6,
            -0.3,
            -0.2,
            0.4,
            -0.4,
            -0.4,
            -0.1,
            0.5,
            0.3,
            0.3,
            0.5,
            -1.0,
        ],
        dtype=np.float64,
    )


def select_action(obs: np.ndarray, params: LunarParams) -> int:
    """Discrete action 0..3 from 8-dim LunarLander observation."""
    x, y, vx, vy, angle, ang_vel, leg1, leg2 = obs
    on_ground = bool(leg1) or bool(leg2)
    near_ground = y < params.y_land_zone or on_ground

    tilt_cmd = params.kp_angle * angle + params.kd_angle * ang_vel
    tilt_cmd += params.kp_x * x + params.kd_x * vx

    if near_ground:
        tilt_cmd *= params.angle_land_max / max(params.tilt_threshold, 1e-3)

    vy_target = -0.15 if near_ground else -0.4
    thrust_cmd = params.kp_y * (0.7 - y) + params.kd_y * (vy_target - vy) + params.main_bias

    if near_ground:
        if vy > params.vy_land_thresh:
            thrust_cmd = min(thrust_cmd, 0.0)

    scores = np.zeros(4, dtype=np.float64)
    scores[0] = params.noop_penalty
    scores[1] = params.left_score_scale * max(0.0, tilt_cmd - params.tilt_threshold)
    scores[3] = params.right_score_scale * max(0.0, -tilt_cmd - params.tilt_threshold)

    use_main = False
    if params.y_main_min < y < params.y_main_max and vy < params.vy_main_thresh:
        use_main = True
    if near_ground and vy < params.vy_land_thresh:
        use_main = True
    if thrust_cmd > 0.25:
        use_main = True

    if use_main:
        scores[2] = params.main_score_scale * max(thrust_cmd, 0.15)

    return int(np.argmax(scores))


def params_to_dict(params: LunarParams) -> dict[str, float]:
    return asdict(params)


def save_params(raw: np.ndarray, path: Path, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "raw": [float(x) for x in raw.reshape(-1)],
        "params": params_to_dict(raw_to_params(raw)),
        "param_names": PARAM_NAMES,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_raw(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["raw"], dtype=np.float64)
