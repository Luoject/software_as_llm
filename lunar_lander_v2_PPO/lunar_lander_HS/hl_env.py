"""LunarLander-v2 compatible env: gym if available, else Gym-scaled NumPy simulator."""

from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np

ENV_ID = "LunarLander-v2"
_USE_GYM = False

try:
    import gym

    _probe = gym.make("LunarLander-v2")
    _probe.close()
    _USE_GYM = True
except Exception:
    gym = None  # type: ignore


class NumpyLunarLander:
    """Box2D-free lander with reward shaping aligned to OpenAI LunarLander-v2 scale."""

    gravity = 10.0
    main_power = 13.0
    side_power = 0.6
    dt = 0.05
    pad_x = 0.0
    pad_y = 0.0

    def __init__(self) -> None:
        self.observation_space = type("Space", (), {"shape": (8,)})()
        self.action_space = type("Space", (), {"n": 4})()
        self.state: np.ndarray | None = None
        self.steps = 0
        self.seed_val = 0
        self.prev_shaping = 0.0

    def seed(self, seed: int | None = None) -> list[int]:
        self.seed_val = int(seed or 0)
        np.random.seed(self.seed_val)
        return [self.seed_val]

    def _shaping(self, s: np.ndarray) -> float:
        x, y, vx, vy, ang, ang_v, c1, c2 = s
        return (
            -0.3 * x * x
            -0.3 * y * y
            -100.0 * math.sqrt(0.01 + vx * vx)
            -0.3 * ang * ang
            -0.3 * ang_v * ang_v
            -10.0 * math.sqrt(0.01 + vy * vy)
            + 10.0 * c1
            + 10.0 * c2
        )

    def reset(self) -> np.ndarray:
        np.random.seed(self.seed_val + self.steps)
        self.steps = 0
        self.state = np.array(
            [
                np.random.uniform(-0.1, 0.1),
                1.4 + np.random.uniform(-0.05, 0.05),
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(-0.5, 0.0),
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(-0.02, 0.02),
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        self.prev_shaping = self._shaping(self.state)
        return self.state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert self.state is not None
        x, y, vx, vy, ang, ang_v, c1, c2 = self.state
        action = int(action)

        if action == 2:
            vy += self.main_power * self.dt
            vx += math.sin(ang) * self.main_power * 0.3 * self.dt
        if action == 1:
            ang_v -= self.side_power * self.dt
        if action == 3:
            ang_v += self.side_power * self.dt

        vy -= self.gravity * self.dt
        x += vx * self.dt
        y += vy * self.dt
        ang += ang_v * self.dt
        ang_v *= 0.995
        vx *= 0.999

        reward = 0.0
        done = False
        self.steps += 1

        if y <= 0.05:
            y = 0.0
            on_pad = abs(x - self.pad_x) < 0.6
            safe = on_pad and abs(vx) < 1.0 and abs(vy) < 1.0 and abs(ang) < 0.4
            if safe:
                c1, c2 = 1.0, 1.0
            else:
                reward -= 100.0
            done = True
        elif y > 1.5 or abs(x) > 2.5:
            reward -= 100.0
            done = True
        elif self.steps >= 300:
            done = True

        self.state = np.array([x, y, vx, vy, ang, ang_v, c1, c2], dtype=np.float64)
        shaping = self._shaping(self.state)
        reward += shaping - self.prev_shaping
        self.prev_shaping = shaping

        if done and c1 > 0.5 and c2 > 0.5:
            reward += 100.0
            reward += max(0.0, 80.0 - 40.0 * (abs(vx) + abs(vy) + abs(ang)))

        return self.state.copy(), float(reward), done, {}

    def close(self) -> None:
        pass

    def render(self, mode: str = "human") -> Any:
        if mode == "rgb_array":
            from PIL import Image, ImageDraw

            w, h = 400, 300
            img = Image.new("RGB", (w, h), (20, 20, 40))
            draw = ImageDraw.Draw(img)
            assert self.state is not None
            px = int(w * 0.5 + self.state[0] * 80)
            py = int(h * 0.85 - self.state[1] * 120)
            draw.polygon(
                [(px, py), (px - 10, py + 18), (px + 10, py + 18)],
                fill=(200, 220, 255),
            )
            draw.rectangle([(w // 2 - 40, h - 30), (w // 2 + 40, h - 20)], fill=(80, 80, 80))
            return np.asarray(img)
        return None


def make_env():
    if _USE_GYM:
        return gym.make(ENV_ID)
    return NumpyLunarLander()


def backend_name() -> str:
    return "gym+box2d" if _USE_GYM else "numpy_fallback"
