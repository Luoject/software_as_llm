#!/usr/bin/env python3
"""Save Lunar Lander HS rollout frames and build GIF."""

import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image

from heuristic_lunar_lander import MAX_EP_LEN, load_raw, raw_to_params, select_action
from hl_env import make_env

POLICY = Path(__file__).resolve().parent / "HS_checkpoints" / "heuristic_lunar_lander_best.json"
ENV_NAME = "LunarLander-v2"


def main() -> None:
    raw = load_raw(POLICY)
    params = raw_to_params(raw)
    env = make_env()

    img_dir = Path("PPO_gif_images") / ENV_NAME
    gif_dir = Path("PPO_gifs") / ENV_NAME
    img_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)

    state = env.reset()
    total_reward = 0.0
    t = 0
    for t in range(1, MAX_EP_LEN + 1):
        action = select_action(np.asarray(state, dtype=np.float64), params)
        state, reward, done, _ = env.step(action)
        total_reward += reward
        frame = env.render(mode="rgb_array")
        Image.fromarray(frame).save(img_dir / f"{t:06d}.jpg")
        if done:
            break
    env.close()
    print(f"Saved {t} frames, reward={total_reward:.2f}")

    paths = sorted(glob.glob(str(img_dir / "*.jpg")))[::10][:300]
    imgs = [Image.open(p) for p in paths]
    gif_path = gif_dir / "HL_LunarLander-v2_gif_0.gif"
    imgs[0].save(
        gif_path,
        save_all=True,
        append_images=imgs[1:],
        duration=150,
        loop=0,
        optimize=True,
    )
    print(f"GIF: {gif_path}")


if __name__ == "__main__":
    main()
