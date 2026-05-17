#!/usr/bin/env python3
"""Test lunar_lander_HS policy with optional render."""

import argparse
from pathlib import Path

from heuristic_lunar_lander import load_raw
from hl_eval import evaluate, rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).resolve().parent / "HS_checkpoints" / "heuristic_lunar_lander_best.json",
    )
    args = parser.parse_args()

    raw = load_raw(args.policy)
    print(f"Loading HS policy: {args.policy}")

    if args.render:
        for ep in range(args.episodes):
            ret, ok = rollout(raw, seed=ep * 31, render=True)
            print(f"Episode {ep + 1}: reward={ret:.2f} landed={ok}")
    else:
        ev = evaluate(raw, args.episodes, seed=42)
        print(f"mean={ev.mean_return:.2f} std={ev.std_return:.2f} success={ev.success_rate:.2%}")
        print("returns:", [round(x, 1) for x in ev.returns])


if __name__ == "__main__":
    main()
