"""Checkpoint and best-model tracking (orchestration layer, not RL math)."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass
class BestModelRecord:
    env_name: str
    random_seed: int
    timestep: int
    episode: int
    avg_reward: float
    checkpoint_path: str
    hyperparameters: dict[str, Any]
    saved_at: str


class BestModelTracker:
    """Track rolling average reward and persist best weights + hyperparameters."""

    def __init__(
        self,
        env_name: str,
        random_seed: int,
        hyperparameters: dict[str, Any],
        base_dir: str = "PPO_preTrained",
        log_dir: str = "PPO_logs",
    ) -> None:
        self.env_name = env_name
        self.random_seed = random_seed
        self.hyperparameters = hyperparameters
        self.base_dir = base_dir
        self.log_dir = log_dir

        self.best_avg_reward = float("-inf")
        self.best_timestep = 0
        self.best_episode = 0

        env_ckpt_dir = os.path.join(base_dir, env_name)
        os.makedirs(env_ckpt_dir, exist_ok=True)
        self.best_checkpoint_path = os.path.join(
            env_ckpt_dir, f"PPO_{env_name}_{random_seed}_best.pth"
        )
        self.best_meta_path = os.path.join(
            env_ckpt_dir, f"PPO_{env_name}_{random_seed}_best_meta.json"
        )

        env_log_dir = os.path.join(log_dir, env_name)
        os.makedirs(env_log_dir, exist_ok=True)
        self.best_log_path = os.path.join(
            env_log_dir, f"PPO_{env_name}_best_model_log.jsonl"
        )

    def maybe_update(
        self,
        avg_reward: float,
        timestep: int,
        episode: int,
        save_fn,
        periodic_checkpoint_path: str | None = None,
    ) -> bool:
        """Save if avg_reward is a new best. Returns True if updated."""
        if avg_reward <= self.best_avg_reward:
            return False

        self.best_avg_reward = avg_reward
        self.best_timestep = timestep
        self.best_episode = episode

        save_fn(self.best_checkpoint_path)
        record = BestModelRecord(
            env_name=self.env_name,
            random_seed=self.random_seed,
            timestep=timestep,
            episode=episode,
            avg_reward=round(avg_reward, 4),
            checkpoint_path=self.best_checkpoint_path,
            hyperparameters=self.hyperparameters,
            saved_at=datetime.now().isoformat(timespec="seconds"),
        )
        with open(self.best_meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(record), f, indent=2, ensure_ascii=False)

        with open(self.best_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        if periodic_checkpoint_path and os.path.isfile(periodic_checkpoint_path):
            shutil.copy2(periodic_checkpoint_path, self.best_checkpoint_path + ".periodic_copy")

        print("--------------------------------------------------------------------------------------------")
        print(f"[BEST] New best avg reward: {avg_reward:.4f} @ episode={episode} timestep={timestep}")
        print(f"[BEST] Saved weights: {self.best_checkpoint_path}")
        print(f"[BEST] Saved metadata: {self.best_meta_path}")
        print(f"[BEST] Appended log: {self.best_log_path}")
        print("--------------------------------------------------------------------------------------------")
        return True
