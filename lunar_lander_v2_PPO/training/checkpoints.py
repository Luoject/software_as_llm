import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from training.config import TrainConfig


def resolve_checkpoint_path(
    env_name: str,
    random_seed: int,
    run_num_pretrained: int,
    checkpoint_root: str = "PPO_preTrained",
    use_best: bool = False,
) -> str:
    """Resolve periodic or best checkpoint path for loading in test / gif scripts."""
    directory = os.path.join(checkpoint_root, env_name)
    if use_best:
        best_dir = os.path.join(directory, "best")
        return os.path.join(best_dir, f"PPO_{env_name}_best.pth")
    return os.path.join(directory, f"PPO_{env_name}_{random_seed}_{run_num_pretrained}.pth")


class CheckpointManager:
    """Periodic and best-model checkpointing with hyperparameter metadata logs."""

    def __init__(self, config: TrainConfig, run_num: int) -> None:
        self.config = config
        self.run_num = run_num
        self.best_avg_reward = float("-inf")

        env_dir = os.path.join(config.checkpoint_root, config.env_name)
        os.makedirs(env_dir, exist_ok=True)

        self.periodic_path = resolve_checkpoint_path(
            config.env_name,
            config.random_seed,
            config.run_num_pretrained,
            config.checkpoint_root,
            use_best=False,
        )

        self.best_dir = os.path.join(env_dir, config.best_checkpoint_subdir)
        os.makedirs(self.best_dir, exist_ok=True)
        self.best_model_path = os.path.join(
            self.best_dir, f"PPO_{config.env_name}_best.pth"
        )
        self.best_metadata_path = os.path.join(self.best_dir, "best_metadata.json")
        self.best_log_path = os.path.join(self.best_dir, "best_checkpoint_log.jsonl")
        self.run_hyperparams_path = os.path.join(
            env_dir, f"run_{run_num}_hyperparameters.json"
        )

        self._write_run_hyperparameters()

    def _write_run_hyperparameters(self) -> None:
        payload = {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "run_num": self.run_num,
            "hyperparameters": self.config.to_dict(),
        }
        with open(self.run_hyperparams_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("run hyperparameters saved at : " + self.run_hyperparams_path)

    def save_periodic(self, ppo_agent, timestep: int) -> None:
        print("--------------------------------------------------------------------------------------------")
        print("saving periodic model at : " + self.periodic_path)
        ppo_agent.save(self.periodic_path)
        print("periodic model saved (timestep {})".format(timestep))
        print("--------------------------------------------------------------------------------------------")

    def maybe_save_best(
        self,
        ppo_agent,
        avg_reward: float,
        timestep: int,
        episode: int,
    ) -> bool:
        if avg_reward <= self.best_avg_reward:
            return False

        previous_best = self.best_avg_reward
        self.best_avg_reward = avg_reward

        ppo_agent.save(self.best_model_path)
        metadata = self._build_metadata(avg_reward, timestep, episode, previous_best)
        self._write_metadata(metadata)
        self._append_best_log(metadata)

        print("--------------------------------------------------------------------------------------------")
        print(
            "new best model saved at : {} (avg reward {:.4f}, was {:.4f})".format(
                self.best_model_path, avg_reward, previous_best
            )
        )
        print("best metadata : " + self.best_metadata_path)
        print("--------------------------------------------------------------------------------------------")
        return True

    def _build_metadata(
        self,
        avg_reward: float,
        timestep: int,
        episode: int,
        previous_best: float,
    ) -> Dict[str, Any]:
        return {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "env_name": self.config.env_name,
            "run_num": self.run_num,
            "episode": episode,
            "timestep": timestep,
            "avg_reward": avg_reward,
            "previous_best_avg_reward": None if previous_best == float("-inf") else previous_best,
            "checkpoint_path": self.best_model_path,
            "hyperparameters": self.config.to_dict(),
        }

    def _write_metadata(self, metadata: Dict[str, Any]) -> None:
        with open(self.best_metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _append_best_log(self, metadata: Dict[str, Any]) -> None:
        with open(self.best_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata) + "\n")
