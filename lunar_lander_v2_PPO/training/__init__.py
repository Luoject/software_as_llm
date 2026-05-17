"""Training orchestration: config, checkpoints, and the environment training loop."""

from training.config import TrainConfig
from training.checkpoints import CheckpointManager, resolve_checkpoint_path

__all__ = ["TrainConfig", "CheckpointManager", "resolve_checkpoint_path", "run_training"]


def run_training(*args, **kwargs):
    from training.runner import run_training as _run_training

    return _run_training(*args, **kwargs)
