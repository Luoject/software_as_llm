from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class TrainConfig:
    env_name: str = "LunarLander-v2"
    has_continuous_action_space: bool = False

    max_ep_len: int = 300
    max_training_timesteps: int = int(1e6)

    print_freq: int = 0
    log_freq: int = 0
    save_model_freq: int = int(5e4)

    action_std: float = 0.6
    action_std_decay_rate: float = 0.05
    min_action_std: float = 0.1
    action_std_decay_freq: int = int(2.5e5)

    update_timestep: int = 0
    K_epochs: int = 80
    eps_clip: float = 0.2
    gamma: float = 0.99
    lr_actor: float = 0.0003
    lr_critic: float = 0.001

    random_seed: int = 42
    run_num_pretrained: int = 0

    log_dir: str = "PPO_logs"
    checkpoint_root: str = "PPO_preTrained"
    best_checkpoint_subdir: str = "best"

    def __post_init__(self) -> None:
        if self.print_freq <= 0:
            self.print_freq = self.max_ep_len * 8
        if self.log_freq <= 0:
            self.log_freq = self.max_ep_len * 2
        if self.update_timestep <= 0:
            self.update_timestep = self.max_ep_len * 4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
