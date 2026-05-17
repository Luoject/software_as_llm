"""Training hyperparameters and environment setup (non-algorithm)."""

from dataclasses import dataclass, asdict


@dataclass
class TrainConfig:
    env_name: str = "LunarLander-v2"
    has_continuous_action_space: bool = False
    max_ep_len: int = 300
    max_training_timesteps: int = int(1e6)
    print_freq: int = 300 * 8
    log_freq: int = 300 * 2
    save_model_freq: int = int(5e4)
    action_std: float = 0.6
    action_std_decay_rate: float = 0.05
    min_action_std: float = 0.1
    action_std_decay_freq: int = int(2.5e5)
    update_timestep: int = 300 * 4
    K_epochs: int = 80
    eps_clip: float = 0.2
    gamma: float = 0.99
    lr_actor: float = 0.0003
    lr_critic: float = 0.001
    random_seed: int = 42

    def to_hyperparameter_dict(self) -> dict:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.print_freq <= self.max_ep_len:
            self.print_freq = self.max_ep_len * 8
        if self.log_freq <= self.max_ep_len:
            self.log_freq = self.max_ep_len * 2
        if self.update_timestep <= self.max_ep_len:
            self.update_timestep = self.max_ep_len * 4
