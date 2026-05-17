import os
from datetime import datetime

import gym
import numpy as np
import torch

from rl.checkpointing import BestModelTracker
from rl.ppo import PPO
from rl.train_config import TrainConfig


def train() -> None:
    cfg = TrainConfig()
    print("============================================================================================")
    print("training environment name : " + cfg.env_name)

    env = gym.make(cfg.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = (
        env.action_space.shape[0]
        if cfg.has_continuous_action_space
        else env.action_space.n
    )

    log_dir = os.path.join("PPO_logs", cfg.env_name)
    os.makedirs(log_dir, exist_ok=True)
    run_num = len(os.listdir(log_dir))
    log_f_name = os.path.join(log_dir, f"PPO_{cfg.env_name}_log_{run_num}.csv")
    print("current logging run number for", cfg.env_name, ":", run_num)
    print("logging at :", log_f_name)

    run_num_pretrained = 0
    ckpt_dir = os.path.join("PPO_preTrained", cfg.env_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        ckpt_dir, f"PPO_{cfg.env_name}_{cfg.random_seed}_{run_num_pretrained}.pth"
    )
    print("save checkpoint path :", checkpoint_path)

    hyperparameters = cfg.to_hyperparameter_dict()
    hyperparameters["state_dim"] = state_dim
    hyperparameters["action_dim"] = action_dim
    best_tracker = BestModelTracker(
        env_name=cfg.env_name,
        random_seed=cfg.random_seed,
        hyperparameters=hyperparameters,
    )

    print("--------------------------------------------------------------------------------------------")
    for k, v in hyperparameters.items():
        print(f"  {k}: {v}")
    if cfg.random_seed:
        torch.manual_seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
    print("============================================================================================")

    ppo_agent = PPO(
        state_dim,
        action_dim,
        cfg.lr_actor,
        cfg.lr_critic,
        cfg.gamma,
        cfg.K_epochs,
        cfg.eps_clip,
        cfg.has_continuous_action_space,
        cfg.action_std,
    )

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) :", start_time)
    print("============================================================================================")

    log_f = open(log_f_name, "w+", encoding="utf-8")
    log_f.write("episode,timestep,reward\n")

    print_running_reward = 0.0
    print_running_episodes = 0
    log_running_reward = 0.0
    log_running_episodes = 0
    time_step = 0
    i_episode = 0

    while time_step <= cfg.max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0.0

        for _t in range(1, cfg.max_ep_len + 1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            time_step += 1
            current_ep_reward += reward

            if time_step % cfg.update_timestep == 0:
                ppo_agent.update()

            if cfg.has_continuous_action_space and time_step % cfg.action_std_decay_freq == 0:
                ppo_agent.decay_action_std(cfg.action_std_decay_rate, cfg.min_action_std)

            if time_step % cfg.log_freq == 0 and log_running_episodes > 0:
                log_avg_reward = round(log_running_reward / log_running_episodes, 4)
                log_f.write(f"{i_episode},{time_step},{log_avg_reward}\n")
                log_f.flush()
                best_tracker.maybe_update(
                    log_avg_reward,
                    time_step,
                    i_episode,
                    ppo_agent.save,
                    periodic_checkpoint_path=checkpoint_path,
                )
                log_running_reward = 0.0
                log_running_episodes = 0

            if time_step % cfg.print_freq == 0 and print_running_episodes > 0:
                print_avg_reward = round(print_running_reward / print_running_episodes, 2)
                print(
                    f"Episode : {i_episode} \t\t Timestep : {time_step} \t\t "
                    f"Average Reward : {print_avg_reward}"
                )
                print_running_reward = 0.0
                print_running_episodes = 0

            if time_step % cfg.save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at :", checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  :", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        log_running_reward += current_ep_reward
        log_running_episodes += 1
        i_episode += 1

    log_f.close()
    env.close()

    end_time = datetime.now().replace(microsecond=0)
    print("============================================================================================")
    print("Started training at (GMT) :", start_time)
    print("Finished training at (GMT) :", end_time)
    print("Total training time  :", end_time - start_time)
    print(
        f"Best avg reward: {best_tracker.best_avg_reward} @ timestep {best_tracker.best_timestep}"
    )
    print("Best checkpoint:", best_tracker.best_checkpoint_path)
    print("============================================================================================")


if __name__ == "__main__":
    train()
