import os
from datetime import datetime
from typing import Optional

import gym
import numpy as np
import torch

from llm_integration.dual_policy import describe_dual_policy_roles
from rl.ppo import PPO
from training.checkpoints import CheckpointManager
from training.config import TrainConfig


def _setup_log_file(config: TrainConfig) -> tuple:
    log_dir = os.path.join(config.log_dir, config.env_name)
    os.makedirs(log_dir, exist_ok=True)

    run_num = len(next(os.walk(log_dir))[2])
    log_f_name = os.path.join(log_dir, f"PPO_{config.env_name}_log_{run_num}.csv")

    print("current logging run number for " + config.env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    return log_f_name, run_num


def _print_hyperparameters(config: TrainConfig, state_dim: int, action_dim: int) -> None:
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", config.max_training_timesteps)
    print("max timesteps per episode : ", config.max_ep_len)
    print("model saving frequency : " + str(config.save_model_freq) + " timesteps")
    print("log frequency : " + str(config.log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(config.print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if config.has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", config.action_std)
        print("decay rate of std of action distribution : ", config.action_std_decay_rate)
        print("minimum std of action distribution : ", config.min_action_std)
        print(
            "decay frequency of std of action distribution : "
            + str(config.action_std_decay_freq)
            + " timesteps"
        )
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(config.update_timestep) + " timesteps")
    print("PPO K epochs : ", config.K_epochs)
    print("PPO epsilon clip : ", config.eps_clip)
    print("discount factor (gamma) : ", config.gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", config.lr_actor)
    print("optimizer learning rate critic : ", config.lr_critic)
    if config.random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", config.random_seed)
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
    print("--------------------------------------------------------------------------------------------")
    print(describe_dual_policy_roles())


def run_training(config: Optional[TrainConfig] = None) -> None:
    if config is None:
        config = TrainConfig()

    print("============================================================================================")
    print("training environment name : " + config.env_name)

    env = gym.make(config.env_name)
    state_dim = env.observation_space.shape[0]
    if config.has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    log_f_name, run_num = _setup_log_file(config)
    checkpoint_mgr = CheckpointManager(config, run_num)

    print("save periodic checkpoint path : " + checkpoint_mgr.periodic_path)
    print("save best checkpoint path : " + checkpoint_mgr.best_model_path)

    _print_hyperparameters(config, state_dim, action_dim)
    print("============================================================================================")

    ppo_agent = PPO(
        state_dim,
        action_dim,
        config.lr_actor,
        config.lr_critic,
        config.gamma,
        config.K_epochs,
        config.eps_clip,
        config.has_continuous_action_space,
        config.action_std,
    )

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    log_f = open(log_f_name, "w+")
    log_f.write("episode,timestep,reward\n")

    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    while time_step <= config.max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0

        for _t in range(1, config.max_ep_len + 1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            if time_step % config.update_timestep == 0:
                ppo_agent.update()

            if config.has_continuous_action_space and time_step % config.action_std_decay_freq == 0:
                ppo_agent.decay_action_std(config.action_std_decay_rate, config.min_action_std)

            if time_step % config.log_freq == 0 and log_running_episodes > 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write("{},{},{}\n".format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                checkpoint_mgr.maybe_save_best(
                    ppo_agent, log_avg_reward, time_step, i_episode
                )

                log_running_reward = 0
                log_running_episodes = 0

            if time_step % config.print_freq == 0 and print_running_episodes > 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print(
                    "Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                        i_episode, time_step, print_avg_reward
                    )
                )

                print_running_reward = 0
                print_running_episodes = 0

            if time_step % config.save_model_freq == 0:
                checkpoint_mgr.save_periodic(ppo_agent, time_step)
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)

            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        log_running_reward += current_ep_reward
        log_running_episodes += 1
        i_episode += 1

    log_f.close()
    env.close()

    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("Best avg reward during run : ", checkpoint_mgr.best_avg_reward)
    print("Best model path : ", checkpoint_mgr.best_model_path)
    print("============================================================================================")
