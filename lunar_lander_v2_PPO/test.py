import os
import time

import gym

from PPO import PPO
from training.checkpoints import resolve_checkpoint_path


def test():
    print("============================================================================================")

    env_name = "LunarLander-v2"
    has_continuous_action_space = False
    max_ep_len = 300
    action_std = None

    render = True
    frame_delay = 0
    total_test_episodes = 10

    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.001

    random_seed = 42
    run_num_pretrained = 0
    use_best_checkpoint = True

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std,
    )

    checkpoint_path = resolve_checkpoint_path(
        env_name,
        random_seed,
        run_num_pretrained,
        use_best=use_best_checkpoint,
    )
    if use_best_checkpoint and not os.path.isfile(checkpoint_path):
        print("best checkpoint not found, falling back to periodic checkpoint")
        checkpoint_path = resolve_checkpoint_path(
            env_name, random_seed, run_num_pretrained, use_best=False
        )

    print("loading network from : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)
    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        ppo_agent.buffer.clear()
        test_running_reward += ep_reward
        print("Episode: {} \t\t Reward: {}".format(ep, round(ep_reward, 2)))

    env.close()

    print("============================================================================================")
    avg_test_reward = round(test_running_reward / total_test_episodes, 2)
    print("average test reward : " + str(avg_test_reward))
    print("============================================================================================")


if __name__ == "__main__":
    test()
