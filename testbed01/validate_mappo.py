import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
logging.getLogger("numba").setLevel(logging.WARNING)

from drone_crowd_env import DroneCrowdEnv


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
        )
        self.mean_layer = nn.Linear(128, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def forward(self, obs):
        x = self.net(obs)
        mean = torch.tanh(self.mean_layer(x))
        bounded_log_std = torch.clamp(self.log_std, min=-2.0, max=0.5)
        std = torch.exp(bounded_log_std).expand_as(mean)
        return Normal(mean, std)

    def get_action_deterministic(self, obs):
        """Returns the mean action for evaluation (no noise)."""
        x = self.net(obs)
        mean = torch.tanh(self.mean_layer(x))
        return torch.clamp(mean, -1.0, 1.0)


def evaluate(
    checkpoint_path="checkpoints/best_actor.pth",
    num_eval_episodes=10,
    render=True
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found!")

    # Initialize environment
    render_mode = "human" if render else None
    env = DroneCrowdEnv(
        num_drones=3,
        num_people=30,
        env_size=(40, 40),
        fov_size=(8.0, 5.0),
        k_nearest=5,
        max_steps=300,
        render_mode=render_mode
    )

    obs, _ = env.reset()
    obs_dim = env.observation_space("drone_0").shape[0]
    act_dim = env.action_space("drone_0").shape[0]

    # Load Actor
    actor = GaussianActor(obs_dim, act_dim)
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    actor.load_state_dict(state_dict)
    actor.eval()

    print(f"=== Running MAPPO Policy Evaluation ({num_eval_episodes} Episodes) ===")
    print(f"Loaded weights from: {checkpoint_path}\n")

    eval_rewards = []

    for ep in range(num_eval_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        step_count = 0

        while env.agents:
            actions = {}
            for agent in env.agents:
                agent_obs = obs[agent]
                obs_tensor = torch.tensor(agent_obs, dtype=torch.float32)

                with torch.no_grad():
                    action = actor.get_action_deterministic(obs_tensor)

                actions[agent] = action.numpy()

            next_obs, rewards, terminations, truncations, _ = env.step(actions)
            ep_reward += sum(rewards.values())
            step_count += 1
            obs = next_obs

        eval_rewards.append(ep_reward)
        print(f"Eval Episode {ep + 1:02d}/{num_eval_episodes:02d} | Total Swarm Reward: {ep_reward:7.1f} | Steps: {step_count}")

    env.close()

    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)

    print("\n" + "=" * 45)
    print(f" Mean Swarm Reward : {mean_reward:7.1f} ± {std_reward:.1f}")
    print(f" Max Swarm Reward  : {np.max(eval_rewards):7.1f}")
    print(f" Min Swarm Reward  : {np.min(eval_rewards):7.1f}")
    print("=" * 45)


if __name__ == "__main__":
    evaluate(
        checkpoint_path="checkpoints/best_actor.pth",
        num_eval_episodes=5,
        render=True
    )