import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
logging.getLogger("numba").setLevel(logging.WARNING)

from drone_crowd_env import DroneCrowdEnv

ENABLE_GUI = True
GUI_INTERVAL = 1


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


class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, global_state):
        return self.net(global_state)


def plot_training_results(episodes, rewards, actor_losses, critic_losses):
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("MAPPO Drone Swarm Training Performance", fontsize=14, fontweight='bold')

    ax1.plot(episodes, rewards, alpha=0.3, color='dodgerblue', label='Episode Reward')
    if len(rewards) >= 10:
        ma_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
        ax1.plot(episodes[9:], ma_rewards, color='blue', linewidth=2, label='10-Ep Moving Avg')
    ax1.set_ylabel("Total Swarm Reward", fontsize=11)
    ax1.legend(loc='upper left')
    ax1.set_title("Reward Convergence", fontsize=12)

    color_actor = 'crimson'
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Actor Loss", color=color_actor, fontsize=11)
    line1 = ax2.plot(episodes, actor_losses, color=color_actor, label='Actor Loss', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color_actor)

    ax2_twin = ax2.twinx()
    color_critic = 'darkorange'
    ax2_twin.set_ylabel("Critic Loss", color=color_critic, fontsize=11)
    line2 = ax2_twin.plot(episodes, critic_losses, color=color_critic, label='Critic Loss', alpha=0.8)
    ax2_twin.tick_params(axis='y', labelcolor=color_critic)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    ax2.set_title("Policy & Value Network Losses (Dual Scale)", fontsize=12)

    plt.tight_layout()
    plt.savefig("checkpoints/training_performance.png", dpi=300)
    print("\n[Graphics] Saved training plot to './checkpoints/training_performance.png'")
    plt.show()


def train():
    os.makedirs("checkpoints", exist_ok=True)

    env = DroneCrowdEnv(
        num_drones=3,
        num_people=30,
        env_size=(40, 40),
        fov_size=(8.0, 5.0),
        k_nearest=5,
        max_steps=300,
        render_mode=None
    )

    obs, _ = env.reset()
    obs_dim = env.observation_space("drone_0").shape[0]
    act_dim = env.action_space("drone_0").shape[0]
    global_state_dim = len(env.get_global_state())

    actor = GaussianActor(obs_dim, act_dim)
    critic = CentralizedCritic(global_state_dim)

    init_lr = 3e-4
    actor_opt = optim.Adam(actor.parameters(), lr=init_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=init_lr)

    num_episodes = 500
    gamma = 0.99
    gae_lambda = 0.95
    ppo_epochs = 4
    mini_batch_size = 128
    clip_eps = 0.2
    entropy_coef = 0.01
    reward_scale = 0.05
    max_grad_norm = 0.5
    target_kl = 0.05
    best_reward = -float("inf")

    history_episodes, history_rewards = [], []
    history_actor_loss, history_critic_loss = [], []

    print("=== Starting Restored MAPPO Training ===")
    print(f"Observation Dim: {obs_dim} | Global State Dim: {global_state_dim}\n")

    for episode in range(num_episodes):
        frac = max(0.1, 1.0 - (episode / float(num_episodes)))
        curr_lr = init_lr * frac
        for param_group in actor_opt.param_groups:
            param_group['lr'] = curr_lr
        for param_group in critic_opt.param_groups:
            param_group['lr'] = curr_lr

        should_render = ENABLE_GUI and ((episode + 1) % GUI_INTERVAL == 0)
        env.render_mode = "human" if should_render else None

        obs, _ = env.reset()

        agent_trajectories = {
            agent: {"obs": [], "act": [], "logprob": [], "reward": [], "val": [], "state": []}
            for agent in env.possible_agents
        }
        raw_ep_rewards = {agent: 0.0 for agent in env.possible_agents}

        while env.agents:
            global_state = env.get_global_state()
            state_tensor = torch.tensor(global_state, dtype=torch.float32)

            with torch.no_grad():
                value = critic(state_tensor).item()

            actions = {}
            step_obs, step_acts, step_logprobs = {}, {}, {}

            for agent in env.agents:
                agent_obs = obs[agent]
                obs_tensor = torch.tensor(agent_obs, dtype=torch.float32)

                with torch.no_grad():
                    dist = actor(obs_tensor)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum()

                action_np = torch.clamp(action, -1.0, 1.0).numpy()
                actions[agent] = action_np

                step_obs[agent] = agent_obs
                step_acts[agent] = action_np
                step_logprobs[agent] = log_prob.item()

            next_obs, rewards, terminations, truncations, _ = env.step(actions)

            for agent in step_obs:
                agent_trajectories[agent]["obs"].append(step_obs[agent])
                agent_trajectories[agent]["state"].append(global_state)
                agent_trajectories[agent]["act"].append(step_acts[agent])
                agent_trajectories[agent]["logprob"].append(step_logprobs[agent])
                agent_trajectories[agent]["reward"].append(rewards[agent] * reward_scale)
                agent_trajectories[agent]["val"].append(value)
                raw_ep_rewards[agent] += rewards[agent]

            obs = next_obs

        all_obs, all_states, all_acts, all_logprobs = [], [], [], []
        all_returns, all_advantages = [], []

        for agent in env.possible_agents:
            traj = agent_trajectories[agent]
            r = traj["reward"]
            v = traj["val"] + [0.0]
            T = len(r)

            adv = np.zeros(T, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T)):
                delta = r[t] + gamma * v[t + 1] - v[t]
                gae = delta + gamma * gae_lambda * gae
                adv[t] = gae

            ret = adv + np.array(traj["val"], dtype=np.float32)

            all_obs.extend(traj["obs"])
            all_states.extend(traj["state"])
            all_acts.extend(traj["act"])
            all_logprobs.extend(traj["logprob"])
            all_advantages.extend(adv)
            all_returns.extend(ret)

        obs_t = torch.tensor(np.array(all_obs), dtype=torch.float32)
        state_t = torch.tensor(np.array(all_states), dtype=torch.float32)
        act_t = torch.tensor(np.array(all_acts), dtype=torch.float32)
        old_logprob_t = torch.tensor(np.array(all_logprobs), dtype=torch.float32)
        returns_t = torch.tensor(np.array(all_returns), dtype=torch.float32)

        adv_np = np.array(all_advantages, dtype=np.float32)
        adv_t = torch.tensor((adv_np - adv_np.mean()) / (adv_np.std() + 1e-8), dtype=torch.float32)

        epoch_actor_losses, epoch_critic_losses = [], []
        dataset_size = obs_t.size(0)

        for epoch in range(ppo_epochs):
            permutation = torch.randperm(dataset_size)
            epoch_kls = []

            for start_idx in range(0, dataset_size, mini_batch_size):
                batch_indices = permutation[start_idx : start_idx + mini_batch_size]

                b_obs = obs_t[batch_indices]
                b_state = state_t[batch_indices]
                b_act = act_t[batch_indices]
                b_old_logprob = old_logprob_t[batch_indices]
                b_adv = adv_t[batch_indices]
                b_returns = returns_t[batch_indices]

                dist = actor(b_obs)
                new_logprob = dist.log_prob(b_act).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                log_ratio = new_logprob - b_old_logprob
                ratios = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl = (b_old_logprob - new_logprob).mean().item()
                    epoch_kls.append(approx_kl)

                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean() - (entropy_coef * entropy)

                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=max_grad_norm)
                actor_opt.step()

                v_pred = critic(b_state).squeeze(-1)
                critic_loss = nn.SmoothL1Loss()(v_pred, b_returns)

                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=max_grad_norm)
                critic_opt.step()

                epoch_actor_losses.append(actor_loss.item())
                epoch_critic_losses.append(critic_loss.item())

            # Evaluate KL divergence at epoch level
            if np.mean(epoch_kls) > target_kl:
                break

        total_ep_reward = sum(raw_ep_rewards.values())

        history_episodes.append(episode + 1)
        history_rewards.append(total_ep_reward)
        history_actor_loss.append(np.mean(epoch_actor_losses) if epoch_actor_losses else 0.0)
        history_critic_loss.append(np.mean(epoch_critic_losses) if epoch_critic_losses else 0.0)

        if total_ep_reward > best_reward:
            best_reward = total_ep_reward
            torch.save(actor.state_dict(), "checkpoints/best_actor.pth")
            save_msg = " [Saved Best Model]"
        else:
            save_msg = ""

        current_std = float(torch.exp(torch.clamp(actor.log_std, -2.0, 0.5))[0].item())
        print(f"Episode {episode + 1:03d}/{num_episodes} | Swarm Reward: {total_ep_reward:6.1f} | Action Std: {current_std:.3f} | Actor Loss: {history_actor_loss[-1]:.4f} | Critic Loss: {history_critic_loss[-1]:.4f}{save_msg}")

    checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "episode": num_episodes,
        "best_reward": best_reward,
    }
    torch.save(checkpoint, "checkpoints/full_checkpoint.pth")
    print("\n=== Training Finished. Checkpoints saved to ./checkpoints/ ===")

    env.close()
    plot_training_results(history_episodes, history_rewards, history_actor_loss, history_critic_loss)


if __name__ == "__main__":
    train()
