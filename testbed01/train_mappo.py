import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Independent
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from drone_crowd_env import DroneCrowdEnv

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
ENABLE_GUI = False          # True = watch every GUI_INTERVAL-th update live (slower)
GUI_INTERVAL = 25
WARM_START_EPOCHS = 0       # behavioural-cloning warm start; 0 = disabled (recommended)

NUM_UPDATES = 400           # PPO updates
EPISODES_PER_UPDATE = 3     # episodes collected per update (=> 1200 episodes total)
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 256
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.03
INIT_LR = 3e-4
FINAL_LR_FRAC = 0.1         # lr decays linearly to INIT_LR * FINAL_LR_FRAC
BEST_MA_WINDOW = 5          # best_actor.pth is chosen by the moving average of this many updates


# ----------------------------------------------------------------------------
# Networks (GaussianActor is imported by validate_mappo.py - keep the class name)
# ----------------------------------------------------------------------------
class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, init_log_std=-1.0):
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
        self.log_std = nn.Parameter(torch.full((act_dim,), float(init_log_std)))

    def forward(self, obs):
        is_1d = (obs.dim() == 1)
        if is_1d:
            obs = obs.unsqueeze(0)

        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        x = self.net(obs)
        mean = torch.tanh(self.mean_layer(x))
        bounded_log_std = torch.clamp(self.log_std, min=-2.0, max=0.5)
        std = torch.exp(bounded_log_std).expand_as(mean)
        std = torch.clamp(std, min=1e-3, max=10.0)

        if is_1d:
            mean = mean.squeeze(0)
            std = std.squeeze(0)

        base_dist = Normal(mean, std)
        return Independent(base_dist, reinterpreted_batch_ndims=1)


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
        if global_state.dim() == 1:
            global_state = global_state.unsqueeze(0)
        return self.net(global_state)


# ----------------------------------------------------------------------------
# Optional behavioural-cloning warm start (disabled by default)
# ----------------------------------------------------------------------------
class CrowdImitationDataset(Dataset):
    def __init__(self, dataset_dir="dataset", obs_dim=55):
        self.samples = []
        for file_path in glob.glob(os.path.join(dataset_dir, "*.npz")):
            data = np.load(file_path)
            h_pos = data["human_positions"]
            surge = data["surge_active"]

            for t in range(len(surge)):
                if not surge[t]:
                    expert_act = np.array([0.0, 0.0, 0.5, 0.0], dtype=np.float32)
                else:
                    expert_act = np.array([0.0, -0.3, -1.0, 0.0], dtype=np.float32)

                dummy_obs = np.zeros(obs_dim, dtype=np.float32)
                dummy_obs[0:2] = np.mean(h_pos[t], axis=0)
                self.samples.append((dummy_obs, expert_act))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (torch.tensor(self.samples[idx][0], dtype=torch.float32),
                torch.tensor(self.samples[idx][1], dtype=torch.float32))


def warm_start_actor(actor, obs_dim, dataset_dir="dataset", epochs=2, lr=1e-4):
    print("\n=== Pre-training Actor via Behavioral Cloning (Warm-Start) ===")
    dataset = CrowdImitationDataset(dataset_dir=dataset_dir, obs_dim=obs_dim)
    if len(dataset) == 0:
        print("[Warning] No dataset found. Skipping warm-start.")
        return

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(actor.parameters(), lr=lr)

    actor.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for states, expert_acts in loader:
            dist = actor(states)
            loss = F.mse_loss(dist.mean, expert_acts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"BC Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")
    print("=== Warm-Start Complete ===\n")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
class RunningStd:
    """Running std of discounted returns -> used to scale rewards so the critic sees O(1) targets."""

    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        bm, bv, bc = x.mean(), x.var(), x.size
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    @property
    def std(self):
        return float(np.sqrt(self.var) + 1e-8)


def discounted_returns(rewards, gamma):
    out = np.zeros(len(rewards), dtype=np.float64)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def compute_gae(rewards, values, last_value, gamma, lam):
    T = len(rewards)
    v = np.append(np.asarray(values, dtype=np.float32), np.float32(last_value))
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * v[t + 1] - v[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv, adv + v[:-1]


def collect_episode(env, actor, critic, render=False):
    env.render_mode = "human" if render else None
    obs, _ = env.reset()
    traj = {a: {"obs": [], "act": [], "logprob": [], "reward": [], "val": [], "state": []}
            for a in env.possible_agents}
    metrics = {"ped": [], "area": [], "ov": []}

    while env.agents:
        global_state = env.get_global_state()
        with torch.no_grad():
            value = critic(torch.tensor(global_state, dtype=torch.float32)).item()

        actions, step_obs, step_acts, step_logprobs = {}, {}, {}, {}
        for agent in env.agents:
            agent_obs = obs[agent]
            with torch.no_grad():
                dist = actor(torch.tensor(agent_obs, dtype=torch.float32))
                action = dist.sample()
                log_prob = dist.log_prob(action)
            action_np = torch.clamp(action, -1.0, 1.0).cpu().numpy()
            actions[agent] = action_np
            step_obs[agent] = agent_obs
            step_acts[agent] = action_np
            step_logprobs[agent] = log_prob.item()

        obs, rewards, _, _, infos = env.step(actions)

        for agent in step_obs:
            traj[agent]["obs"].append(step_obs[agent])
            traj[agent]["state"].append(global_state)
            traj[agent]["act"].append(step_acts[agent])
            traj[agent]["logprob"].append(step_logprobs[agent])
            traj[agent]["reward"].append(rewards[agent])
            traj[agent]["val"].append(value)

        if infos:
            m = next(iter(infos.values()))
            metrics["ped"].append(m["ped_coverage"])
            metrics["area"].append(m["area_coverage"])
            metrics["ov"].append(m["overlap_of_union"])

    # The episode ends on a time limit, so bootstrap from the critic instead of assuming V=0.
    with torch.no_grad():
        last_value = critic(torch.tensor(env.get_global_state(), dtype=torch.float32)).item()

    return traj, last_value, {k: float(np.mean(v)) for k, v in metrics.items()}


def plot_training_results(h):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("MAPPO Drone Swarm Training Performance", fontsize=14, fontweight="bold")
    x = h["update"]

    ax1.plot(x, h["reward"], alpha=0.3, color="dodgerblue", label="Mean episode reward (per update)")
    if len(h["reward"]) >= 10:
        ma = np.convolve(h["reward"], np.ones(10) / 10, mode="valid")
        ax1.plot(x[9:], ma, color="blue", linewidth=2, label="10-update moving avg")
    ax1.set_ylabel("Total Swarm Reward")
    ax1.set_title("Reward Convergence")
    ax1.legend(loc="upper left")

    ax2.plot(x, np.array(h["ped"]) * 100, label="Pedestrian coverage %", color="green")
    ax2.plot(x, np.array(h["area"]) * 100, label="Area coverage %", color="teal")
    ax2.plot(x, np.array(h["ov"]) * 100, label="Overlap % (of covered area)", color="crimson")
    ax2.set_ylabel("%")
    ax2.set_title("Coverage vs Overlap")
    ax2.legend(loc="best")

    ax3.set_xlabel("PPO update")
    ax3.set_ylabel("Actor Loss", color="crimson")
    ax3.plot(x, h["actor_loss"], color="crimson", alpha=0.8)
    ax3t = ax3.twinx()
    ax3t.set_ylabel("Critic Loss", color="darkorange")
    ax3t.plot(x, h["critic_loss"], color="darkorange", alpha=0.8)

    plt.tight_layout()
    os.makedirs("checkpoints", exist_ok=True)
    plt.savefig("checkpoints/training_performance.png", dpi=300)
    plt.show()


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(num_updates=NUM_UPDATES, episodes_per_update=EPISODES_PER_UPDATE,
          dataset_dir="dataset", seed=0):
    os.makedirs("checkpoints", exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = DroneCrowdEnv(
        num_drones=3,
        env_size=(25, 25),
        base_fov_size=(8.0, 4.5),
        k_nearest=5,
        dataset_dir=dataset_dir,
        render_mode=None,
    )

    env.reset()
    obs_dim = env.observation_space("drone_0").shape[0]
    act_dim = env.action_space("drone_0").shape[0]
    global_state_dim = len(env.get_global_state())

    actor = GaussianActor(obs_dim, act_dim)
    critic = CentralizedCritic(global_state_dim)

    if WARM_START_EPOCHS > 0:
        warm_start_actor(actor, obs_dim, dataset_dir=dataset_dir, epochs=WARM_START_EPOCHS)

    actor_opt = optim.Adam(actor.parameters(), lr=INIT_LR)
    critic_opt = optim.Adam(critic.parameters(), lr=INIT_LR)

    ret_rms = RunningStd()
    best_ma = -float("inf")
    hist = {k: [] for k in ["update", "reward", "ped", "area", "ov", "actor_loss", "critic_loss"]}

    print("=== Starting MAPPO Training ===")
    print(f"Observation Dim: {obs_dim} | Global State Dim: {global_state_dim} | "
          f"{num_updates} updates x {episodes_per_update} episodes\n")

    for update in range(num_updates):
        frac = update / max(num_updates - 1, 1)
        lr = INIT_LR * (1.0 - frac * (1.0 - FINAL_LR_FRAC))
        for opt in (actor_opt, critic_opt):
            for g in opt.param_groups:
                g["lr"] = lr

        reward_scale = 1.0 / ret_rms.std   # frozen for the whole update
        render = ENABLE_GUI and ((update + 1) % GUI_INTERVAL == 0)

        all_obs, all_states, all_acts, all_logprobs = [], [], [], []
        all_returns, all_advantages = [], []
        ep_rewards, ep_ped, ep_area, ep_ov = [], [], [], []
        raw_discounted = []

        for ep in range(episodes_per_update):
            traj, last_value, m = collect_episode(env, actor, critic, render=(render and ep == 0))
            ep_total = 0.0
            for agent in env.possible_agents:
                t = traj[agent]
                raw_r = np.array(t["reward"], dtype=np.float64)
                ep_total += raw_r.sum()
                raw_discounted.append(discounted_returns(raw_r, GAMMA))

                adv, ret = compute_gae((raw_r * reward_scale).astype(np.float32),
                                       t["val"], last_value, GAMMA, GAE_LAMBDA)
                all_obs.extend(t["obs"])
                all_states.extend(t["state"])
                all_acts.extend(t["act"])
                all_logprobs.extend(t["logprob"])
                all_advantages.extend(adv)
                all_returns.extend(ret)

            ep_rewards.append(ep_total)
            ep_ped.append(m["ped"]); ep_area.append(m["area"]); ep_ov.append(m["ov"])

        ret_rms.update(np.concatenate(raw_discounted))   # used from the NEXT update on

        obs_t = torch.tensor(np.array(all_obs), dtype=torch.float32)
        state_t = torch.tensor(np.array(all_states), dtype=torch.float32)
        act_t = torch.tensor(np.array(all_acts), dtype=torch.float32)
        old_logprob_t = torch.tensor(np.array(all_logprobs), dtype=torch.float32)
        returns_t = torch.tensor(np.array(all_returns), dtype=torch.float32)
        adv_np = np.array(all_advantages, dtype=np.float32)
        adv_t = torch.tensor((adv_np - adv_np.mean()) / (adv_np.std() + 1e-8), dtype=torch.float32)

        actor_losses, critic_losses = [], []
        n = obs_t.size(0)

        for _ in range(PPO_EPOCHS):
            perm = torch.randperm(n)
            kls = []
            for s in range(0, n, MINI_BATCH_SIZE):
                idx = perm[s:s + MINI_BATCH_SIZE]

                dist = actor(obs_t[idx])
                new_logprob = dist.log_prob(act_t[idx])
                entropy = dist.entropy().mean()
                log_ratio = new_logprob - old_logprob_t[idx]
                ratios = torch.exp(log_ratio)
                with torch.no_grad():
                    kls.append(((ratios - 1) - log_ratio).mean().item())

                surr1 = ratios * adv_t[idx]
                surr2 = torch.clamp(ratios, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_t[idx]
                actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
                actor_opt.step()

                v_pred = critic(state_t[idx]).squeeze(-1)
                critic_loss = nn.SmoothL1Loss()(v_pred, returns_t[idx])
                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
                critic_opt.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

            if np.mean(kls) > TARGET_KL:
                break

        mean_reward = float(np.mean(ep_rewards))
        hist["update"].append(update + 1)
        hist["reward"].append(mean_reward)
        hist["ped"].append(float(np.mean(ep_ped)))
        hist["area"].append(float(np.mean(ep_area)))
        hist["ov"].append(float(np.mean(ep_ov)))
        hist["actor_loss"].append(float(np.mean(actor_losses)))
        hist["critic_loss"].append(float(np.mean(critic_losses)))

        ma = float(np.mean(hist["reward"][-BEST_MA_WINDOW:]))
        if len(hist["reward"]) >= BEST_MA_WINDOW and ma > best_ma:
            best_ma = ma
            torch.save(actor.state_dict(), "checkpoints/best_actor.pth")
        torch.save(actor.state_dict(), "checkpoints/last_actor.pth")
        torch.save(critic.state_dict(), "checkpoints/last_critic.pth")

        print(f"Update {update+1:03d}/{num_updates} | R {mean_reward:8.0f} | MA{BEST_MA_WINDOW} {ma:8.0f} | "
              f"ped {hist['ped'][-1]*100:4.0f}% area {hist['area'][-1]*100:4.0f}% "
              f"overlap {hist['ov'][-1]*100:4.0f}% | ret_std {ret_rms.std:6.1f} | "
              f"std {torch.exp(actor.log_std).mean().item():.2f}")

    env.close()
    plot_training_results(hist)


if __name__ == "__main__":
    train()
