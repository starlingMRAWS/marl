import os
import torch
import numpy as np
import time
from drone_crowd_env import DroneCrowdEnv
from train_mappo import GaussianActor

def evaluate():
    checkpoint_path = "checkpoints/best_actor.pth"
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found at '{checkpoint_path}'. Please run training first.")
        return

    # Initialize environment with human rendering enabled to watch live
    env = DroneCrowdEnv(
        num_drones=3,
        env_size=(25, 25),
        base_fov_size=(8.0, 4.5),
        k_nearest=5,
        dataset_dir="dataset",
        render_mode="human"
    )

    obs, _ = env.reset()
    obs_dim = env.observation_space("drone_0").shape[0]
    act_dim = env.action_space("drone_0").shape[0]

    # Load trained actor weights
    actor = GaussianActor(obs_dim, act_dim)
    actor.load_state_dict(torch.load(checkpoint_path))
    actor.eval()

    print(f"\n=== Loaded Actor Checkpoint from {checkpoint_path} ===")
    print("=== Starting Live Evaluation Rollout ===\n")

    total_episode_reward = 0.0
    step_count = 0

    while env.agents:
        actions = {}
        with torch.no_grad():
            for agent in env.agents:
                agent_obs = obs[agent]
                obs_tensor = torch.tensor(agent_obs, dtype=torch.float32)
                
                # Sample deterministic or mean actions for evaluation (using mean for robust behavior)
                dist = actor(obs_tensor)
                action = dist.mean  # Use mean action for smooth, optimal evaluation
                
                action_np = torch.clamp(action, -1.0, 1.0).cpu().numpy()
                actions[agent] = action_np

        next_obs, rewards, terminations, truncations, _ = env.step(actions)
        
        step_reward = sum(rewards.values())
        total_episode_reward += step_reward
        step_count += 1
        
        obs = next_obs
        
        # Control playback speed for human viewing
        time.sleep(0.03)

    print(f"=== Evaluation Complete ===")
    print(f"Total Steps: {step_count} | Final Episode Reward: {total_episode_reward:.2f}")
    env.close()

if __name__ == "__main__":
    evaluate()
