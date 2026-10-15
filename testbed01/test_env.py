import time
import numpy as np
from drone_crowd_env import DroneCrowdEnv

if __name__ == "__main__":
    env = DroneCrowdEnv(
        num_drones=3,
        num_people=35,
        env_size=(40, 40),
        fov_size=(8.0, 5.0),
        render_mode="human"
    )

    obs, infos = env.reset()
    print("Environment reset successful!")
    print(f"Observation space per agent: {env.observation_space('drone_0')}")

    for step in range(200):
        # Sample random actions: continuous velocity [dx, dy]
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        obs, rewards, terminations, truncations, infos = env.step(actions)

        if step % 20 == 0:
            total_reward = sum(rewards.values())
            print(f"Step {step:03d} | Total covered heads across 3 drones: {total_reward}")

        if not env.agents:
            break

    env.close()
    print("Test run completed.")