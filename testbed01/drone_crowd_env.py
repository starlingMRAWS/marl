import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
import pysocialforce as psf


class DroneCrowdEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "drone_crowd_v0",
    }

    def __init__(
        self,
        num_drones=3,
        num_people=30,
        env_size=(40, 40),
        fov_size=(8.0, 5.0),
        k_nearest=5,
        max_steps=300,
        render_mode=None,
    ):
        super().__init__()
        self.num_drones = num_drones
        self.num_people = num_people
        self.env_width, self.env_height = env_size
        self.fov_w, self.fov_h = fov_size
        self.k_nearest = k_nearest
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.possible_agents = [f"drone_{i}" for i in range(num_drones)]
        self.agents = self.possible_agents[:]

        # Speed and kinematic properties
        self.dt = 0.05
        self.drone_max_speed = 3.0
        self.human_desired_speed = 1.3
        self.momentum = 0.85

        # Continuous action space: [ax, ay] in [-1, 1]
        self._action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Observation dimension: state (6) + teammates (2*(N-1)) + mean velocity (2) + K-nearest (4*K)
        obs_dim = 6 + (2 * (self.num_drones - 1)) + 2 + (4 * self.k_nearest)
        self._observation_spaces = {
            agent: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        # Rendering
        self.screen_size = (800, 800)
        self.scale_x = self.screen_size[0] / self.env_width
        self.scale_y = self.screen_size[1] / self.env_height
        self.screen = None
        self.clock = None

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def _init_crowd(self):
        """Initializes crowd with 3 distinct groups, local initial scatter targets."""
        pos = np.random.uniform(
            low=[5.0, 5.0],
            high=[self.env_width - 5.0, self.env_height - 5.0],
            size=(self.num_people, 2)
        )

        self.num_groups = 3
        self.group_assignments = np.random.choice(self.num_groups, size=self.num_people)

        self.group_targets = np.random.uniform(
            low=[5.0, 5.0],
            high=[self.env_width - 5.0, self.env_height - 5.0],
            size=(self.num_groups, 2)
        )

        # Phase 1 local scatter initialization
        targets = np.zeros((self.num_people, 2))
        for i in range(self.num_people):
            targets[i] = pos[i] + np.random.uniform(-3.0, 3.0, size=2)

        directions = targets - pos
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        vel = (directions / np.maximum(norms, 1e-6)) * self.human_desired_speed

        ped_states = np.hstack([pos, vel, targets])
        self.crowd_sim = psf.Simulator(ped_states)

    def _update_crowd_dynamics(self):
        """Phased dynamics: Phase 1 (steps 0-40) Scatter; Phase 2 (>40) Directed migration."""
        if self.step_count == 40:
            for i in range(self.num_people):
                grp = self.group_assignments[i]
                self.crowd_sim.peds.state[i, 4:6] = self.group_targets[grp]

        if self.step_count > 40 and self.step_count % 60 == 0:
            for g in range(self.num_groups):
                self.group_targets[g] = np.random.uniform(
                    low=[4.0, 4.0],
                    high=[self.env_width - 4.0, self.env_height - 4.0]
                )
            for i in range(self.num_people):
                grp = self.group_assignments[i]
                self.crowd_sim.peds.state[i, 4:6] = self.group_targets[grp]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.prev_actions = {a: np.zeros(2, dtype=np.float32) for a in self.agents}

        self.drone_positions = {
            agent: np.array([(i + 1) * self.env_width / (self.num_drones + 1), self.env_height / 2], dtype=np.float32)
            for i, agent in enumerate(self.agents)
        }
        self.drone_velocities = {agent: np.zeros(2, dtype=np.float32) for agent in self.agents}

        self._init_crowd()

        obs = self._get_obs()
        infos = {agent: {} for agent in self.agents}

        if self.render_mode == "human":
            self._init_pygame()

        return obs, infos

    def _get_obs(self):
        human_positions = self.crowd_sim.peds.state[:, :2]
        human_velocities = self.crowd_sim.peds.state[:, 2:4]
        observations = {}

        for agent in self.agents:
            pos = self.drone_positions[agent]
            vel = self.drone_velocities[agent]

            teammate_rel = []
            for other in self.possible_agents:
                if other != agent:
                    rel = self.drone_positions[other] - pos
                    teammate_rel.extend(rel)

            diffs = human_positions - pos
            dists = np.linalg.norm(diffs, axis=1)

            k = min(self.k_nearest, len(dists))
            nearest_indices = np.argsort(dists)[:k]

            nearest_k_info = []
            for idx in nearest_indices:
                rel_pos = diffs[idx]
                ped_vel = human_velocities[idx]
                nearest_k_info.extend([rel_pos[0], rel_pos[1], ped_vel[0], ped_vel[1]])

            if len(nearest_indices) < self.k_nearest:
                pad_count = self.k_nearest - len(nearest_indices)
                nearest_k_info.extend([0.0] * (4 * pad_count))

            nearest_k_info = np.array(nearest_k_info, dtype=np.float32)

            nearest_vels = human_velocities[nearest_indices]
            mean_group_vel = np.mean(nearest_vels, axis=0) if len(nearest_vels) > 0 else np.zeros(2)

            obs_vec = np.concatenate([
                pos, vel, [self.fov_w, self.fov_h],
                teammate_rel,
                mean_group_vel,
                nearest_k_info
            ]).astype(np.float32)

            observations[agent] = obs_vec

        return observations

    def get_global_state(self):
        all_drone_states = np.concatenate([self.drone_positions[a] for a in self.possible_agents])
        human_positions = self.crowd_sim.peds.state[:, :2].flatten()
        return np.concatenate([all_drone_states, human_positions]).astype(np.float32)

    def step(self, actions):
        self.step_count += 1

        # 1. Update crowd dynamics using Social Force Model
        self._update_crowd_dynamics()
        prev_pos = self.crowd_sim.peds.state[:, :2].copy()
        self.crowd_sim.step()
        raw_next_pos = self.crowd_sim.peds.state[:, :2].copy()

        disp = raw_next_pos - prev_pos
        disp_norms = np.linalg.norm(disp, axis=1, keepdims=True)
        max_step_dist = self.human_desired_speed * self.dt
        scaled_disp = np.where(
            disp_norms > max_step_dist,
            (disp / np.maximum(disp_norms, 1e-6)) * max_step_dist,
            disp
        )

        actual_pos = prev_pos + scaled_disp
        actual_pos[:, 0] = np.clip(actual_pos[:, 0], 1.0, self.env_width - 1.0)
        actual_pos[:, 1] = np.clip(actual_pos[:, 1], 1.0, self.env_height - 1.0)
        self.crowd_sim.peds.state[:, :2] = actual_pos
        human_positions = actual_pos

        # 2. Kinematic calculation for drones
        for agent in self.agents:
            action = np.clip(actions[agent], -1.0, 1.0)
            target_vel = action * self.drone_max_speed
            current_vel = self.drone_velocities[agent]
            new_vel = (self.momentum * current_vel) + ((1.0 - self.momentum) * target_vel)

            self.drone_velocities[agent] = new_vel
            new_pos = self.drone_positions[agent] + new_vel * self.dt

            new_pos[0] = np.clip(new_pos[0], 0, self.env_width)
            new_pos[1] = np.clip(new_pos[1], 0, self.env_height)
            self.drone_positions[agent] = new_pos

        # 3. Reward formulation
        all_covered = set()
        agent_covered = {}

        for agent in self.agents:
            px, py = self.drone_positions[agent]
            x_min, x_max = px - self.fov_w / 2, px + self.fov_w / 2
            y_min, y_max = py - self.fov_h / 2, py + self.fov_h / 2

            inside = np.where(
                (human_positions[:, 0] >= x_min) & (human_positions[:, 0] <= x_max) &
                (human_positions[:, 1] >= y_min) & (human_positions[:, 1] <= y_max)
            )[0]

            agent_covered[agent] = set(inside)
            all_covered.update(inside)

        rewards = {}
        optimal_dist = np.sqrt(self.fov_w**2 + self.fov_h**2)

        for agent in self.agents:
            px, py = self.drone_positions[agent]

            r_local_cov = 1.5 * len(agent_covered[agent])
            r_team_cov = 0.5 * len(all_covered)

            dists_to_humans = np.linalg.norm(human_positions - np.array([px, py]), axis=1)
            min_dist = np.min(dists_to_humans) if len(dists_to_humans) > 0 else 0.0
            r_dist = -0.1 * min_dist

            r_overlap = 0.0
            for other in self.agents:
                if other != agent:
                    dist_other = np.linalg.norm(self.drone_positions[agent] - self.drone_positions[other])
                    if dist_other < optimal_dist:
                        r_overlap += 0.1 * ((optimal_dist - dist_other) ** 2)

            a_curr = actions[agent]
            a_prev = self.prev_actions[agent]
            r_jitter = (0.02 * np.sum(a_curr ** 2)) + (0.05 * np.sum((a_curr - a_prev) ** 2))

            margin = 2.0
            viol_x = max(0, margin - px) + max(0, px - (self.env_width - margin))
            viol_y = max(0, margin - py) + max(0, py - (self.env_height - margin))
            r_boundary = 0.5 * (viol_x + viol_y)

            rewards[agent] = float(r_local_cov + r_team_cov + r_dist - r_overlap - r_jitter - r_boundary)

        self.prev_actions = {a: actions[a].copy() for a in self.agents}

        terminated = self.step_count >= self.max_steps
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if terminated:
            self.agents = []

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminations, truncations, infos

    def _init_pygame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("MAPPO Drone Swarm Coverage")
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def render(self):
        if self.render_mode is None:
            return

        self._init_pygame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((240, 240, 240))

        # Pedestrians and Velocity Vectors
        human_positions = self.crowd_sim.peds.state[:, :2]
        human_velocities = self.crowd_sim.peds.state[:, 2:4]

        for h, v in zip(human_positions, human_velocities):
            sx = int(h[0] * self.scale_x)
            sy = int(h[1] * self.scale_y)
            pygame.draw.circle(self.screen, (20, 40, 100), (sx, sy), 5)
            end_x = int((h[0] + v[0] * 1.2) * self.scale_x)
            end_y = int((h[1] + v[1] * 1.2) * self.scale_y)
            pygame.draw.line(self.screen, (220, 60, 60), (sx, sy), (end_x, end_y), 2)

        # Drone Positions & FOV Rectangles
        drone_colors = [(220, 50, 50), (50, 180, 50), (50, 50, 220)]
        for i, agent in enumerate(self.possible_agents):
            if agent not in self.drone_positions:
                continue
            pos = self.drone_positions[agent]
            color = drone_colors[i % len(drone_colors)]

            rect_w = int(self.fov_w * self.scale_x)
            rect_h = int(self.fov_h * self.scale_y)
            center_x = int(pos[0] * self.scale_x)
            center_y = int(pos[1] * self.scale_y)

            fov_rect = pygame.Rect(
                center_x - rect_w // 2,
                center_y - rect_h // 2,
                rect_w,
                rect_h
            )
            pygame.draw.rect(self.screen, color, fov_rect, width=2)
            pygame.draw.circle(self.screen, color, (center_x, center_y), 6)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
