import os
import glob
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv


# ----------------------------------------------------------------------------
# Reward weights (all per-step, per-agent). Tune these first if behaviour is off.
# ----------------------------------------------------------------------------
DEFAULT_WEIGHTS = dict(
    fleet_ped=2.0,        # shared: fraction of ALL pedestrians seen by >=1 drone
    own_unique=1.0,       # fraction of pedestrians seen ONLY by this drone
    ped_redundancy=0.5,   # penalty: fraction of pedestrians this drone shares with another
    area_cov=0.5,         # shared: fraction of arena area covered by the union of FOVs
    overlap=4.0,          # penalty: fraction of MY footprint that another drone also covers
    overlap_tol=0.05,     #   ...ignored below this fraction (lets FOVs touch edge-to-edge)
    waste=1.5,            # penalty: fraction of MY FOV that falls outside the arena
    waste_tol=0.10,
    zoom=3.0,             # penalty: |scale - density-dependent target scale|^2
    center=0.5,           # penalty: drone centre far from the choke point it is looking at
    boundary=1.0,         # penalty: drone centre hugging the arena wall
    collision=3.0,        # penalty: drone centres too close
    action_mag=0.005,     # penalty: |a|^2
    jitter=0.05,          # penalty: |a_t - a_{t-1}|^2
)


class DroneCrowdEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "drone_crowd_v0",
    }

    def __init__(
        self,
        num_drones=3,
        env_size=(25, 25),
        base_fov_size=(8.0, 4.5),  # 16:9 ratio
        k_nearest=5,
        dataset_dir="dataset",
        render_mode=None,
        reward_weights=None,
        choke_density=4.0,   # people per density-bin that counts as a full bottleneck
        choke_scale=1.0,     # scale drones should zoom down to at a full bottleneck
        density_grid=4,      # coarse density map (g x g) given to every drone as obs
        area_grid=50,        # resolution of the area-coverage raster
    ):
        super().__init__()
        self.num_drones = num_drones
        self.env_width, self.env_height = env_size
        self.base_fov_w, self.base_fov_h = base_fov_size
        self.k_nearest = k_nearest
        self.render_mode = render_mode

        self.w = dict(DEFAULT_WEIGHTS)
        if reward_weights:
            self.w.update(reward_weights)
        self.choke_density = choke_density
        self.choke_scale = choke_scale
        self.density_grid = density_grid

        # Load pre-recorded episodes from dataset
        self.dataset_dir = dataset_dir
        self.episodes = []
        self._load_dataset()

        self.possible_agents = [f"drone_{i}" for i in range(num_drones)]
        self.agents = self.possible_agents[:]

        self.dt = 0.05
        self.drone_max_speed = 3.0
        self.drone_max_scale_speed = 1.0
        self.drone_max_rot_speed = 0.5
        self.momentum = 0.85

        self.min_scale = 0.5
        self.max_scale = 2.5  # hard limit (previous 1.5x overshoot removed)

        self.collision_threshold = 2.0
        self.boundary_margin = 1.0

        # Raster used for exact (rotation-aware) area coverage / overlap.
        g = area_grid
        xs = (np.arange(g) + 0.5) * self.env_width / g
        ys = (np.arange(g) + 0.5) * self.env_height / g
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self._grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
        self._cell_area = (self.env_width / g) * (self.env_height / g)
        self._bin_pts = None
        self._bin_shape = None

        # 4D Action space: [ax, ay, scale_delta, rotation_delta] in [-1, 1]
        self._action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # own(7) + teammates(5 each) + mean group vel(2) + kNN(4k) + coarse density(g*g)
        obs_dim = 7 + 5 * (self.num_drones - 1) + 2 + 4 * self.k_nearest + density_grid ** 2
        self._observation_spaces = {
            agent: spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

        # Canvas viewport with padding
        self.canvas_size = (1000, 1000)
        self.padding = 150
        self.render_area_size = self.canvas_size[0] - (2 * self.padding)

        self.scale_x = self.render_area_size / self.env_width
        self.scale_y = self.render_area_size / self.env_height

        self.screen = None
        self.clock = None
        self.font = None

        self.current_ep_data = None
        self.max_steps = 0
        self.num_people = 0
        self.density_map = None
        self.metrics = {}

    # ------------------------------------------------------------------ data
    def _load_dataset(self):
        episode_files = glob.glob(os.path.join(self.dataset_dir, "*.npz"))
        if not episode_files:
            raise FileNotFoundError(
                f"No recorded .npz episodes found in '{self.dataset_dir}/'! "
                "Please run your crowd recording script first."
            )

        for file_path in episode_files:
            data = np.load(file_path)
            self.episodes.append({
                "human_positions": data["human_positions"],  # (T, num_people, 2)
                "human_velocities": data["human_velocities"],# (T, num_people, 2)
                "density_maps": data["density_maps"],        # (T, bins_x, bins_y)
                "surge_active": data["surge_active"]         # (T,)
            })
        print(f"[Dataset] Loaded {len(self.episodes)} pre-recorded episode(s) successfully.")

    def _setup_density_bins(self, shape):
        if self._bin_shape == shape:
            return
        bx, by = shape
        xs = (np.arange(bx) + 0.5) * self.env_width / bx
        ys = (np.arange(by) + 0.5) * self.env_height / by
        gx, gy = np.meshgrid(xs, ys, indexing="ij")  # ravel order == density_map[x, y].ravel()
        self._bin_pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
        self._bin_shape = shape

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    # ----------------------------------------------------------------- reset
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.step_count = 0
        self.prev_actions = {a: np.zeros(4, dtype=np.float32) for a in self.agents}

        ep_idx = np.random.randint(0, len(self.episodes))
        self.current_ep_data = self.episodes[ep_idx]
        self.max_steps = len(self.current_ep_data["surge_active"])
        self.num_people = self.current_ep_data["human_positions"].shape[1]

        self.human_positions = self.current_ep_data["human_positions"][0].copy()
        self.human_velocities = self.current_ep_data["human_velocities"][0].copy()
        self.density_map = self.current_ep_data["density_maps"][0]
        self._setup_density_bins(self.density_map.shape)

        # Spread-out start (small jitter so the policy doesn't overfit one layout)
        base = [(6.5, 12.5), (12.5, 12.5), (18.5, 12.5)]
        self.drone_positions = {}
        for i, agent in enumerate(self.agents):
            bx, by = base[i % len(base)]
            jitter = np.random.uniform(-1.0, 1.0, size=2)
            self.drone_positions[agent] = np.array([bx, by], dtype=np.float32) + jitter.astype(np.float32)
        self.drone_velocities = {agent: np.zeros(2, dtype=np.float32) for agent in self.agents}
        self.drone_scales = {agent: 1.5 for agent in self.agents}
        self.drone_thetas = {agent: 0.0 for agent in self.agents}

        self.visual_positions = self.human_positions.copy()
        self.metrics = {}

        obs = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        return obs, infos

    # ----------------------------------------------------------- observations
    def _coarse_density(self):
        g = self.density_grid
        dm = self.density_map
        bx, by = dm.shape
        xe = np.linspace(0, bx, g + 1).astype(int)
        ye = np.linspace(0, by, g + 1).astype(int)
        out = np.zeros((g, g), dtype=np.float32)
        for i in range(g):
            for j in range(g):
                blk = dm[xe[i]:max(xe[i + 1], xe[i] + 1), ye[j]:max(ye[j + 1], ye[j] + 1)]
                out[i, j] = blk.mean() if blk.size else 0.0
        out = np.nan_to_num(out)
        return np.clip(out / self.choke_density, 0.0, 2.0).ravel()

    def _get_obs(self):
        observations = {}
        W, H = self.env_width, self.env_height
        wh = np.array([W, H], dtype=np.float32)
        coarse = self._coarse_density()

        for agent in self.agents:
            pos = self.drone_positions[agent]
            vel = self.drone_velocities[agent]
            scale = self.drone_scales[agent]
            theta = self.drone_thetas[agent]

            own = [
                pos[0] / W * 2 - 1, pos[1] / H * 2 - 1,
                vel[0] / self.drone_max_speed, vel[1] / self.drone_max_speed,
                scale / self.max_scale, np.sin(theta), np.cos(theta),
            ]

            teammate_rel = []
            for other in self.possible_agents:
                if other != agent:
                    rel_pos = (self.drone_positions[other] - pos) / wh
                    rel_scale = (self.drone_scales[other] - scale) / self.max_scale
                    d_theta = self.drone_thetas[other] - theta
                    teammate_rel.extend([rel_pos[0], rel_pos[1], rel_scale,
                                         np.sin(d_theta), np.cos(d_theta)])

            diffs = self.human_positions - pos
            dists = np.linalg.norm(diffs, axis=1)
            k = min(self.k_nearest, len(dists))
            nearest_indices = np.argsort(dists)[:k]

            nearest_k_info = []
            for idx in nearest_indices:
                rel_pos = diffs[idx] / wh
                ped_vel = self.human_velocities[idx]
                nearest_k_info.extend([rel_pos[0], rel_pos[1], ped_vel[0], ped_vel[1]])

            if len(nearest_indices) < self.k_nearest:
                pad_count = self.k_nearest - len(nearest_indices)
                nearest_k_info.extend([0.0] * (4 * pad_count))

            nearest_vels = self.human_velocities[nearest_indices]
            mean_group_vel = np.mean(nearest_vels, axis=0) if len(nearest_vels) > 0 else np.zeros(2)

            obs_vec = np.concatenate([
                own, teammate_rel, mean_group_vel, nearest_k_info, coarse
            ]).astype(np.float32)
            observations[agent] = obs_vec

        return observations

    def get_global_state(self):
        W, H = self.env_width, self.env_height
        drone_states = []
        for a in self.possible_agents:
            pos = self.drone_positions[a]
            vel = self.drone_velocities[a]
            drone_states.extend([
                pos[0] / W * 2 - 1, pos[1] / H * 2 - 1,
                vel[0] / self.drone_max_speed, vel[1] / self.drone_max_speed,
                self.drone_scales[a] / self.max_scale,
                np.sin(self.drone_thetas[a]), np.cos(self.drone_thetas[a]),
            ])

        human_flat = (self.human_positions / np.array([W, H], dtype=np.float32)).flatten()
        return np.concatenate([drone_states, human_flat, self._coarse_density()]).astype(np.float32)

    # --------------------------------------------------------------- geometry
    def _fov_mask(self, pts, center, scale, theta):
        """Boolean mask of points inside the rotated FOV rectangle."""
        w = self.base_fov_w * scale
        h = self.base_fov_h * scale
        d = pts - center
        c, s = np.cos(theta), np.sin(theta)
        rx = c * d[:, 0] + s * d[:, 1]
        ry = -s * d[:, 0] + c * d[:, 1]
        return (np.abs(rx) <= w / 2.0) & (np.abs(ry) <= h / 2.0)

    def _local_choke(self, pos, scale, theta):
        """Peak density visible in the FOV (-> pressure in [0,1]) and where it is."""
        m = self._fov_mask(self._bin_pts, pos, scale, theta)
        if not m.any():
            return 0.0, None
        vals = self.density_map.ravel()[m]
        j = int(np.argmax(vals))
        pressure = float(np.clip(vals[j] / self.choke_density, 0.0, 1.0))
        return pressure, self._bin_pts[m][j]

    # ------------------------------------------------------------------- step
    def step(self, actions):
        self.step_count += 1

        t_idx = min(self.step_count, self.max_steps - 1)
        self.human_positions = self.current_ep_data["human_positions"][t_idx]
        self.human_velocities = self.current_ep_data["human_velocities"][t_idx]
        self.density_map = self.current_ep_data["density_maps"][t_idx]

        # ---- dynamics
        for agent in self.agents:
            act = np.clip(actions[agent], -1.0, 1.0)

            target_vel = act[:2] * self.drone_max_speed
            current_vel = self.drone_velocities[agent]
            new_vel = (self.momentum * current_vel) + ((1.0 - self.momentum) * target_vel)
            self.drone_velocities[agent] = new_vel

            new_pos = self.drone_positions[agent] + new_vel * self.dt
            new_pos[0] = np.clip(new_pos[0], 0.0, self.env_width)
            new_pos[1] = np.clip(new_pos[1], 0.0, self.env_height)
            self.drone_positions[agent] = new_pos

            scale_change = act[2] * self.drone_max_scale_speed * self.dt
            self.drone_scales[agent] = float(
                np.clip(self.drone_scales[agent] + scale_change, self.min_scale, self.max_scale)
            )

            rot_change = act[3] * self.drone_max_rot_speed * self.dt
            self.drone_thetas[agent] = (self.drone_thetas[agent] + rot_change) % (2 * np.pi)

        self.visual_positions += (self.human_positions - self.visual_positions) * 0.2

        # ---- coverage masks (pedestrians + arena raster)
        ped_cov = np.stack([
            self._fov_mask(self.human_positions, self.drone_positions[a],
                           self.drone_scales[a], self.drone_thetas[a])
            for a in self.agents
        ])                                                    # (A, N)
        area_cov = np.stack([
            self._fov_mask(self._grid_pts, self.drone_positions[a],
                           self.drone_scales[a], self.drone_thetas[a])
            for a in self.agents
        ])                                                    # (A, G*G)

        N = max(self.num_people, 1)
        ped_count = ped_cov.sum(axis=0)
        area_count = area_cov.sum(axis=0)
        overlap_cells = area_count >= 2

        fleet_ped_frac = float((ped_count >= 1).sum()) / N
        union_area_frac = float((area_count >= 1).mean())
        overlap_of_union = float(overlap_cells.sum()) / max(int((area_count >= 1).sum()), 1)

        W = self.w
        s_range = self.max_scale - self.min_scale
        rewards = {}
        per_agent_overlap = {}

        for i, agent in enumerate(self.agents):
            pos = self.drone_positions[agent]
            scale = self.drone_scales[agent]
            theta = self.drone_thetas[agent]

            # --- crowd coverage
            unique_frac = float((ped_cov[i] & (ped_count == 1)).sum()) / N
            shared_frac = float((ped_cov[i] & (ped_count >= 2)).sum()) / N
            r_fleet = W["fleet_ped"] * fleet_ped_frac + W["area_cov"] * union_area_frac
            r_unique = W["own_unique"] * unique_frac
            r_redundant = -W["ped_redundancy"] * shared_frac

            # --- geometric overlap (exact, rotation-aware): share of MY footprint
            #     that is also inside another drone's footprint
            my_cells = max(int(area_cov[i].sum()), 1)
            ov = float((area_cov[i] & overlap_cells).sum()) / my_cells
            per_agent_overlap[agent] = ov
            r_overlap = -W["overlap"] * max(0.0, ov - W["overlap_tol"])

            # --- FOV spilling outside the arena is wasted coverage
            fov_area = (self.base_fov_w * scale) * (self.base_fov_h * scale)
            in_arena = int(area_cov[i].sum()) * self._cell_area
            waste = float(np.clip(1.0 - in_arena / fov_area, 0.0, 1.0))
            r_waste = -W["waste"] * max(0.0, waste - W["waste_tol"])

            # --- altitude/zoom: big when calm, small when a choke point is in view
            pressure, peak_pt = self._local_choke(pos, scale, theta)
            target_scale = self.max_scale - pressure * (self.max_scale - self.choke_scale)
            r_zoom = -W["zoom"] * ((scale - target_scale) / s_range) ** 2

            r_center = 0.0
            if peak_pt is not None and pressure > 0.0:
                half_diag = 0.5 * np.hypot(self.base_fov_w * scale, self.base_fov_h * scale)
                r_center = -W["center"] * pressure * min(1.0, float(np.linalg.norm(peak_pt - pos)) / half_diag)

            # --- boundary: drone centre hugging a wall
            d_wall = min(pos[0], self.env_width - pos[0], pos[1], self.env_height - pos[1])
            r_boundary = 0.0
            if d_wall < self.boundary_margin:
                r_boundary = -W["boundary"] * ((self.boundary_margin - d_wall) / self.boundary_margin) ** 2

            # --- collision between drone centres (ramps up as they close in)
            r_collision = 0.0
            for other in self.agents:
                if other != agent:
                    dist = float(np.linalg.norm(pos - self.drone_positions[other]))
                    if dist < self.collision_threshold:
                        r_collision -= W["collision"] * (self.collision_threshold - dist) / self.collision_threshold

            # --- jitter / effort
            a_curr = np.asarray(actions[agent], dtype=np.float32)
            a_prev = self.prev_actions[agent]
            r_smooth = -(W["action_mag"] * np.sum(a_curr ** 2) + W["jitter"] * np.sum((a_curr - a_prev) ** 2))

            rewards[agent] = float(
                r_fleet + r_unique + r_redundant + r_overlap + r_waste
                + r_zoom + r_center + r_boundary + r_collision + r_smooth
            )

        self.prev_actions = {a: np.asarray(actions[a], dtype=np.float32).copy() for a in self.agents}

        self.metrics = {
            "ped_coverage": fleet_ped_frac,
            "area_coverage": union_area_frac,
            "overlap_of_union": overlap_of_union,
        }

        terminated = self.step_count >= self.max_steps
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {
            agent: {**self.metrics, "overlap": per_agent_overlap[agent]}
            for agent in self.agents
        }

        if terminated:
            self.agents = []

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminations, truncations, infos

    # ----------------------------------------------------------------- render
    def render(self):
        if self.render_mode is None:
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.canvas_size)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((50, 50, 50))

        arena_left = self.padding
        arena_top = self.padding
        arena_rect = pygame.Rect(arena_left, arena_top, self.render_area_size, self.render_area_size)
        pygame.draw.rect(self.screen, (240, 240, 240), arena_rect)

        t_idx = min(self.step_count, len(self.current_ep_data["density_maps"]) - 1)
        density_map = self.current_ep_data["density_maps"][t_idx]
        grid_w = self.render_area_size / density_map.shape[0]
        grid_h = self.render_area_size / density_map.shape[1]

        for x in range(density_map.shape[0]):
            for y in range(density_map.shape[1]):
                val = density_map[x, y]
                if val > 0:
                    intensity = min(255, int(val * 35))
                    surf = pygame.Surface((grid_w, grid_h), pygame.SRCALPHA)
                    surf.fill((255, 60, 60, intensity))
                    self.screen.blit(surf, (arena_left + (x * grid_w), arena_top + (y * grid_h)))

        for h in self.visual_positions:
            sx = int(arena_left + (h[0] * self.scale_x))
            sy = int(arena_top + (h[1] * self.scale_y))
            pygame.draw.circle(self.screen, (20, 40, 100), (sx, sy), 5)

        drone_colors = [(220, 50, 50), (50, 180, 50), (50, 50, 220)]
        for i, agent in enumerate(self.possible_agents):
            pos = self.drone_positions[agent]
            scale = self.drone_scales[agent]
            theta = self.drone_thetas[agent]
            color = drone_colors[i % len(drone_colors)]

            w = self.base_fov_w * scale
            h = self.base_fov_h * scale
            rect_w = int(w * self.scale_x)
            rect_h = int(h * self.scale_y)
            center_x = int(arena_left + (pos[0] * self.scale_x))
            center_y = int(arena_top + (pos[1] * self.scale_y))

            surf_size = max(rect_w, rect_h) * 2
            surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
            pygame.draw.rect(surf, (*color, 80), ((surf_size - rect_w)//2, (surf_size - rect_h)//2, rect_w, rect_h))
            pygame.draw.rect(surf, color, ((surf_size - rect_w)//2, (surf_size - rect_h)//2, rect_w, rect_h), width=2)

            rotated_surf = pygame.transform.rotate(surf, np.degrees(-theta))
            rot_rect = rotated_surf.get_rect(center=(center_x, center_y))
            self.screen.blit(rotated_surf, rot_rect.topleft)
            pygame.draw.circle(self.screen, color, (center_x, center_y), 6)

        pygame.draw.rect(self.screen, (80, 80, 80), arena_rect, 4)

        elapsed_seconds = self.step_count * self.dt
        m = self.metrics
        hud = f"Step: {self.step_count} | Time: {elapsed_seconds:.1f}s | Density-Driven MAPPO Active"
        if m:
            hud2 = (f"Ped cov: {m['ped_coverage']*100:.0f}% | Area cov: {m['area_coverage']*100:.0f}% "
                    f"| Overlap: {m['overlap_of_union']*100:.0f}%")
            self.screen.blit(self.font.render(hud2, True, (20, 20, 20)), (20, 45))
        self.screen.blit(self.font.render(hud, True, (20, 20, 20)), (20, 20))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
