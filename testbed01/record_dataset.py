import os
import glob
import numpy as np
import pygame


def compute_density_grid(human_positions, env_size=(25, 25), grid_resolution=1.0):
    width, height = env_size
    bins_x = int(width / grid_resolution)
    bins_y = int(height / grid_resolution)
    
    density_map, _, _ = np.histogram2d(
        human_positions[:, 0],
        human_positions[:, 1],
        bins=[bins_x, bins_y],
        range=[[0, width], [0, height]]
    )
    return density_map


class SmoothCrowdSimulator:
    def __init__(self, num_people=None, env_size=(25, 25), max_steps_real_sec=30):
        self.env_width, self.env_height = env_size
        self.num_people = num_people if num_people is not None else np.random.randint(120, 180)
        
        self.fps = 60  
        self.dt = 1.0 / self.fps
        self.max_steps = max_steps_real_sec * self.fps
        self.step_count = 0

        self.normal_speed = 1.0
        self.surge_speed = np.random.uniform(1.8, 2.5)

        # Initialize positions across the 25x25 arena
        self.positions = np.random.uniform(
            low=[3.0, 3.0],
            high=[self.env_width - 3.0, self.env_height - 3.0],
            size=(self.num_people, 2)
        )
        
        self.velocities = np.zeros((self.num_people, 2), dtype=np.float32)
        self.targets = self.positions + np.random.uniform(-6.0, 6.0, size=(self.num_people, 2))
        self.targets[:, 0] = np.clip(self.targets[:, 0], 1.5, self.env_width - 1.5)
        self.targets[:, 1] = np.clip(self.targets[:, 1], 1.5, self.env_height - 1.5)

        # Diverse potential hotspots / bottlenecks in the arena
        self.hotspots = [
            np.array([12.5, 23.0], dtype=np.float32),  # Bottom bottleneck
            np.array([12.5, 2.0], dtype=np.float32),   # Top bottleneck
            np.array([2.0, 12.5], dtype=np.float32),   # Left checkpoint
            np.array([23.0, 12.5], dtype=np.float32),  # Right checkpoint
            np.array([12.5, 12.5], dtype=np.float32)   # Center plaza
        ]

        # Scenario types:
        # 0: Wander -> Single Convergence -> Dissolve
        # 1: Multiple successive convergence waves
        # 2: Pure organic wandering (no convergence)
        # 3: Dual-cluster split convergence -> Dissolve
        self.scenario_type = np.random.choice([0, 1, 2, 3], p=[0.35, 0.30, 0.15, 0.20])
        
        # Agent states: 0 = wandering, >0 = heading to hotspot (state_idx - 1)
        self.agent_states = np.zeros(self.num_people, dtype=int)
        self.phase_triggered = False
        
        # Schedule timeline events
        self._setup_scenario_schedule()

    def _setup_scenario_schedule(self):
        self.events = []
        if self.scenario_type == 0:
            t_start = np.random.uniform(4.0, 9.0)
            t_dissolve = t_start + np.random.uniform(8.0, 14.0)
            self.events = [
                (t_start, "converge", {"fraction": np.random.uniform(0.4, 0.7)}),
                (t_dissolve, "dissolve", {})
            ]
        elif self.scenario_type == 1:
            t1 = np.random.uniform(3.0, 6.0)
            t1_end = t1 + np.random.uniform(5.0, 7.0)
            t2 = t1_end + np.random.uniform(2.0, 4.0)
            t2_end = t2 + np.random.uniform(5.0, 7.0)
            self.events = [
                (t1, "converge", {"fraction": np.random.uniform(0.3, 0.5)}),
                (t1_end, "dissolve", {}),
                (t2, "converge", {"fraction": np.random.uniform(0.4, 0.7)}),
                (t2_end, "dissolve", {})
            ]
        elif self.scenario_type == 2:
            self.events = []  # Pure organic wandering
        elif self.scenario_type == 3:
            t_start = np.random.uniform(4.0, 8.0)
            t_dissolve = t_start + np.random.uniform(9.0, 14.0)
            self.events = [
                (t_start, "split_converge", {}),
                (t_dissolve, "dissolve", {})
            ]

        self.events.sort(key=lambda x: x[0])
        self.current_event_idx = 0

    def step(self):
        self.step_count += 1
        elapsed_seconds = self.step_count / self.fps

        # Trigger scheduled events
        while (self.current_event_idx < len(self.events) and 
               elapsed_seconds >= self.events[self.current_event_idx][0]):
            t, action, kwargs = self.events[self.current_event_idx]
            self._execute_event(action, kwargs, elapsed_seconds)
            self.current_event_idx += 1

        self.phase_triggered = any(state > 0 for state in self.agent_states)

        for i in range(self.num_people):
            pos = self.positions[i]
            target = self.targets[i]
            state = self.agent_states[i]

            active_speed = self.surge_speed if state > 0 else self.normal_speed

            # Destination logic
            if state == 0 and np.linalg.norm(target - pos) < 2.0:
                new_target = pos + np.random.uniform(-6.0, 6.0, size=2)
                new_target[0] = np.clip(new_target[0], 1.5, self.env_width - 1.5)
                new_target[1] = np.clip(new_target[1], 1.5, self.env_height - 1.5)
                self.targets[i] = new_target
            elif state > 0 and np.linalg.norm(target - pos) < 1.5:
                hs_idx = state - 1
                if hs_idx < len(self.hotspots):
                    self.targets[i] = self.hotspots[hs_idx] + np.random.uniform(-1.5, 1.5, size=2)

            to_target = self.targets[i] - pos
            dist = np.linalg.norm(to_target)
            desired_vel = (to_target / dist) * active_speed if dist > 1e-3 else np.zeros(2)

            # Social repulsion (collision avoidance)
            repulsion = np.zeros(2, dtype=np.float32)
            diffs = pos - self.positions
            dists = np.linalg.norm(diffs, axis=1)
            close_neighbors = np.where((dists > 0) & (dists < 1.2))[0]
            for nb in close_neighbors:
                d = dists[nb]
                repulsion += (diffs[nb] / (d * d + 1e-5)) * 0.3

            steering = desired_vel + repulsion
            s_len = np.linalg.norm(steering)
            if s_len > 1e-3:
                steering = (steering / s_len) * active_speed

            momentum = 0.92
            self.velocities[i] = (momentum * self.velocities[i]) + ((1.0 - momentum) * steering)

            v_len = np.linalg.norm(self.velocities[i])
            if v_len > active_speed:
                self.velocities[i] = (self.velocities[i] / v_len) * active_speed

        self.positions += self.velocities * self.dt
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0.5, self.env_width - 0.5)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0.5, self.env_height - 0.5)

        return self.step_count >= self.max_steps

    def _execute_event(self, action, kwargs, elapsed_seconds):
        if action == "converge":
            fraction = kwargs.get("fraction", 0.5)
            num_surgers = int(self.num_people * fraction)
            surging_indices = np.random.choice(self.num_people, size=num_surgers, replace=False)
            hs_idx = np.random.randint(0, len(self.hotspots))
            print(f"[Event] T={elapsed_seconds:.1f}s: Converging {num_surgers} people toward hotspot {hs_idx}.")
            for i in surging_indices:
                self.agent_states[i] = hs_idx + 1
                self.targets[i] = self.hotspots[hs_idx] + np.random.uniform(-1.0, 1.0, size=2)

        elif action == "split_converge":
            hs1, hs2 = np.random.choice(len(self.hotspots), size=2, replace=False)
            indices = np.arange(self.num_people)
            np.random.shuffle(indices)
            mid = self.num_people // 2
            print(f"[Event] T={elapsed_seconds:.1f}s: Split convergence toward hotspots {hs1} & {hs2}.")
            for i in indices[:mid]:
                self.agent_states[i] = hs1 + 1
                self.targets[i] = self.hotspots[hs1] + np.random.uniform(-1.0, 1.0, size=2)
            for i in indices[mid:]:
                self.agent_states[i] = hs2 + 1
                self.targets[i] = self.hotspots[hs2] + np.random.uniform(-1.0, 1.0, size=2)

        elif action == "dissolve":
            print(f"[Event] T={elapsed_seconds:.1f}s: Crowd dissolving back to wander.")
            for i in range(self.num_people):
                self.agent_states[i] = 0
                new_target = self.positions[i] + np.random.uniform(-6.0, 6.0, size=2)
                new_target[0] = np.clip(new_target[0], 1.5, self.env_width - 1.5)
                new_target[1] = np.clip(new_target[1], 1.5, self.env_height - 1.5)
                self.targets[i] = new_target


def run_and_record_batch(num_episodes=20, render=False):
    os.makedirs("dataset", exist_ok=True)
    
    canvas_size = (1000, 1000)
    padding = 150
    render_area_size = canvas_size[0] - (2 * padding)
    scale_x = render_area_size / 25.0
    scale_y = render_area_size / 25.0

    if render:
        pygame.init()
        screen = pygame.display.set_mode(canvas_size)
        pygame.display.set_caption("Batch Crowd Recording")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 24)

    for ep_idx in range(num_episodes):
        sim = SmoothCrowdSimulator(max_steps_real_sec=30)
        print(f"\n--- Recording Episode {ep_idx+1}/{num_episodes} (Scenario: {sim.scenario_type}) ---")

        ep_data = {
            "timestamps": [],
            "human_positions": [],
            "human_velocities": [],
            "density_maps": [],
            "surge_active": []
        }

        running = True
        while running and sim.step_count < sim.max_steps:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            sim.step()
            elapsed_seconds = sim.step_count / sim.fps

            human_pos = sim.positions.copy()
            human_vel = sim.velocities.copy()
            density_map = compute_density_grid(human_pos, env_size=(25, 25), grid_resolution=1.0)
            surge_active = sim.phase_triggered

            ep_data["timestamps"].append(sim.step_count)
            ep_data["human_positions"].append(human_pos)
            ep_data["human_velocities"].append(human_vel)
            ep_data["density_maps"].append(density_map)
            ep_data["surge_active"].append(surge_active)

            if render:
                screen.fill((50, 50, 50))
                arena_rect = pygame.Rect(padding, padding, render_area_size, render_area_size)
                pygame.draw.rect(screen, (240, 240, 240), arena_rect)
                
                grid_w = render_area_size / density_map.shape[0]
                grid_h = render_area_size / density_map.shape[1]
                for x in range(density_map.shape[0]):
                    for y in range(density_map.shape[1]):
                        val = density_map[x, y]
                        if val > 0:
                            intensity = min(255, int(val * 35))
                            surf = pygame.Surface((grid_w, grid_h), pygame.SRCALPHA)
                            surf.fill((255, 60, 60, intensity))
                            screen.blit(surf, (padding + (x * grid_w), padding + (y * grid_h)))

                for h in human_pos:
                    sx = int(padding + (h[0] * scale_x))
                    sy = int(padding + (h[1] * scale_y))
                    pygame.draw.circle(screen, (20, 40, 100), (sx, sy), 5)

                pygame.draw.rect(screen, (80, 80, 80), arena_rect, 4)
                phase_str = "ACTIVE CONVERGENCE" if surge_active else "WANDERING / DISSOLVED"
                text_surf = font.render(f"Ep {ep_idx+1} | Time: {elapsed_seconds:.1f}s | {phase_str}", True, (20, 20, 20))
                screen.blit(text_surf, (20, 20))
                pygame.display.flip()
                clock.tick(sim.fps)

        file_path = f"dataset/episode_{ep_idx:03d}.npz"
        np.savez_compressed(
            file_path,
            timestamps=np.array(ep_data["timestamps"]),
            human_positions=np.array(ep_data["human_positions"]),
            human_velocities=np.array(ep_data["human_velocities"]),
            density_maps=np.array(ep_data["density_maps"]),
            surge_active=np.array(ep_data["surge_active"])
        )
        print(f"[Success] Saved {file_path}")

    if render:
        pygame.quit()
    print(f"\n[Batch Complete] All {num_episodes} generalized episodes saved successfully to 'dataset/'!")


if __name__ == "__main__":
    # Set render=True if you want to watch them generate live; set render=False to record all 20 instantly in the background.
    run_and_record_batch(num_episodes=20, render=False)
