import os
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
    def __init__(self, num_people=150, env_size=(25, 25), max_steps_real_sec=30):
        self.num_people = num_people
        self.env_width, self.env_height = env_size
        
        self.fps = 60  
        self.dt = 1.0 / self.fps
        self.max_steps = max_steps_real_sec * self.fps
        self.step_count = 0

        self.normal_speed = 1.0
        self.surge_speed = 2.0

        # Initialize positions across the 25x25 arena
        self.positions = np.random.uniform(
            low=[3.0, 3.0],
            high=[self.env_width - 3.0, self.env_height - 3.0],
            size=(self.num_people, 2)
        )
        
        self.velocities = np.zeros((self.num_people, 2), dtype=np.float32)
        self.is_surging_agent = np.zeros(self.num_people, dtype=bool)

        self.targets = self.positions + np.random.uniform(-6.0, 6.0, size=(self.num_people, 2))
        self.targets[:, 0] = np.clip(self.targets[:, 0], 1.5, self.env_width - 1.5)
        self.targets[:, 1] = np.clip(self.targets[:, 1], 1.5, self.env_height - 1.5)

        # Bottom-center bottleneck target in the 25x25 arena
        self.bottleneck_target = np.array([12.5, 23.0], dtype=np.float32)
        self.phase_triggered = False

    def step(self):
        self.step_count += 1
        elapsed_seconds = self.step_count / self.fps

        # Phase Transition at T = 10 seconds: 75 people converge to bottleneck
        if elapsed_seconds >= 10.0 and not self.phase_triggered:
            print("[Simulation Event] T = 10s: Bottleneck active. 75 people converging.")
            self.phase_triggered = True
            
            surging_indices = np.random.choice(self.num_people, size=75, replace=False)
            self.is_surging_agent[surging_indices] = True
            
            for i in surging_indices:
                self.targets[i] = self.bottleneck_target + np.random.uniform(-1.0, 1.0, size=2)

        for i in range(self.num_people):
            pos = self.positions[i]
            target = self.targets[i]
            is_surger = self.is_surging_agent[i]

            active_speed = self.surge_speed if (is_surger and elapsed_seconds >= 10.0) else self.normal_speed

            if not is_surger and np.linalg.norm(target - pos) < 2.0:
                new_target = pos + np.random.uniform(-6.0, 6.0, size=2)
                new_target[0] = np.clip(new_target[0], 1.5, self.env_width - 1.5)
                new_target[1] = np.clip(new_target[1], 1.5, self.env_height - 1.5)
                self.targets[i] = new_target
                target = new_target
            elif is_surger and elapsed_seconds >= 10.0 and np.linalg.norm(target - pos) < 1.0:
                self.targets[i] = self.bottleneck_target + np.random.uniform(-1.0, 1.0, size=2)
                target = self.targets[i]

            to_target = target - pos
            dist = np.linalg.norm(to_target)
            desired_vel = (to_target / dist) * active_speed if dist > 1e-3 else np.zeros(2)

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


def run_and_record():
    os.makedirs("dataset", exist_ok=True)
    sim = SmoothCrowdSimulator(num_people=150, max_steps_real_sec=30)
    
    canvas_size = (1000, 1000)
    padding = 150
    render_area_size = canvas_size[0] - (2 * padding)
    scale_x = render_area_size / 25.0
    scale_y = render_area_size / 25.0

    pygame.init()
    screen = pygame.display.set_mode(canvas_size)
    pygame.display.set_caption("Recording 25x25 Crowd Episode")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    ep_data = {
        "timestamps": [],
        "human_positions": [],
        "human_velocities": [],
        "density_maps": [],
        "surge_active": []
    }

    running = True
    while running and sim.step_count < sim.max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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

        phase_str = "SURGE: Bottleneck Convergence" if surge_active else "MACRO: Wandering"
        text_surf = font.render(f"Time: {elapsed_seconds:.1f}s | {phase_str}", True, (20, 20, 20))
        screen.blit(text_surf, (20, 20))

        pygame.display.flip()
        clock.tick(sim.fps)

    file_path = "dataset/episode_000.npz"
    np.savez_compressed(
        file_path,
        timestamps=np.array(ep_data["timestamps"]),
        human_positions=np.array(ep_data["human_positions"]),
        human_velocities=np.array(ep_data["human_velocities"]),
        density_maps=np.array(ep_data["density_maps"]),
        surge_active=np.array(ep_data["surge_active"])
    )
    print(f"\n[Success] 25x25 episode recorded and saved to {file_path}!")

    pygame.quit()

if __name__ == "__main__":
    run_and_record()