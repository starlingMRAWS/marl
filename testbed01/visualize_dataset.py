import os
import numpy as np
import pygame


def visualize_episode(episode_path="dataset/episode_000.npz"):
    if not os.path.exists(episode_path):
        print(f"Error: {episode_path} not found. Please record a dataset first!")
        return

    # Load recorded generalized episode data
    data = np.load(episode_path)
    timestamps = data["timestamps"]
    human_positions = data["human_positions"]
    density_maps = data["density_maps"]
    surge_active = data["surge_active"]

    env_width, env_height = 25.0, 25.0  # Corrected to 25x25 arena
    screen_size = (800, 800)
    scale_x = screen_size[0] / env_width
    scale_y = screen_size[1] / env_height

    pygame.init()
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption(f"Generalized Crowd Visualizer - {episode_path}")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 24)
    num_frames = len(timestamps)
    frame_idx = 0
    paused = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    frame_idx = min(num_frames - 1, frame_idx + 1)
                elif event.key == pygame.K_LEFT:
                    frame_idx = max(0, frame_idx - 1)
                elif event.key == pygame.K_UP:
                    frame_idx = min(num_frames - 1, frame_idx + 30)  # Skip forward ~1 sec
                elif event.key == pygame.K_DOWN:
                    frame_idx = max(0, frame_idx - 30)             # Skip backward ~1 sec

        screen.fill((240, 240, 240))

        # Render density map grid overlay
        current_density = density_maps[frame_idx]
        grid_w = screen_size[0] / current_density.shape[0]
        grid_h = screen_size[1] / current_density.shape[1]
        
        for x in range(current_density.shape[0]):
            for y in range(current_density.shape[1]):
                val = current_density[x, y]
                if val > 0:
                    intensity = min(255, int(val * 35))
                    surf = pygame.Surface((grid_w, grid_h), pygame.SRCALPHA)
                    surf.fill((255, 50, 50, intensity))
                    screen.blit(surf, (x * grid_w, y * grid_h))

        # Render humans
        h_pos = human_positions[frame_idx]
        for h in h_pos:
            sx = int(h[0] * scale_x)
            sy = int(h[1] * scale_y)
            pygame.draw.circle(screen, (20, 40, 100), (sx, sy), 5)

        # Render HUD Overlay Text
        is_surging = surge_active[frame_idx]
        phase_str = "ACTIVE CONVERGENCE (Surge Wave)" if is_surging else "WANDERING / DISSOLVED"
        phase_color = (200, 30, 30) if is_surging else (30, 150, 30)

        elapsed_sec = timestamps[frame_idx] / 60.0
        text_surf = font.render(f"Frame: {timestamps[frame_idx]} ({elapsed_sec:.1f}s) | {phase_str}", True, phase_color)
        screen.blit(text_surf, (20, 20))
        
        instructions = font.render("Space: Pause | Left/Right: Step Frame | Up/Down: Skip ±1s", True, (100, 100, 100))
        screen.blit(instructions, (20, 50))

        pygame.display.flip()

        if not paused:
            frame_idx = (frame_idx + 1) % num_frames
            
        clock.tick(30)  # 30 FPS playback

    pygame.quit()


if __name__ == "__main__":
    visualize_episode("dataset/episode_000.npz")
