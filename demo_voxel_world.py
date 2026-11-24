"""Interactive 3D Voxel & Polygonal Showcase built purely with PyForge3D and Pygame."""

import pygame
from pyforge3d import (
    Camera3D,
    DirectionalLight,
    FirstPersonController,
    Fog3D,
    Primitives,
    Renderer3D,
    Scene3D,
    VoxelWorld,
)


def main() -> None:
    pygame.init()
    width, height = 960, 540
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("PyForge3D — Interactive 3D Voxel & Mesh Engine Showcase")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    camera = Camera3D(position=(5.0, 6.5, -5.5), yaw=0.0, pitch=-0.42, fov=72.0)
    controller = FirstPersonController(camera, move_speed=7.0)
    controller.set_mouse_capture(True)

    renderer = Renderer3D(
        width=width,
        height=height,
        light=DirectionalLight(direction=(-0.45, -1.0, -0.35), ambient=0.35),
        fog=Fog3D(color=(12, 18, 34), near=14.0, far=40.0),
    )

    world = VoxelWorld()
    world.generate_hills_terrain(width=10, depth=10, max_height=3)

    scene = Scene3D()
    crystal = Primitives.pyramid(base_size=1.1, height=1.6, color=(56, 189, 248), position=(5.0, 5.5, 5.0))
    orb = Primitives.cube(size=0.9, color=(168, 85, 247), position=(2.5, 5.0, 5.0), outline_color=(255, 199, 51))
    scene.add_mesh(crystal, orb)

    palette = [
        (56, 189, 248),   # Cyan
        (34, 197, 94),    # Emerald
        (255, 199, 51),   # Gold
        (168, 85, 247),   # Purple
        (239, 68, 68),    # Crimson
    ]
    selected_color_idx = 0

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        hit = world.raycast(camera.position, camera.forward, max_distance=10.0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif controller.handle_event(event):
                continue
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if controller.mouse_captured:
                        controller.set_mouse_capture(False)
                    else:
                        running = False
                elif event.key == pygame.K_TAB:
                    selected_color_idx = (selected_color_idx + 1) % len(palette)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not controller.mouse_captured:
                    controller.set_mouse_capture(True)
                elif hit is not None:
                    if event.button == 1:  # Left click: Break block
                        world.remove_block(*hit.block_pos)
                    elif event.button == 3:  # Right click: Place block
                        world.set_block(*hit.adjacent_pos, color=palette[selected_color_idx])

        controller.update(dt, fly_mode=True)
        crystal.transform.rotation_y += 1.4 * dt
        orb.transform.rotation_x += 1.1 * dt
        orb.transform.rotation_y += 0.9 * dt

        screen.fill((12, 18, 34))
        faces = renderer.render_scene(screen, camera, scene=scene, voxel_world=world, highlight_hit=hit)

        # Crosshair
        cx, cy = width // 2, height // 2
        pygame.draw.line(screen, (255, 199, 51), (cx - 8, cy), (cx + 8, cy), 2)
        pygame.draw.line(screen, (255, 199, 51), (cx, cy - 8), (cx, cy + 8), 2)

        hud = font.render(
            f"FPS: {clock.get_fps():.0f} | Faces: {faces} | Blocks: {len(world.blocks)} | WASD+Space/Shift: Move | Arrows: Look | L/R Click: Break/Place",
            True,
            (235, 248, 255),
        )
        screen.blit(hud, (12, 12))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
