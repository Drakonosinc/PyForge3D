# 🧊 PyForge3D — Vectorized 3D Software Renderer, Voxel Engine & 3D Physics for Pygame

[![Ask DeepWiki](https://img.shields.io/badge/Ask-DeepWiki-4B32C3?logo=read-the-docs&logoColor=white)](https://deepwiki.com/Drakonosinc/PyForge3D)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame--CE-3D_Software_Engine-00B4AB)
![License](https://img.shields.io/badge/License-MIT-green)

**PyForge3D** (`pyforge3d`) is a high-performance, zero-OpenGL **3D Software Renderer, Voxel World Engine (Minecraft-style), Procedural Mesh Library, 3D DDA Raycaster, and 3D AABB Physics engine** built purely on top of `pygame` / `pygame-ce` and `numpy`.

Because it renders directly onto standard `pygame.Surface` canvases using vectorized NumPy matrix transformations, back-face culling, neighbor voxel face culling, and near-plane clipping, it integrates seamlessly with **`PyForgeUI`**, **`PyForgeFX`**, and **`PyForgeAI`**.

---

## 🚀 Key Features

| Module | Classes | Description |
| :--- | :--- | :--- |
| **`pyforge3d.math`** | `Matrix4`, `Transform3D`, `clip_polygon_near_plane` | Vectorized 4x4 perspective, view, translation, Euler rotation, and near-plane polygon clipping in NumPy. |
| **`pyforge3d.camera`** | `Camera3D`, `FirstPersonController`, `OrbitCamera` | Perspective 3D camera with Yaw/Pitch/Roll, Frustum FOV, plug-and-play FPS/Minecraft controller, and orbital camera. |
| **`pyforge3d.meshes`** | `Mesh3D`, `Primitives`, `OBJLoader` | Procedural 3D primitives (`Cube`, `Pyramid`, `Sphere`, `Cylinder`, `Plane3D`) and Wavefront `.obj` 3D mesh loader. |
| **`pyforge3d.voxels`** | `VoxelWorld`, `VoxelHit` | Minecraft-style block world with automatic **Neighbor Face Culling** (~85% polygon reduction) and **3D DDA Voxel Raycasting** for breaking/placing blocks. |
| **`pyforge3d.physics`** | `AABB3D`, `VoxelPhysicsBody` | 3D Axis-Aligned Bounding Box collision detection, gravity, jumping, and sliding against `VoxelWorld` blocks. |
| **`pyforge3d.lighting`** | `DirectionalLight`, `PointLight`, `Fog3D` | Per-face flat directional/point shading, ambient illumination, and distance-based atmospheric fog interpolation. |
| **`pyforge3d.renderer`** | `Renderer3D`, `Scene3D` | Vectorized back-face culling, Painter's Algorithm depth sorting, solid/wireframe/outlined rendering, and 3D voxel cursor highlighting. |

---

## 📦 Installation

```bash
pip install pyforge3d
```

---

## ⚡ Quickstart Example (Minecraft-style 3D Voxel World in Pygame)

```python
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

pygame.init()
screen = pygame.display.set_mode((960, 540))
clock = pygame.time.Clock()

camera = Camera3D(position=(4.0, 5.0, -6.0), yaw=0.0, pitch=-0.35, fov=70.0)
controller = FirstPersonController(camera, move_speed=6.0)
renderer = Renderer3D(
    width=960,
    height=540,
    light=DirectionalLight(direction=(-0.5, -1.0, -0.4), ambient=0.35),
    fog=Fog3D(color=(14, 20, 36), near=12.0, far=36.0),
)

world = VoxelWorld()
world.generate_flat_terrain(width=10, depth=10, ground_color=(56, 189, 248))
world.set_block(4, 1, 4, color=(255, 199, 51))

scene = Scene3D()
floating_cube = Primitives.cube(size=1.0, color=(168, 85, 247), position=(4.5, 3.0, 4.5))
scene.add_mesh(floating_cube)

running = True
while running:
    dt = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    floating_cube.transform.rotation_y += 1.2 * dt
    screen.fill((14, 20, 36))
    renderer.render_scene(screen, camera, scene=scene, voxel_world=world)
    pygame.display.flip()

pygame.quit()
```

---

## 🧪 Running Tests

```bash
python -m unittest discover -s tests -v
```
