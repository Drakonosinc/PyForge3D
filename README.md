# 🧊 PyForge3D — High-Performance 3D Software Renderer, Voxel Chunk Engine & 3D Physics for Pygame

[![Ask DeepWiki](https://img.shields.io/badge/Ask-DeepWiki-4B32C3?logo=read-the-docs&logoColor=white)](https://deepwiki.com/Drakonosinc/PyForge3D)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame--CE-3D_Software_Engine-00B4AB)
![Version](https://img.shields.io/badge/Version-1.6.0-22C55E)
![License](https://img.shields.io/badge/License-MIT-green)

**PyForge3D** (`pyforge3d`) is a 100% game-agnostic, zero-OpenGL **3D Software Renderer, Multi-Threaded C99 Textured Z-Buffer Engine, Infinite Chunked Voxel Engine with Native 2D Greedy Meshing, Procedural Perlin Noise Library, Atmospheric Sky Dome, 3D DDA Raycaster, and 3D AABB Physics engine** built on top of `pygame` / `pygame-ce` and `numpy`.

It features a **Dual-Backend Multi-Threaded Scanline Z-Buffer Pipeline** (a zero-dependency ISO C99 native engine loaded via `ctypes` with a persistent Win32/POSIX band thread pool, zero-copy chunk pointer dispatch, span-level early-Z rejection, and single-call C99 greedy chunk meshing — plus a 100% portable pure-NumPy fallback with pixel parity), delivering **145+ FPS** even at 12-chunk (`96m` radius, 480+ chunks) Native resolutions with perspective-correct tiled textures, translucent Water/Glass blending, and Ambient Occlusion.

---

## 🚀 Key Features

| Module | Classes & Functions | Description |
| :--- | :--- | :--- |
| **`pyforge3d.renderer`** | `Renderer3D`, `Scene3D`, `SoftwareTexturedRasterizer`, `HAS_NATIVE_RASTERIZER`, `build_native_rasterizer` | Dual-backend (C99 native + Pure-NumPy) per-pixel `1/Z` Z-Buffer rasterizer, perspective-correct tiled UV mapping, translucent alpha blending, `block_overlays` voxel decals, and Painter solid fallback. |
| **`pyforge3d.voxels`** | `ChunkedVoxelWorld`, `Chunk3D`, `VoxelWorld`, `VoxelMaterial`, `VoxelHit` | Infinite streamable 3D chunk worlds with **2D Greedy Meshing** (4×–6× fewer polygons), dense-halo neighbor face culling, per-face **Ambient Occlusion**, per-chunk 3D frustum culling, **3D DDA Raycasting**, and compact JSON override serialization (`export_overrides`/`import_overrides`). |
| **`pyforge3d.textures`** | `Texture2D`, `TextureAtlas` | RGB/RGBA 2D texture wrapper (from NumPy, PNG, or `pygame.Surface`) with cutout/alpha detection and contiguous atlas packing for fast indexed rendering. |
| **`pyforge3d.procedural`** | `PerlinNoise2D`, `FractalNoise2D`, `hash_2d` | Pure-NumPy vectorized 2D Perlin noise, multi-octave fBm fractal noise, and deterministic 2D coordinate hashing. |
| **`pyforge3d.lighting`** | `SkyDome3D`, `DirectionalLight`, `PointLight`, `Fog3D` | Continuous day/night sky dome (orbiting Sun/Moon, twinkling stars, infinite 3D clouds), directional/point lighting, and atmospheric distance fog. |
| **`pyforge3d.meshes`** | `Mesh3D`, `Primitives`, `OBJLoader` | Solid & textured 3D meshes (`Primitives.cube`, `Primitives.textured_cube`, `pyramid`, `sphere`, `cylinder`, `plane`) and Wavefront `.obj` parser. |
| **`pyforge3d.camera`** | `Camera3D`, `FirstPersonController`, `OrbitCamera` | Perspective 3D camera with Yaw/Pitch/Roll, configurable FOV/near/far planes, mouse-captured FPS controller, and orbital camera. |
| **`pyforge3d.physics`** | `AABB3D`, `VoxelPhysicsBody` | 3D Axis-Aligned Bounding Box collisions, gravity, jumping, and non-solid liquid traversal against `VoxelWorld` and `ChunkedVoxelWorld`. |
| **`pyforge3d.math`** | `Matrix4`, `Transform3D`, `clip_polygon_near_plane`, `compute_face_normals` | Vectorized 3D transforms, Sutherland-Hodgman 3D & UV near-plane clipping, and fast face normal computation. |

---

## 📦 Installation

```bash
pip install pyforge3d
```

Or install in editable mode from source:

```bash
git clone https://github.com/Drakonosinc/PyForge3D.git
cd PyForge3D
pip install -e .
```

*(If `gcc`/`clang` is available on PATH, the zero-dependency C99 native rasterizer builds automatically on first import for 10×+ higher FPS; otherwise the pure-NumPy backend runs seamlessly with zero setup.)*

---

## ⚡ Quickstart Example (Textured Infinite Chunk World + SkyDome3D)

```python
import numpy as np
import pygame
from pyforge3d import (
    Camera3D,
    ChunkedVoxelWorld,
    FirstPersonController,
    FractalNoise2D,
    Renderer3D,
    SkyDome3D,
    Texture2D,
    VoxelMaterial,
)

pygame.init()
screen = pygame.display.set_mode((960, 540))
clock = pygame.time.Clock()

# Create a simple procedural 16x16 texture (100% game-defined!)
rng = np.random.default_rng(7)
top_tex = Texture2D(rng.integers(60, 175, (16, 16, 3), dtype=np.uint8))
side_tex = Texture2D(rng.integers(85, 135, (16, 16, 3), dtype=np.uint8))
terrain_mat = VoxelMaterial(top_texture=top_tex, side_texture=side_tex, bottom_texture=side_tex)

noise = FractalNoise2D(seed=42, octaves=4, scale=24.0)

def generate_chunk(cx: int, cz: int, cs: int, w: ChunkedVoxelWorld) -> None:
    xs = np.arange(cx * cs, (cx + 1) * cs, dtype=np.float32)
    zs = np.arange(cz * cs, (cz + 1) * cs, dtype=np.float32)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    heights = np.floor(noise.sample(gx, gz, normalized=True) * 8.0).astype(int)
    for ix in range(cs):
        for iz in range(cs):
            for y in range(int(heights[ix, iz]) + 1):
                w.set_block_raw(int(xs[ix]), y, int(zs[iz]), terrain_mat)

world = ChunkedVoxelWorld(chunk_size=8, generator_fn=generate_chunk, greedy_meshing=True)
world.update_chunks_around(0, 0, load_radius=5, unload_radius=7, max_new_per_call=120)

camera = Camera3D(position=(4.0, 9.5, 4.0), fov=75.0, near=0.06, far=64.0)
controller = FirstPersonController(camera, move_speed=8.0)
sky = SkyDome3D(time_of_day=0.25)
renderer = Renderer3D(960, 540, textured=True, render_scale=0.72)
sky.sync_lighting(renderer.light, renderer.fog)

running = True
while running:
    dt = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        controller.handle_event(event)

    sky.update(dt)
    sky.sync_lighting(renderer.light, renderer.fog)
    world.update_chunks_around(camera.x, camera.z, load_radius=5, unload_radius=7, max_new_per_call=2)
    renderer.render_scene(screen, camera, voxel_world=world, sky_dome=sky)
    pygame.display.flip()

pygame.quit()
```

---

## 🧪 Running Tests

```bash
python -m unittest discover -s tests -v
```
