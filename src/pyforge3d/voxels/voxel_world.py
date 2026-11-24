from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class VoxelHit:
    """Result of a 3D DDA voxel raycast query."""

    block_pos: tuple[int, int, int]
    face_normal: tuple[int, int, int]
    adjacent_pos: tuple[int, int, int]
    distance: float
    color: tuple[int, int, int]


# Unit cube face definitions on [0, 1]^3 with outward normals
_VOXEL_FACES: list[tuple[tuple[int, int, int], np.ndarray]] = [
    (
        (0, 0, -1),  # Front (-Z)
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    ),
    (
        (0, 0, +1),  # Back (+Z)
        np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
    ),
    (
        (+1, 0, 0),  # Right (+X)
        np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float32),
    ),
    (
        (-1, 0, 0),  # Left (-X)
        np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
    ),
    (
        (0, +1, 0),  # Top (+Y)
        np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float32),
    ),
    (
        (0, -1, 0),  # Bottom (-Y)
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32),
    ),
]


class VoxelWorld:
    """
    Minecraft-style 3D Voxel block manager with automatic Neighbor Face Culling,
    vectorized quad buffer caching, procedural terrain generation, and 3D DDA Raycasting.
    Each block at integer coordinate (bx, by, bz) occupies [bx, bx+1] x [by, by+1] x [bz, bz+1].
    """

    def __init__(self) -> None:
        self.blocks: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        self._dirty: bool = True
        self._cached_quads: np.ndarray = np.empty((0, 4, 3), dtype=np.float32)
        self._cached_normals: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self._cached_colors: np.ndarray = np.empty((0, 3), dtype=np.float32)

    def set_block(self, x: int, y: int, z: int, color: tuple[int, int, int] = (56, 189, 248)) -> None:
        pos = (int(x), int(y), int(z))
        rgb = (int(color[0]), int(color[1]), int(color[2]))
        if self.blocks.get(pos) != rgb:
            self.blocks[pos] = rgb
            self._dirty = True

    def remove_block(self, x: int, y: int, z: int) -> bool:
        pos = (int(x), int(y), int(z))
        if pos in self.blocks:
            del self.blocks[pos]
            self._dirty = True
            return True
        return False

    def has_block(self, x: int, y: int, z: int) -> bool:
        return (int(x), int(y), int(z)) in self.blocks

    def get_block(self, x: int, y: int, z: int) -> tuple[int, int, int] | None:
        return self.blocks.get((int(x), int(y), int(z)))

    def clear(self) -> None:
        self.blocks.clear()
        self._dirty = True

    def generate_flat_terrain(
        self,
        width: int = 8,
        depth: int = 8,
        y_level: int = 0,
        ground_color: tuple[int, int, int] = (34, 197, 94),
        alt_color: tuple[int, int, int] | None = (22, 163, 74),
    ) -> None:
        """Populate a flat terrain grid of blocks on y = y_level."""
        for x in range(int(width)):
            for z in range(int(depth)):
                col = ground_color if (alt_color is None or (x + z) % 2 == 0) else alt_color
                self.set_block(x, y_level, z, col)

    def generate_hills_terrain(
        self,
        width: int = 10,
        depth: int = 10,
        max_height: int = 3,
        palette: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Populate a multi-layer stepped hills voxel terrain."""
        if not palette:
            palette = [
                (30, 64, 175),   # Deep layer
                (22, 163, 74),   # Grass layer
                (34, 197, 94),   # Upper grass
                (255, 199, 51),  # Peak gold
            ]
        for x in range(int(width)):
            for z in range(int(depth)):
                wave = 0.5 * (math.sin(x * 0.65) + math.cos(z * 0.65))
                h = max(0, min(int(max_height), int(round((wave + 1.0) * 0.5 * max_height))))
                for y in range(h + 1):
                    col = palette[min(y, len(palette) - 1)]
                    self.set_block(x, y, z, col)

    def get_culled_quads(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return cached `(quad_vertices (M, 4, 3), quad_normals (M, 3), quad_colors (M, 3))`
        containing ONLY exposed faces whose neighboring voxel cell is empty.
        """
        if not self._dirty:
            return self._cached_quads, self._cached_normals, self._cached_colors

        quads_list: list[np.ndarray] = []
        normals_list: list[tuple[int, int, int]] = []
        colors_list: list[tuple[int, int, int]] = []

        blocks_dict = self.blocks
        for (bx, by, bz), color in blocks_dict.items():
            offset = np.array([bx, by, bz], dtype=np.float32)
            for (nx, ny, nz), unit_quad in _VOXEL_FACES:
                if (bx + nx, by + ny, bz + nz) not in blocks_dict:
                    quads_list.append(unit_quad + offset)
                    normals_list.append((nx, ny, nz))
                    colors_list.append(color)

        if quads_list:
            self._cached_quads = np.asarray(quads_list, dtype=np.float32)
            self._cached_normals = np.asarray(normals_list, dtype=np.float32)
            self._cached_colors = np.asarray(colors_list, dtype=np.float32)
        else:
            self._cached_quads = np.empty((0, 4, 3), dtype=np.float32)
            self._cached_normals = np.empty((0, 3), dtype=np.float32)
            self._cached_colors = np.empty((0, 3), dtype=np.float32)

        self._dirty = False
        return self._cached_quads, self._cached_normals, self._cached_colors

    def raycast(
        self,
        origin: tuple[float, float, float] | np.ndarray,
        direction: tuple[float, float, float] | np.ndarray,
        max_distance: float = 8.0,
    ) -> VoxelHit | None:
        """
        Fast 3D DDA Voxel Raycast (Amanatides & Woo algorithm).
        Finds the first solid voxel block intersected by the ray `origin + t * direction` within `max_distance`.
        """
        if not self.blocks:
            return None

        ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
        dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-8:
            return None
        dx, dy, dz = dx / length, dy / length, dz / length

        vx, vy, vz = math.floor(ox), math.floor(oy), math.floor(oz)

        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
        step_z = 1 if dz > 0 else (-1 if dz < 0 else 0)

        t_delta_x = abs(1.0 / dx) if abs(dx) > 1e-8 else float("inf")
        t_delta_y = abs(1.0 / dy) if abs(dy) > 1e-8 else float("inf")
        t_delta_z = abs(1.0 / dz) if abs(dz) > 1e-8 else float("inf")

        t_max_x = ((vx + 1.0 - ox) if step_x > 0 else (ox - vx)) * t_delta_x if step_x != 0 else float("inf")
        t_max_y = ((vy + 1.0 - oy) if step_y > 0 else (oy - vy)) * t_delta_y if step_y != 0 else float("inf")
        t_max_z = ((vz + 1.0 - oz) if step_z > 0 else (oz - vz)) * t_delta_z if step_z != 0 else float("inf")

        dist = 0.0
        last_normal = (0, 1, 0)

        while dist <= max_distance:
            pos = (vx, vy, vz)
            if pos in self.blocks:
                adj = (vx + last_normal[0], vy + last_normal[1], vz + last_normal[2])
                return VoxelHit(
                    block_pos=pos,
                    face_normal=last_normal,
                    adjacent_pos=adj,
                    distance=dist,
                    color=self.blocks[pos],
                )

            if t_max_x < t_max_y:
                if t_max_x < t_max_z:
                    vx += step_x
                    dist = t_max_x
                    t_max_x += t_delta_x
                    last_normal = (-step_x, 0, 0)
                else:
                    vz += step_z
                    dist = t_max_z
                    t_max_z += t_delta_z
                    last_normal = (0, 0, -step_z)
            else:
                if t_max_y < t_max_z:
                    vy += step_y
                    dist = t_max_y
                    t_max_y += t_delta_y
                    last_normal = (0, -step_y, 0)
                else:
                    vz += step_z
                    dist = t_max_z
                    t_max_z += t_delta_z
                    last_normal = (0, 0, -step_z)

        return None
