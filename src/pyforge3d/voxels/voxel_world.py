from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any
import numpy as np
from ..textures.texture2d import Texture2D


@dataclass(frozen=True)
class VoxelMaterial:
    """
    Multi-face color & optional 2D texture definition for voxel blocks
    (distinct top, side, and bottom colors/textures, optional alpha translucency, solidity, and emissive boost).
    """

    top: tuple[int, int, int] | None = None
    side: tuple[int, int, int] | None = None
    bottom: tuple[int, int, int] | None = None
    top_texture: Texture2D | None = None
    side_texture: Texture2D | None = None
    bottom_texture: Texture2D | None = None
    alpha: int = 255
    solid: bool = True
    emissive: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if self.top is None:
            col = self.top_texture.avg_color if self.top_texture is not None else (
                self.side_texture.avg_color if self.side_texture is not None else (180, 180, 180)
            )
            object.__setattr__(self, "top", col)
        if self.side is None:
            col = self.side_texture.avg_color if self.side_texture is not None else self.top
            object.__setattr__(self, "side", col)
        if self.bottom is None and self.bottom_texture is not None:
            object.__setattr__(self, "bottom", self.bottom_texture.avg_color)

    @property
    def is_transparent(self) -> bool:
        if self.alpha < 255:
            return True
        for tex in (self.top_texture, self.side_texture, self.bottom_texture):
            if tex is not None and tex.has_cutout:
                return True
        return False

    def color_for_normal(self, normal: tuple[int, int, int]) -> tuple[int, int, int]:
        ny = normal[1]
        if ny > 0:
            return self.top
        if ny < 0:
            return self.bottom if self.bottom is not None else self.side
        return self.side

    def texture_for_normal(self, normal: tuple[int, int, int]) -> Texture2D | None:
        ny = normal[1]
        if ny > 0:
            return self.top_texture
        if ny < 0:
            return self.bottom_texture if self.bottom_texture is not None else self.side_texture
        return self.side_texture


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
        self.blocks: dict[tuple[int, int, int], tuple[int, int, int] | VoxelMaterial] = {}
        self._dirty: bool = True
        self._cached_quads: np.ndarray = np.empty((0, 4, 3), dtype=np.float32)
        self._cached_normals: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self._cached_colors: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self._cached_textures: list[Texture2D | None] = []
        self._cached_alphas: np.ndarray = np.empty((0,), dtype=np.uint8)
        self._cached_emissive: np.ndarray = np.empty((0,), dtype=np.float32)
        self._cached_ao: np.ndarray = np.empty((0,), dtype=np.float32)

    @staticmethod
    def _coerce_material(color: tuple[int, int, int] | VoxelMaterial) -> tuple[int, int, int] | VoxelMaterial:
        if isinstance(color, VoxelMaterial):
            return color
        return (int(color[0]), int(color[1]), int(color[2]))

    def set_block(self, x: int, y: int, z: int, color: tuple[int, int, int] | VoxelMaterial = (56, 189, 248)) -> None:
        pos = (int(x), int(y), int(z))
        mat = self._coerce_material(color)
        if self.blocks.get(pos) != mat:
            self.blocks[pos] = mat
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

    def is_solid(self, x: int, y: int, z: int) -> bool:
        mat = self.blocks.get((int(x), int(y), int(z)))
        if mat is None:
            return False
        return bool(mat.solid) if isinstance(mat, VoxelMaterial) else True

    def get_block(self, x: int, y: int, z: int) -> tuple[int, int, int] | VoxelMaterial | None:
        return self.blocks.get((int(x), int(y), int(z)))

    def get_highest_solid_y(self, x: int, z: int) -> int | None:
        """Return the highest solid block Y index at column (x, z), or None if empty."""
        ix, iz = int(x), int(z)
        ys = [by for (bx, by, bz) in self.blocks.keys() if bx == ix and bz == iz]
        return max(ys) if ys else None

    def place_tree(
        self,
        x: int,
        base_y: int,
        z: int,
        trunk_color: tuple[int, int, int] | VoxelMaterial = VoxelMaterial((175, 138, 92), (112, 74, 42), (175, 138, 92)),
        leaf_color: tuple[int, int, int] | VoxelMaterial = (34, 175, 85),
        height: int = 4,
    ) -> None:
        """Place a Minecraft-style voxel tree with trunk and layered leaf canopy."""
        tx, ty, tz = int(x), int(base_y), int(z)
        h = max(2, int(height))
        for dy in range(h):
            self.set_block(tx, ty + dy, tz, trunk_color)
        crown_base = ty + h - 1
        for dy in (0, 1):
            radius = 2 if dy == 0 else 1
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if dy == 0 and abs(dx) == 2 and abs(dz) == 2:
                        continue
                    if dx == 0 and dz == 0 and dy == 0:
                        continue
                    self.set_block(tx + dx, crown_base + dy, tz + dz, leaf_color)
        self.set_block(tx, crown_base + 2, tz, leaf_color)

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

    def _rebuild_cache(self) -> None:
        quads_list: list[np.ndarray] = []
        normals_list: list[tuple[int, int, int]] = []
        colors_list: list[tuple[int, int, int]] = []
        textures_list: list[Texture2D | None] = []
        alphas_list: list[int] = []
        emissive_list: list[float] = []
        ao_list: list[float] = []

        blocks_dict = self.blocks
        for (bx, by, bz), mat in blocks_dict.items():
            offset = np.array([bx, by, bz], dtype=np.float32)
            is_mat = isinstance(mat, VoxelMaterial)
            mat_trans = mat.is_transparent if is_mat else False
            mat_alpha = mat.alpha if is_mat else 255
            mat_emissive = mat.emissive if is_mat else 0.0

            for (nx, ny, nz), unit_quad in _VOXEL_FACES:
                neighbor = blocks_dict.get((bx + nx, by + ny, bz + nz))
                if neighbor is not None:
                    n_trans = neighbor.is_transparent if isinstance(neighbor, VoxelMaterial) else False
                    if not n_trans:
                        continue
                    if mat_trans and n_trans:
                        continue

                quads_list.append(unit_quad + offset)
                normals_list.append((nx, ny, nz))
                face_rgb = mat.color_for_normal((nx, ny, nz)) if is_mat else mat
                colors_list.append(face_rgb)
                textures_list.append(mat.texture_for_normal((nx, ny, nz)) if is_mat else None)
                alphas_list.append(mat_alpha)
                emissive_list.append(mat_emissive)

                # Lightweight face Ambient Occlusion from upper-adjacent occluders
                if ny == 1:
                    occ = 0
                    for ox, oz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        if (bx + ox, by + 1, bz + oz) in blocks_dict:
                            nb = blocks_dict[(bx + ox, by + 1, bz + oz)]
                            if not (isinstance(nb, VoxelMaterial) and nb.is_transparent):
                                occ += 1
                    ao_list.append(max(0.72, 1.0 - occ * 0.065))
                elif ny == 0:
                    has_overhang = (bx + nx, by + 1, bz + nz) in blocks_dict
                    has_floor_edge = (bx + nx, by - 1, bz + nz) in blocks_dict
                    ao_val = 1.0 - (0.12 if has_overhang else 0.0) - (0.06 if has_floor_edge else 0.0)
                    ao_list.append(ao_val)
                else:
                    ao_list.append(0.88)

        if quads_list:
            self._cached_quads = np.asarray(quads_list, dtype=np.float32)
            self._cached_normals = np.asarray(normals_list, dtype=np.float32)
            self._cached_colors = np.asarray(colors_list, dtype=np.float32)
            self._cached_textures = textures_list
            self._cached_alphas = np.asarray(alphas_list, dtype=np.uint8)
            self._cached_emissive = np.asarray(emissive_list, dtype=np.float32)
            self._cached_ao = np.asarray(ao_list, dtype=np.float32)
        else:
            self._cached_quads = np.empty((0, 4, 3), dtype=np.float32)
            self._cached_normals = np.empty((0, 3), dtype=np.float32)
            self._cached_colors = np.empty((0, 3), dtype=np.float32)
            self._cached_textures = []
            self._cached_alphas = np.empty((0,), dtype=np.uint8)
            self._cached_emissive = np.empty((0,), dtype=np.float32)
            self._cached_ao = np.empty((0,), dtype=np.float32)

        self._dirty = False

    def get_culled_quads(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return cached `(quad_vertices (M, 4, 3), quad_normals (M, 3), quad_colors (M, 3))`
        containing ONLY exposed faces whose neighboring voxel cell is empty or transparent.
        """
        if self._dirty:
            self._rebuild_cache()
        return self._cached_quads, self._cached_normals, self._cached_colors

    def get_culled_render_data(self) -> dict[str, Any]:
        """Return full culled render batch including textures, alphas, emissive boost, and Ambient Occlusion."""
        if self._dirty:
            self._rebuild_cache()
        return {
            "quads": self._cached_quads,
            "normals": self._cached_normals,
            "colors": self._cached_colors,
            "textures": self._cached_textures,
            "alphas": self._cached_alphas,
            "emissive": self._cached_emissive,
            "ao": self._cached_ao,
        }

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
                raw_mat = self.blocks[pos]
                hit_rgb = raw_mat.color_for_normal(last_normal) if isinstance(raw_mat, VoxelMaterial) else raw_mat
                return VoxelHit(
                    block_pos=pos,
                    face_normal=last_normal,
                    adjacent_pos=adj,
                    distance=dist,
                    color=hit_rgb,
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
