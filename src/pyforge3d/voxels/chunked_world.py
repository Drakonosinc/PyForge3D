from __future__ import annotations

import math
from typing import Any, Callable
import numpy as np
from ..textures.texture2d import Texture2D
from .voxel_world import VoxelHit, VoxelMaterial, _VOXEL_FACES


class Chunk3D:
    """Single horizontal column chunk (chunk_size x Y x chunk_size) with its own cached culled face buffers."""

    def __init__(self, cx: int, cz: int, chunk_size: int = 8) -> None:
        self.cx: int = int(cx)
        self.cz: int = int(cz)
        self.chunk_size: int = int(chunk_size)
        self.blocks: dict[tuple[int, int, int], tuple[int, int, int] | VoxelMaterial] = {}
        self.dirty: bool = True
        self.generated: bool = False
        self.modified_by_user: bool = False

        self.quads: np.ndarray = np.empty((0, 4, 3), dtype=np.float32)
        self.normals: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.colors: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.textures: list[Texture2D | None] = []
        self.alphas: np.ndarray = np.empty((0,), dtype=np.uint8)
        self.emissive: np.ndarray = np.empty((0,), dtype=np.float32)
        self.ao: np.ndarray = np.empty((0,), dtype=np.float32)

    @property
    def center_xz(self) -> tuple[float, float]:
        s = float(self.chunk_size)
        return ((self.cx + 0.5) * s, (self.cz + 0.5) * s)


class ChunkedVoxelWorld:
    """
    Infinite, game-agnostic procedural Voxel World organized into streamable 3D Chunks.
    - Supports cross-chunk Neighbor Face Culling and per-face Ambient Occlusion (AO).
    - Streams chunks incrementally around any (x, z) focus point (`update_chunks_around`).
    - Preserves user edits in a persistent override dictionary even when distant chunks unload.
    - Drop-in compatible with `Renderer3D`, `VoxelPhysicsBody`, and 3D DDA `raycast`.
    """

    def __init__(
        self,
        chunk_size: int = 8,
        generator_fn: Callable[[int, int, int, ChunkedVoxelWorld], None] | None = None,
    ) -> None:
        self.chunk_size: int = max(4, int(chunk_size))
        self.generator_fn = generator_fn
        self.chunks: dict[tuple[int, int], Chunk3D] = {}
        # Persistent user edits: (x, y, z) -> material or None (deleted)
        self._overrides: dict[tuple[int, int, int], tuple[int, int, int] | VoxelMaterial | None] = {}
        self._aggregate_dirty: bool = True
        self._cached_render_data: dict[str, Any] = {}

    @property
    def blocks(self) -> dict[tuple[int, int, int], tuple[int, int, int] | VoxelMaterial]:
        """Flat view of loaded blocks (for inspection/stats)."""
        out: dict[tuple[int, int, int], tuple[int, int, int] | VoxelMaterial] = {}
        for ch in self.chunks.values():
            out.update(ch.blocks)
        return out

    @property
    def total_blocks(self) -> int:
        return sum(len(ch.blocks) for ch in self.chunks.values())

    def _chunk_coords(self, x: int, z: int) -> tuple[int, int]:
        return (math.floor(int(x) / self.chunk_size), math.floor(int(z) / self.chunk_size))

    def _mark_dirty_with_neighbors(self, x: int, z: int) -> None:
        cx, cz = self._chunk_coords(x, z)
        ch = self.chunks.get((cx, cz))
        if ch is not None:
            ch.dirty = True
        lx = int(x) - cx * self.chunk_size
        lz = int(z) - cz * self.chunk_size
        if lx == 0 and (cx - 1, cz) in self.chunks:
            self.chunks[(cx - 1, cz)].dirty = True
        elif lx == self.chunk_size - 1 and (cx + 1, cz) in self.chunks:
            self.chunks[(cx + 1, cz)].dirty = True
        if lz == 0 and (cx, cz - 1) in self.chunks:
            self.chunks[(cx, cz - 1)].dirty = True
        elif lz == self.chunk_size - 1 and (cx, cz + 1) in self.chunks:
            self.chunks[(cx, cz + 1)].dirty = True
        self._aggregate_dirty = True

    def _ensure_chunk(self, cx: int, cz: int) -> Chunk3D:
        key = (int(cx), int(cz))
        ch = self.chunks.get(key)
        if ch is None:
            ch = Chunk3D(cx, cz, self.chunk_size)
            self.chunks[key] = ch
        return ch

    def set_block_raw(self, x: int, y: int, z: int, material: tuple[int, int, int] | VoxelMaterial) -> None:
        """Internal/generator setter that does not record a persistent player override."""
        ix, iy, iz = int(x), int(y), int(z)
        cx, cz = self._chunk_coords(ix, iz)
        ch = self._ensure_chunk(cx, cz)
        mat = material if isinstance(material, VoxelMaterial) else (int(material[0]), int(material[1]), int(material[2]))
        ch.blocks[(ix, iy, iz)] = mat
        ch.dirty = True
        self._aggregate_dirty = True

    def set_block(self, x: int, y: int, z: int, color: tuple[int, int, int] | VoxelMaterial = (56, 189, 248)) -> None:
        ix, iy, iz = int(x), int(y), int(z)
        mat = color if isinstance(color, VoxelMaterial) else (int(color[0]), int(color[1]), int(color[2]))
        cx, cz = self._chunk_coords(ix, iz)
        ch = self._ensure_chunk(cx, cz)
        ch.blocks[(ix, iy, iz)] = mat
        ch.modified_by_user = True
        self._overrides[(ix, iy, iz)] = mat
        self._mark_dirty_with_neighbors(ix, iz)

    def remove_block(self, x: int, y: int, z: int) -> bool:
        ix, iy, iz = int(x), int(y), int(z)
        cx, cz = self._chunk_coords(ix, iz)
        ch = self.chunks.get((cx, cz))
        if ch is not None and (ix, iy, iz) in ch.blocks:
            del ch.blocks[(ix, iy, iz)]
            ch.modified_by_user = True
            self._overrides[(ix, iy, iz)] = None
            self._mark_dirty_with_neighbors(ix, iz)
            return True
        return False

    def get_block(self, x: int, y: int, z: int) -> tuple[int, int, int] | VoxelMaterial | None:
        ix, iy, iz = int(x), int(y), int(z)
        ch = self.chunks.get(self._chunk_coords(ix, iz))
        return ch.blocks.get((ix, iy, iz)) if ch is not None else None

    def has_block(self, x: int, y: int, z: int) -> bool:
        return self.get_block(x, y, z) is not None

    def is_solid(self, x: int, y: int, z: int) -> bool:
        mat = self.get_block(x, y, z)
        if mat is None:
            return False
        return bool(mat.solid) if isinstance(mat, VoxelMaterial) else True

    def get_highest_solid_y(self, x: int, z: int) -> int | None:
        ix, iz = int(x), int(z)
        ch = self.chunks.get(self._chunk_coords(ix, iz))
        if ch is None:
            return None
        ys = [
            by
            for (bx, by, bz), mat in ch.blocks.items()
            if bx == ix and bz == iz and (not isinstance(mat, VoxelMaterial) or mat.solid)
        ]
        return max(ys) if ys else None

    def clear(self) -> None:
        self.chunks.clear()
        self._overrides.clear()
        self._aggregate_dirty = True

    def generate_chunk(self, cx: int, cz: int) -> Chunk3D:
        key = (int(cx), int(cz))
        existing = self.chunks.get(key)
        if existing is not None and existing.generated:
            return existing
        ch = self._ensure_chunk(cx, cz)
        ch.generated = True
        if self.generator_fn is not None:
            self.generator_fn(cx, cz, self.chunk_size, self)

        # Re-apply any persistent user edits belonging to this chunk
        x0 = cx * self.chunk_size
        x1 = x0 + self.chunk_size
        z0 = cz * self.chunk_size
        z1 = z0 + self.chunk_size
        for (ox, oy, oz), mat in self._overrides.items():
            if x0 <= ox < x1 and z0 <= oz < z1:
                if mat is None:
                    ch.blocks.pop((ox, oy, oz), None)
                else:
                    ch.blocks[(ox, oy, oz)] = mat
                ch.modified_by_user = True

        # Mark neighbor chunks dirty so cross-chunk borders cull cleanly
        for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = self.chunks.get((cx + dx, cz + dz))
            if nb is not None:
                nb.dirty = True
        ch.dirty = True
        self._aggregate_dirty = True
        return ch

    def update_chunks_around(
        self,
        world_x: float,
        world_z: float,
        load_radius: int = 2,
        unload_radius: int = 4,
        max_new_per_call: int = 2,
    ) -> int:
        """
        Stream chunks dynamically around `(world_x, world_z)`.
        Generates closest missing chunks first (up to `max_new_per_call`) and unloads chunks beyond `unload_radius`.
        Returns the number of newly generated chunks.
        """
        pcx = math.floor(float(world_x) / self.chunk_size)
        pcz = math.floor(float(world_z) / self.chunk_size)

        # Unload far chunks
        to_remove = [
            key
            for key in self.chunks.keys()
            if max(abs(key[0] - pcx), abs(key[1] - pcz)) > unload_radius
        ]
        if to_remove:
            for key in to_remove:
                del self.chunks[key]
            self._aggregate_dirty = True

        # Find missing chunks inside load_radius sorted by distance to player
        missing: list[tuple[int, int, int]] = []
        for dx in range(-load_radius, load_radius + 1):
            for dz in range(-load_radius, load_radius + 1):
                if dx * dx + dz * dz <= load_radius * load_radius + 1:
                    key = (pcx + dx, pcz + dz)
                    existing = self.chunks.get(key)
                    if existing is None or not existing.generated:
                        missing.append((dx * dx + dz * dz, key[0], key[1]))
        missing.sort(key=lambda item: item[0])

        created = 0
        for _, cx, cz in missing[: max(1, int(max_new_per_call))]:
            self.generate_chunk(cx, cz)
            created += 1
        return created

    def _rebuild_chunk_mesh(self, ch: Chunk3D) -> None:
        quads_list: list[np.ndarray] = []
        normals_list: list[tuple[int, int, int]] = []
        colors_list: list[tuple[int, int, int]] = []
        textures_list: list[Texture2D | None] = []
        alphas_list: list[int] = []
        emissive_list: list[float] = []
        ao_list: list[float] = []

        get_block = self.get_block
        local_blocks = ch.blocks
        cs = self.chunk_size
        x0 = ch.cx * cs
        x1 = x0 + cs - 1
        z0 = ch.cz * cs
        z1 = z0 + cs - 1

        for (bx, by, bz), mat in local_blocks.items():
            offset = np.array([bx, by, bz], dtype=np.float32)
            is_mat = isinstance(mat, VoxelMaterial)
            mat_trans = mat.is_transparent if is_mat else False
            mat_alpha = mat.alpha if is_mat else 255
            mat_emissive = mat.emissive if is_mat else 0.0
            is_interior = (x0 < bx < x1) and (z0 < bz < z1)

            for (nx, ny, nz), unit_quad in _VOXEL_FACES:
                nb_pos = (bx + nx, by + ny, bz + nz)
                neighbor = local_blocks.get(nb_pos) if is_interior else get_block(*nb_pos)
                if neighbor is not None:
                    n_trans = neighbor.is_transparent if isinstance(neighbor, VoxelMaterial) else False
                    if not n_trans:
                        continue
                    if mat_trans and n_trans:
                        continue

                quads_list.append(unit_quad + offset)
                normals_list.append((nx, ny, nz))
                colors_list.append(mat.color_for_normal((nx, ny, nz)) if is_mat else mat)
                textures_list.append(mat.texture_for_normal((nx, ny, nz)) if is_mat else None)
                alphas_list.append(mat_alpha)
                emissive_list.append(mat_emissive)

                if ny == 1:
                    occ = 0
                    for ox, oz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nb = local_blocks.get((bx + ox, by + 1, bz + oz)) if is_interior else get_block(bx + ox, by + 1, bz + oz)
                        if nb is not None and not (isinstance(nb, VoxelMaterial) and nb.is_transparent):
                            occ += 1
                    ao_list.append(max(0.72, 1.0 - occ * 0.065))
                elif ny == 0:
                    ov = local_blocks.get((bx + nx, by + 1, bz + nz)) if is_interior else get_block(bx + nx, by + 1, bz + nz)
                    fl = local_blocks.get((bx + nx, by - 1, bz + nz)) if is_interior else get_block(bx + nx, by - 1, bz + nz)
                    ao_val = 1.0 - (0.12 if (ov is not None and not (isinstance(ov, VoxelMaterial) and ov.is_transparent)) else 0.0)
                    ao_val -= 0.06 if (fl is not None and not (isinstance(fl, VoxelMaterial) and fl.is_transparent)) else 0.0
                    ao_list.append(ao_val)
                else:
                    ao_list.append(0.88)

        if quads_list:
            ch.quads = np.asarray(quads_list, dtype=np.float32)
            ch.normals = np.asarray(normals_list, dtype=np.float32)
            ch.colors = np.asarray(colors_list, dtype=np.float32)
            ch.textures = textures_list
            ch.alphas = np.asarray(alphas_list, dtype=np.uint8)
            ch.emissive = np.asarray(emissive_list, dtype=np.float32)
            ch.ao = np.asarray(ao_list, dtype=np.float32)
        else:
            ch.quads = np.empty((0, 4, 3), dtype=np.float32)
            ch.normals = np.empty((0, 3), dtype=np.float32)
            ch.colors = np.empty((0, 3), dtype=np.float32)
            ch.textures = []
            ch.alphas = np.empty((0,), dtype=np.uint8)
            ch.emissive = np.empty((0,), dtype=np.float32)
            ch.ao = np.empty((0,), dtype=np.float32)
        ch.dirty = False

    def get_culled_render_data(self) -> dict[str, Any]:
        """Return aggregated culled face arrays across all currently loaded chunks (rebuilding only dirty chunks)."""
        any_rebuilt = False
        for ch in self.chunks.values():
            if ch.dirty:
                self._rebuild_chunk_mesh(ch)
                any_rebuilt = True

        if not self._aggregate_dirty and not any_rebuilt and self._cached_render_data:
            return self._cached_render_data

        non_empty = [ch for ch in self.chunks.values() if len(ch.quads) > 0]
        if non_empty:
            all_textures: list[Texture2D | None] = []
            for ch in non_empty:
                all_textures.extend(ch.textures)
            self._cached_render_data = {
                "quads": np.concatenate([ch.quads for ch in non_empty], axis=0),
                "normals": np.concatenate([ch.normals for ch in non_empty], axis=0),
                "colors": np.concatenate([ch.colors for ch in non_empty], axis=0),
                "textures": all_textures,
                "alphas": np.concatenate([ch.alphas for ch in non_empty], axis=0),
                "emissive": np.concatenate([ch.emissive for ch in non_empty], axis=0),
                "ao": np.concatenate([ch.ao for ch in non_empty], axis=0),
            }
        else:
            self._cached_render_data = {
                "quads": np.empty((0, 4, 3), dtype=np.float32),
                "normals": np.empty((0, 3), dtype=np.float32),
                "colors": np.empty((0, 3), dtype=np.float32),
                "textures": [],
                "alphas": np.empty((0,), dtype=np.uint8),
                "emissive": np.empty((0,), dtype=np.float32),
                "ao": np.empty((0,), dtype=np.float32),
            }
        self._aggregate_dirty = False
        return self._cached_render_data

    def get_culled_quads(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = self.get_culled_render_data()
        return data["quads"], data["normals"], data["colors"]

    def raycast(
        self,
        origin: tuple[float, float, float] | np.ndarray,
        direction: tuple[float, float, float] | np.ndarray,
        max_distance: float = 8.0,
        skip_non_solid: bool = True,
    ) -> VoxelHit | None:
        """3D DDA Voxel Raycast across chunks."""
        if not self.chunks:
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
        get_block = self.get_block

        while dist <= max_distance:
            mat = get_block(vx, vy, vz)
            if mat is not None:
                is_solid = mat.solid if isinstance(mat, VoxelMaterial) else True
                if is_solid or not skip_non_solid:
                    adj = (vx + last_normal[0], vy + last_normal[1], vz + last_normal[2])
                    hit_rgb = mat.color_for_normal(last_normal) if isinstance(mat, VoxelMaterial) else mat
                    return VoxelHit(
                        block_pos=(vx, vy, vz),
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
