from __future__ import annotations

import math
from typing import Any, Callable
import numpy as np
from ..textures.atlas import TextureAtlas
from ..textures.texture2d import Texture2D
from .voxel_world import VoxelHit, VoxelMaterial, _VOXEL_FACES


def _greedy_mesh_slice_python(keys: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Pure-Python fallback 2D greedy rectangle merger over `(rows, cols)` int32 key grid."""
    rows, cols = keys.shape
    rects: list[tuple[int, int, int, int, int]] = []
    for r in range(rows):
        c = 0
        while c < cols:
            k = int(keys[r, c])
            if k == 0:
                c += 1
                continue
            w = 1
            while c + w < cols and int(keys[r, c + w]) == k:
                w += 1
            h = 1
            while r + h < rows:
                ok = True
                for dx in range(w):
                    if int(keys[r + h, c + dx]) != k:
                        ok = False
                        break
                if not ok:
                    break
                h += 1
            keys[r : r + h, c : c + w] = 0
            rects.append((r, c, h, w, k))
            c += w
    return rects


class Chunk3D:
    """Single horizontal column chunk (chunk_size x Y x chunk_size) with cached greedy-meshed face buffers."""

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
        self.uv_scales: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.textures: list[Texture2D | None] = []
        self.tex_ids: np.ndarray = np.empty((0,), dtype=np.int32)
        self.alphas: np.ndarray = np.empty((0,), dtype=np.uint8)
        self.emissive: np.ndarray = np.empty((0,), dtype=np.float32)
        self.ao: np.ndarray = np.empty((0,), dtype=np.float32)
        self.is_translucent: np.ndarray = np.empty((0,), dtype=bool)

        s = float(self.chunk_size)
        self.center_3d: np.ndarray = np.array([(cx + 0.5) * s, 4.0, (cz + 0.5) * s], dtype=np.float32)
        self.radius_3d: float = s * 0.9

    @property
    def center_xz(self) -> tuple[float, float]:
        s = float(self.chunk_size)
        return ((self.cx + 0.5) * s, (self.cz + 0.5) * s)


class ChunkedVoxelWorld:
    """
    High-performance, game-agnostic procedural Voxel World organized into streamable 3D Chunks.
    - 2D Greedy Meshing (`greedy_meshing=True`) merges adjacent coplanar same-material/same-AO faces
      into seamlessly tiled quads (4x-6x fewer polygons with identical pixel-art visuals!).
    - Vectorized dense-grid halo culling & Ambient Occlusion (~25x faster chunk rebuilds).
    - Per-chunk 3D bounding-sphere frustum culling (`get_visible_render_data`).
    - Persistent user edit overrides & compact JSON-friendly serialization (`export_overrides`/`import_overrides`).
    """

    def __init__(
        self,
        chunk_size: int = 8,
        generator_fn: Callable[[int, int, int, ChunkedVoxelWorld], None] | None = None,
        greedy_meshing: bool = True,
    ) -> None:
        self.chunk_size: int = max(4, int(chunk_size))
        self.generator_fn = generator_fn
        self.greedy_meshing: bool = bool(greedy_meshing)
        self.chunks: dict[tuple[int, int], Chunk3D] = {}
        self.atlas: TextureAtlas = TextureAtlas()
        self._overrides: dict[tuple[int, int, int], tuple[int, int, int] | VoxelMaterial | None] = {}
        self._aggregate_dirty: bool = True
        self._cached_render_data: dict[str, Any] = {}
        self._greedy_rect_buf: np.ndarray = np.empty((256, 5), dtype=np.int32)

    @property
    def blocks(self) -> dict[tuple[int, int, int], tuple[int, int, int] | VoxelMaterial]:
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

    def export_overrides(self, material_to_id: Callable[[Any], str | int]) -> list[list[Any]]:
        out: list[list[Any]] = []
        for (bx, by, bz), mat in self._overrides.items():
            out.append([int(bx), int(by), int(bz), None if mat is None else material_to_id(mat)])
        return out

    def import_overrides(
        self,
        records: list[list[Any]],
        id_to_material: Callable[[str | int], Any],
    ) -> int:
        count = 0
        for rec in records:
            if len(rec) < 4:
                continue
            bx, by, bz, mat_id = int(rec[0]), int(rec[1]), int(rec[2]), rec[3]
            if mat_id is None:
                self._overrides[(bx, by, bz)] = None
                ch = self.chunks.get(self._chunk_coords(bx, bz))
                if ch is not None:
                    ch.blocks.pop((bx, by, bz), None)
                    ch.modified_by_user = True
                    self._mark_dirty_with_neighbors(bx, bz)
            else:
                mat = id_to_material(mat_id)
                if mat is None:
                    continue
                self._overrides[(bx, by, bz)] = mat
                ch = self.chunks.get(self._chunk_coords(bx, bz))
                if ch is not None:
                    ch.blocks[(bx, by, bz)] = mat
                    ch.modified_by_user = True
                    self._mark_dirty_with_neighbors(bx, bz)
            count += 1
        self._aggregate_dirty = True
        return count

    def generate_chunk(self, cx: int, cz: int) -> Chunk3D:
        key = (int(cx), int(cz))
        existing = self.chunks.get(key)
        if existing is not None and existing.generated:
            return existing
        ch = self._ensure_chunk(cx, cz)
        ch.generated = True
        if self.generator_fn is not None:
            self.generator_fn(cx, cz, self.chunk_size, self)

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
        pcx = math.floor(float(world_x) / self.chunk_size)
        pcz = math.floor(float(world_z) / self.chunk_size)

        to_remove = [
            key
            for key in self.chunks.keys()
            if max(abs(key[0] - pcx), abs(key[1] - pcz)) > unload_radius
        ]
        if to_remove:
            for key in to_remove:
                del self.chunks[key]
            self._aggregate_dirty = True

        missing: list[tuple[int, int, int]] = []
        r2 = load_radius * load_radius + 1
        for dx in range(-load_radius, load_radius + 1):
            for dz in range(-load_radius, load_radius + 1):
                d2 = dx * dx + dz * dz
                if d2 <= r2:
                    key = (pcx + dx, pcz + dz)
                    existing = self.chunks.get(key)
                    if existing is None or not existing.generated:
                        missing.append((d2, key[0], key[1]))
        missing.sort(key=lambda item: item[0])

        created = 0
        for _, cx, cz in missing[: max(1, int(max_new_per_call))]:
            self.generate_chunk(cx, cz)
            created += 1
        return created

    def _run_greedy_2d(self, keys: np.ndarray) -> np.ndarray:
        """Run C99 native or Python fallback 2D greedy rectangle merging. Returns `(K, 5)` int32 `[r, c, h, w, key]`."""
        rows, cols = keys.shape
        from ..renderer.native_bridge import HAS_NATIVE_RASTERIZER, NATIVE_LIB

        if HAS_NATIVE_RASTERIZER and NATIVE_LIB is not None:
            k_c = np.ascontiguousarray(keys, dtype=np.int32)
            max_r = rows * cols
            if len(self._greedy_rect_buf) < max_r:
                self._greedy_rect_buf = np.empty((max_r, 5), dtype=np.int32)
            n = NATIVE_LIB.pf3d_greedy_mesh_2d(
                k_c.ctypes.data, int(rows), int(cols),
                self._greedy_rect_buf.ctypes.data, int(len(self._greedy_rect_buf)),
            )
            return self._greedy_rect_buf[:n].copy()
        rects = _greedy_mesh_slice_python(keys.copy())
        if not rects:
            return np.empty((0, 5), dtype=np.int32)
        return np.asarray(rects, dtype=np.int32)

    def _rebuild_chunk_mesh(self, ch: Chunk3D) -> None:
        """
        Vectorized Dense-Halo + 2D Greedy Chunk Mesher:
        Extracts culled faces + Ambient Occlusion on an L1-resident halo grid, then merges coplanar
        identical-material/identical-AO faces into tiled quads (4x-6x polygon reduction!).
        """
        local_blocks = ch.blocks
        if not local_blocks:
            ch.quads = np.empty((0, 4, 3), dtype=np.float32)
            ch.normals = np.empty((0, 3), dtype=np.float32)
            ch.colors = np.empty((0, 3), dtype=np.float32)
            ch.uv_scales = np.empty((0, 2), dtype=np.float32)
            ch.textures = []
            ch.tex_ids = np.empty((0,), dtype=np.int32)
            ch.alphas = np.empty((0,), dtype=np.uint8)
            ch.emissive = np.empty((0,), dtype=np.float32)
            ch.ao = np.empty((0,), dtype=np.float32)
            ch.is_translucent = np.empty((0,), dtype=bool)
            ch.dirty = False
            return

        cs = self.chunk_size
        x0 = ch.cx * cs
        z0 = ch.cz * cs
        x_end = x0 + cs - 1
        z_end = z0 + cs - 1

        ys_all = [pos[1] for pos in local_blocks.keys()]
        min_y = min(ys_all)
        max_y = max(ys_all)
        h_span = max_y - min_y + 1

        half_s = cs * 0.5
        half_h = h_span * 0.5
        ch.center_3d = np.array([x0 + half_s, min_y + half_h, z0 + half_s], dtype=np.float32)
        ch.radius_3d = math.sqrt(half_s * half_s * 2.0 + half_h * half_h) + 0.5

        mat_to_id: dict[Any, int] = {}
        mats_list: list[Any] = [None]
        grid = np.zeros((cs + 2, h_span + 2, cs + 2), dtype=np.int16)

        for (bx, by, bz), mat in local_blocks.items():
            mid = mat_to_id.get(mat)
            if mid is None:
                mid = len(mats_list)
                mat_to_id[mat] = mid
                mats_list.append(mat)
            grid[bx - x0 + 1, by - min_y + 1, bz - z0 + 1] = mid

        nb_configs = (
            (self.chunks.get((ch.cx - 1, ch.cz)), 0, None, x0 - 1, None),
            (self.chunks.get((ch.cx + 1, ch.cz)), cs + 1, None, x_end + 1, None),
            (self.chunks.get((ch.cx, ch.cz - 1)), None, 0, None, z0 - 1),
            (self.chunks.get((ch.cx, ch.cz + 1)), None, cs + 1, None, z_end + 1),
        )
        for nb_ch, gx_fixed, gz_fixed, wx_match, wz_match in nb_configs:
            if nb_ch is None or not nb_ch.blocks:
                continue
            for (bx, by, bz), mat in nb_ch.blocks.items():
                if by < min_y - 1 or by > max_y + 1:
                    continue
                if wx_match is not None:
                    if bx != wx_match:
                        continue
                    gx = gx_fixed
                    gz = bz - z0 + 1
                    if gz < 1 or gz > cs:
                        continue
                else:
                    if bz != wz_match:
                        continue
                    gx = bx - x0 + 1
                    gz = gz_fixed
                    if gx < 1 or gx > cs:
                        continue
                mid = mat_to_id.get(mat)
                if mid is None:
                    mid = len(mats_list)
                    mat_to_id[mat] = mid
                    mats_list.append(mat)
                grid[gx, by - min_y + 1, gz] = mid

        n_mats = len(mats_list)
        mat_trans = np.zeros(n_mats, dtype=bool)
        mat_alpha = np.full(n_mats, 255, dtype=np.uint8)
        mat_emissive = np.zeros(n_mats, dtype=np.float32)

        mat_colors = np.zeros((3, n_mats, 3), dtype=np.float32)
        mat_tex_ids = np.full((3, n_mats), -1, dtype=np.int32)
        mat_tex_objs: list[list[Texture2D | None]] = [[None] * n_mats for _ in range(3)]
        mat_face_translucent = np.zeros((3, n_mats), dtype=bool)

        atlas_reg = self.atlas.register
        for mid in range(1, n_mats):
            m = mats_list[mid]
            if isinstance(m, VoxelMaterial):
                mat_trans[mid] = m.is_transparent
                mat_alpha[mid] = m.alpha
                mat_emissive[mid] = m.emissive
                c_side = m.color_for_normal((1, 0, 0))
                c_top = m.color_for_normal((0, 1, 0))
                c_bot = m.color_for_normal((0, -1, 0))
                t_side = m.texture_for_normal((1, 0, 0))
                t_top = m.texture_for_normal((0, 1, 0))
                t_bot = m.texture_for_normal((0, -1, 0))
                for ftype, col, tex in ((0, c_side, t_side), (1, c_top, t_top), (2, c_bot, t_bot)):
                    mat_colors[ftype, mid] = col
                    mat_tex_objs[ftype][mid] = tex
                    mat_tex_ids[ftype, mid] = atlas_reg(tex)
                    mat_face_translucent[ftype, mid] = (m.alpha < 255) or (tex is not None and tex.has_cutout)
            else:
                for ftype in (0, 1, 2):
                    mat_colors[ftype, mid] = m

        core = grid[1 : cs + 1, 1 : h_span + 1, 1 : cs + 1]
        core_present = core > 0
        core_tr = mat_trans[core]
        occ_grid = (grid > 0) & (~mat_trans[grid])

        quads_parts: list[np.ndarray] = []
        normals_parts: list[np.ndarray] = []
        colors_parts: list[np.ndarray] = []
        uv_scales_parts: list[np.ndarray] = []
        tex_ids_parts: list[np.ndarray] = []
        textures_all: list[Texture2D | None] = []
        alphas_parts: list[np.ndarray] = []
        emissive_parts: list[np.ndarray] = []
        ao_parts: list[np.ndarray] = []
        translucent_parts: list[np.ndarray] = []

        for (nx, ny, nz), unit_quad in _VOXEL_FACES:
            nb = grid[
                1 + nx : cs + 1 + nx,
                1 + ny : h_span + 1 + ny,
                1 + nz : cs + 1 + nz,
            ]
            nb_tr = mat_trans[nb]
            vis_mask = core_present & ((nb == 0) | (nb_tr & (~core_tr)))
            if not vis_mask.any():
                continue

            ftype = 1 if ny > 0 else (2 if ny < 0 else 0)
            tex_lookup = mat_tex_objs[ftype]

            if not self.greedy_meshing:
                ix, iy, iz = np.nonzero(vis_mask)
                mids = core[ix, iy, iz]
                k = len(mids)
                coords = np.empty((k, 3), dtype=np.float32)
                coords[:, 0] = ix + x0
                coords[:, 1] = iy + min_y
                coords[:, 2] = iz + z0

                quads_parts.append(unit_quad[None, :, :] + coords[:, None, :])
                normals_parts.append(np.broadcast_to(np.array([nx, ny, nz], dtype=np.float32), (k, 3)).copy())
                colors_parts.append(mat_colors[ftype, mids])
                uv_scales_parts.append(np.ones((k, 2), dtype=np.float32))
                tex_ids_parts.append(mat_tex_ids[ftype, mids])
                alphas_parts.append(mat_alpha[mids])
                emissive_parts.append(mat_emissive[mids])
                translucent_parts.append(mat_face_translucent[ftype, mids])
                textures_all.extend(tex_lookup[int(mid)] for mid in mids)

                gx, gy, gz = ix + 1, iy + 1, iz + 1
                if ny == 1:
                    occ = (
                        occ_grid[gx + 1, gy + 1, gz].astype(np.int8)
                        + occ_grid[gx - 1, gy + 1, gz].astype(np.int8)
                        + occ_grid[gx, gy + 1, gz + 1].astype(np.int8)
                        + occ_grid[gx, gy + 1, gz - 1].astype(np.int8)
                    )
                    ao_arr = np.maximum(0.72, 1.0 - occ * 0.065).astype(np.float32)
                elif ny == 0:
                    ov = occ_grid[gx + nx, gy + 1, gz + nz].astype(np.float32)
                    fl = occ_grid[gx + nx, gy - 1, gz + nz].astype(np.float32)
                    ao_arr = (1.0 - ov * 0.12 - fl * 0.06).astype(np.float32)
                else:
                    ao_arr = np.full(k, 0.88, dtype=np.float32)
                ao_parts.append(ao_arr)
                continue

            # --- 2D GREEDY MESHING ---
            # Compute quantized AO byte (0..255 representing 0.00..1.00) over the 3D slice volume
            if ny == 1:
                occ3 = (
                    occ_grid[2 : cs + 2, 2 : h_span + 2, 1 : cs + 1].astype(np.int16)
                    + occ_grid[0:cs, 2 : h_span + 2, 1 : cs + 1].astype(np.int16)
                    + occ_grid[1 : cs + 1, 2 : h_span + 2, 2 : cs + 2].astype(np.int16)
                    + occ_grid[1 : cs + 1, 2 : h_span + 2, 0:cs].astype(np.int16)
                )
                ao_u8 = np.maximum(184, 255 - occ3 * 17).astype(np.int32)
            elif ny == 0:
                ov3 = occ_grid[1 + nx : cs + 1 + nx, 2 : h_span + 2, 1 + nz : cs + 1 + nz].astype(np.int32)
                fl3 = occ_grid[1 + nx : cs + 1 + nx, 0:h_span, 1 + nz : cs + 1 + nz].astype(np.int32)
                ao_u8 = 255 - ov3 * 31 - fl3 * 15
            else:
                ao_u8 = np.full((cs, h_span, cs), 224, dtype=np.int32)

            key_vol = np.where(vis_mask, (core.astype(np.int32) << 8) | ao_u8, 0)

            dir_quads: list[np.ndarray] = []
            dir_uvs: list[tuple[float, float]] = []
            dir_mids: list[int] = []
            dir_aos: list[float] = []

            if ny != 0:
                # Top (+Y) or Bottom (-Y): slice along iy; rows=iz (dz), cols=ix (dx)
                for iy in range(h_span):
                    sl = key_vol[:, iy, :].T  # shape (cs_z, cs_x)
                    if not sl.any():
                        continue
                    rects = self._run_greedy_2d(sl)
                    if len(rects) == 0:
                        continue
                    wy = float(min_y + iy)
                    for r0, c0, dz, dx, k_val in rects:
                        wx = float(x0 + c0)
                        wz = float(z0 + r0)
                        wf, hf = float(dx), float(dz)
                        if ny == 1:
                            q = np.array(
                                [[wx, wy + 1.0, wz], [wx, wy + 1.0, wz + hf], [wx + wf, wy + 1.0, wz + hf], [wx + wf, wy + 1.0, wz]],
                                dtype=np.float32,
                            )
                        else:
                            q = np.array(
                                [[wx, wy, wz + hf], [wx, wy, wz], [wx + wf, wy, wz], [wx + wf, wy, wz + hf]],
                                dtype=np.float32,
                            )
                        dir_quads.append(q)
                        dir_uvs.append((wf, hf))
                        dir_mids.append(int(k_val >> 8))
                        dir_aos.append((k_val & 0xFF) / 255.0)
            elif nz != 0:
                # Front (-Z) or Back (+Z): slice along iz; rows=iy (dy), cols=ix (dx)
                for iz in range(cs):
                    sl = key_vol[:, :, iz].T  # shape (h_span, cs_x)
                    if not sl.any():
                        continue
                    rects = self._run_greedy_2d(sl)
                    if len(rects) == 0:
                        continue
                    wz = float(z0 + iz)
                    for r0, c0, dy, dx, k_val in rects:
                        wx = float(x0 + c0)
                        wy = float(min_y + r0)
                        wf, hf = float(dx), float(dy)
                        if nz == -1:
                            q = np.array(
                                [[wx, wy, wz], [wx, wy + hf, wz], [wx + wf, wy + hf, wz], [wx + wf, wy, wz]],
                                dtype=np.float32,
                            )
                        else:
                            q = np.array(
                                [[wx + wf, wy, wz + 1.0], [wx + wf, wy + hf, wz + 1.0], [wx, wy + hf, wz + 1.0], [wx, wy, wz + 1.0]],
                                dtype=np.float32,
                            )
                        dir_quads.append(q)
                        dir_uvs.append((wf, hf))
                        dir_mids.append(int(k_val >> 8))
                        dir_aos.append((k_val & 0xFF) / 255.0)
            else:
                # Right (+X) or Left (-X): slice along ix; rows=iy (dy), cols=iz (dz)
                for ix in range(cs):
                    sl = key_vol[ix, :, :]  # shape (h_span, cs_z)
                    if not sl.any():
                        continue
                    rects = self._run_greedy_2d(sl)
                    if len(rects) == 0:
                        continue
                    wx = float(x0 + ix)
                    for r0, c0, dy, dz, k_val in rects:
                        wy = float(min_y + r0)
                        wz = float(z0 + c0)
                        wf, hf = float(dz), float(dy)
                        if nx == 1:
                            q = np.array(
                                [[wx + 1.0, wy, wz], [wx + 1.0, wy + hf, wz], [wx + 1.0, wy + hf, wz + wf], [wx + 1.0, wy, wz + wf]],
                                dtype=np.float32,
                            )
                        else:
                            q = np.array(
                                [[wx, wy, wz + wf], [wx, wy + hf, wz + wf], [wx, wy + hf, wz], [wx, wy, wz]],
                                dtype=np.float32,
                            )
                        dir_quads.append(q)
                        dir_uvs.append((wf, hf))
                        dir_mids.append(int(k_val >> 8))
                        dir_aos.append((k_val & 0xFF) / 255.0)

            if dir_quads:
                k = len(dir_quads)
                mids_arr = np.asarray(dir_mids, dtype=np.int32)
                quads_parts.append(np.asarray(dir_quads, dtype=np.float32))
                normals_parts.append(np.broadcast_to(np.array([nx, ny, nz], dtype=np.float32), (k, 3)).copy())
                colors_parts.append(mat_colors[ftype, mids_arr])
                uv_scales_parts.append(np.asarray(dir_uvs, dtype=np.float32))
                tex_ids_parts.append(mat_tex_ids[ftype, mids_arr])
                alphas_parts.append(mat_alpha[mids_arr])
                emissive_parts.append(mat_emissive[mids_arr])
                ao_parts.append(np.asarray(dir_aos, dtype=np.float32))
                translucent_parts.append(mat_face_translucent[ftype, mids_arr])
                textures_all.extend(tex_lookup[m] for m in dir_mids)

        if quads_parts:
            ch.quads = np.concatenate(quads_parts, axis=0)
            ch.normals = np.concatenate(normals_parts, axis=0)
            ch.colors = np.concatenate(colors_parts, axis=0)
            ch.uv_scales = np.concatenate(uv_scales_parts, axis=0)
            ch.tex_ids = np.concatenate(tex_ids_parts, axis=0)
            ch.textures = textures_all
            ch.alphas = np.concatenate(alphas_parts, axis=0)
            ch.emissive = np.concatenate(emissive_parts, axis=0)
            ch.ao = np.concatenate(ao_parts, axis=0)
            ch.is_translucent = np.concatenate(translucent_parts, axis=0)
        else:
            ch.quads = np.empty((0, 4, 3), dtype=np.float32)
            ch.normals = np.empty((0, 3), dtype=np.float32)
            ch.colors = np.empty((0, 3), dtype=np.float32)
            ch.uv_scales = np.empty((0, 2), dtype=np.float32)
            ch.tex_ids = np.empty((0,), dtype=np.int32)
            ch.textures = []
            ch.alphas = np.empty((0,), dtype=np.uint8)
            ch.emissive = np.empty((0,), dtype=np.float32)
            ch.ao = np.empty((0,), dtype=np.float32)
            ch.is_translucent = np.empty((0,), dtype=bool)
        ch.dirty = False

    def _aggregate_chunks(self, chunk_list: list[Chunk3D], include_textures: bool = True) -> dict[str, Any]:
        non_empty = [ch for ch in chunk_list if len(ch.quads) > 0]
        self.atlas.ensure_packed()
        if not non_empty:
            return {
                "quads": np.empty((0, 4, 3), dtype=np.float32),
                "normals": np.empty((0, 3), dtype=np.float32),
                "colors": np.empty((0, 3), dtype=np.float32),
                "uv_scales": np.empty((0, 2), dtype=np.float32),
                "textures": [],
                "tex_ids": np.empty((0,), dtype=np.int32),
                "alphas": np.empty((0,), dtype=np.uint8),
                "emissive": np.empty((0,), dtype=np.float32),
                "ao": np.empty((0,), dtype=np.float32),
                "is_translucent": np.empty((0,), dtype=bool),
                "atlas": self.atlas,
            }
        if len(non_empty) == 1:
            ch = non_empty[0]
            return {
                "quads": ch.quads,
                "normals": ch.normals,
                "colors": ch.colors,
                "uv_scales": ch.uv_scales,
                "textures": ch.textures if include_textures else [],
                "tex_ids": ch.tex_ids,
                "alphas": ch.alphas,
                "emissive": ch.emissive,
                "ao": ch.ao,
                "is_translucent": ch.is_translucent,
                "atlas": self.atlas,
            }
        all_textures: list[Texture2D | None] = []
        if include_textures:
            for ch in non_empty:
                all_textures.extend(ch.textures)
        return {
            "quads": np.concatenate([ch.quads for ch in non_empty], axis=0),
            "normals": np.concatenate([ch.normals for ch in non_empty], axis=0),
            "colors": np.concatenate([ch.colors for ch in non_empty], axis=0),
            "uv_scales": np.concatenate([ch.uv_scales for ch in non_empty], axis=0),
            "textures": all_textures,
            "tex_ids": np.concatenate([ch.tex_ids for ch in non_empty], axis=0),
            "alphas": np.concatenate([ch.alphas for ch in non_empty], axis=0),
            "emissive": np.concatenate([ch.emissive for ch in non_empty], axis=0),
            "ao": np.concatenate([ch.ao for ch in non_empty], axis=0),
            "is_translucent": np.concatenate([ch.is_translucent for ch in non_empty], axis=0),
            "atlas": self.atlas,
        }

    def get_visible_render_data(
        self,
        cam_pos: np.ndarray,
        view_mat: np.ndarray,
        near_z: float,
        far_z: float,
        half_w: float,
        half_h: float,
        proj_scale: float,
        include_textures: bool = True,
    ) -> dict[str, Any]:
        visible_chunks: list[Chunk3D] = []
        tan_hx = half_w / max(1e-5, proj_scale) + 0.12
        tan_hy = half_h / max(1e-5, proj_scale) + 0.12
        sec_hx = math.sqrt(1.0 + tan_hx * tan_hx)
        sec_hy = math.sqrt(1.0 + tan_hy * tan_hy)

        for ch in self.chunks.values():
            if ch.dirty:
                self._rebuild_chunk_mesh(ch)
            if len(ch.quads) == 0:
                continue
            rel = ch.center_3d - cam_pos
            cz = float(rel[0] * view_mat[0, 2] + rel[1] * view_mat[1, 2] + rel[2] * view_mat[2, 2])
            r = ch.radius_3d
            if cz + r < near_z or cz - r > far_z:
                continue
            cx_v = float(rel[0] * view_mat[0, 0] + rel[1] * view_mat[1, 0] + rel[2] * view_mat[2, 0])
            if abs(cx_v) > cz * tan_hx + r * sec_hx:
                continue
            cy_v = float(rel[0] * view_mat[0, 1] + rel[1] * view_mat[1, 1] + rel[2] * view_mat[2, 1])
            if abs(cy_v) > cz * tan_hy + r * sec_hy:
                continue
            visible_chunks.append(ch)

        return self._aggregate_chunks(visible_chunks, include_textures=include_textures)

    def get_culled_render_data(self) -> dict[str, Any]:
        any_rebuilt = False
        for ch in self.chunks.values():
            if ch.dirty:
                self._rebuild_chunk_mesh(ch)
                any_rebuilt = True

        if not self._aggregate_dirty and not any_rebuilt and self._cached_render_data:
            return self._cached_render_data

        self._cached_render_data = self._aggregate_chunks(list(self.chunks.values()), include_textures=True)
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
