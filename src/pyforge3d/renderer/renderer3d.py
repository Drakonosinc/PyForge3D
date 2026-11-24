from __future__ import annotations

import math
import numpy as np
import pygame
from ..camera.camera3d import Camera3D
from ..lighting.lights import DirectionalLight, Fog3D, PointLight
from ..lighting.sky_dome import SkyDome3D
from ..math.transforms import clip_polygon_near_plane, compute_face_normals
from ..meshes.mesh3d import Mesh3D
from ..textures.atlas import TextureAtlas
from ..textures.texture2d import Texture2D
from ..voxels.chunked_world import ChunkedVoxelWorld
from ..voxels.voxel_world import VoxelHit, VoxelWorld, _VOXEL_FACES
from .native_bridge import HAS_NATIVE_RASTERIZER, NATIVE_LIB
from .rasterizer import CANONICAL_QUAD_UV, SoftwareTexturedRasterizer, clip_textured_polygon_near_plane


class Scene3D:
    """Container holding 3D meshes for a 3D scene."""

    def __init__(self) -> None:
        self.meshes: list[Mesh3D] = []

    def add_mesh(self, *meshes: Mesh3D) -> None:
        for m in meshes:
            if m not in self.meshes:
                self.meshes.append(m)

    def remove_mesh(self, mesh: Mesh3D) -> None:
        if mesh in self.meshes:
            self.meshes.remove(mesh)

    def clear(self) -> None:
        self.meshes.clear()


class Renderer3D:
    """
    High-performance Dual-Backend Software 3D Renderer for Pygame.
    Supports:
    1. **Textured Z-Buffer Pipeline (`textured=True`)**:
       - Automatically engages the compiled zero-copy C99 scanline Z-Buffer backend when available
         (100+ FPS at crisp HD resolution with 1/Z occlusion, perspective-correct UV mapping, AO, fog,
         and translucent Water/Glass blending).
       - Seamlessly falls back to the pure-NumPy vectorized planar Z-Buffer backend if native C99 is unavailable
         or disabled via `use_native=False`.
    2. **Solid Polygon Pipeline (`textured=False`)**: Fast Painter's Algorithm polygons.
    Works seamlessly with `VoxelWorld`, infinite `ChunkedVoxelWorld` (with per-chunk 3D frustum culling),
    `Scene3D` meshes, and `SkyDome3D`.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        light: DirectionalLight | PointLight | None = None,
        fog: Fog3D | None = None,
        voxel_outline_color: tuple[int, int, int] | None = (18, 28, 48),
        textured: bool = False,
        render_scale: float = 0.65,
        texture_lod_distance: float = 28.0,
        use_native: bool = True,
    ) -> None:
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.light: DirectionalLight | PointLight = light if light is not None else DirectionalLight()
        self.fog: Fog3D = fog if fog is not None else Fog3D(enabled=True)
        self.voxel_outline_color: tuple[int, int, int] | None = voxel_outline_color
        self.textured: bool = bool(textured)
        self.render_scale: float = max(0.2, min(1.0, float(render_scale)))
        self.texture_lod_distance: float = max(4.0, float(texture_lod_distance))
        self.use_native: bool = bool(use_native)
        self.chunk_rebuild_budget_ms: float | None = None
        self.dynamic_lights: list[PointLight] = []
        self._dyn_light_buf: np.ndarray = np.empty((32, 6), dtype=np.float32)
        self.last_rendered_faces: int = 0
        self._sky_cache_key = None
        self._sky_cache_surf: pygame.Surface | None = None
        self._rasterizer: SoftwareTexturedRasterizer | None = None
        self._empty_atlas: TextureAtlas = TextureAtlas()
        self._empty_atlas.ensure_packed()
        self._ptr_cap: int = 0
        self._c_quads: np.ndarray = np.empty((0,), dtype=np.uintp)
        self._c_normals: np.ndarray = np.empty((0,), dtype=np.uintp)
        self._c_colors: np.ndarray = np.empty((0,), dtype=np.uintp)
        self._c_uvs: np.ndarray = np.empty((0,), dtype=np.uintp)
        self._c_tids: np.ndarray = np.empty((0,), dtype=np.uintp)
        self._c_alphas: np.ndarray = np.empty((0,), dtype=np.uintp)
        self._c_emissive: np.ndarray = np.empty((0,), dtype=np.uintp)
        self._c_ao: np.ndarray = np.empty((0,), dtype=np.uintp)
        self._c_is_tr: np.ndarray = np.empty((0,), dtype=np.uintp)
        self._c_counts: np.ndarray = np.empty((0,), dtype=np.int32)

    def _ensure_chunk_ptr_bufs(self, n: int) -> None:
        if self._ptr_cap >= n:
            return
        cap = max(64, n * 2)
        self._ptr_cap = cap
        self._c_quads = np.empty((cap,), dtype=np.uintp)
        self._c_normals = np.empty((cap,), dtype=np.uintp)
        self._c_colors = np.empty((cap,), dtype=np.uintp)
        self._c_uvs = np.empty((cap,), dtype=np.uintp)
        self._c_tids = np.empty((cap,), dtype=np.uintp)
        self._c_alphas = np.empty((cap,), dtype=np.uintp)
        self._c_emissive = np.empty((cap,), dtype=np.uintp)
        self._c_ao = np.empty((cap,), dtype=np.uintp)
        self._c_is_tr = np.empty((cap,), dtype=np.uintp)
        self._c_counts = np.empty((cap,), dtype=np.int32)

    @property
    def native_active(self) -> bool:
        return bool(self.use_native and HAS_NATIVE_RASTERIZER and NATIVE_LIB is not None)

    def _get_rasterizer(self, screen_w: int, screen_h: int) -> SoftwareTexturedRasterizer:
        rw = max(64, int(round(screen_w * self.render_scale)))
        rh = max(48, int(round(screen_h * self.render_scale)))
        if self._rasterizer is None:
            self._rasterizer = SoftwareTexturedRasterizer(rw, rh)
        else:
            self._rasterizer.resize(rw, rh)
        return self._rasterizer

    def draw_sky_gradient(
        self,
        screen: pygame.Surface,
        camera: Camera3D,
        zenith_color: tuple[int, int, int] = (14, 24, 54),
        horizon_color: tuple[int, int, int] = (48, 88, 148),
        ground_color: tuple[int, int, int] = (10, 16, 28),
    ) -> None:
        """Draw a dynamic 3D sky & horizon gradient that shifts smoothly with camera.pitch."""
        w, h = screen.get_size()
        horizon_y = int(h * 0.5 + math.sin(camera.pitch) * h * 0.55)
        horizon_y = max(-h, min(h * 2, horizon_y))

        cache_key = (w, h, horizon_y // 4, zenith_color, horizon_color, ground_color)
        if self._sky_cache_key != cache_key or self._sky_cache_surf is None:
            surf = pygame.Surface((w, h))
            step = 4
            for y in range(0, h, step):
                if y <= horizon_y:
                    t = max(0.0, min(1.0, (horizon_y - y) / max(1.0, h * 0.65)))
                    c = (
                        int(horizon_color[0] + (zenith_color[0] - horizon_color[0]) * t),
                        int(horizon_color[1] + (zenith_color[1] - horizon_color[1]) * t),
                        int(horizon_color[2] + (zenith_color[2] - horizon_color[2]) * t),
                    )
                else:
                    t = max(0.0, min(1.0, (y - horizon_y) / max(1.0, h * 0.45)))
                    c = (
                        int(horizon_color[0] + (ground_color[0] - horizon_color[0]) * t),
                        int(horizon_color[1] + (ground_color[1] - horizon_color[1]) * t),
                        int(horizon_color[2] + (ground_color[2] - horizon_color[2]) * t),
                    )
                pygame.draw.rect(surf, c, (0, y, w, step))
            self._sky_cache_key = cache_key
            self._sky_cache_surf = surf
        screen.blit(self._sky_cache_surf, (0, 0))

    def _project_point_list(
        self,
        poly_cam: np.ndarray,
        scale: float,
        half_w: float,
        half_h: float,
        near_z: float,
    ) -> list[tuple[int, int]]:
        z_safe = np.maximum(poly_cam[:, 2], near_z * 0.5)
        sx = half_w + (poly_cam[:, 0] / z_safe) * scale
        sy = half_h - (poly_cam[:, 1] / z_safe) * scale
        return [(int(round(float(x))), int(round(float(y)))) for x, y in zip(sx, sy)]

    def _project_point_list_invz(
        self,
        poly_cam: np.ndarray,
        scale: float,
        half_w: float,
        half_h: float,
        near_z: float,
    ) -> list[tuple[float, float, float]]:
        z_safe = np.maximum(poly_cam[:, 2], near_z * 0.5)
        inv_z = 1.0 / z_safe
        sx = half_w + poly_cam[:, 0] * inv_z * scale
        sy = half_h - poly_cam[:, 1] * inv_z * scale
        return [(float(x), float(y), float(iz)) for x, y, iz in zip(sx, sy, inv_z)]

    def _pack_dynamic_lights(
        self,
        cam_pos: np.ndarray,
        dynamic_lights: list[PointLight] | None = None,
    ) -> tuple[int, np.ndarray]:
        lights = dynamic_lights if dynamic_lights is not None else self.dynamic_lights
        if not lights:
            return 0, self._dyn_light_buf
        if len(lights) > 32:
            cx, cy, cz = float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])
            lights = sorted(
                lights,
                key=lambda pl: (float(pl.position[0]) - cx) ** 2 + (float(pl.position[1]) - cy) ** 2 + (float(pl.position[2]) - cz) ** 2,
            )[:32]
        n = len(lights)
        buf = self._dyn_light_buf
        for i, pl in enumerate(lights):
            buf[i, 0] = float(pl.position[0])
            buf[i, 1] = float(pl.position[1])
            buf[i, 2] = float(pl.position[2])
            buf[i, 3] = float(pl.intensity)
            r = max(0.5, float(getattr(pl, "radius", 14.0)))
            buf[i, 4] = r * r
            buf[i, 5] = float(pl.attenuation)
        return n, buf

    @staticmethod
    def _sample_dynamic_lights_single(
        center: np.ndarray,
        normal: np.ndarray,
        n_dyn: int,
        dyn_buf: np.ndarray,
    ) -> float:
        if n_dyn <= 0:
            return 0.0
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
        boost = 0.0
        for i in range(n_dyn):
            dlx = float(dyn_buf[i, 0]) - cx
            dly = float(dyn_buf[i, 1]) - cy
            dlz = float(dyn_buf[i, 2]) - cz
            dist2 = dlx * dlx + dly * dly + dlz * dlz
            r2 = float(dyn_buf[i, 4])
            if dist2 >= r2:
                continue
            falloff = 1.0 - (dist2 / r2)
            att = 1.0 / (1.0 + float(dyn_buf[i, 5]) * dist2)
            inv_d = 1.0 / (math.sqrt(dist2) + 1e-4)
            nd = max(0.0, (nx * dlx + ny * dly + nz * dlz) * inv_d)
            boost += float(dyn_buf[i, 3]) * falloff * att * (0.45 + 0.55 * nd)
        return boost

    def render_scene(
        self,
        screen: pygame.Surface,
        camera: Camera3D,
        scene: Scene3D | None = None,
        voxel_world: VoxelWorld | ChunkedVoxelWorld | None = None,
        highlight_hit: VoxelHit | None = None,
        sky_dome: SkyDome3D | None = None,
        block_overlays: dict[tuple[int, int, int], Texture2D] | None = None,
        dynamic_lights: list[PointLight] | None = None,
    ) -> int:
        """
        Render both a `VoxelWorld` / `ChunkedVoxelWorld` and a `Scene3D` of `Mesh3D` instances onto `screen`.
        Optional `block_overlays` maps `(bx, by, bz) -> Texture2D` for Z-buffered voxel decals (e.g. damage cracks).
        Optional `dynamic_lights` supplies local `PointLight` emitters (e.g. torches, lanterns, lava) in addition to `self.light`.
        """
        if self.textured:
            if self.native_active:
                return self._render_scene_textured_native(
                    screen,
                    camera,
                    scene=scene,
                    voxel_world=voxel_world,
                    highlight_hit=highlight_hit,
                    sky_dome=sky_dome,
                    block_overlays=block_overlays,
                    dynamic_lights=dynamic_lights,
                )
            return self._render_scene_textured_numpy(
                screen,
                camera,
                scene=scene,
                voxel_world=voxel_world,
                highlight_hit=highlight_hit,
                sky_dome=sky_dome,
                block_overlays=block_overlays,
                dynamic_lights=dynamic_lights,
            )
        if sky_dome is not None:
            sky_dome.draw(screen, camera)
        return self._render_scene_solid_painter(
            screen,
            camera,
            scene=scene,
            voxel_world=voxel_world,
            highlight_hit=highlight_hit,
        )

    def _fetch_voxel_data(
        self,
        voxel_world: VoxelWorld | ChunkedVoxelWorld,
        cam_pos: np.ndarray,
        view_mat: np.ndarray,
        near_z: float,
        far_z: float,
        half_w: float,
        half_h: float,
        proj_scale: float,
        include_textures: bool = True,
    ) -> dict:
        if hasattr(voxel_world, "get_visible_render_data"):
            return voxel_world.get_visible_render_data(
                cam_pos, view_mat, near_z, far_z, half_w, half_h, proj_scale, include_textures=include_textures
            )
        return voxel_world.get_culled_render_data()

    def _render_scene_textured_native(
        self,
        screen: pygame.Surface,
        camera: Camera3D,
        scene: Scene3D | None = None,
        voxel_world: VoxelWorld | ChunkedVoxelWorld | None = None,
        highlight_hit: VoxelHit | None = None,
        sky_dome: SkyDome3D | None = None,
        block_overlays: dict[tuple[int, int, int], Texture2D] | None = None,
        dynamic_lights: list[PointLight] | None = None,
    ) -> int:
        """
        Zero-copy SDL + Single-Call C99 Batched Scanline Z-Buffer Pipeline.
        Eliminates all per-face Python loops and all full-screen array3d/blit_array transposes!
        """
        assert NATIVE_LIB is not None
        scr_w, scr_h = screen.get_size()
        self.width, self.height = scr_w, scr_h
        rast = self._get_rasterizer(scr_w, scr_h)
        rw, rh = rast.width, rast.height
        surf = rast.surface

        # 1. Draw sky directly onto the 32-bit SDL surface (zero array3d copy!)
        if sky_dome is not None:
            sky_dome.draw(surf, camera)
        else:
            if screen.get_size() != (rw, rh):
                pygame.transform.scale(screen, (rw, rh), surf)
            else:
                surf.blit(screen, (0, 0))

        half_w = rw * 0.5
        half_h = rh * 0.5
        focal = 1.0 / math.tan(math.radians(camera.fov) * 0.5)
        proj_scale = focal * half_h

        cam_pos = camera.position
        view_mat = np.ascontiguousarray(camera.get_view_matrix_3x3(), dtype=np.float32)
        near_z = camera.near
        far_z = camera.far
        fog_enabled = self.fog.enabled
        fog_near = self.fog.near
        fog_span = max(1e-5, self.fog.far - self.fog.near)
        fog_inv_span = 1.0 / fog_span
        fog_r = float(self.fog.color[0])
        fog_g = float(self.fog.color[1])
        fog_b = float(self.fog.color[2])
        lod_dist = self.texture_lod_distance

        shifts = surf.get_shifts()
        r_off = int(shifts[0]) >> 3
        g_off = int(shifts[1]) >> 3
        b_off = int(shifts[2]) >> 3
        pitch = int(surf.get_pitch())
        z_ptr = rast.z_buf.ctypes.data

        atlas: TextureAtlas = self._empty_atlas
        total_faces = 0
        straddle_tr: list[tuple] = []

        # Collect visible voxel chunks with ZERO np.concatenate & ZERO per-quad NumPy allocations!
        n_chunks = 0
        _keep_refs: list[Any] = []
        if voxel_world is not None:
            if hasattr(voxel_world, "get_visible_chunks"):
                vis_chunks, atlas = voxel_world.get_visible_chunks(
                    cam_pos,
                    view_mat,
                    near_z,
                    far_z,
                    half_w,
                    half_h,
                    proj_scale,
                    rebuild_budget_ms=self.chunk_rebuild_budget_ms,
                )
                n_chunks = len(vis_chunks)
                if n_chunks > 0:
                    self._ensure_chunk_ptr_bufs(n_chunks)
                    cq = self._c_quads
                    cn = self._c_normals
                    cc = self._c_colors
                    cu = self._c_uvs
                    ct = self._c_tids
                    ca = self._c_alphas
                    ce = self._c_emissive
                    cao = self._c_ao
                    ctr = self._c_is_tr
                    cnt = self._c_counts
                    for idx_c, ch in enumerate(vis_chunks):
                        cq[idx_c] = ch.quads.ctypes.data
                        cn[idx_c] = ch.normals.ctypes.data
                        cc[idx_c] = ch.colors.ctypes.data
                        cu[idx_c] = ch.uv_scales.ctypes.data
                        ct[idx_c] = ch.tex_ids.ctypes.data
                        ca[idx_c] = ch.alphas.ctypes.data
                        ce[idx_c] = ch.emissive.ctypes.data
                        cao[idx_c] = ch.ao.ctypes.data
                        ctr[idx_c] = ch.is_translucent.ctypes.data
                        cnt[idx_c] = len(ch.quads)
            else:
                data = voxel_world.get_culled_render_data()
                atlas = data.get("atlas", self._empty_atlas)
                quads_w = data["quads"]
                if len(quads_w) > 0:
                    self._ensure_chunk_ptr_bufs(1)
                    n_chunks = 1
                    q_c = np.ascontiguousarray(quads_w, dtype=np.float32)
                    n_c = np.ascontiguousarray(data["normals"], dtype=np.float32)
                    c_c = np.ascontiguousarray(data["colors"], dtype=np.float32)
                    uv_raw = data.get("uv_scales")
                    uv_c = np.ascontiguousarray(uv_raw, dtype=np.float32) if uv_raw is not None else None
                    t_c = np.ascontiguousarray(data["tex_ids"], dtype=np.int32)
                    a_c = np.ascontiguousarray(data["alphas"], dtype=np.uint8)
                    e_c = np.ascontiguousarray(data["emissive"], dtype=np.float32)
                    ao_c = np.ascontiguousarray(data["ao"], dtype=np.float32)
                    tr_c = np.ascontiguousarray(data["is_translucent"], dtype=np.uint8)
                    _keep_refs.extend((q_c, n_c, c_c, uv_c, t_c, a_c, e_c, ao_c, tr_c))
                    self._c_quads[0] = q_c.ctypes.data
                    self._c_normals[0] = n_c.ctypes.data
                    self._c_colors[0] = c_c.ctypes.data
                    self._c_uvs[0] = uv_c.ctypes.data if uv_c is not None else 0
                    self._c_tids[0] = t_c.ctypes.data
                    self._c_alphas[0] = a_c.ctypes.data
                    self._c_emissive[0] = e_c.ctypes.data
                    self._c_ao[0] = ao_c.ctypes.data
                    self._c_is_tr[0] = tr_c.ctypes.data
                    self._c_counts[0] = len(q_c)

        # Extract light scalars (supports both DirectionalLight and PointLight agnostically)
        if isinstance(self.light, PointLight):
            light_type = 1
            lx, ly, lz = float(self.light.position[0]), float(self.light.position[1]), float(self.light.position[2])
            l_ambient = float(self.light.ambient)
            l_diffuse = float(self.light.intensity)
            l_atten = float(self.light.attenuation)
        else:
            light_type = 0
            tl = self.light.to_light_dir
            lx, ly, lz = float(tl[0]), float(tl[1]), float(tl[2])
            l_ambient = float(self.light.ambient)
            l_diffuse = float(self.light.diffuse)
            l_atten = 0.0

        n_dyn_lights, dyn_buf = self._pack_dynamic_lights(cam_pos, dynamic_lights)

        # Prepare Scene3D Mesh3D polygons (supports both solid & textured Mesh3D!)
        mesh_polys: list[tuple] = []
        mesh_tr_polys: list[tuple] = []
        n_dyn_lights, dyn_buf = self._pack_dynamic_lights(cam_pos, dynamic_lights)
        if scene is not None and scene.meshes:
            for mesh in scene.meshes:
                if not mesh.visible or len(mesh.faces) == 0:
                    continue
                verts_w = mesh.get_world_vertices()
                verts_c = (verts_w - cam_pos[None, :]) @ view_mat
                mesh_alp = int(getattr(mesh, "alpha", 255))
                face_textures = getattr(mesh, "face_textures", None)
                face_uvs = getattr(mesh, "face_uvs", None)

                for f_idx, face in enumerate(mesh.faces):
                    poly_w = verts_w[list(face)]
                    center_w = poly_w.mean(axis=0)
                    normal_w = compute_face_normals(poly_w[0], poly_w[1], poly_w[2])
                    if mesh.backface_culling and not mesh.wireframe:
                        if float(np.dot(normal_w, center_w - cam_pos)) >= 0.0:
                            continue

                    poly_c = verts_c[list(face)]
                    z_c = poly_c[:, 2]
                    if float(np.max(z_c)) < near_z or float(np.min(z_c)) > far_z:
                        continue

                    f_tex = face_textures[f_idx] if face_textures is not None else None
                    f_uv = face_uvs[f_idx] if face_uvs is not None else None
                    if f_tex is not None and f_uv is None and len(poly_c) == 4:
                        f_uv = CANONICAL_QUAD_UV

                    if float(np.min(z_c)) < near_z:
                        if f_tex is not None and f_uv is not None:
                            poly_c, f_uv = clip_textured_polygon_near_plane(poly_c, f_uv, near_z)
                        else:
                            poly_c = clip_polygon_near_plane(poly_c, near_z)
                        if len(poly_c) < 3 or len(poly_c) > 8:
                            continue

                    d_val = float(np.linalg.norm(poly_c.mean(axis=0)))
                    z_s = np.maximum(poly_c[:, 2], near_z * 0.5)
                    iz_a = (1.0 / z_s).astype(np.float32)
                    px_a = (half_w + poly_c[:, 0] * iz_a * proj_scale).astype(np.float32)
                    py_a = (half_h - poly_c[:, 1] * iz_a * proj_scale).astype(np.float32)
                    if float(px_a.max()) < 0.0 or float(px_a.min()) >= rw or float(py_a.max()) < 0.0 or float(py_a.min()) >= rh:
                        continue

                    base_rgb = mesh.face_colors[f_idx]
                    intensity = float(self.light.compute_intensities(normal_w, center_w)[0])
                    if n_dyn_lights > 0:
                        intensity = min(1.45, intensity + self._sample_dynamic_lights_single(center_w, normal_w, n_dyn_lights, dyn_buf))
                    fog_f_m = min(1.0, max(0.0, (d_val - fog_near) / fog_span)) if fog_enabled else 0.0
                    shaded = self.fog.apply_to_colors(base_rgb * intensity, np.array([d_val], dtype=np.float32))

                    if f_tex is not None and f_uv is not None:
                        tid_m = atlas.register(f_tex)
                        u_a = np.ascontiguousarray(f_uv[:, 0], dtype=np.float32)
                        v_a = np.ascontiguousarray(f_uv[:, 1], dtype=np.float32)
                        is_tr_m = (mesh_alp < 255) or f_tex.has_cutout
                        entry_m = (
                            d_val,
                            np.ascontiguousarray(px_a),
                            np.ascontiguousarray(py_a),
                            np.ascontiguousarray(iz_a),
                            u_a,
                            v_a,
                            len(poly_c),
                            tid_m,
                            max(0.25, min(1.25, intensity * 1.05)),
                            fog_f_m,
                            int(shaded[0]),
                            int(shaded[1]),
                            int(shaded[2]),
                            mesh_alp,
                        )
                        if is_tr_m:
                            mesh_tr_polys.append(entry_m)
                        else:
                            mesh_polys.append(entry_m)
                    else:
                        entry_m = (
                            d_val,
                            np.ascontiguousarray(px_a),
                            np.ascontiguousarray(py_a),
                            np.ascontiguousarray(iz_a),
                            None,
                            None,
                            len(poly_c),
                            -1,
                            1.0,
                            0.0,
                            int(shaded[0]),
                            int(shaded[1]),
                            int(shaded[2]),
                            mesh_alp,
                        )
                        if mesh_alp < 255:
                            mesh_tr_polys.append(entry_m)
                        else:
                            mesh_polys.append(entry_m)
                    total_faces += 1

        # Prepare Z-buffered Voxel Block Overlays / Decals (e.g. damage cracks, frost, runes)
        if block_overlays:
            pad = 0.008
            for (bx, by, bz), ov_tex in block_overlays.items():
                if ov_tex is None:
                    continue
                tid_ov = atlas.register(ov_tex)
                base_off = np.array([float(bx), float(by), float(bz)], dtype=np.float32)
                for (nx, ny, nz), unit_quad in _VOXEL_FACES:
                    n_vec = np.array([float(nx), float(ny), float(nz)], dtype=np.float32)
                    q_w = unit_quad + base_off[None, :] + n_vec[None, :] * pad
                    cent_w = q_w.mean(axis=0)
                    if float(np.dot(n_vec, cent_w - cam_pos)) >= -1e-5:
                        continue
                    q_c = (q_w - cam_pos[None, :]) @ view_mat
                    z_c = q_c[:, 2]
                    if float(np.max(z_c)) < near_z or float(np.min(z_c)) > far_z:
                        continue
                    ov_uv = CANONICAL_QUAD_UV
                    if float(np.min(z_c)) < near_z:
                        q_c, ov_uv = clip_textured_polygon_near_plane(q_c, ov_uv, near_z)
                        if len(q_c) < 3 or len(q_c) > 8:
                            continue
                    d_val = float(np.linalg.norm(q_c.mean(axis=0)))
                    z_s = np.maximum(q_c[:, 2], near_z * 0.5)
                    iz_a = (1.0 / z_s).astype(np.float32)
                    px_a = (half_w + q_c[:, 0] * iz_a * proj_scale).astype(np.float32)
                    py_a = (half_h - q_c[:, 1] * iz_a * proj_scale).astype(np.float32)
                    if float(px_a.max()) < 0.0 or float(px_a.min()) >= rw or float(py_a.max()) < 0.0 or float(py_a.min()) >= rh:
                        continue
                    straddle_tr.append((
                        d_val,
                        np.ascontiguousarray(px_a),
                        np.ascontiguousarray(py_a),
                        np.ascontiguousarray(iz_a),
                        np.ascontiguousarray(ov_uv[:, 0], dtype=np.float32),
                        np.ascontiguousarray(ov_uv[:, 1], dtype=np.float32),
                        len(q_c),
                        tid_ov,
                        1.0,
                        0.0,
                        255,
                        255,
                        255,
                        255,
                    ))
                    total_faces += 1

        # 2. Lock SDL pixel memory directly and execute C99 native passes!
        atlas.ensure_packed()
        a_rgb_ptr = atlas.rgb_flat.ctypes.data
        a_alp_ptr = atlas.alpha_flat.ctypes.data
        a_off_ptr = atlas.offsets.ctypes.data
        a_w_ptr = atlas.widths.ctypes.data
        a_h_ptr = atlas.heights.ctypes.data
        a_cut_ptr = atlas.has_cutout.ctypes.data

        surf_buf = surf.get_buffer()
        fb_arr = np.frombuffer(surf_buf, dtype=np.uint8)
        fb_ptr = fb_arr.ctypes.data
        try:
            # Pass 1: ALL Visible Voxel Chunks (Transform + Cull + Light + Clip + Multi-Threaded Band Z-Buffer!)
            if n_chunks > 0:
                drawn_voxels = NATIVE_LIB.pf3d_render_voxel_chunks(
                    fb_ptr, pitch, z_ptr, rw, rh, r_off, g_off, b_off,
                    int(n_chunks),
                    self._c_quads.ctypes.data,
                    self._c_normals.ctypes.data,
                    self._c_colors.ctypes.data,
                    self._c_uvs.ctypes.data,
                    self._c_tids.ctypes.data,
                    self._c_alphas.ctypes.data,
                    self._c_emissive.ctypes.data,
                    self._c_ao.ctypes.data,
                    self._c_is_tr.ctypes.data,
                    self._c_counts.ctypes.data,
                    float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]),
                    view_mat.ctypes.data,
                    float(near_z), float(far_z),
                    float(half_w), float(half_h), float(proj_scale),
                    float(lod_dist),
                    int(light_type),
                    lx, ly, lz,
                    l_ambient, l_diffuse, l_atten,
                    1 if fog_enabled else 0, float(fog_near), float(fog_inv_span),
                    fog_r, fog_g, fog_b,
                    a_rgb_ptr, a_alp_ptr, a_off_ptr, a_w_ptr, a_h_ptr, a_cut_ptr,
                    1,  # parallel Z-buffer clear across threads!
                    int(n_dyn_lights),
                    dyn_buf.ctypes.data if n_dyn_lights > 0 else 0,
                )
                total_faces += int(drawn_voxels)
            else:
                NATIVE_LIB.pf3d_clear_zbuf(z_ptr, rast.z_buf.size)

            # Pass 2: Scene3D Opaque Meshes (Dropped items, entities, custom 3D models)
            if mesh_polys:
                mesh_polys.sort(key=lambda e: e[0])
                for _, px_a, py_a, iz_a, u_a, v_a, nv, tid, sh, ff, sr, sg, sb, alp in mesh_polys:
                    NATIVE_LIB.pf3d_rasterize_polygon(
                        fb_ptr, pitch, z_ptr, rw, rh, r_off, g_off, b_off,
                        px_a.ctypes.data, py_a.ctypes.data, iz_a.ctypes.data,
                        u_a.ctypes.data if u_a is not None else 0,
                        v_a.ctypes.data if v_a is not None else 0,
                        nv, tid, sh, ff,
                        fog_r, fog_g, fog_b, sr, sg, sb, alp,
                        a_rgb_ptr, a_alp_ptr, a_off_ptr, a_w_ptr, a_h_ptr, a_cut_ptr,
                    )

            # Pass 3: Translucent Meshes & Voxel Block Overlays / Decals (mining cracks, runes)
            if mesh_tr_polys:
                straddle_tr.extend(mesh_tr_polys)
            if straddle_tr:
                straddle_tr.sort(key=lambda e: e[0], reverse=True)
                for _, px_a, py_a, iz_a, u_a, v_a, nv, tid, sh, ff, sr, sg, sb, alp in straddle_tr:
                    NATIVE_LIB.pf3d_rasterize_polygon(
                        fb_ptr, pitch, z_ptr, rw, rh, r_off, g_off, b_off,
                        px_a.ctypes.data, py_a.ctypes.data, iz_a.ctypes.data,
                        u_a.ctypes.data if u_a is not None else 0,
                        v_a.ctypes.data if v_a is not None else 0,
                        nv, tid, sh, ff,
                        fog_r, fog_g, fog_b, sr, sg, sb, alp,
                        a_rgb_ptr, a_alp_ptr, a_off_ptr, a_w_ptr, a_h_ptr, a_cut_ptr,
                    )
        finally:
            del fb_arr
            del surf_buf

        # 3. Direct SDL blit/scale (zero surfarray transpose!)
        if (rw, rh) == (scr_w, scr_h):
            screen.blit(surf, (0, 0))
        else:
            pygame.transform.scale(surf, (scr_w, scr_h), screen)

        if highlight_hit is not None:
            self.draw_voxel_highlight(screen, camera, highlight_hit.block_pos)

        self.last_rendered_faces = total_faces
        return total_faces

    def _render_scene_textured_numpy(
        self,
        screen: pygame.Surface,
        camera: Camera3D,
        scene: Scene3D | None = None,
        voxel_world: VoxelWorld | ChunkedVoxelWorld | None = None,
        highlight_hit: VoxelHit | None = None,
        sky_dome: SkyDome3D | None = None,
        block_overlays: dict[tuple[int, int, int], Texture2D] | None = None,
        dynamic_lights: list[PointLight] | None = None,
    ) -> int:
        """100% Pure-NumPy portable fallback Z-Buffer textured pipeline (used when C99 binary is unavailable/disabled)."""
        scr_w, scr_h = screen.get_size()
        self.width, self.height = scr_w, scr_h
        rast = self._get_rasterizer(scr_w, scr_h)
        rw, rh = rast.width, rast.height

        if sky_dome is not None:
            sky_dome.draw(rast.surface, camera)
            rast.begin_frame(bg_surface=rast.surface)
        else:
            rast.begin_frame(bg_surface=screen)

        half_w = rw * 0.5
        half_h = rh * 0.5
        focal = 1.0 / math.tan(math.radians(camera.fov) * 0.5)
        proj_scale = focal * half_h

        cam_pos = camera.position
        view_mat = camera.get_view_matrix_3x3()
        near_z = camera.near
        far_z = camera.far
        fog_enabled = self.fog.enabled
        fog_near = self.fog.near
        fog_span = max(1e-5, self.fog.far - self.fog.near)
        fog_color = self.fog.color
        lod_dist = self.texture_lod_distance

        opaque_items: list[tuple] = []
        translucent_items: list[tuple] = []

        if voxel_world is not None:
            data = self._fetch_voxel_data(
                voxel_world, cam_pos, view_mat, near_z, far_z, half_w, half_h, proj_scale
            )
            quads_w = data["quads"]
            if len(quads_w) > 0:
                normals_w = data["normals"]
                colors_w = data["colors"]
                textures_w = data["textures"]
                alphas_w = data["alphas"]
                emissive_w = data["emissive"]
                ao_w = data["ao"]

                centers_w = quads_w.mean(axis=1)
                to_face = centers_w - cam_pos[None, :]
                front_idx = np.flatnonzero(np.einsum("ij,ij->i", normals_w, to_face) < -1e-5)

                if len(front_idx) > 0:
                    q_vis = quads_w[front_idx]
                    n_vis = normals_w[front_idx]
                    c_vis = colors_w[front_idx]
                    cent_vis = centers_w[front_idx]
                    a_vis = alphas_w[front_idx]
                    em_vis = emissive_w[front_idx]
                    ao_vis = ao_w[front_idx]

                    k_vis = len(q_vis)
                    q_cam = ((q_vis.reshape(-1, 3) - cam_pos[None, :]) @ view_mat).reshape(k_vis, 4, 3)

                    z_vals = q_cam[:, :, 2]
                    max_z = np.max(z_vals, axis=1)
                    min_z = np.min(z_vals, axis=1)
                    x_vals = q_cam[:, :, 0]
                    y_vals = q_cam[:, :, 1]
                    max_z_pos = np.maximum(max_z, 1.0)
                    max_x_allow = max_z_pos * ((half_w / proj_scale) + 0.55) + 1.4
                    max_y_allow = max_z_pos * ((half_h / proj_scale) + 0.55) + 1.4
                    keep_sub = np.flatnonzero(
                        (max_z >= near_z)
                        & (min_z <= far_z)
                        & (np.min(x_vals, axis=1) <= max_x_allow)
                        & (np.max(x_vals, axis=1) >= -max_x_allow)
                        & (np.min(y_vals, axis=1) <= max_y_allow)
                        & (np.max(y_vals, axis=1) >= -max_y_allow)
                    )

                    if len(keep_sub) > 0:
                        orig_indices = front_idx[keep_sub]
                        q_cam = q_cam[keep_sub]
                        n_vis = n_vis[keep_sub]
                        c_vis = c_vis[keep_sub]
                        cent_vis = cent_vis[keep_sub]
                        a_vis = a_vis[keep_sub]
                        em_vis = em_vis[keep_sub]
                        ao_vis = ao_vis[keep_sub]
                        min_z = min_z[keep_sub]

                        depths = np.linalg.norm(q_cam.mean(axis=1), axis=1)
                        intensities = self.light.compute_intensities(n_vis, cent_vis)
                        n_dyn_np, dyn_buf_np = self._pack_dynamic_lights(cam_pos, dynamic_lights)
                        if n_dyn_np > 0:
                            q_min = np.min(quads_w[orig_indices], axis=1)
                            q_max = np.max(quads_w[orig_indices], axis=1)
                            for li in range(n_dyn_np):
                                lp = dyn_buf_np[li, :3]
                                samp = 0.5 * (cent_vis + np.clip(lp[None, :], q_min, q_max))
                                dv = lp[None, :] - samp
                                d2 = np.einsum("ij,ij->i", dv, dv)
                                r2 = float(dyn_buf_np[li, 4])
                                active = d2 < r2
                                if np.any(active):
                                    d2_a = d2[active]
                                    fall = 1.0 - d2_a / r2
                                    att = 1.0 / (1.0 + float(dyn_buf_np[li, 5]) * d2_a)
                                    inv_d = 1.0 / (np.sqrt(d2_a) + 1e-4)
                                    nd = np.maximum(0.0, np.einsum("ij,ij->i", n_vis[active], dv[active]) * inv_d)
                                    intensities[active] += float(dyn_buf_np[li, 3]) * fall * att * (0.45 + 0.55 * nd)
                        face_bias = np.where(
                            n_vis[:, 1] > 0.5,
                            1.06,
                            np.where(
                                n_vis[:, 1] < -0.5,
                                0.64,
                                np.where(np.abs(n_vis[:, 0]) > 0.5, 0.84, 0.94),
                            ),
                        ).astype(np.float32)
                        shades = np.clip(intensities * face_bias * ao_vis + em_vis, 0.15, 1.45)
                        fog_factors = (
                            np.clip((depths - fog_near) / fog_span, 0.0, 1.0)
                            if fog_enabled
                            else np.zeros_like(depths)
                        )
                        lit_colors = c_vis * shades[:, None]
                        final_colors = self.fog.apply_to_colors(lit_colors, depths).astype(np.int32)

                        all_in_front = min_z >= near_z
                        fast_sub = np.flatnonzero(all_in_front)
                        if len(fast_sub) > 0:
                            q_fast = q_cam[fast_sub]
                            inv_z_fast = 1.0 / np.maximum(q_fast[:, :, 2], near_z * 0.5)
                            sx_fast = half_w + q_fast[:, :, 0] * inv_z_fast * proj_scale
                            sy_fast = half_h - q_fast[:, :, 1] * inv_z_fast * proj_scale

                            valid_fast = np.flatnonzero(
                                (np.max(sx_fast, axis=1) >= 0.0)
                                & (np.min(sx_fast, axis=1) < rw)
                                & (np.max(sy_fast, axis=1) >= 0.0)
                                & (np.min(sy_fast, axis=1) < rh)
                            )
                            uv_list = [(0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]

                            uv_scales_w = data.get("uv_scales")
                            for k in valid_fast:
                                local_i = fast_sub[k]
                                orig_i = orig_indices[local_i]
                                d_val = float(depths[local_i])
                                tex = textures_w[orig_i] if d_val <= lod_dist else None
                                alp = int(a_vis[local_i])
                                uvs_pair = (float(uv_scales_w[orig_i, 0]), float(uv_scales_w[orig_i, 1])) if uv_scales_w is not None else (1.0, 1.0)
                                pts_invz = [
                                    (float(sx_fast[k, 0]), float(sy_fast[k, 0]), float(inv_z_fast[k, 0])),
                                    (float(sx_fast[k, 1]), float(sy_fast[k, 1]), float(inv_z_fast[k, 1])),
                                    (float(sx_fast[k, 2]), float(sy_fast[k, 2]), float(inv_z_fast[k, 2])),
                                    (float(sx_fast[k, 3]), float(sy_fast[k, 3]), float(inv_z_fast[k, 3])),
                                ]
                                rgb = (
                                    int(final_colors[local_i, 0]),
                                    int(final_colors[local_i, 1]),
                                    int(final_colors[local_i, 2]),
                                )
                                item = (
                                    d_val, pts_invz, uv_list if tex is not None else None,
                                    tex, rgb, float(shades[local_i]), float(fog_factors[local_i]), alp, tex is not None, uvs_pair,
                                )
                                if alp < 255 or (tex is not None and tex.has_cutout):
                                    translucent_items.append(item)
                                else:
                                    opaque_items.append(item)

                        straddle_sub = np.flatnonzero(~all_in_front)
                        uv_scales_w = data.get("uv_scales")
                        for local_i in straddle_sub:
                            orig_i = orig_indices[local_i]
                            scaled_uv = CANONICAL_QUAD_UV * uv_scales_w[orig_i][None, :] if uv_scales_w is not None else CANONICAL_QUAD_UV
                            clipped_p, clipped_uv = clip_textured_polygon_near_plane(
                                q_cam[local_i], scaled_uv, near_z
                            )
                            if len(clipped_p) < 3:
                                continue
                            pts_invz = self._project_point_list_invz(clipped_p, proj_scale, half_w, half_h, near_z)
                            xs = [p[0] for p in pts_invz]
                            ys = [p[1] for p in pts_invz]
                            if max(xs) < 0.0 or min(xs) >= rw or max(ys) < 0.0 or min(ys) >= rh:
                                continue
                            d_val = float(depths[local_i])
                            tex = textures_w[orig_i] if d_val <= lod_dist else None
                            alp = int(a_vis[local_i])
                            uv_l = [(float(u[0]), float(u[1])) for u in clipped_uv] if tex is not None else None
                            rgb = (
                                int(final_colors[local_i, 0]),
                                int(final_colors[local_i, 1]),
                                int(final_colors[local_i, 2]),
                            )
                            item = (
                                d_val, pts_invz, uv_l, tex, rgb,
                                float(shades[local_i]), float(fog_factors[local_i]), alp, False, (1.0, 1.0),
                            )
                            if alp < 255 or (tex is not None and tex.has_cutout):
                                translucent_items.append(item)
                            else:
                                opaque_items.append(item)

        n_dyn_lights, dyn_buf = self._pack_dynamic_lights(cam_pos, dynamic_lights)
        if scene is not None and scene.meshes:
            for mesh in scene.meshes:
                if not mesh.visible or len(mesh.faces) == 0:
                    continue
                verts_w = mesh.get_world_vertices()
                verts_c = (verts_w - cam_pos[None, :]) @ view_mat
                mesh_alp = int(getattr(mesh, "alpha", 255))
                face_textures = getattr(mesh, "face_textures", None)
                face_uvs = getattr(mesh, "face_uvs", None)

                for f_idx, face in enumerate(mesh.faces):
                    poly_w = verts_w[list(face)]
                    center_w = poly_w.mean(axis=0)
                    normal_w = compute_face_normals(poly_w[0], poly_w[1], poly_w[2])
                    if mesh.backface_culling and not mesh.wireframe:
                        if float(np.dot(normal_w, center_w - cam_pos)) >= 0.0:
                            continue

                    poly_c = verts_c[list(face)]
                    z_c = poly_c[:, 2]
                    if float(np.max(z_c)) < near_z or float(np.min(z_c)) > far_z:
                        continue

                    f_tex = face_textures[f_idx] if face_textures is not None else None
                    f_uv = face_uvs[f_idx] if face_uvs is not None else None
                    if f_tex is not None and f_uv is None and len(poly_c) == 4:
                        f_uv = CANONICAL_QUAD_UV

                    if float(np.min(z_c)) < near_z:
                        if f_tex is not None and f_uv is not None:
                            poly_c, f_uv = clip_textured_polygon_near_plane(poly_c, f_uv, near_z)
                        else:
                            poly_c = clip_polygon_near_plane(poly_c, near_z)
                        if len(poly_c) < 3:
                            continue

                    d_val = float(np.linalg.norm(poly_c.mean(axis=0)))
                    pts_invz = self._project_point_list_invz(poly_c, proj_scale, half_w, half_h, near_z)
                    xs = [p[0] for p in pts_invz]
                    ys = [p[1] for p in pts_invz]
                    if max(xs) < 0.0 or min(xs) >= rw or max(ys) < 0.0 or min(ys) >= rh:
                        continue

                    base_rgb = mesh.face_colors[f_idx]
                    intensity = float(self.light.compute_intensities(normal_w, center_w)[0])
                    if n_dyn_lights > 0:
                        intensity = min(1.45, intensity + self._sample_dynamic_lights_single(center_w, normal_w, n_dyn_lights, dyn_buf))
                    fog_f_m = min(1.0, max(0.0, (d_val - fog_near) / fog_span)) if fog_enabled else 0.0
                    shaded = self.fog.apply_to_colors(base_rgb * intensity, np.array([d_val], dtype=np.float32))
                    rgb_tuple = (int(shaded[0]), int(shaded[1]), int(shaded[2]))
                    if f_tex is not None and f_uv is not None:
                        uv_l = [(float(u[0]), float(u[1])) for u in f_uv]
                        item_m = (d_val, pts_invz, uv_l, f_tex, rgb_tuple, max(0.25, min(1.25, intensity * 1.05)), fog_f_m, mesh_alp, False, (1.0, 1.0))
                        if mesh_alp < 255 or f_tex.has_cutout:
                            translucent_items.append(item_m)
                        else:
                            opaque_items.append(item_m)
                    else:
                        item_m = (d_val, pts_invz, None, None, rgb_tuple, 1.0, 0.0, mesh_alp, False, (1.0, 1.0))
                        if mesh_alp < 255:
                            translucent_items.append(item_m)
                        else:
                            opaque_items.append(item_m)

        if block_overlays:
            pad = 0.008
            for (bx, by, bz), ov_tex in block_overlays.items():
                if ov_tex is None:
                    continue
                base_off = np.array([float(bx), float(by), float(bz)], dtype=np.float32)
                for (nx, ny, nz), unit_quad in _VOXEL_FACES:
                    n_vec = np.array([float(nx), float(ny), float(nz)], dtype=np.float32)
                    q_w = unit_quad + base_off[None, :] + n_vec[None, :] * pad
                    cent_w = q_w.mean(axis=0)
                    if float(np.dot(n_vec, cent_w - cam_pos)) >= -1e-5:
                        continue
                    q_c = (q_w - cam_pos[None, :]) @ view_mat
                    z_c = q_c[:, 2]
                    if float(np.max(z_c)) < near_z or float(np.min(z_c)) > far_z:
                        continue
                    ov_uv = CANONICAL_QUAD_UV
                    if float(np.min(z_c)) < near_z:
                        q_c, ov_uv = clip_textured_polygon_near_plane(q_c, ov_uv, near_z)
                        if len(q_c) < 3:
                            continue
                    d_val = float(np.linalg.norm(q_c.mean(axis=0)))
                    pts_invz = self._project_point_list_invz(q_c, proj_scale, half_w, half_h, near_z)
                    xs = [p[0] for p in pts_invz]
                    ys = [p[1] for p in pts_invz]
                    if max(xs) < 0.0 or min(xs) >= rw or max(ys) < 0.0 or min(ys) >= rh:
                        continue
                    uv_l = [(float(u[0]), float(u[1])) for u in ov_uv]
                    translucent_items.append((d_val, pts_invz, uv_l, ov_tex, (255, 255, 255), 1.0, 0.0, 255, False, (1.0, 1.0)))

        draw_poly = rast.draw_planar_polygon
        opaque_items.sort(key=lambda it: it[0])
        for _, pts_invz, uvs, tex, rgb, shade, fog_f, alp, is_canon, uvs_pair in opaque_items:
            draw_poly(
                pts_invz, uvs=uvs, texture=tex, solid_color=rgb,
                shade=shade, fog_factor=fog_f, fog_color=fog_color, alpha=255, canonical_uv=is_canon, uv_scale=uvs_pair,
            )

        translucent_items.sort(key=lambda it: it[0], reverse=True)
        for _, pts_invz, uvs, tex, rgb, shade, fog_f, alp, is_canon, uvs_pair in translucent_items:
            draw_poly(
                pts_invz, uvs=uvs, texture=tex, solid_color=rgb,
                shade=shade, fog_factor=fog_f, fog_color=fog_color, alpha=alp, canonical_uv=is_canon, uv_scale=uvs_pair,
            )

        rast.blit_to_screen(screen)

        if highlight_hit is not None:
            self.draw_voxel_highlight(screen, camera, highlight_hit.block_pos)

        self.last_rendered_faces = len(opaque_items) + len(translucent_items)
        return self.last_rendered_faces

    def _render_scene_solid_painter(
        self,
        screen: pygame.Surface,
        camera: Camera3D,
        scene: Scene3D | None = None,
        voxel_world: VoxelWorld | ChunkedVoxelWorld | None = None,
        highlight_hit: VoxelHit | None = None,
    ) -> int:
        scr_w, scr_h = screen.get_size()
        self.width, self.height = scr_w, scr_h
        half_w = scr_w * 0.5
        half_h = scr_h * 0.5
        focal = 1.0 / math.tan(math.radians(camera.fov) * 0.5)
        proj_scale = focal * half_h

        cam_pos = camera.position
        view_mat = camera.get_view_matrix_3x3()
        near_z = camera.near
        far_z = camera.far

        render_queue: list[tuple[float, list[tuple[int, int]], tuple[int, int, int], tuple[int, int, int] | None, bool]] = []

        if voxel_world is not None:
            data = self._fetch_voxel_data(
                voxel_world, cam_pos, view_mat, near_z, far_z, half_w, half_h, proj_scale
            )
            quads_w, normals_w, colors_w = data["quads"], data["normals"], data["colors"]
            if len(quads_w) > 0:
                centers_w = quads_w.mean(axis=1)
                to_face = centers_w - cam_pos[None, :]
                front_mask = np.einsum("ij,ij->i", normals_w, to_face) < -1e-5

                if np.any(front_mask):
                    q_vis = quads_w[front_mask]
                    n_vis = normals_w[front_mask]
                    c_vis = colors_w[front_mask]
                    cent_vis = centers_w[front_mask]

                    k_vis = len(q_vis)
                    q_cam = ((q_vis.reshape(-1, 3) - cam_pos[None, :]) @ view_mat).reshape(k_vis, 4, 3)

                    z_vals = q_cam[:, :, 2]
                    max_z = np.max(z_vals, axis=1)
                    min_z = np.min(z_vals, axis=1)
                    x_vals = q_cam[:, :, 0]
                    max_x_allow = np.maximum(max_z, 1.0) * ((half_w / proj_scale) + 0.6) + 1.5
                    in_horiz = (np.min(x_vals, axis=1) <= max_x_allow) & (np.max(x_vals, axis=1) >= -max_x_allow)
                    depth_mask = (max_z >= near_z) & (min_z <= far_z) & in_horiz

                    if np.any(depth_mask):
                        q_cam = q_cam[depth_mask]
                        n_vis = n_vis[depth_mask]
                        c_vis = c_vis[depth_mask]
                        cent_vis = cent_vis[depth_mask]
                        min_z = min_z[depth_mask]

                        cam_centers = q_cam.mean(axis=1)
                        depths = np.linalg.norm(cam_centers, axis=1)

                        intensities = self.light.compute_intensities(n_vis, cent_vis)
                        face_bias = np.where(
                            n_vis[:, 1] > 0.5,
                            1.06,
                            np.where(
                                n_vis[:, 1] < -0.5,
                                0.65,
                                np.where(np.abs(n_vis[:, 0]) > 0.5, 0.84, 0.94),
                            ),
                        ).astype(np.float32)
                        lit_colors = c_vis * (intensities * face_bias)[:, None]
                        final_colors = self.fog.apply_to_colors(lit_colors, depths).astype(np.int32)

                        all_in_front = min_z >= near_z
                        if np.any(all_in_front):
                            q_fast = q_cam[all_in_front]
                            d_fast = depths[all_in_front]
                            col_fast = final_colors[all_in_front]

                            z_safe = q_fast[:, :, 2]
                            sx = np.rint(half_w + (q_fast[:, :, 0] / z_safe) * proj_scale).astype(np.int32)
                            sy = np.rint(half_h - (q_fast[:, :, 1] / z_safe) * proj_scale).astype(np.int32)

                            on_screen = (
                                (np.max(sx, axis=1) >= 0)
                                & (np.min(sx, axis=1) < scr_w)
                                & (np.max(sy, axis=1) >= 0)
                                & (np.min(sy, axis=1) < scr_h)
                            )
                            sx = sx[on_screen]
                            sy = sy[on_screen]
                            d_fast = d_fast[on_screen]
                            col_fast = col_fast[on_screen]

                            outline_c = self.voxel_outline_color
                            for i in range(len(d_fast)):
                                pts = [
                                    (int(sx[i, 0]), int(sy[i, 0])),
                                    (int(sx[i, 1]), int(sy[i, 1])),
                                    (int(sx[i, 2]), int(sy[i, 2])),
                                    (int(sx[i, 3]), int(sy[i, 3])),
                                ]
                                rgb = (int(col_fast[i, 0]), int(col_fast[i, 1]), int(col_fast[i, 2]))
                                render_queue.append((float(d_fast[i]), pts, rgb, outline_c, False))

                        straddle_indices = np.flatnonzero(~all_in_front)
                        for idx in straddle_indices:
                            clipped = clip_polygon_near_plane(q_cam[idx], near_z)
                            if len(clipped) >= 3:
                                pts = self._project_point_list(clipped, proj_scale, half_w, half_h, near_z)
                                xs = [p[0] for p in pts]
                                ys = [p[1] for p in pts]
                                if max(xs) >= 0 and min(xs) < scr_w and max(ys) >= 0 and min(ys) < scr_h:
                                    rgb = (
                                        int(final_colors[idx, 0]),
                                        int(final_colors[idx, 1]),
                                        int(final_colors[idx, 2]),
                                    )
                                    render_queue.append((float(depths[idx]), pts, rgb, self.voxel_outline_color, False))

        if scene is not None and scene.meshes:
            for mesh in scene.meshes:
                if not mesh.visible or len(mesh.faces) == 0:
                    continue
                verts_w = mesh.get_world_vertices()
                verts_c = (verts_w - cam_pos[None, :]) @ view_mat

                for f_idx, face in enumerate(mesh.faces):
                    poly_w = verts_w[list(face)]
                    center_w = poly_w.mean(axis=0)
                    normal_w = compute_face_normals(poly_w[0], poly_w[1], poly_w[2])

                    if mesh.backface_culling and not mesh.wireframe:
                        if float(np.dot(normal_w, center_w - cam_pos)) >= 0.0:
                            continue

                    poly_c = verts_c[list(face)]
                    z_c = poly_c[:, 2]
                    if float(np.max(z_c)) < near_z or float(np.min(z_c)) > far_z:
                        continue

                    if float(np.min(z_c)) < near_z:
                        poly_c = clip_polygon_near_plane(poly_c, near_z)
                        if len(poly_c) < 3:
                            continue

                    depth = float(np.linalg.norm(poly_c.mean(axis=0)))
                    pts = self._project_point_list(poly_c, proj_scale, half_w, half_h, near_z)
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    if max(xs) < 0 or min(xs) >= scr_w or max(ys) < 0 or min(ys) >= scr_h:
                        continue

                    base_rgb = mesh.face_colors[f_idx]
                    intensity = float(self.light.compute_intensities(normal_w, center_w)[0])
                    lit_rgb = base_rgb * intensity
                    shaded = self.fog.apply_to_colors(lit_rgb, np.array([depth], dtype=np.float32))
                    rgb_tuple = (int(shaded[0]), int(shaded[1]), int(shaded[2]))

                    render_queue.append((depth, pts, rgb_tuple, mesh.outline_color, mesh.wireframe))

        render_queue.sort(key=lambda item: item[0], reverse=True)

        for _, pts, fill_rgb, outline_rgb, is_wireframe in render_queue:
            if is_wireframe:
                pygame.draw.polygon(screen, outline_rgb or fill_rgb, pts, 1)
            else:
                pygame.draw.polygon(screen, fill_rgb, pts)
                if outline_rgb is not None:
                    pygame.draw.polygon(screen, outline_rgb, pts, 1)

        if highlight_hit is not None:
            self.draw_voxel_highlight(screen, camera, highlight_hit.block_pos)

        self.last_rendered_faces = len(render_queue)
        return self.last_rendered_faces

    def draw_voxel_highlight(
        self,
        screen: pygame.Surface,
        camera: Camera3D,
        block_pos: tuple[int, int, int],
        color: tuple[int, int, int] = (20, 20, 20),
        width: int = 2,
    ) -> None:
        """Draws a crisp 3D wireframe bounding box around a targeted voxel block at (bx, by, bz)."""
        bx, by, bz = float(block_pos[0]), float(block_pos[1]), float(block_pos[2])
        pad = 0.006
        x0, x1 = bx - pad, bx + 1.0 + pad
        y0, y1 = by - pad, by + 1.0 + pad
        z0, z1 = bz - pad, bz + 1.0 + pad

        corners_w = np.array(
            [
                [x0, y0, z0],
                [x0, y1, z0],
                [x1, y1, z0],
                [x1, y0, z0],
                [x0, y0, z1],
                [x0, y1, z1],
                [x1, y1, z1],
                [x1, y0, z1],
            ],
            dtype=np.float32,
        )
        corners_c = camera.world_to_camera(corners_w)
        if float(np.min(corners_c[:, 2])) < camera.near:
            return
        scr_w, scr_h = screen.get_size()
        pts_2d = camera.project_camera_vertices(corners_c, scr_w, scr_h)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i0, i1 in edges:
            p0 = (int(round(float(pts_2d[i0, 0]))), int(round(float(pts_2d[i0, 1]))))
            p1 = (int(round(float(pts_2d[i1, 0]))), int(round(float(pts_2d[i1, 1]))))
            pygame.draw.line(screen, color, p0, p1, width)
