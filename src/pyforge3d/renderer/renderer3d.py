from __future__ import annotations

import math
import numpy as np
import pygame
from ..camera.camera3d import Camera3D
from ..lighting.lights import DirectionalLight, Fog3D, PointLight
from ..lighting.sky_dome import SkyDome3D
from ..math.transforms import clip_polygon_near_plane, compute_face_normals
from ..meshes.mesh3d import Mesh3D
from ..voxels.chunked_world import ChunkedVoxelWorld
from ..voxels.voxel_world import VoxelHit, VoxelWorld
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
    High-performance Vectorized Software 3D Renderer for Pygame.
    Supports both:
    1. **Textured Z-Buffer Software Pipeline** (`textured=True`): Perspective-correct 2D texture mapping,
       per-pixel 1/Z depth occlusion, translucent alpha blending (water/glass), AO, and distance fog.
    2. **Fast Solid Polygon Pipeline** (`textured=False`): Painter's Algorithm depth-sorted polygons.
    Works seamlessly with both `VoxelWorld` and infinite `ChunkedVoxelWorld`, plus `Scene3D` meshes and `SkyDome3D`.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        light: DirectionalLight | PointLight | None = None,
        fog: Fog3D | None = None,
        voxel_outline_color: tuple[int, int, int] | None = (18, 28, 48),
        textured: bool = False,
        render_scale: float = 0.45,
        texture_lod_distance: float = 20.0,
    ) -> None:
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.light: DirectionalLight | PointLight = light if light is not None else DirectionalLight()
        self.fog: Fog3D = fog if fog is not None else Fog3D(enabled=True)
        self.voxel_outline_color: tuple[int, int, int] | None = voxel_outline_color
        self.textured: bool = bool(textured)
        self.render_scale: float = max(0.2, min(1.0, float(render_scale)))
        self.texture_lod_distance: float = max(4.0, float(texture_lod_distance))
        self.last_rendered_faces: int = 0
        self._sky_cache_key = None
        self._sky_cache_surf: pygame.Surface | None = None
        self._rasterizer: SoftwareTexturedRasterizer | None = None

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

    def render_scene(
        self,
        screen: pygame.Surface,
        camera: Camera3D,
        scene: Scene3D | None = None,
        voxel_world: VoxelWorld | ChunkedVoxelWorld | None = None,
        highlight_hit: VoxelHit | None = None,
        sky_dome: SkyDome3D | None = None,
    ) -> int:
        """
        Render both a `VoxelWorld` / `ChunkedVoxelWorld` and a `Scene3D` of `Mesh3D` instances onto `screen`.
        Automatically engages the perspective-correct Z-Buffer rasterizer when `self.textured=True`.
        Returns the total number of visible polygons drawn.
        """
        if self.textured:
            return self._render_scene_textured_zbuf(
                screen,
                camera,
                scene=scene,
                voxel_world=voxel_world,
                highlight_hit=highlight_hit,
                sky_dome=sky_dome,
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

    def _render_scene_textured_zbuf(
        self,
        screen: pygame.Surface,
        camera: Camera3D,
        scene: Scene3D | None = None,
        voxel_world: VoxelWorld | ChunkedVoxelWorld | None = None,
        highlight_hit: VoxelHit | None = None,
        sky_dome: SkyDome3D | None = None,
    ) -> int:
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

        # 1. Voxel World (VoxelWorld or ChunkedVoxelWorld)
        if voxel_world is not None:
            data = voxel_world.get_culled_render_data()
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
                front_mask = np.einsum("ij,ij->i", normals_w, to_face) < -1e-5
                front_idx = np.where(front_mask)[0]

                if len(front_idx) > 0:
                    q_vis = quads_w[front_idx]
                    n_vis = normals_w[front_idx]
                    c_vis = colors_w[front_idx]
                    cent_vis = centers_w[front_idx]
                    a_vis = alphas_w[front_idx]
                    em_vis = emissive_w[front_idx]
                    ao_vis = ao_w[front_idx]

                    k_vis = len(q_vis)
                    flat_rel = q_vis.reshape(-1, 3) - cam_pos[None, :]
                    q_cam = (flat_rel @ view_mat).reshape(k_vis, 4, 3)

                    z_vals = q_cam[:, :, 2]
                    max_z = np.max(z_vals, axis=1)
                    min_z = np.min(z_vals, axis=1)
                    x_vals = q_cam[:, :, 0]
                    y_vals = q_cam[:, :, 1]
                    max_z_pos = np.maximum(max_z, 1.0)
                    max_x_allow = max_z_pos * ((half_w / proj_scale) + 0.55) + 1.4
                    max_y_allow = max_z_pos * ((half_h / proj_scale) + 0.55) + 1.4
                    in_frustum = (
                        (np.min(x_vals, axis=1) <= max_x_allow)
                        & (np.max(x_vals, axis=1) >= -max_x_allow)
                        & (np.min(y_vals, axis=1) <= max_y_allow)
                        & (np.max(y_vals, axis=1) >= -max_y_allow)
                    )
                    keep_mask = (max_z >= near_z) & (min_z <= far_z) & in_frustum
                    keep_sub = np.where(keep_mask)[0]

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
                        face_bias = np.where(
                            n_vis[:, 1] > 0.5,
                            1.06,
                            np.where(
                                n_vis[:, 1] < -0.5,
                                0.64,
                                np.where(np.abs(n_vis[:, 0]) > 0.5, 0.84, 0.94),
                            ),
                        ).astype(np.float32)
                        shades = np.clip(intensities * face_bias * ao_vis + em_vis, 0.15, 1.35)
                        fog_factors = (
                            np.clip((depths - fog_near) / fog_span, 0.0, 1.0)
                            if fog_enabled
                            else np.zeros_like(depths)
                        )
                        lit_colors = c_vis * shades[:, None]
                        final_colors = self.fog.apply_to_colors(lit_colors, depths).astype(np.int32)

                        all_in_front = min_z >= near_z
                        fast_sub = np.where(all_in_front)[0]
                        if len(fast_sub) > 0:
                            q_fast = q_cam[fast_sub]
                            inv_z_fast = 1.0 / np.maximum(q_fast[:, :, 2], near_z * 0.5)
                            sx_fast = half_w + q_fast[:, :, 0] * inv_z_fast * proj_scale
                            sy_fast = half_h - q_fast[:, :, 1] * inv_z_fast * proj_scale

                            on_scr = (
                                (np.max(sx_fast, axis=1) >= 0.0)
                                & (np.min(sx_fast, axis=1) < rw)
                                & (np.max(sy_fast, axis=1) >= 0.0)
                                & (np.min(sy_fast, axis=1) < rh)
                            )
                            valid_fast = np.where(on_scr)[0]
                            uv_list = [(0.0, 1.0), (0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]

                            for k in valid_fast:
                                local_i = fast_sub[k]
                                orig_i = orig_indices[local_i]
                                d_val = float(depths[local_i])
                                tex = textures_w[orig_i] if d_val <= lod_dist else None
                                alp = int(a_vis[local_i])
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
                                    d_val,
                                    pts_invz,
                                    uv_list if tex is not None else None,
                                    tex,
                                    rgb,
                                    float(shades[local_i]),
                                    float(fog_factors[local_i]),
                                    alp,
                                    tex is not None,
                                )
                                if alp < 255 or (tex is not None and tex.has_cutout):
                                    translucent_items.append(item)
                                else:
                                    opaque_items.append(item)

                        straddle_sub = np.where(~all_in_front)[0]
                        for local_i in straddle_sub:
                            clipped_p, clipped_uv = clip_textured_polygon_near_plane(
                                q_cam[local_i], CANONICAL_QUAD_UV, near_z
                            )
                            if len(clipped_p) < 3:
                                continue
                            pts_invz = self._project_point_list_invz(clipped_p, proj_scale, half_w, half_h, near_z)
                            xs = [p[0] for p in pts_invz]
                            ys = [p[1] for p in pts_invz]
                            if max(xs) < 0.0 or min(xs) >= rw or max(ys) < 0.0 or min(ys) >= rh:
                                continue
                            orig_i = orig_indices[local_i]
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
                                d_val,
                                pts_invz,
                                uv_l,
                                tex,
                                rgb,
                                float(shades[local_i]),
                                float(fog_factors[local_i]),
                                alp,
                                False,
                            )
                            if alp < 255 or (tex is not None and tex.has_cutout):
                                translucent_items.append(item)
                            else:
                                opaque_items.append(item)

        # 2. General Mesh3D Objects (Z-tested against the same voxel Z-Buffer!)
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

                    d_val = float(np.linalg.norm(poly_c.mean(axis=0)))
                    pts_invz = self._project_point_list_invz(poly_c, proj_scale, half_w, half_h, near_z)
                    xs = [p[0] for p in pts_invz]
                    ys = [p[1] for p in pts_invz]
                    if max(xs) < 0.0 or min(xs) >= rw or max(ys) < 0.0 or min(ys) >= rh:
                        continue

                    base_rgb = mesh.face_colors[f_idx]
                    intensity = float(self.light.compute_intensities(normal_w, center_w)[0])
                    shaded = self.fog.apply_to_colors(base_rgb * intensity, np.array([d_val], dtype=np.float32))
                    rgb_tuple = (int(shaded[0]), int(shaded[1]), int(shaded[2]))
                    opaque_items.append((d_val, pts_invz, None, None, rgb_tuple, 1.0, 0.0, 255, False))

        # 3. Rasterize opaque faces FRONT-TO-BACK (maximizes early-Z rejection!)
        draw_poly = rast.draw_planar_polygon
        opaque_items.sort(key=lambda it: it[0])
        for _, pts_invz, uvs, tex, rgb, shade, fog_f, alp, is_canon in opaque_items:
            draw_poly(
                pts_invz,
                uvs=uvs,
                texture=tex,
                solid_color=rgb,
                shade=shade,
                fog_factor=fog_f,
                fog_color=fog_color,
                alpha=255,
                canonical_uv=is_canon,
            )

        # 4. Rasterize translucent faces BACK-TO-FRONT (water, glass, cutouts)
        translucent_items.sort(key=lambda it: it[0], reverse=True)
        for _, pts_invz, uvs, tex, rgb, shade, fog_f, alp, is_canon in translucent_items:
            draw_poly(
                pts_invz,
                uvs=uvs,
                texture=tex,
                solid_color=rgb,
                shade=shade,
                fog_factor=fog_f,
                fog_color=fog_color,
                alpha=alp,
                canonical_uv=is_canon,
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
            quads_w, normals_w, colors_w = voxel_world.get_culled_quads()
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
                    flat_rel = q_vis.reshape(-1, 3) - cam_pos[None, :]
                    q_cam = (flat_rel @ view_mat).reshape(k_vis, 4, 3)

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

                        straddle_indices = np.where(~all_in_front)[0]
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
