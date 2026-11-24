from __future__ import annotations

import math
import numpy as np
import pygame
from ..camera.camera3d import Camera3D
from ..lighting.lights import DirectionalLight, Fog3D, PointLight
from ..math.transforms import clip_polygon_near_plane, compute_face_normals
from ..meshes.mesh3d import Mesh3D
from ..voxels.voxel_world import VoxelHit, VoxelWorld


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
    Combines world-space back-face culling, near-plane Sutherland-Hodgman polygon clipping,
    vectorized flat lighting + distance fog, Painter's Algorithm depth sorting, and voxel block outline highlighting.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        light: DirectionalLight | PointLight | None = None,
        fog: Fog3D | None = None,
        voxel_outline_color: tuple[int, int, int] | None = (18, 28, 48),
    ) -> None:
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.light: DirectionalLight | PointLight = light if light is not None else DirectionalLight()
        self.fog: Fog3D = fog if fog is not None else Fog3D(enabled=True)
        self.voxel_outline_color: tuple[int, int, int] | None = voxel_outline_color
        self.last_rendered_faces: int = 0

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

    def render_scene(
        self,
        screen: pygame.Surface,
        camera: Camera3D,
        scene: Scene3D | None = None,
        voxel_world: VoxelWorld | None = None,
        highlight_hit: VoxelHit | None = None,
    ) -> int:
        """
        Render both a `VoxelWorld` and a `Scene3D` of `Mesh3D` instances onto `screen`.
        Returns the total number of visible polygons drawn.
        """
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

        # Render queue items: (depth, screen_pts, fill_rgb, outline_rgb, is_wireframe)
        render_queue: list[tuple[float, list[tuple[int, int]], tuple[int, int, int], tuple[int, int, int] | None, bool]] = []

        # 1. Vectorized VoxelWorld Pipeline
        if voxel_world is not None and len(voxel_world.blocks) > 0:
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

                    # Transform all visible quad vertices to camera space in one matrix multiply
                    k_vis = len(q_vis)
                    flat_rel = q_vis.reshape(-1, 3) - cam_pos[None, :]
                    q_cam = (flat_rel @ view_mat).reshape(k_vis, 4, 3)

                    z_vals = q_cam[:, :, 2]
                    max_z = np.max(z_vals, axis=1)
                    min_z = np.min(z_vals, axis=1)
                    depth_mask = (max_z >= near_z) & (min_z <= far_z)

                    if np.any(depth_mask):
                        q_cam = q_cam[depth_mask]
                        n_vis = n_vis[depth_mask]
                        c_vis = c_vis[depth_mask]
                        cent_vis = cent_vis[depth_mask]
                        min_z = min_z[depth_mask]

                        cam_centers = q_cam.mean(axis=1)
                        depths = np.linalg.norm(cam_centers, axis=1)

                        intensities = self.light.compute_intensities(n_vis, cent_vis)
                        lit_colors = c_vis * intensities[:, None]
                        final_colors = self.fog.apply_to_colors(lit_colors, depths).astype(np.int32)

                        # Bulk project quads that are completely in front of near_z
                        all_in_front = min_z >= near_z
                        if np.any(all_in_front):
                            q_fast = q_cam[all_in_front]
                            d_fast = depths[all_in_front]
                            col_fast = final_colors[all_in_front]

                            z_safe = q_fast[:, :, 2]
                            sx = np.rint(half_w + (q_fast[:, :, 0] / z_safe) * proj_scale).astype(np.int32)
                            sy = np.rint(half_h - (q_fast[:, :, 1] / z_safe) * proj_scale).astype(np.int32)

                            # Screen frustum bounds check
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

                        # Near-plane clipping for straddling quads
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

        # 2. General Mesh3D Pipeline
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

        # 3. Sort back-to-front (Painter's Algorithm) and draw
        render_queue.sort(key=lambda item: item[0], reverse=True)

        for _, pts, fill_rgb, outline_rgb, is_wireframe in render_queue:
            if is_wireframe:
                pygame.draw.polygon(screen, outline_rgb or fill_rgb, pts, 1)
            else:
                pygame.draw.polygon(screen, fill_rgb, pts)
                if outline_rgb is not None:
                    pygame.draw.polygon(screen, outline_rgb, pts, 1)

        # 4. Highlight targeted voxel block (if highlight_hit is provided)
        if highlight_hit is not None:
            self.draw_voxel_highlight(screen, camera, highlight_hit.block_pos)

        self.last_rendered_faces = len(render_queue)
        return self.last_rendered_faces

    def draw_voxel_highlight(
        self,
        screen: pygame.Surface,
        camera: Camera3D,
        block_pos: tuple[int, int, int],
        color: tuple[int, int, int] = (255, 199, 51),
        width: int = 2,
    ) -> None:
        """Draws a glowing 3D wireframe bounding box around a targeted voxel block at (bx, by, bz)."""
        bx, by, bz = float(block_pos[0]), float(block_pos[1]), float(block_pos[2])
        pad = 0.008
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
