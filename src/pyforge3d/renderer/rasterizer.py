from __future__ import annotations

import numpy as np
import pygame
from ..textures.texture2d import Texture2D


CANONICAL_QUAD_UV = np.array(
    [
        [0.0, 1.0],  # v0: bottom-left
        [0.0, 0.0],  # v1: top-left
        [1.0, 0.0],  # v2: top-right
        [1.0, 1.0],  # v3: bottom-right
    ],
    dtype=np.float32,
)


def clip_textured_polygon_near_plane(
    poly_cam: np.ndarray,
    uvs: np.ndarray,
    near_z: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sutherland-Hodgman 3D polygon + UV clipping against the camera near plane `z >= near_z`.
    Returns `(clipped_xyz (K, 3), clipped_uv (K, 2))`.
    """
    pts = np.asarray(poly_cam, dtype=np.float32)
    uv = np.asarray(uvs, dtype=np.float32)
    n = len(pts)
    if n < 3:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    out_p: list[np.ndarray] = []
    out_u: list[np.ndarray] = []

    prev_p = pts[-1]
    prev_u = uv[-1]
    prev_in = float(prev_p[2]) >= near_z

    for i in range(n):
        curr_p = pts[i]
        curr_u = uv[i]
        curr_in = float(curr_p[2]) >= near_z

        if curr_in:
            if not prev_in:
                dz = float(curr_p[2] - prev_p[2])
                t = (near_z - float(prev_p[2])) / dz if abs(dz) > 1e-7 else 0.0
                inter_p = prev_p + t * (curr_p - prev_p)
                inter_p[2] = near_z
                inter_u = prev_u + t * (curr_u - prev_u)
                out_p.append(inter_p)
                out_u.append(inter_u)
            out_p.append(curr_p)
            out_u.append(curr_u)
        elif prev_in:
            dz = float(curr_p[2] - prev_p[2])
            t = (near_z - float(prev_p[2])) / dz if abs(dz) > 1e-7 else 0.0
            inter_p = prev_p + t * (curr_p - prev_p)
            inter_p[2] = near_z
            inter_u = prev_u + t * (curr_u - prev_u)
            out_p.append(inter_p)
            out_u.append(inter_u)

        prev_p = curr_p
        prev_u = curr_u
        prev_in = curr_in

    if len(out_p) < 3:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    return np.asarray(out_p, dtype=np.float32), np.asarray(out_u, dtype=np.float32)


class SoftwareTexturedRasterizer:
    """
    High-performance Pure-NumPy vectorized software rasterizer featuring:
    - Exact planar perspective-correct N-gon rasterization (single pass per convex quad or clipped N-gon!).
    - Per-pixel 1/Z depth buffer (true pixel-accurate 3D occlusion, zero polygon sorting artifacts).
    - Pre-shaded & fogged texture LUT cache (eliminates per-pixel float color multiplication and clipping).
    - Translucent alpha-blended & cutout pass (for water, glass, leaves, energy shields).
    - Solid-color fast path for untextured faces, distant LOD faces, and sub-pixel polygons.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = max(16, int(width))
        self.height = max(16, int(height))
        self.surface = pygame.Surface((self.width, self.height))
        self.color_buf = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.z_buf = np.zeros((self.height, self.width), dtype=np.float32)
        self._xs = np.arange(self.width, dtype=np.float32) + 0.5
        self._ys = np.arange(self.height, dtype=np.float32) + 0.5
        self._shade_cache: dict[tuple, np.ndarray] = {}
        self._shade_cache_fog_key: tuple[int, int, int] = (-1, -1, -1)

    def resize(self, width: int, height: int) -> None:
        w, h = max(16, int(width)), max(16, int(height))
        if w != self.width or h != self.height:
            self.width = w
            self.height = h
            self.surface = pygame.Surface((w, h))
            self.color_buf = np.zeros((h, w, 3), dtype=np.uint8)
            self.z_buf = np.zeros((h, w), dtype=np.float32)
            self._xs = np.arange(w, dtype=np.float32) + 0.5
            self._ys = np.arange(h, dtype=np.float32) + 0.5

    def begin_frame(
        self,
        bg_surface: pygame.Surface | None = None,
        clear_color: tuple[int, int, int] = (14, 20, 36),
    ) -> None:
        """Reset Z-Buffer (1/z = 0.0 -> infinity) and initialize color buffer from `bg_surface` or `clear_color`."""
        self.z_buf.fill(0.0)
        if bg_surface is not None:
            if bg_surface is not self.surface:
                if bg_surface.get_size() != (self.width, self.height):
                    pygame.transform.scale(bg_surface, (self.width, self.height), self.surface)
                else:
                    self.surface.blit(bg_surface, (0, 0))
            self.color_buf[:, :, :] = pygame.surfarray.array3d(self.surface).swapaxes(0, 1)
        else:
            self.color_buf[:, :, 0] = clear_color[0]
            self.color_buf[:, :, 1] = clear_color[1]
            self.color_buf[:, :, 2] = clear_color[2]

    def _get_shaded_texture(
        self,
        texture: Texture2D,
        shade: float,
        fog_factor: float,
        fog_color: np.ndarray | None,
    ) -> np.ndarray:
        """
        Return a cached `(H, W, 3) uint8` version of `texture.rgb` pre-lit with `shade` and blended with `fog`.
        Quantizes shade & fog into smooth 1/20 steps so 16x16 texels are shaded once instead of per screen pixel.
        """
        if fog_color is not None and fog_factor > 1e-3:
            fk = (int(fog_color[0]) >> 2, int(fog_color[1]) >> 2, int(fog_color[2]) >> 2)
            if fk != self._shade_cache_fog_key:
                self._shade_cache.clear()
                self._shade_cache_fog_key = fk
            fb = int(fog_factor * 20.0 + 0.5)
        else:
            fb = 0
        sb = int(shade * 24.0 + 0.5)
        key = (id(texture), sb, fb)
        cached = self._shade_cache.get(key)
        if cached is not None:
            return cached

        if len(self._shade_cache) > 512:
            self._shade_cache.clear()

        s_val = sb / 24.0
        src = texture.rgb.astype(np.float32) * s_val
        if fb > 0 and fog_color is not None:
            f_val = min(1.0, fb / 20.0)
            src = src * (1.0 - f_val) + fog_color * f_val
        out = np.clip(src, 0.0, 255.0).astype(np.uint8)
        self._shade_cache[key] = out
        return out

    def draw_planar_polygon(
        self,
        screen_pts_invz: list[tuple[float, float, float]],
        uvs: list[tuple[float, float]] | None = None,
        texture: Texture2D | None = None,
        solid_color: tuple[int, int, int] = (200, 200, 200),
        shade: float = 1.0,
        fog_factor: float = 0.0,
        fog_color: np.ndarray | None = None,
        alpha: int = 255,
        canonical_uv: bool = False,
    ) -> None:
        """
        Single-pass exact perspective-correct rasterizer for ANY convex planar 3D polygon (triangle, quad, or clipped N-gon).
        Exploits the projective invariant that `1/z`, `u/z`, and `v/z` are exact linear functions of screen `(x, y)`
        across the entire 3D plane, eliminating fan triangulation overhead for quads!
        """
        n = len(screen_pts_invz)
        if n < 3:
            return

        # Fast bounding box & plane basis selection (pick first 3 vertices with non-zero area)
        p0 = screen_pts_invz[0]
        x0, y0, iz0 = p0
        min_xf = max_xf = x0
        min_yf = max_yf = y0
        max_iz = iz0
        for idx in range(1, n):
            px, py, piz = screen_pts_invz[idx]
            if px < min_xf:
                min_xf = px
            elif px > max_xf:
                max_xf = px
            if py < min_yf:
                min_yf = py
            elif py > max_yf:
                max_yf = py
            if piz > max_iz:
                max_iz = piz

        min_x = 0 if min_xf < 0.0 else int(min_xf)
        max_x = self.width - 1 if max_xf >= self.width else int(max_xf)
        min_y = 0 if min_yf < 0.0 else int(min_yf)
        max_y = self.height - 1 if max_yf >= self.height else int(max_yf)
        if min_x > max_x or min_y > max_y:
            return

        # Find a non-degenerate triangle (0, i1, i2) to compute exact screen-space linear plane gradients
        p1 = screen_pts_invz[1]
        p2 = screen_pts_invz[2]
        x1, y1, iz1 = p1
        x2, y2, iz2 = p2
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        i1, i2 = 1, 2
        if abs(denom) < 1e-3 and n > 3:
            for k in range(2, n - 1):
                tx1, ty1, tiz1 = screen_pts_invz[k]
                tx2, ty2, tiz2 = screen_pts_invz[k + 1]
                td = (ty1 - ty2) * (x0 - tx2) + (tx2 - tx1) * (y0 - ty2)
                if abs(td) > abs(denom):
                    denom = td
                    x1, y1, iz1 = tx1, ty1, tiz1
                    x2, y2, iz2 = tx2, ty2, tiz2
                    i1, i2 = k, k + 1
        if abs(denom) < 1e-3:
            return

        z_slice = self.z_buf[min_y : max_y + 1, min_x : max_x + 1]
        bw = max_x - min_x + 1
        bh = max_y - min_y + 1

        # Early-Z rejection for opaque faces (opaque faces are sorted front-to-back!)
        if bw * bh >= 16 and max_iz <= float(z_slice.min()):
            return

        xs_1d = self._xs[min_x : max_x + 1]
        ys_1d = self._ys[min_y : max_y + 1]
        xs = xs_1d[None, :]
        ys = ys_1d[:, None]

        # Orientation sign from signed area of (0, i1, i2): ensures interior satisfies e >= 0 for all edges
        # Edge e_ab(x, y) = (xb - xa)*(y - ya) - (yb - ya)*(x - xa)
        # Note: (x1 - x0)*(y2 - y0) - (y1 - y0)*(x2 - x0) == +denom
        orient = 1.0 if denom > 0.0 else -1.0

        # Build convex N-gon interior mask via broadcasted 1D+1D edge half-space tests
        if n == 4:
            q0x, q0y, _ = screen_pts_invz[0]
            q1x, q1y, _ = screen_pts_invz[1]
            q2x, q2y, _ = screen_pts_invz[2]
            q3x, q3y, _ = screen_pts_invz[3]

            ex0 = (q0y - q1y) * orient
            ey0 = (q1x - q0x) * orient
            ec0 = -(ex0 * q0x + ey0 * q0y)

            ex1 = (q1y - q2y) * orient
            ey1 = (q2x - q1x) * orient
            ec1 = -(ex1 * q1x + ey1 * q1y)

            ex2 = (q2y - q3y) * orient
            ey2 = (q3x - q2x) * orient
            ec2 = -(ex2 * q2x + ey2 * q2y)

            ex3 = (q3y - q0y) * orient
            ey3 = (q0x - q3x) * orient
            ec3 = -(ex3 * q3x + ey3 * q3y)

            mask = (
                ((ex0 * xs) + (ey0 * ys + ec0) >= -1e-3)
                & ((ex1 * xs) + (ey1 * ys + ec1) >= -1e-3)
                & ((ex2 * xs) + (ey2 * ys + ec2) >= -1e-3)
                & ((ex3 * xs) + (ey3 * ys + ec3) >= -1e-3)
            )
        elif n == 3:
            ex0 = (y0 - y1) * orient
            ey0 = (x1 - x0) * orient
            ec0 = -(ex0 * x0 + ey0 * y0)

            ex1 = (y1 - y2) * orient
            ey1 = (x2 - x1) * orient
            ec1 = -(ex1 * x1 + ey1 * y1)

            ex2 = (y2 - y0) * orient
            ey2 = (x0 - x2) * orient
            ec2 = -(ex2 * x2 + ey2 * y2)

            mask = (
                ((ex0 * xs) + (ey0 * ys + ec0) >= -1e-3)
                & ((ex1 * xs) + (ey1 * ys + ec1) >= -1e-3)
                & ((ex2 * xs) + (ey2 * ys + ec2) >= -1e-3)
            )
        else:
            mask = np.ones((bh, bw), dtype=bool)
            for idx in range(n):
                ax, ay, _ = screen_pts_invz[idx]
                bx, by, _ = screen_pts_invz[(idx + 1) % n]
                ex = (ay - by) * orient
                ey = (bx - ax) * orient
                ec = -(ex * ax + ey * ay)
                mask &= (ex * xs) + (ey * ys + ec) >= -1e-3

        # Exact planar linear interpolation of 1/z across the whole polygon!
        inv_denom = 1.0 / denom
        a0 = (y1 - y2) * inv_denom
        b0 = (x2 - x1) * inv_denom
        c0 = -(a0 * x2 + b0 * y2)

        a1 = (y2 - y0) * inv_denom
        b1 = (x0 - x2) * inv_denom
        c1 = -(a1 * x2 + b1 * y2)

        a2 = -a0 - a1
        b2 = -b0 - b1
        c2 = 1.0 - c0 - c1

        az = a0 * iz0 + a1 * iz1 + a2 * iz2
        bz = b0 * iz0 + b1 * iz1 + b2 * iz2
        cz = c0 * iz0 + c1 * iz1 + c2 * iz2

        inv_z = (az * xs) + (bz * ys + cz)
        vis = mask & (inv_z > z_slice)
        if not vis.any():
            return

        c_slice = self.color_buf[min_y : max_y + 1, min_x : max_x + 1]

        # Solid-color fast path (untextured, distant LOD, or tiny <=4px face)
        use_tex = (
            texture is not None
            and (canonical_uv or uvs is not None)
            and (bw > 4 or bh > 4 or texture.has_cutout)
        )
        if not use_tex:
            if alpha < 255:
                a = alpha / 255.0
                dst = c_slice[vis].astype(np.float32)
                c_slice[vis] = (dst * (1.0 - a) + np.array(solid_color, dtype=np.float32) * a).astype(np.uint8)
            else:
                c_slice[vis] = solid_color
                z_slice[vis] = inv_z[vis]
            return

        assert texture is not None
        vy, vx = np.nonzero(vis)
        px = xs_1d[vx]
        py = ys_1d[vy]
        iz_v = inv_z[vy, vx]
        inv_iz = 1.0 / iz_v

        if canonical_uv and i1 == 1 and i2 == 2:
            # Canonical quad UV: uv0=(0,1), uv1=(0,0), uv2=(1,0) -> single-term u/z and v/z!
            u_over_z = (a2 * iz2) * px + (b2 * iz2) * py + (c2 * iz2)
            v_over_z = (a0 * iz0) * px + (b0 * iz0) * py + (c0 * iz0)
        else:
            assert uvs is not None
            u0, v0 = uvs[0]
            u1, v1 = uvs[i1]
            u2, v2 = uvs[i2]
            u0z, u1z, u2z = u0 * iz0, u1 * iz1, u2 * iz2
            v0z, v1z, v2z = v0 * iz0, v1 * iz1, v2 * iz2
            u_over_z = (a0 * u0z + a1 * u1z + a2 * u2z) * px + (b0 * u0z + b1 * u1z + b2 * u2z) * py + (
                c0 * u0z + c1 * u1z + c2 * u2z
            )
            v_over_z = (a0 * v0z + a1 * v1z + a2 * v2z) * px + (b0 * v0z + b1 * v1z + b2 * v2z) * py + (
                c0 * v0z + c1 * v1z + c2 * v2z
            )

        tw = texture.width
        th = texture.height
        tx = (u_over_z * (inv_iz * tw)).astype(np.int32)
        ty = (v_over_z * (inv_iz * th)).astype(np.int32)
        np.maximum(tx, 0, out=tx)
        np.minimum(tx, tw - 1, out=tx)
        np.maximum(ty, 0, out=ty)
        np.minimum(ty, th - 1, out=ty)

        shaded_rgb = self._get_shaded_texture(texture, shade, fog_factor, fog_color)

        if texture.has_cutout and texture.alpha is not None:
            tex_a = texture.alpha[ty, tx]
            keep = tex_a >= 32
            if not keep.any():
                return
            vy = vy[keep]
            vx = vx[keep]
            tx = tx[keep]
            ty = ty[keep]
            iz_v = iz_v[keep]
            tex_a = tex_a[keep]
            src_u8 = shaded_rgb[ty, tx]
            if alpha < 255 or (tex_a < 250).any():
                pixel_alpha = (tex_a.astype(np.float32) * (alpha / 65025.0))[:, None]
                dst = c_slice[vy, vx].astype(np.float32)
                c_slice[vy, vx] = (dst * (1.0 - pixel_alpha) + src_u8.astype(np.float32) * pixel_alpha).astype(np.uint8)
                opaque_px = pixel_alpha[:, 0] >= 0.96
                if opaque_px.any():
                    z_slice[vy[opaque_px], vx[opaque_px]] = iz_v[opaque_px]
            else:
                c_slice[vy, vx] = src_u8
                z_slice[vy, vx] = iz_v
        elif alpha < 255:
            a = alpha / 255.0
            dst = c_slice[vy, vx].astype(np.float32)
            c_slice[vy, vx] = (dst * (1.0 - a) + shaded_rgb[ty, tx].astype(np.float32) * a).astype(np.uint8)
        else:
            c_slice[vy, vx] = shaded_rgb[ty, tx]
            z_slice[vy, vx] = iz_v

    def draw_triangle(
        self,
        p0: tuple[float, float, float],
        p1: tuple[float, float, float],
        p2: tuple[float, float, float],
        uv0: tuple[float, float] | None = None,
        uv1: tuple[float, float] | None = None,
        uv2: tuple[float, float] | None = None,
        texture: Texture2D | None = None,
        solid_color: tuple[int, int, int] = (200, 200, 200),
        shade: float = 1.0,
        fog_factor: float = 0.0,
        fog_color: np.ndarray | None = None,
        alpha: int = 255,
    ) -> None:
        """Rasterize a single 3D screen-space triangle `(x, y, inv_z)`."""
        uvs = [uv0, uv1, uv2] if (uv0 is not None and uv1 is not None and uv2 is not None) else None
        self.draw_planar_polygon(
            [p0, p1, p2],
            uvs=uvs,
            texture=texture,
            solid_color=solid_color,
            shade=shade,
            fog_factor=fog_factor,
            fog_color=fog_color,
            alpha=alpha,
        )

    def draw_polygon(
        self,
        screen_pts_invz: list[tuple[float, float, float]],
        uvs: list[tuple[float, float]] | None = None,
        texture: Texture2D | None = None,
        solid_color: tuple[int, int, int] = (200, 200, 200),
        shade: float = 1.0,
        fog_factor: float = 0.0,
        fog_color: np.ndarray | None = None,
        alpha: int = 255,
        canonical_uv: bool = False,
    ) -> None:
        """Rasterize a convex planar N-gon (N >= 3) in a single vectorized Z-Buffer pass."""
        self.draw_planar_polygon(
            screen_pts_invz,
            uvs=uvs,
            texture=texture,
            solid_color=solid_color,
            shade=shade,
            fog_factor=fog_factor,
            fog_color=fog_color,
            alpha=alpha,
            canonical_uv=canonical_uv,
        )

    def blit_to_screen(self, screen: pygame.Surface) -> None:
        """Upload the NumPy framebuffer to pygame and scale crisply onto `screen`."""
        pygame.surfarray.blit_array(self.surface, self.color_buf.swapaxes(0, 1))
        if screen.get_size() == (self.width, self.height):
            screen.blit(self.surface, (0, 0))
        else:
            pygame.transform.scale(self.surface, screen.get_size(), screen)
