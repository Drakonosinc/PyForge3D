from __future__ import annotations

import math
import random
import numpy as np
import pygame
from ..camera.camera3d import Camera3D
from ..math.transforms import clip_polygon_near_plane, normalize_vectors
from .lights import DirectionalLight, Fog3D


def _lerp_rgb(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


class SkyDome3D:
    """
    Game-agnostic 3D atmospheric sky dome with continuous day/night cycle (`time_of_day` in [0, 1)),
    orbiting square Sun & Moon, twinkling night stars, drifting 3D flat clouds, and automatic
    DirectionalLight + Fog3D synchronization.
    """

    def __init__(
        self,
        time_of_day: float = 0.25,
        day_speed: float = 0.004,
        cloud_height: float = 22.0,
        cloud_count: int = 18,
        seed: int = 7,
    ) -> None:
        self.time_of_day: float = float(time_of_day) % 1.0  # 0.0=dawn, 0.25=noon, 0.5=sunset, 0.75=midnight
        self.day_speed: float = float(day_speed)
        self.cloud_height: float = float(cloud_height)
        self._cloud_drift: float = 0.0

        rng = random.Random(seed)
        # Deterministic stars on the upper sky hemisphere
        self._stars: list[tuple[float, float, float, float]] = []
        for _ in range(70):
            az = rng.uniform(0.0, math.tau)
            el = rng.uniform(0.12, 1.45)
            ce = math.cos(el)
            self._stars.append((math.sin(az) * ce, math.sin(el), math.cos(az) * ce, rng.uniform(0.5, 1.0)))

        # Drifting flat rectangular clouds in world offsets around camera
        self._clouds: list[tuple[float, float, float, float]] = []
        for _ in range(int(cloud_count)):
            cx = rng.uniform(-45.0, 45.0)
            cz = rng.uniform(-45.0, 45.0)
            cw = rng.uniform(4.5, 10.5)
            cd = rng.uniform(3.0, 7.5)
            self._clouds.append((cx, cz, cw, cd))

        self._sky_cache_key = None
        self._sky_cache_surf: pygame.Surface | None = None
        self._cloud_surf_size: tuple[int, int] | None = None
        self._cloud_surf: pygame.Surface | None = None

    def update(self, dt: float) -> None:
        dt = max(0.0, float(dt))
        self.time_of_day = (self.time_of_day + self.day_speed * dt) % 1.0
        self._cloud_drift = (self._cloud_drift + 1.2 * dt) % 90.0

    def get_sun_direction(self) -> np.ndarray:
        """Unit direction vector pointing FROM origin TOWARD the Sun."""
        angle = self.time_of_day * math.tau
        return normalize_vectors(np.array([math.cos(angle), math.sin(angle), 0.28], dtype=np.float32))

    def get_atmosphere_palette(self) -> dict[str, tuple[int, int, int] | float]:
        """Return current zenith, horizon, ground, cloud, fog colors, and daylight factor [0..1]."""
        sun_y = math.sin(self.time_of_day * math.tau)  # +1 at noon, -1 at midnight
        day_t = max(0.0, min(1.0, (sun_y + 0.15) / 0.85))
        sunset_t = max(0.0, 1.0 - abs(sun_y) * 3.8)

        night_zenith = (8, 13, 32)
        day_zenith = (82, 148, 245)
        zenith = _lerp_rgb(night_zenith, day_zenith, day_t)

        night_horizon = (18, 28, 56)
        day_horizon = (174, 214, 255)
        sunset_horizon = (242, 128, 76)
        horizon = _lerp_rgb(night_horizon, day_horizon, day_t)
        horizon = _lerp_rgb(horizon, sunset_horizon, sunset_t * 0.78)

        ground = _lerp_rgb((8, 12, 22), (76, 114, 152), day_t)
        cloud_col = _lerp_rgb((68, 78, 108), (248, 252, 255), day_t)
        cloud_col = _lerp_rgb(cloud_col, (255, 186, 142), sunset_t * 0.55)

        return {
            "zenith": zenith,
            "horizon": horizon,
            "ground": ground,
            "fog": horizon,
            "cloud": cloud_col,
            "day_factor": day_t,
            "night_factor": max(0.0, min(1.0, (-sun_y - 0.05) / 0.55)),
        }

    def sync_lighting(self, light: DirectionalLight, fog: Fog3D | None = None) -> dict:
        """Synchronize a DirectionalLight and Fog3D to the current sky time of day."""
        pal = self.get_atmosphere_palette()
        sun_dir = self.get_sun_direction()
        day_t = float(pal["day_factor"])
        # Light rays travel opposite to sun position when sun is up, or gentle moonlight from opposite side at night
        active_to_light = sun_dir if sun_dir[1] >= -0.05 else -sun_dir
        light.to_light_dir = normalize_vectors(active_to_light)
        light.ambient = 0.24 + 0.24 * day_t
        light.diffuse = 0.30 + 0.44 * day_t
        if fog is not None:
            fog_rgb = pal["fog"]
            fog.color = np.array([fog_rgb[0], fog_rgb[1], fog_rgb[2]], dtype=np.float32)
        return pal

    def draw(self, screen: pygame.Surface, camera: Camera3D) -> None:
        """Render the dynamic gradient sky, night stars, square Sun & Moon, and drifting 3D clouds onto `screen`."""
        w, h = screen.get_size()
        pal = self.get_atmosphere_palette()
        zenith = pal["zenith"]
        horizon = pal["horizon"]
        ground = pal["ground"]

        horizon_y = int(h * 0.5 + math.sin(camera.pitch) * h * 0.55)
        horizon_y = max(-h, min(h * 2, horizon_y))

        cache_key = (w, h, horizon_y // 4, zenith, horizon, ground)
        if self._sky_cache_key != cache_key or self._sky_cache_surf is None:
            surf = pygame.Surface((w, h))
            step = 4
            for y in range(0, h, step):
                if y <= horizon_y:
                    t = max(0.0, min(1.0, (horizon_y - y) / max(1.0, h * 0.65)))
                    c = _lerp_rgb(horizon, zenith, t)
                else:
                    t = max(0.0, min(1.0, (y - horizon_y) / max(1.0, h * 0.45)))
                    c = _lerp_rgb(horizon, ground, t)
                pygame.draw.rect(surf, c, (0, y, w, step))
            self._sky_cache_key = cache_key
            self._sky_cache_surf = surf
        screen.blit(self._sky_cache_surf, (0, 0))

        view_mat = camera.get_view_matrix_3x3()
        focal = 1.0 / math.tan(math.radians(camera.fov) * 0.5)
        half_w = w * 0.5
        half_h = h * 0.5
        scale = focal * half_h

        # 1. Twinkling stars at night
        night_f = float(pal["night_factor"])
        if night_f > 0.05:
            star_dirs = np.array([[s[0], s[1], s[2]] for s in self._stars], dtype=np.float32)
            star_cam = star_dirs @ view_mat
            in_front = star_cam[:, 2] > 0.15
            if np.any(in_front):
                sc = star_cam[in_front]
                sx = half_w + (sc[:, 0] / sc[:, 2]) * scale
                sy = half_h - (sc[:, 1] / sc[:, 2]) * scale
                bri = int(120 + 135 * night_f)
                star_col = (bri, bri, min(255, bri + 20))
                for px, py in zip(sx, sy):
                    ix, iy = int(px), int(py)
                    if 0 <= ix < w and 0 <= iy < h:
                        pygame.draw.rect(screen, star_col, (ix, iy, 2, 2))

        # 2. Square Sun & Moon celestial bodies
        sun_dir = self.get_sun_direction()
        for body_dir, body_col, glow_col, sz in (
            (sun_dir, (255, 248, 195), (255, 210, 100), max(14, int(h * 0.065))),
            (-sun_dir, (226, 236, 255), (140, 170, 225), max(11, int(h * 0.052))),
        ):
            bc = body_dir @ view_mat
            if float(bc[2]) > 0.15:
                bx = int(half_w + (float(bc[0]) / float(bc[2])) * scale)
                by = int(half_h - (float(bc[1]) / float(bc[2])) * scale)
                if -sz * 2 < bx < w + sz * 2 and -sz * 2 < by < h + sz * 2:
                    glow_s = pygame.Surface((sz * 3, sz * 3), pygame.SRCALPHA)
                    pygame.draw.rect(glow_s, (*glow_col, 70), (sz // 2, sz // 2, sz * 2, sz * 2))
                    screen.blit(glow_s, (bx - int(sz * 1.5), by - int(sz * 1.5)))
                    pygame.draw.rect(screen, body_col, (bx - sz // 2, by - sz // 2, sz, sz))

        # 3. Drifting flat 3D blocky clouds overhead
        cloud_col = pal["cloud"]
        cloud_y = self.cloud_height
        cam_x, cam_z = camera.x, camera.z
        near_z = 0.5
        if self._cloud_surf is None or self._cloud_surf_size != (w, h):
            self._cloud_surf = pygame.Surface((w, h), pygame.SRCALPHA)
            self._cloud_surf_size = (w, h)
        else:
            self._cloud_surf.fill((0, 0, 0, 0))
        cloud_surf = self._cloud_surf
        for ox, oz, cw, cd in self._clouds:
            # Wrap clouds infinitely around the camera
            rx = ((ox + self._cloud_drift - cam_x + 45.0) % 90.0) - 45.0
            rz = ((oz - cam_z + 45.0) % 90.0) - 45.0
            wx = cam_x + rx
            wz = cam_z + rz
            quad_w = np.array(
                [
                    [wx - cw * 0.5, cloud_y, wz - cd * 0.5],
                    [wx + cw * 0.5, cloud_y, wz - cd * 0.5],
                    [wx + cw * 0.5, cloud_y, wz + cd * 0.5],
                    [wx - cw * 0.5, cloud_y, wz + cd * 0.5],
                ],
                dtype=np.float32,
            )
            quad_c = (quad_w - camera.position[None, :]) @ view_mat
            if float(np.max(quad_c[:, 2])) < near_z:
                continue
            if float(np.min(quad_c[:, 2])) < near_z:
                quad_c = clip_polygon_near_plane(quad_c, near_z)
                if len(quad_c) < 3:
                    continue
            pts = [
                (
                    int(half_w + (float(p[0]) / float(p[2])) * scale),
                    int(half_h - (float(p[1]) / float(p[2])) * scale),
                )
                for p in quad_c
            ]
            pygame.draw.polygon(cloud_surf, (*cloud_col, 185), pts)
        screen.blit(cloud_surf, (0, 0))
