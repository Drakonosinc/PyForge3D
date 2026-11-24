from __future__ import annotations

import numpy as np
from ..math.transforms import normalize_vectors


class DirectionalLight:
    """Directional sun/sky light computing vectorized Lambertian flat shading across 3D faces."""

    def __init__(
        self,
        direction: tuple[float, float, float] = (-0.45, -1.0, -0.35),
        ambient: float = 0.35,
        diffuse: float = 0.65,
    ) -> None:
        raw_dir = np.array(direction, dtype=np.float32)
        # Light vector points FROM surface TOWARD light source (-direction)
        self.to_light_dir: np.ndarray = normalize_vectors(-raw_dir)
        self.ambient: float = max(0.0, min(1.0, float(ambient)))
        self.diffuse: float = max(0.0, min(1.5, float(diffuse)))

    def compute_intensities(self, normals: np.ndarray, centers: np.ndarray | None = None) -> np.ndarray:
        """Compute (N,) shading multipliers in [ambient, 1.0+] for an (N, 3) array of unit normals."""
        n = np.asarray(normals, dtype=np.float32)
        if n.ndim == 1:
            dot = max(0.0, float(np.dot(n, self.to_light_dir)))
            return np.array([min(1.15, self.ambient + self.diffuse * dot)], dtype=np.float32)
        dots = np.maximum(0.0, n @ self.to_light_dir)
        return np.minimum(1.15, self.ambient + self.diffuse * dots)


class PointLight:
    """Omnidirectional 3D point light with distance attenuation and optional finite influence radius."""

    def __init__(
        self,
        position: tuple[float, float, float] = (0.0, 6.0, 0.0),
        ambient: float = 0.3,
        intensity: float = 1.2,
        attenuation: float = 0.04,
        radius: float = 14.0,
        color: tuple[int, int, int] = (255, 240, 210),
    ) -> None:
        self.position = np.array(position, dtype=np.float32)
        self.ambient = max(0.0, min(1.0, float(ambient)))
        self.intensity = max(0.0, float(intensity))
        self.attenuation = max(0.0, float(attenuation))
        self.radius = max(0.5, float(radius))
        self.color = (int(color[0]), int(color[1]), int(color[2]))

    def compute_intensities(self, normals: np.ndarray, centers: np.ndarray) -> np.ndarray:
        n = np.asarray(normals, dtype=np.float32)
        c = np.asarray(centers, dtype=np.float32)
        if n.ndim == 1:
            n = n.reshape(1, 3)
            c = c.reshape(1, 3)
        to_light = self.position - c
        dists = np.linalg.norm(to_light, axis=1)
        dirs = to_light / np.maximum(dists[:, None], 1e-6)
        dots = np.maximum(0.0, np.einsum("ij,ij->i", n, dirs))
        atten = 1.0 / (1.0 + self.attenuation * (dists ** 2))
        return np.minimum(1.2, self.ambient + self.intensity * dots * atten)


class Fog3D:
    """Distance-based atmospheric fog interpolating polygon colors smoothly toward the horizon color."""

    def __init__(
        self,
        color: tuple[int, int, int] = (14, 20, 36),
        near: float = 14.0,
        far: float = 42.0,
        enabled: bool = True,
    ) -> None:
        self.color = np.array([int(color[0]), int(color[1]), int(color[2])], dtype=np.float32)
        self.near = max(0.5, float(near))
        self.far = max(self.near + 1.0, float(far))
        self.enabled = bool(enabled)

    def apply_to_colors(self, rgb_colors: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """Blend (N, 3) RGB colors toward `self.color` based on (N,) camera distances."""
        colors = np.asarray(rgb_colors, dtype=np.float32)
        if not self.enabled:
            return np.clip(colors, 0.0, 255.0)
        d = np.asarray(depths, dtype=np.float32)
        span = max(1e-5, self.far - self.near)
        factor = np.clip((d - self.near) / span, 0.0, 1.0)
        if colors.ndim == 1:
            f = float(factor) if factor.ndim == 0 else float(factor[0])
            return np.clip(colors * (1.0 - f) + self.color * f, 0.0, 255.0)
        f_col = factor[:, None]
        return np.clip(colors * (1.0 - f_col) + self.color[None, :] * f_col, 0.0, 255.0)
