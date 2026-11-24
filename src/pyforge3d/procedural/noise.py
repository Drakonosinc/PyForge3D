from __future__ import annotations

import numpy as np


def hash_2d(x: int | np.ndarray, z: int | np.ndarray, seed: int = 0) -> float | np.ndarray:
    """Deterministic fast hash mapping integer 2D coordinates (x, z) to [0.0, 1.0)."""
    is_scalar = np.ndim(x) == 0 and np.ndim(z) == 0
    ix = np.asarray(x, dtype=np.int64)
    iz = np.asarray(z, dtype=np.int64)
    h = (ix * np.int64(374761393) + iz * np.int64(668265263) + np.int64(seed) * np.int64(1442695041)) & np.int64(0xFFFFFFFF)
    h = ((h ^ (h >> np.int64(13))) * np.int64(1274126177)) & np.int64(0xFFFFFFFF)
    h = h ^ (h >> np.int64(16))
    val = (h.astype(np.float64) / 4294967296.0).astype(np.float32)
    return float(val) if is_scalar else val


class PerlinNoise2D:
    """Vectorized 2D Gradient Perlin Noise generator in pure NumPy."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        p = np.arange(256, dtype=np.int32)
        rng.shuffle(p)
        self._perm = np.concatenate([p, p])
        # 8 unit-length 2D gradient directions
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False, dtype=np.float32)
        self._grads = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def sample(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        """Evaluate 2D Perlin noise at scalar or NumPy array coordinates, returning values in [-1.0, 1.0]."""
        is_scalar = np.ndim(x) == 0 and np.ndim(y) == 0
        xa = np.asarray(x, dtype=np.float32)
        ya = np.asarray(y, dtype=np.float32)

        x0 = np.floor(xa).astype(np.int32)
        y0 = np.floor(ya).astype(np.int32)
        xf = xa - x0
        yf = ya - y0

        xi = x0 & 255
        yi = y0 & 255

        p = self._perm
        g = self._grads

        g00 = g[p[p[xi] + yi] & 7]
        g10 = g[p[p[xi + 1] + yi] & 7]
        g01 = g[p[p[xi] + yi + 1] & 7]
        g11 = g[p[p[xi + 1] + yi + 1] & 7]

        n00 = g00[..., 0] * xf + g00[..., 1] * yf
        n10 = g10[..., 0] * (xf - 1.0) + g10[..., 1] * yf
        n01 = g01[..., 0] * xf + g01[..., 1] * (yf - 1.0)
        n11 = g11[..., 0] * (xf - 1.0) + g11[..., 1] * (yf - 1.0)

        u = self._fade(xf)
        v = self._fade(yf)

        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        res = (nx0 + v * (nx1 - nx0)) * 1.4142135
        res = np.clip(res, -1.0, 1.0)
        return float(res) if is_scalar else res


class FractalNoise2D:
    """Multi-octave Fractal Brownian Motion (fBm) 2D noise built on top of PerlinNoise2D."""

    def __init__(
        self,
        seed: int = 0,
        octaves: int = 4,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        scale: float = 24.0,
    ) -> None:
        self.seed = int(seed)
        self.octaves = max(1, int(octaves))
        self.persistence = float(persistence)
        self.lacunarity = float(lacunarity)
        self.scale = max(1e-4, float(scale))
        self._layers = [PerlinNoise2D(seed=self.seed + i * 1013) for i in range(self.octaves)]

    def sample(self, x: float | np.ndarray, y: float | np.ndarray, normalized: bool = False) -> float | np.ndarray:
        """
        Evaluate fBm noise at coordinates (x, y).
        If `normalized=False`, returns values in [-1.0, 1.0]; if `True`, returns [0.0, 1.0].
        """
        is_scalar = np.ndim(x) == 0 and np.ndim(y) == 0
        xa = np.asarray(x, dtype=np.float32) / self.scale
        ya = np.asarray(y, dtype=np.float32) / self.scale

        total = np.zeros_like(xa, dtype=np.float32)
        amplitude = 1.0
        frequency = 1.0
        max_amp = 0.0

        for layer in self._layers:
            total = total + layer.sample(xa * frequency, ya * frequency) * amplitude
            max_amp += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity

        res = np.clip(total / max(1e-6, max_amp), -1.0, 1.0)
        if normalized:
            res = (res + 1.0) * 0.5
        return float(res) if is_scalar else res
