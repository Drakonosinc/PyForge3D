from __future__ import annotations

import numpy as np


def hash_2d(x: int | np.ndarray, z: int | np.ndarray, seed: int = 0) -> float | np.ndarray:
    """Deterministic fast hash mapping integer 2D coordinates (x, z) to [0.0, 1.0)."""
    if isinstance(x, (int, np.integer)) and isinstance(z, (int, np.integer)):
        h = (int(x) * 374761393 + int(z) * 668265263 + int(seed) * 1442695041) & 0xFFFFFFFF
        h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
        h = h ^ (h >> 16)
        return h * (1.0 / 4294967296.0)
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


def hash_3d(
    x: int | np.ndarray,
    y: int | np.ndarray,
    z: int | np.ndarray,
    seed: int = 0,
) -> float | np.ndarray:
    """Deterministic fast hash mapping integer 3D coordinates (x, y, z) to [0.0, 1.0)."""
    if (
        isinstance(x, (int, np.integer))
        and isinstance(y, (int, np.integer))
        and isinstance(z, (int, np.integer))
    ):
        h = (
            int(x) * 374761393
            + int(y) * 668265263
            + int(z) * 2147483647
            + int(seed) * 1442695041
        ) & 0xFFFFFFFF
        h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
        h = h ^ (h >> 16)
        return h * (1.0 / 4294967296.0)
    is_scalar = np.ndim(x) == 0 and np.ndim(y) == 0 and np.ndim(z) == 0
    ix = np.asarray(x, dtype=np.int64)
    iy = np.asarray(y, dtype=np.int64)
    iz = np.asarray(z, dtype=np.int64)
    h = (
        ix * np.int64(374761393)
        + iy * np.int64(668265263)
        + iz * np.int64(2147483647)
        + np.int64(seed) * np.int64(1442695041)
    ) & np.int64(0xFFFFFFFF)
    h = ((h ^ (h >> np.int64(13))) * np.int64(1274126177)) & np.int64(0xFFFFFFFF)
    h = h ^ (h >> np.int64(16))
    val = (h.astype(np.float64) / 4294967296.0).astype(np.float32)
    return float(val) if is_scalar else val


_GRADS_3D = np.array(
    [
        [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1],
    ],
    dtype=np.float32,
)


class PerlinNoise3D:
    """Vectorized 3D Gradient Perlin Noise generator in pure NumPy (ideal for 3D caves, overhangs, and ore veins)."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        p = np.arange(256, dtype=np.int32)
        rng.shuffle(p)
        self._perm = np.concatenate([p, p])
        self._grads = _GRADS_3D

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def sample(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        z: float | np.ndarray,
    ) -> float | np.ndarray:
        """Evaluate 3D Perlin noise at scalar or NumPy array coordinates, returning values in [-1.0, 1.0]."""
        is_scalar = np.ndim(x) == 0 and np.ndim(y) == 0 and np.ndim(z) == 0
        xa = np.asarray(x, dtype=np.float32)
        ya = np.asarray(y, dtype=np.float32)
        za = np.asarray(z, dtype=np.float32)

        x0 = np.floor(xa).astype(np.int32)
        y0 = np.floor(ya).astype(np.int32)
        z0 = np.floor(za).astype(np.int32)
        xf = xa - x0
        yf = ya - y0
        zf = za - z0

        xi = x0 & 255
        yi = y0 & 255
        zi = z0 & 255

        p = self._perm
        g = self._grads

        px0 = p[xi]
        px1 = p[xi + 1]
        p00 = p[px0 + yi]
        p10 = p[px1 + yi]
        p01 = p[px0 + yi + 1]
        p11 = p[px1 + yi + 1]

        g000 = g[p[p00 + zi] & 15]
        g100 = g[p[p10 + zi] & 15]
        g010 = g[p[p01 + zi] & 15]
        g110 = g[p[p11 + zi] & 15]
        g001 = g[p[p00 + zi + 1] & 15]
        g101 = g[p[p10 + zi + 1] & 15]
        g011 = g[p[p01 + zi + 1] & 15]
        g111 = g[p[p11 + zi + 1] & 15]

        xf1 = xf - 1.0
        yf1 = yf - 1.0
        zf1 = zf - 1.0

        n000 = g000[..., 0] * xf + g000[..., 1] * yf + g000[..., 2] * zf
        n100 = g100[..., 0] * xf1 + g100[..., 1] * yf + g100[..., 2] * zf
        n010 = g010[..., 0] * xf + g010[..., 1] * yf1 + g010[..., 2] * zf
        n110 = g110[..., 0] * xf1 + g110[..., 1] * yf1 + g110[..., 2] * zf
        n001 = g001[..., 0] * xf + g001[..., 1] * yf + g001[..., 2] * zf1
        n101 = g101[..., 0] * xf1 + g101[..., 1] * yf + g101[..., 2] * zf1
        n011 = g011[..., 0] * xf + g011[..., 1] * yf1 + g011[..., 2] * zf1
        n111 = g111[..., 0] * xf1 + g111[..., 1] * yf1 + g111[..., 2] * zf1

        u = self._fade(xf)
        v = self._fade(yf)
        w = self._fade(zf)

        x00 = n000 + u * (n100 - n000)
        x10 = n010 + u * (n110 - n010)
        x01 = n001 + u * (n101 - n001)
        x11 = n011 + u * (n111 - n011)
        xy0 = x00 + v * (x10 - x00)
        xy1 = x01 + v * (x11 - x01)
        res = np.clip((xy0 + w * (xy1 - xy0)) * 1.15, -1.0, 1.0)
        return float(res) if is_scalar else res


class FractalNoise3D:
    """Multi-octave 3D Fractal Brownian Motion (fBm) noise built on top of PerlinNoise3D."""

    def __init__(
        self,
        seed: int = 0,
        octaves: int = 3,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        scale: float = 18.0,
    ) -> None:
        self.seed = int(seed)
        self.octaves = max(1, int(octaves))
        self.persistence = float(persistence)
        self.lacunarity = float(lacunarity)
        self.scale = max(1e-4, float(scale))
        self._layers = [PerlinNoise3D(seed=self.seed + i * 2027) for i in range(self.octaves)]

    def sample(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        z: float | np.ndarray,
        normalized: bool = False,
    ) -> float | np.ndarray:
        """Evaluate 3D fBm noise at (x, y, z). Returns [-1, 1] or [0, 1] when `normalized=True`."""
        is_scalar = np.ndim(x) == 0 and np.ndim(y) == 0 and np.ndim(z) == 0
        xa = np.asarray(x, dtype=np.float32) / self.scale
        ya = np.asarray(y, dtype=np.float32) / self.scale
        za = np.asarray(z, dtype=np.float32) / self.scale

        total = np.zeros_like(xa, dtype=np.float32)
        amplitude = 1.0
        frequency = 1.0
        max_amp = 0.0

        for layer in self._layers:
            total = total + layer.sample(xa * frequency, ya * frequency, za * frequency) * amplitude
            max_amp += amplitude
            amplitude *= self.persistence
            frequency *= self.lacunarity

        res = np.clip(total / max(1e-6, max_amp), -1.0, 1.0)
        if normalized:
            res = (res + 1.0) * 0.5
        return float(res) if is_scalar else res
