from __future__ import annotations

import numpy as np
import pygame


class Texture2D:
    """
    Lightweight, game-agnostic 2D texture wrapper for PyForge3D software rasterization.
    Stores contiguous RGB uint8 pixel arrays `(H, W, 3)`, optional per-pixel alpha `(H, W)`,
    and precomputed average RGB for fast LOD / untextured fallback.
    """

    def __init__(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray | None = None,
        name: str = "",
    ) -> None:
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError("Texture2D array must have shape (H, W, 3) or (H, W, 4).")
        if arr.shape[2] == 4:
            self.rgb: np.ndarray = np.ascontiguousarray(arr[:, :, :3])
            self.alpha: np.ndarray | None = np.ascontiguousarray(arr[:, :, 3])
        else:
            self.rgb = np.ascontiguousarray(arr)
            self.alpha = np.ascontiguousarray(np.asarray(alpha, dtype=np.uint8)) if alpha is not None else None

        self.height: int = int(self.rgb.shape[0])
        self.width: int = int(self.rgb.shape[1])
        self.name: str = str(name)
        mean_rgb = self.rgb.reshape(-1, 3).mean(axis=0)
        self.avg_color: tuple[int, int, int] = (
            int(round(float(mean_rgb[0]))),
            int(round(float(mean_rgb[1]))),
            int(round(float(mean_rgb[2]))),
        )
        self.has_cutout: bool = bool(self.alpha is not None and np.any(self.alpha < 245))
        self._surface_cache: pygame.Surface | None = None

    @classmethod
    def from_array(cls, arr: np.ndarray, name: str = "") -> Texture2D:
        return cls(arr, name=name)

    @classmethod
    def from_surface(cls, surface: pygame.Surface, name: str = "") -> Texture2D:
        rgb = pygame.surfarray.array3d(surface).swapaxes(0, 1)
        alpha = None
        if surface.get_flags() & pygame.SRCALPHA:
            try:
                alpha = pygame.surfarray.array_alpha(surface).swapaxes(0, 1)
            except Exception:
                alpha = None
        return cls(rgb=rgb, alpha=alpha, name=name)

    @classmethod
    def from_file(cls, path: str, size: tuple[int, int] | None = None) -> Texture2D:
        surf = pygame.image.load(path)
        if size is not None:
            surf = pygame.transform.scale(surf, size)
        return cls.from_surface(surf, name=path)

    def to_surface(self, size: tuple[int, int] | None = None) -> pygame.Surface:
        if self._surface_cache is None:
            if self.alpha is not None:
                surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.surfarray.blit_array(surf, self.rgb.swapaxes(0, 1))
                alpha_view = pygame.surfarray.pixels_alpha(surf)
                alpha_view[:, :] = self.alpha.swapaxes(0, 1)
                del alpha_view
            else:
                surf = pygame.Surface((self.width, self.height))
                pygame.surfarray.blit_array(surf, self.rgb.swapaxes(0, 1))
            self._surface_cache = surf
        if size is not None and size != (self.width, self.height):
            return pygame.transform.scale(self._surface_cache, size)
        return self._surface_cache
