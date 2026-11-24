from __future__ import annotations

import numpy as np
from .texture2d import Texture2D


class TextureAtlas:
    """
    Game-agnostic contiguous texture atlas packing arbitrary `Texture2D` assets
    into flat C-contiguous NumPy buffers for zero-overhead indexed lookup in
    both native C99 and pure-NumPy rendering backends.
    """

    def __init__(self) -> None:
        self._id_by_obj: dict[int, int] = {}
        self._textures: list[Texture2D] = []
        self._rgb_chunks: list[np.ndarray] = []
        self._alpha_chunks: list[np.ndarray] = []
        self._offsets: list[int] = []
        self._widths: list[int] = []
        self._heights: list[int] = []
        self._has_cutout: list[int] = []
        self._total_texels: int = 0
        self._dirty: bool = True

        self.rgb_flat: np.ndarray = np.empty((0, 3), dtype=np.uint8)
        self.alpha_flat: np.ndarray = np.empty((0,), dtype=np.uint8)
        self.offsets: np.ndarray = np.empty((0,), dtype=np.int32)
        self.widths: np.ndarray = np.empty((0,), dtype=np.int32)
        self.heights: np.ndarray = np.empty((0,), dtype=np.int32)
        self.has_cutout: np.ndarray = np.empty((0,), dtype=np.uint8)

    def register(self, texture: Texture2D | None) -> int:
        """Return integer texture ID for `texture` (`-1` if `None`), packing it once if new."""
        if texture is None:
            return -1
        key = id(texture)
        tid = self._id_by_obj.get(key)
        if tid is not None:
            return tid

        tid = len(self._textures)
        self._id_by_obj[key] = tid
        self._textures.append(texture)

        h, w = int(texture.height), int(texture.width)
        n_texels = w * h
        self._offsets.append(self._total_texels)
        self._widths.append(w)
        self._heights.append(h)
        self._has_cutout.append(1 if texture.has_cutout else 0)
        self._total_texels += n_texels

        rgb_c = np.ascontiguousarray(texture.rgb.reshape(n_texels, 3), dtype=np.uint8)
        self._rgb_chunks.append(rgb_c)
        if texture.alpha is not None:
            alp_c = np.ascontiguousarray(texture.alpha.reshape(n_texels), dtype=np.uint8)
        else:
            alp_c = np.full(n_texels, 255, dtype=np.uint8)
        self._alpha_chunks.append(alp_c)
        self._dirty = True
        return tid

    def get_texture(self, tid: int) -> Texture2D | None:
        if 0 <= tid < len(self._textures):
            return self._textures[tid]
        return None

    def ensure_packed(self) -> None:
        if not self._dirty:
            return
        if self._rgb_chunks:
            self.rgb_flat = np.ascontiguousarray(np.concatenate(self._rgb_chunks, axis=0), dtype=np.uint8)
            self.alpha_flat = np.ascontiguousarray(np.concatenate(self._alpha_chunks, axis=0), dtype=np.uint8)
            self.offsets = np.ascontiguousarray(self._offsets, dtype=np.int32)
            self.widths = np.ascontiguousarray(self._widths, dtype=np.int32)
            self.heights = np.ascontiguousarray(self._heights, dtype=np.int32)
            self.has_cutout = np.ascontiguousarray(self._has_cutout, dtype=np.uint8)
        else:
            self.rgb_flat = np.empty((1, 3), dtype=np.uint8)
            self.alpha_flat = np.full((1,), 255, dtype=np.uint8)
            self.offsets = np.zeros((1,), dtype=np.int32)
            self.widths = np.ones((1,), dtype=np.int32)
            self.heights = np.ones((1,), dtype=np.int32)
            self.has_cutout = np.zeros((1,), dtype=np.uint8)
        self._dirty = False

    def __len__(self) -> int:
        return len(self._textures)
