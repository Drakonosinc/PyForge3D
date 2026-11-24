from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a 1D (3,) or 2D (N, 3) array of 3D vectors."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > eps else np.zeros_like(arr)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > eps, norms, 1.0)
    return arr / norms


def compute_face_normals(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute unit surface normals for triangles/quads given their first 3 vertices (3,) or (N, 3) in left-handed winding."""
    e1 = v1 - v0
    e2 = v2 - v0
    if e1.ndim == 1:
        nx = float(e1[1] * e2[2] - e1[2] * e2[1])
        ny = float(e1[2] * e2[0] - e1[0] * e2[2])
        nz = float(e1[0] * e2[1] - e1[1] * e2[0])
        inv = math.sqrt(nx * nx + ny * ny + nz * nz)
        if inv > 1e-8:
            inv = 1.0 / inv
            return np.array([nx * inv, ny * inv, nz * inv], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)
    nx = e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1]
    ny = e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2]
    nz = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    cross = np.stack((nx, ny, nz), axis=1).astype(np.float32, copy=False)
    return normalize_vectors(cross)


def clip_polygon_near_plane(poly_cam: np.ndarray, near_z: float = 0.15) -> np.ndarray:
    """
    Sutherland-Hodgman 3D polygon clipping against the camera near plane `z >= near_z`.
    Accepts an (M, 3) array of camera-space vertices and returns the clipped (K, 3) array.
    """
    pts = np.asarray(poly_cam, dtype=np.float32)
    n = len(pts)
    if n < 3:
        return np.empty((0, 3), dtype=np.float32)

    out: list[np.ndarray] = []
    prev_pt = pts[-1]
    prev_in = float(prev_pt[2]) >= near_z

    for i in range(n):
        curr_pt = pts[i]
        curr_in = float(curr_pt[2]) >= near_z

        if curr_in:
            if not prev_in:
                dz = float(curr_pt[2] - prev_pt[2])
                t = (near_z - float(prev_pt[2])) / dz if abs(dz) > 1e-7 else 0.0
                inter = prev_pt + t * (curr_pt - prev_pt)
                inter[2] = near_z
                out.append(inter)
            out.append(curr_pt)
        elif prev_in:
            dz = float(curr_pt[2] - prev_pt[2])
            t = (near_z - float(prev_pt[2])) / dz if abs(dz) > 1e-7 else 0.0
            inter = prev_pt + t * (curr_pt - prev_pt)
            inter[2] = near_z
            out.append(inter)

        prev_pt = curr_pt
        prev_in = curr_in

    if len(out) < 3:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


class Matrix4:
    """Vectorized 3D rotation, translation, scale, and view matrix builders for row-vector multiplication (V @ M)."""

    @staticmethod
    def identity() -> np.ndarray:
        return np.eye(4, dtype=np.float32)

    @staticmethod
    def rotation_x(angle_rad: float) -> np.ndarray:
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, s],
                [0.0, -s, c],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def rotation_y(angle_rad: float) -> np.ndarray:
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return np.array(
            [
                [c, 0.0, -s],
                [0.0, 1.0, 0.0],
                [s, 0.0, c],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def rotation_z(angle_rad: float) -> np.ndarray:
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        return np.array(
            [
                [c, s, 0.0],
                [-s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @classmethod
    def euler_rotation_3x3(cls, rx: float, ry: float, rz: float) -> np.ndarray:
        """Combined 3x3 rotation matrix for row vectors: v_rot = v @ (Rz @ Rx @ Ry)."""
        return cls.rotation_z(rz) @ cls.rotation_x(rx) @ cls.rotation_y(ry)

    @classmethod
    def view_rotation_3x3(cls, yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
        """
        Camera view rotation matrix for row vectors: transforms world-relative (V - cam_pos)
        into camera space where +Z is forward, +X is right, +Y is up.
        """
        cy, sy = math.cos(-yaw), math.sin(-yaw)
        ry = np.array(
            [
                [cy, 0.0, -sy],
                [0.0, 1.0, 0.0],
                [sy, 0.0, cy],
            ],
            dtype=np.float32,
        )
        cp, sp = math.cos(-pitch), math.sin(-pitch)
        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cp, -sp],
                [0.0, sp, cp],
            ],
            dtype=np.float32,
        )
        mat = ry @ rx
        if abs(roll) > 1e-6:
            mat = mat @ cls.rotation_z(-roll)
        return mat


@dataclass
class Transform3D:
    """Represents 3D position, Euler rotation (radians), and scale for any Mesh3D object."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @position.setter
    def position(self, value: tuple[float, float, float] | list[float] | np.ndarray) -> None:
        self.x, self.y, self.z = float(value[0]), float(value[1]), float(value[2])

    @property
    def rotation(self) -> tuple[float, float, float]:
        return (self.rotation_x, self.rotation_y, self.rotation_z)

    @rotation.setter
    def rotation(self, value: tuple[float, float, float] | list[float] | np.ndarray) -> None:
        self.rotation_x, self.rotation_y, self.rotation_z = float(value[0]), float(value[1]), float(value[2])

    @property
    def scale(self) -> tuple[float, float, float]:
        return (self.scale_x, self.scale_y, self.scale_z)

    @scale.setter
    def scale(self, value: float | tuple[float, float, float] | list[float]) -> None:
        if isinstance(value, (int, float)):
            s = float(value)
            self.scale_x = self.scale_y = self.scale_z = s
        else:
            self.scale_x, self.scale_y, self.scale_z = float(value[0]), float(value[1]), float(value[2])

    def apply_to_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """Transform local (N, 3) vertices into world space (scale -> rotate -> translate)."""
        v = np.asarray(vertices, dtype=np.float32)
        if self.scale_x != 1.0 or self.scale_y != 1.0 or self.scale_z != 1.0:
            v = v * np.array([self.scale_x, self.scale_y, self.scale_z], dtype=np.float32)
        if abs(self.rotation_x) > 1e-6 or abs(self.rotation_y) > 1e-6 or abs(self.rotation_z) > 1e-6:
            rot = Matrix4.euler_rotation_3x3(self.rotation_x, self.rotation_y, self.rotation_z)
            v = v @ rot
        if self.x != 0.0 or self.y != 0.0 or self.z != 0.0:
            v = v + np.array([self.x, self.y, self.z], dtype=np.float32)
        return v
