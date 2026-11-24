from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from ..voxels.voxel_world import VoxelWorld


@dataclass
class AABB3D:
    """3D Axis-Aligned Bounding Box defined by center (x, y, z) and full dimensions (width, height, depth)."""

    x: float
    y: float
    z: float
    width: float = 0.6
    height: float = 1.7
    depth: float = 0.6

    @property
    def min_x(self) -> float:
        return self.x - self.width * 0.5

    @property
    def max_x(self) -> float:
        return self.x + self.width * 0.5

    @property
    def min_y(self) -> float:
        return self.y - self.height * 0.5

    @property
    def max_y(self) -> float:
        return self.y + self.height * 0.5

    @property
    def min_z(self) -> float:
        return self.z - self.depth * 0.5

    @property
    def max_z(self) -> float:
        return self.z + self.depth * 0.5

    def intersects(self, other: AABB3D) -> bool:
        return (
            self.min_x < other.max_x
            and self.max_x > other.min_x
            and self.min_y < other.max_y
            and self.max_y > other.min_y
            and self.min_z < other.max_z
            and self.max_z > other.min_z
        )


class VoxelPhysicsBody:
    """
    3D physics character body with gravity, jumping, and per-axis sliding collision resolution
    against a VoxelWorld so players or AI agents can walk and jump across 3D voxel terrain.
    """

    def __init__(
        self,
        position: tuple[float, float, float] = (2.5, 4.0, 2.5),
        width: float = 0.6,
        height: float = 1.7,
        depth: float = 0.6,
        gravity: float = 20.0,
        jump_speed: float = 7.5,
        eye_offset_y: float = 0.65,
    ) -> None:
        self.aabb = AABB3D(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            width=float(width),
            height=float(height),
            depth=float(depth),
        )
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.vz: float = 0.0
        self.gravity: float = float(gravity)
        self.jump_speed: float = float(jump_speed)
        self.eye_offset_y: float = float(eye_offset_y)
        self.on_ground: bool = False

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.aabb.x, self.aabb.y, self.aabb.z)

    @position.setter
    def position(self, value: tuple[float, float, float] | list[float] | np.ndarray) -> None:
        self.aabb.x, self.aabb.y, self.aabb.z = float(value[0]), float(value[1]), float(value[2])

    @property
    def eye_position(self) -> tuple[float, float, float]:
        return (self.aabb.x, self.aabb.y + self.eye_offset_y, self.aabb.z)

    def jump(self) -> bool:
        if self.on_ground:
            self.vy = self.jump_speed
            self.on_ground = False
            return True
        return False

    def _collides_with_world(self, world: VoxelWorld) -> bool:
        eps = 1e-4
        x0 = math.floor(self.aabb.min_x + eps)
        x1 = math.floor(self.aabb.max_x - eps)
        y0 = math.floor(self.aabb.min_y + eps)
        y1 = math.floor(self.aabb.max_y - eps)
        z0 = math.floor(self.aabb.min_z + eps)
        z1 = math.floor(self.aabb.max_z - eps)

        for bx in range(x0, x1 + 1):
            for by in range(y0, y1 + 1):
                for bz in range(z0, z1 + 1):
                    if (bx, by, bz) in world.blocks:
                        return True
        return False

    def step(self, world: VoxelWorld, dt: float, move_x: float = 0.0, move_z: float = 0.0) -> tuple[float, float, float]:
        """Advance physics by dt seconds with per-axis voxel collision resolution."""
        dt = max(0.0, min(0.1, float(dt)))
        self.vx = float(move_x)
        self.vz = float(move_z)
        self.vy -= self.gravity * dt

        # Resolve X axis
        dx = self.vx * dt
        self.aabb.x += dx
        if self._collides_with_world(world):
            self.aabb.x -= dx
            self.vx = 0.0

        # Resolve Z axis
        dz = self.vz * dt
        self.aabb.z += dz
        if self._collides_with_world(world):
            self.aabb.z -= dz
            self.vz = 0.0

        # Resolve Y axis
        dy = self.vy * dt
        self.aabb.y += dy
        self.on_ground = False
        if self._collides_with_world(world):
            self.aabb.y -= dy
            if self.vy < 0.0:
                self.on_ground = True
            self.vy = 0.0

        return self.position
