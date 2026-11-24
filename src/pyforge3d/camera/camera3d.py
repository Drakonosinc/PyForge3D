from __future__ import annotations

import math
import numpy as np
import pygame
from ..math.transforms import Matrix4


class Camera3D:
    """Perspective 3D camera supporting Yaw/Pitch/Roll orientation, view transformation, and screen projection."""

    def __init__(
        self,
        position: tuple[float, float, float] = (0.0, 2.0, -5.0),
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        fov: float = 70.0,
        near: float = 0.15,
        far: float = 80.0,
    ) -> None:
        self.x: float = float(position[0])
        self.y: float = float(position[1])
        self.z: float = float(position[2])
        self.yaw: float = float(yaw)
        self.pitch: float = float(pitch)
        self.roll: float = float(roll)
        self.fov: float = max(20.0, min(140.0, float(fov)))
        self.near: float = max(0.05, float(near))
        self.far: float = max(self.near + 1.0, float(far))

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    @position.setter
    def position(self, value: tuple[float, float, float] | list[float] | np.ndarray) -> None:
        self.x, self.y, self.z = float(value[0]), float(value[1]), float(value[2])

    @property
    def forward(self) -> np.ndarray:
        """Unit forward direction vector in world space."""
        cp = math.cos(self.pitch)
        return np.array(
            [
                math.sin(self.yaw) * cp,
                math.sin(self.pitch),
                math.cos(self.yaw) * cp,
            ],
            dtype=np.float32,
        )

    @property
    def forward_flat(self) -> np.ndarray:
        """Horizontal forward vector on the XZ plane (useful for FPS walking)."""
        return np.array([math.sin(self.yaw), 0.0, math.cos(self.yaw)], dtype=np.float32)

    @property
    def right(self) -> np.ndarray:
        """Unit right direction vector on the XZ plane."""
        return np.array([math.cos(self.yaw), 0.0, -math.sin(self.yaw)], dtype=np.float32)

    @property
    def up(self) -> np.ndarray:
        """Unit up direction vector in world space."""
        r = self.right
        f = self.forward
        u = np.cross(f, r)
        norm = float(np.linalg.norm(u))
        return u / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def look_at(self, target: tuple[float, float, float] | np.ndarray) -> None:
        """Orient the camera's yaw and pitch toward a 3D world point."""
        dx = float(target[0]) - self.x
        dy = float(target[1]) - self.y
        dz = float(target[2]) - self.z
        horiz = math.hypot(dx, dz)
        self.yaw = math.atan2(dx, dz)
        self.pitch = math.atan2(dy, max(1e-6, horiz))

    def get_view_matrix_3x3(self) -> np.ndarray:
        return Matrix4.view_rotation_3x3(self.yaw, self.pitch, self.roll)

    def world_to_camera(self, vertices_world: np.ndarray) -> np.ndarray:
        """Transform (N, 3) world-space vertices into camera space."""
        rel = np.asarray(vertices_world, dtype=np.float32) - self.position
        return rel @ self.get_view_matrix_3x3()

    def project_camera_vertices(
        self,
        vertices_cam: np.ndarray,
        screen_width: int,
        screen_height: int,
    ) -> np.ndarray:
        """
        Project (N, 3) camera-space vertices (with z >= near) into 2D screen coordinates (N, 2).
        """
        pts = np.asarray(vertices_cam, dtype=np.float32)
        focal = 1.0 / math.tan(math.radians(self.fov) * 0.5)
        half_w = screen_width * 0.5
        half_h = screen_height * 0.5
        scale = focal * half_h

        z_safe = np.maximum(pts[:, 2], self.near * 0.5)
        sx = half_w + (pts[:, 0] / z_safe) * scale
        sy = half_h - (pts[:, 1] / z_safe) * scale
        return np.column_stack((sx, sy))


class FirstPersonController:
    """Plug-and-play FPS / Minecraft camera controller supporting WASD movement, mouse look, and flying/walking."""

    def __init__(
        self,
        camera: Camera3D,
        move_speed: float = 6.0,
        mouse_sensitivity: float = 0.0025,
        turn_speed: float = 2.2,
        max_pitch: float = 1.48,
    ) -> None:
        self.camera = camera
        self.move_speed = float(move_speed)
        self.mouse_sensitivity = float(mouse_sensitivity)
        self.turn_speed = float(turn_speed)
        self.max_pitch = float(max_pitch)

    def rotate(self, delta_yaw: float, delta_pitch: float) -> None:
        self.camera.yaw = (self.camera.yaw + delta_yaw) % math.tau
        self.camera.pitch = max(-self.max_pitch, min(self.max_pitch, self.camera.pitch + delta_pitch))

    def handle_mouse_motion(self, rel_x: int, rel_y: int) -> None:
        self.rotate(rel_x * self.mouse_sensitivity, -rel_y * self.mouse_sensitivity)

    def update(self, dt: float, keys=None, fly_mode: bool = True) -> np.ndarray:
        """
        Process keyboard input (WASD for movement, Arrow keys for camera rotation, Space/LShift for vertical).
        Returns the world-space displacement vector applied to the camera.
        """
        if keys is None:
            keys = pygame.key.get_pressed()

        # Arrow keys rotate camera even without mouse lock
        yaw_input = float(bool(keys[pygame.K_RIGHT])) - float(bool(keys[pygame.K_LEFT]))
        pitch_input = float(bool(keys[pygame.K_UP])) - float(bool(keys[pygame.K_DOWN]))
        if yaw_input != 0.0 or pitch_input != 0.0:
            self.rotate(yaw_input * self.turn_speed * dt, pitch_input * self.turn_speed * dt)

        fwd_input = float(bool(keys[pygame.K_w])) - float(bool(keys[pygame.K_s]))
        right_input = float(bool(keys[pygame.K_d])) - float(bool(keys[pygame.K_a]))
        up_input = float(bool(keys[pygame.K_SPACE])) - float(
            bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or keys[pygame.K_c])
        )

        move_vec = self.camera.forward_flat * fwd_input + self.camera.right * right_input
        horiz_norm = float(np.linalg.norm(move_vec))
        if horiz_norm > 1e-6:
            move_vec = move_vec / horiz_norm

        if fly_mode:
            move_vec[1] += up_input

        displacement = move_vec * (self.move_speed * dt)
        self.camera.position = self.camera.position + displacement
        return displacement


class OrbitCamera:
    """Orbital controller rotating a Camera3D smoothly around a 3D target point."""

    def __init__(
        self,
        camera: Camera3D,
        target: tuple[float, float, float] = (0.0, 0.0, 0.0),
        distance: float = 8.0,
        azimuth: float = 0.0,
        elevation: float = 0.45,
    ) -> None:
        self.camera = camera
        self.target = np.array(target, dtype=np.float32)
        self.distance = max(0.5, float(distance))
        self.azimuth = float(azimuth)
        self.elevation = max(-1.45, min(1.45, float(elevation)))
        self.sync_camera()

    def orbit(self, delta_azimuth: float, delta_elevation: float = 0.0, delta_distance: float = 0.0) -> None:
        self.azimuth = (self.azimuth + delta_azimuth) % math.tau
        self.elevation = max(-1.45, min(1.45, self.elevation + delta_elevation))
        self.distance = max(0.5, self.distance + delta_distance)
        self.sync_camera()

    def sync_camera(self) -> None:
        ce = math.cos(self.elevation)
        offset = np.array(
            [
                -math.sin(self.azimuth) * ce * self.distance,
                math.sin(self.elevation) * self.distance,
                -math.cos(self.azimuth) * ce * self.distance,
            ],
            dtype=np.float32,
        )
        self.camera.position = self.target + offset
        self.camera.look_at(self.target)
