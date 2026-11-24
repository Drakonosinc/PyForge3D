from __future__ import annotations

import math
from typing import Sequence
import numpy as np
from ..math.transforms import Transform3D


class Mesh3D:
    """
    Represents a 3D polygonal mesh with local vertices (V, 3), polygonal faces (triangles or quads),
    per-face RGB colors, and a Transform3D for world-space positioning, rotation, and scaling.
    """

    def __init__(
        self,
        vertices: np.ndarray | Sequence[Sequence[float]],
        faces: Sequence[Sequence[int]],
        face_colors: Sequence[tuple[int, int, int]] | tuple[int, int, int] = (56, 189, 248),
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float | tuple[float, float, float] = 1.0,
        wireframe: bool = False,
        outline_color: tuple[int, int, int] | None = None,
        backface_culling: bool = True,
    ) -> None:
        self.vertices: np.ndarray = np.asarray(vertices, dtype=np.float32)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("Mesh3D vertices must have shape (N, 3).")

        self.faces: list[tuple[int, ...]] = [tuple(int(idx) for idx in f) for f in faces if len(f) >= 3]
        num_faces = len(self.faces)

        if (
            isinstance(face_colors, (tuple, list))
            and len(face_colors) == 3
            and all(isinstance(c, (int, float)) for c in face_colors)
        ):
            rgb = (int(face_colors[0]), int(face_colors[1]), int(face_colors[2]))
            self.face_colors = np.tile(np.array(rgb, dtype=np.float32), (num_faces, 1))
        else:
            colors_list = [
                (int(c[0]), int(c[1]), int(c[2]))
                for c in face_colors  # type: ignore[union-attr]
            ]
            if len(colors_list) < num_faces:
                default_c = colors_list[-1] if colors_list else (56, 189, 248)
                colors_list.extend([default_c] * (num_faces - len(colors_list)))
            self.face_colors = np.asarray(colors_list[:num_faces], dtype=np.float32)

        self.transform = Transform3D(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            rotation_x=float(rotation[0]),
            rotation_y=float(rotation[1]),
            rotation_z=float(rotation[2]),
        )
        self.transform.scale = scale
        self.wireframe: bool = bool(wireframe)
        self.outline_color: tuple[int, int, int] | None = outline_color
        self.backface_culling: bool = bool(backface_culling)
        self.visible: bool = True

    def get_world_vertices(self) -> np.ndarray:
        """Return (V, 3) vertices transformed into world space."""
        return self.transform.apply_to_vertices(self.vertices)

    def set_color(self, color: tuple[int, int, int]) -> None:
        rgb = np.array([int(color[0]), int(color[1]), int(color[2])], dtype=np.float32)
        self.face_colors = np.tile(rgb, (len(self.faces), 1))

    @classmethod
    def from_obj(
        cls,
        path_or_text: str,
        color: tuple[int, int, int] = (56, 189, 248),
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float | tuple[float, float, float] = 1.0,
    ) -> Mesh3D:
        """Load a Wavefront .obj file from disk path or raw OBJ string."""
        return OBJLoader.load(path_or_text, color=color, position=position, scale=scale)


class OBJLoader:
    """Pure-Python Wavefront .obj parser supporting triangular and polygonal faces."""

    @staticmethod
    def parse_obj_text(
        obj_text: str,
        color: tuple[int, int, int] = (56, 189, 248),
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float | tuple[float, float, float] = 1.0,
    ) -> Mesh3D:
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, ...]] = []

        for raw_line in obj_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            prefix = parts[0].lower()
            if prefix == "v" and len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif prefix == "f" and len(parts) >= 4:
                face_indices: list[int] = []
                for token in parts[1:]:
                    v_str = token.split("/")[0]
                    idx = int(v_str)
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx = idx - 1
                    face_indices.append(idx)
                if len(face_indices) >= 3:
                    faces.append(tuple(face_indices))

        if not vertices:
            raise ValueError("OBJ data contains no valid 'v' vertices.")
        return Mesh3D(vertices=vertices, faces=faces, face_colors=color, position=position, scale=scale)

    @classmethod
    def load(
        cls,
        path_or_text: str,
        color: tuple[int, int, int] = (56, 189, 248),
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale: float | tuple[float, float, float] = 1.0,
    ) -> Mesh3D:
        if "\n" in path_or_text or path_or_text.lstrip().startswith(("v ", "#")):
            return cls.parse_obj_text(path_or_text, color=color, position=position, scale=scale)
        with open(path_or_text, "r", encoding="utf-8") as f:
            return cls.parse_obj_text(f.read(), color=color, position=position, scale=scale)


class Primitives:
    """Factory methods for standard 3D geometric primitives with consistent outward-facing normals."""

    @staticmethod
    def cube(
        size: float | tuple[float, float, float] = 1.0,
        color: tuple[int, int, int] = (56, 189, 248),
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        outline_color: tuple[int, int, int] | None = None,
    ) -> Mesh3D:
        if isinstance(size, (int, float)):
            hx = hy = hz = float(size) * 0.5
        else:
            hx, hy, hz = float(size[0]) * 0.5, float(size[1]) * 0.5, float(size[2]) * 0.5

        vertices = np.array(
            [
                [-hx, -hy, -hz],  # 0: bottom-left-front
                [-hx, +hy, -hz],  # 1: top-left-front
                [+hx, +hy, -hz],  # 2: top-right-front
                [+hx, -hy, -hz],  # 3: bottom-right-front
                [-hx, -hy, +hz],  # 4: bottom-left-back
                [-hx, +hy, +hz],  # 5: top-left-back
                [+hx, +hy, +hz],  # 6: top-right-back
                [+hx, -hy, +hz],  # 7: bottom-right-back
            ],
            dtype=np.float32,
        )
        faces = [
            (0, 1, 2, 3),  # Front (-Z)
            (7, 6, 5, 4),  # Back (+Z)
            (3, 2, 6, 7),  # Right (+X)
            (4, 5, 1, 0),  # Left (-X)
            (1, 5, 6, 2),  # Top (+Y)
            (4, 0, 3, 7),  # Bottom (-Y)
        ]
        return Mesh3D(
            vertices=vertices,
            faces=faces,
            face_colors=color,
            position=position,
            rotation=rotation,
            outline_color=outline_color,
        )

    @staticmethod
    def pyramid(
        base_size: float = 1.0,
        height: float = 1.2,
        color: tuple[int, int, int] = (255, 199, 51),
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        outline_color: tuple[int, int, int] | None = None,
    ) -> Mesh3D:
        h = float(base_size) * 0.5
        hy = float(height) * 0.5
        vertices = np.array(
            [
                [-h, -hy, -h],  # 0: front-left
                [+h, -hy, -h],  # 1: front-right
                [+h, -hy, +h],  # 2: back-right
                [-h, -hy, +h],  # 3: back-left
                [0.0, +hy, 0.0],  # 4: apex
            ],
            dtype=np.float32,
        )
        faces = [
            (0, 4, 1),  # Front
            (1, 4, 2),  # Right
            (2, 4, 3),  # Back
            (3, 4, 0),  # Left
            (3, 0, 1, 2),  # Bottom
        ]
        return Mesh3D(
            vertices=vertices,
            faces=faces,
            face_colors=color,
            position=position,
            outline_color=outline_color,
        )

    @staticmethod
    def plane(
        width: float = 4.0,
        depth: float = 4.0,
        subdivisions_x: int = 2,
        subdivisions_z: int = 2,
        color: tuple[int, int, int] = (34, 197, 94),
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        double_sided: bool = True,
    ) -> Mesh3D:
        nx = max(1, int(subdivisions_x))
        nz = max(1, int(subdivisions_z))
        xs = np.linspace(-width * 0.5, width * 0.5, nx + 1, dtype=np.float32)
        zs = np.linspace(-depth * 0.5, depth * 0.5, nz + 1, dtype=np.float32)

        vertices: list[tuple[float, float, float]] = []
        for z in zs:
            for x in xs:
                vertices.append((float(x), 0.0, float(z)))

        faces: list[tuple[int, ...]] = []
        stride = nx + 1
        for iz in range(nz):
            for ix in range(nx):
                i0 = iz * stride + ix
                i1 = i0 + 1
                i2 = (iz + 1) * stride + ix + 1
                i3 = (iz + 1) * stride + ix
                faces.append((i0, i3, i2, i1))  # Top (+Y)

        return Mesh3D(
            vertices=vertices,
            faces=faces,
            face_colors=color,
            position=position,
            backface_culling=not double_sided,
        )

    @staticmethod
    def sphere(
        radius: float = 1.0,
        rings: int = 8,
        sectors: int = 12,
        color: tuple[int, int, int] = (168, 85, 247),
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Mesh3D:
        r = float(radius)
        n_rings = max(3, int(rings))
        n_sectors = max(4, int(sectors))

        vertices: list[tuple[float, float, float]] = []
        for i in range(n_rings + 1):
            phi = math.pi * i / n_rings
            y = r * math.cos(phi)
            sin_phi = math.sin(phi)
            for j in range(n_sectors):
                theta = math.tau * j / n_sectors
                x = r * sin_phi * math.cos(theta)
                z = r * sin_phi * math.sin(theta)
                vertices.append((x, y, z))

        faces: list[tuple[int, ...]] = []
        for i in range(n_rings):
            for j in range(n_sectors):
                next_j = (j + 1) % n_sectors
                i0 = i * n_sectors + j
                i1 = i * n_sectors + next_j
                i2 = (i + 1) * n_sectors + next_j
                i3 = (i + 1) * n_sectors + j
                faces.append((i0, i1, i2, i3))

        return Mesh3D(vertices=vertices, faces=faces, face_colors=color, position=position)

    @staticmethod
    def cylinder(
        radius: float = 0.6,
        height: float = 1.5,
        segments: int = 12,
        color: tuple[int, int, int] = (56, 189, 248),
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Mesh3D:
        r = float(radius)
        hy = float(height) * 0.5
        n_seg = max(4, int(segments))

        vertices: list[tuple[float, float, float]] = []
        for j in range(n_seg):
            theta = math.tau * j / n_seg
            vertices.append((r * math.cos(theta), -hy, r * math.sin(theta)))
        for j in range(n_seg):
            theta = math.tau * j / n_seg
            vertices.append((r * math.cos(theta), +hy, r * math.sin(theta)))

        faces: list[tuple[int, ...]] = []
        for j in range(n_seg):
            nj = (j + 1) % n_seg
            b0, b1 = j, nj
            t0, t1 = n_seg + j, n_seg + nj
            faces.append((b0, t0, t1, b1))

        # Top & bottom caps
        top_ring = tuple(n_seg + j for j in range(n_seg))
        bot_ring = tuple(j for j in reversed(range(n_seg)))
        faces.append(top_ring)
        faces.append(bot_ring)

        return Mesh3D(vertices=vertices, faces=faces, face_colors=color, position=position)
