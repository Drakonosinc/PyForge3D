"""PyForge3D: Vectorized 3D Software Renderer, Voxel Engine, Meshes, Raycasting & AABB Physics for Pygame."""

__version__ = "1.0.0"
__author__ = "Drakonosinc"

from .camera import Camera3D, FirstPersonController, OrbitCamera
from .lighting import DirectionalLight, Fog3D, PointLight
from .math import (
    Matrix4,
    Transform3D,
    clip_polygon_near_plane,
    compute_face_normals,
    normalize_vectors,
)
from .meshes import Mesh3D, OBJLoader, Primitives
from .physics import AABB3D, VoxelPhysicsBody
from .renderer import Renderer3D, Scene3D
from .voxels import VoxelHit, VoxelMaterial, VoxelWorld

__all__ = [
    "__version__",
    "AABB3D",
    "Camera3D",
    "DirectionalLight",
    "FirstPersonController",
    "Fog3D",
    "Matrix4",
    "Mesh3D",
    "OBJLoader",
    "OrbitCamera",
    "PointLight",
    "Primitives",
    "Renderer3D",
    "Scene3D",
    "Transform3D",
    "VoxelHit",
    "VoxelMaterial",
    "VoxelPhysicsBody",
    "VoxelWorld",
    "clip_polygon_near_plane",
    "compute_face_normals",
    "normalize_vectors",
]
