"""PyForge3D: High-Performance Dual-Backend (C99 Native + Vectorized NumPy) 3D Software Renderer, Textured Z-Buffer, Chunked Voxel Engine & Physics for Pygame."""

__version__ = "1.5.0"
__author__ = "Drakonosinc"

from .camera import Camera3D, FirstPersonController, OrbitCamera
from .lighting import DirectionalLight, Fog3D, PointLight, SkyDome3D
from .math import (
    Matrix4,
    Transform3D,
    clip_polygon_near_plane,
    compute_face_normals,
    normalize_vectors,
)
from .meshes import Mesh3D, OBJLoader, Primitives
from .physics import AABB3D, VoxelPhysicsBody
from .procedural import FractalNoise2D, PerlinNoise2D, hash_2d
from .renderer import (
    CANONICAL_QUAD_UV,
    HAS_NATIVE_RASTERIZER,
    Renderer3D,
    Scene3D,
    SoftwareTexturedRasterizer,
    build_native_rasterizer,
    clip_textured_polygon_near_plane,
)
from .textures import Texture2D, TextureAtlas
from .voxels import Chunk3D, ChunkedVoxelWorld, VoxelHit, VoxelMaterial, VoxelWorld

__all__ = [
    "__version__",
    "AABB3D",
    "CANONICAL_QUAD_UV",
    "HAS_NATIVE_RASTERIZER",
    "Camera3D",
    "Chunk3D",
    "ChunkedVoxelWorld",
    "DirectionalLight",
    "FirstPersonController",
    "Fog3D",
    "FractalNoise2D",
    "Matrix4",
    "Mesh3D",
    "OBJLoader",
    "OrbitCamera",
    "PerlinNoise2D",
    "PointLight",
    "Primitives",
    "Renderer3D",
    "Scene3D",
    "SkyDome3D",
    "SoftwareTexturedRasterizer",
    "Texture2D",
    "TextureAtlas",
    "Transform3D",
    "VoxelHit",
    "VoxelMaterial",
    "VoxelPhysicsBody",
    "VoxelWorld",
    "build_native_rasterizer",
    "clip_polygon_near_plane",
    "clip_textured_polygon_near_plane",
    "compute_face_normals",
    "hash_2d",
    "normalize_vectors",
]
