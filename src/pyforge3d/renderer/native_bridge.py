from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys


_DIR = os.path.dirname(os.path.abspath(__file__))
_C_SOURCE = os.path.join(_DIR, "_native_rasterizer.c")
_LIB_EXT = ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
_LIB_PATH = os.path.join(_DIR, f"_pyforge3d_native{_LIB_EXT}")


def _configure_signatures(lib: ctypes.CDLL) -> ctypes.CDLL:
    c_u8_p = ctypes.c_void_p
    c_f32_p = ctypes.c_void_p
    c_i32_p = ctypes.c_void_p
    c_ptr_p = ctypes.c_void_p

    lib.pf3d_clear_zbuf.argtypes = [c_f32_p, ctypes.c_int]
    lib.pf3d_clear_zbuf.restype = None

    lib.pf3d_get_num_threads.argtypes = []
    lib.pf3d_get_num_threads.restype = ctypes.c_int

    lib.pf3d_set_num_threads.argtypes = [ctypes.c_int]
    lib.pf3d_set_num_threads.restype = ctypes.c_int

    lib.pf3d_greedy_mesh_2d.argtypes = [c_i32_p, ctypes.c_int, ctypes.c_int, c_i32_p, ctypes.c_int]
    lib.pf3d_greedy_mesh_2d.restype = ctypes.c_int

    lib.pf3d_mesh_chunk_greedy.argtypes = [
        c_ptr_p,                                    # grid (int16)
        ctypes.c_int, ctypes.c_int,                 # cs, h_span
        ctypes.c_int, ctypes.c_int, ctypes.c_int,   # x0, min_y, z0
        ctypes.c_int, ctypes.c_int,                 # greedy_enabled, n_mats
        c_u8_p, c_u8_p, c_f32_p,                    # mat_trans, mat_alpha, mat_emissive
        c_f32_p, c_i32_p, c_u8_p,                   # mat_colors, mat_tex_ids, mat_face_tr
        c_f32_p, c_f32_p, c_f32_p, c_f32_p,         # out_quads, normals, colors, uv_scales
        c_i32_p, c_u8_p, c_f32_p, c_f32_p,          # out_tex_ids, alphas, emissive, ao
        c_u8_p, c_i32_p, c_u8_p,                    # out_is_tr, mids, ftypes
        ctypes.c_int,                               # max_faces
    ]
    lib.pf3d_mesh_chunk_greedy.restype = ctypes.c_int

    lib.pf3d_render_voxel_chunks.argtypes = [
        c_u8_p, ctypes.c_int,                       # fb_bgra, pitch_bytes
        c_f32_p,                                    # z_buf
        ctypes.c_int, ctypes.c_int,                 # width, height
        ctypes.c_int, ctypes.c_int, ctypes.c_int,   # r_off, g_off, b_off
        ctypes.c_int,                               # n_chunks
        c_ptr_p, c_ptr_p, c_ptr_p, c_ptr_p,         # c_quads, normals, colors, uv_scales
        c_ptr_p, c_ptr_p, c_ptr_p, c_ptr_p, c_ptr_p,# c_tex_ids, alphas, emissive, ao, is_tr
        c_i32_p,                                    # c_counts
        ctypes.c_float, ctypes.c_float, ctypes.c_float,  # cam_x, cam_y, cam_z
        c_f32_p,                                    # view_mat (9 floats)
        ctypes.c_float, ctypes.c_float,             # near_z, far_z
        ctypes.c_float, ctypes.c_float, ctypes.c_float,  # half_w, half_h, proj_scale
        ctypes.c_float,                             # lod_dist
        ctypes.c_int,                               # light_type
        ctypes.c_float, ctypes.c_float, ctypes.c_float,  # lx, ly, lz
        ctypes.c_float, ctypes.c_float, ctypes.c_float,  # l_ambient, l_diffuse, l_atten
        ctypes.c_int, ctypes.c_float, ctypes.c_float,    # fog_enabled, fog_near, fog_inv_span
        ctypes.c_float, ctypes.c_float, ctypes.c_float,  # fog_r, fog_g, fog_b
        c_u8_p, c_u8_p,                             # atlas_rgb, atlas_alpha
        c_i32_p, c_i32_p, c_i32_p, c_u8_p,          # tex_offsets, widths, heights, cutout
        ctypes.c_int,                               # clear_z
    ]
    lib.pf3d_render_voxel_chunks.restype = ctypes.c_int

    lib.pf3d_rasterize_quads.argtypes = [
        c_u8_p, ctypes.c_int,
        c_f32_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        c_f32_p, c_f32_p, c_f32_p,
        c_f32_p,
        c_i32_p,
        c_f32_p, c_f32_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        c_u8_p, c_u8_p,
        c_u8_p, c_u8_p,
        c_i32_p, c_i32_p, c_i32_p, c_u8_p,
        c_i32_p, ctypes.c_int,
    ]
    lib.pf3d_rasterize_quads.restype = None

    lib.pf3d_rasterize_polygon.argtypes = [
        c_u8_p, ctypes.c_int,
        c_f32_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        c_f32_p, c_f32_p, c_f32_p, c_f32_p, c_f32_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8, ctypes.c_uint8,
        c_u8_p, c_u8_p,
        c_i32_p, c_i32_p, c_i32_p, c_u8_p,
    ]
    lib.pf3d_rasterize_polygon.restype = None
    return lib


def build_native_rasterizer(force: bool = False) -> bool:
    """
    Compile the self-contained C99 native rasterizer into a shared library next to this module.
    Returns `True` on success, `False` if no C compiler is available (pure-NumPy fallback stays active).
    """
    if not force and os.path.exists(_LIB_PATH):
        if os.path.getmtime(_LIB_PATH) >= os.path.getmtime(_C_SOURCE):
            return True

    compiler = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        return False

    cmd = [compiler, "-O3", "-ffast-math", "-shared"]
    if sys.platform == "win32":
        cmd.append("-static-libgcc")
    else:
        cmd.extend(["-fPIC", "-pthread"])
    cmd.extend([_C_SOURCE, "-o", _LIB_PATH])
    if sys.platform != "win32":
        cmd.append("-lm")

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return res.returncode == 0 and os.path.exists(_LIB_PATH)
    except Exception:
        return False


def _load_native_lib() -> ctypes.CDLL | None:
    if not os.path.exists(_LIB_PATH) or (
        os.path.exists(_C_SOURCE) and os.path.getmtime(_LIB_PATH) < os.path.getmtime(_C_SOURCE)
    ):
        build_native_rasterizer(force=True)
    if not os.path.exists(_LIB_PATH):
        return None
    try:
        lib = ctypes.CDLL(_LIB_PATH)
        return _configure_signatures(lib)
    except Exception:
        return None


NATIVE_LIB: ctypes.CDLL | None = _load_native_lib()
HAS_NATIVE_RASTERIZER: bool = NATIVE_LIB is not None
