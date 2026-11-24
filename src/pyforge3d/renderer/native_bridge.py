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

    lib.pf3d_clear_zbuf.argtypes = [c_f32_p, ctypes.c_int]
    lib.pf3d_clear_zbuf.restype = None

    lib.pf3d_rasterize_quads.argtypes = [
        c_u8_p, ctypes.c_int,   # fb_bgra, pitch_bytes
        c_f32_p,                # z_buf
        ctypes.c_int, ctypes.c_int,  # width, height
        ctypes.c_int, ctypes.c_int, ctypes.c_int,  # r_off, g_off, b_off
        c_f32_p, c_f32_p, c_f32_p,  # sx, sy, iz
        c_i32_p,                # tex_ids
        c_f32_p, c_f32_p,       # shades, fog_f
        ctypes.c_float, ctypes.c_float, ctypes.c_float,  # fog_r, fog_g, fog_b
        c_u8_p, c_u8_p,         # solid_rgb, alphas
        c_u8_p, c_u8_p,         # atlas_rgb, atlas_alpha
        c_i32_p, c_i32_p, c_i32_p, c_u8_p,  # tex_offsets, widths, heights, cutout
        c_i32_p, ctypes.c_int,  # order, num_draw
    ]
    lib.pf3d_rasterize_quads.restype = None

    lib.pf3d_rasterize_polygon.argtypes = [
        c_u8_p, ctypes.c_int,
        c_f32_p,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        c_f32_p, c_f32_p, c_f32_p, c_f32_p, c_f32_p,  # px, py, iz, u, v
        ctypes.c_int,           # n_verts
        ctypes.c_int,           # tid
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
        cmd.append("-fPIC")
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
