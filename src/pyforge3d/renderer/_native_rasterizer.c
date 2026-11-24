/*
 * PyForge3D Native C99 Scanline Z-Buffer Software Rasterizer
 * 100% Game-Agnostic | Zero External Dependencies | Cross-Platform ISO C99
 *
 * Features:
 * - Direct zero-copy writes into 32-bit row-major (H, W) SDL pixel buffers (dynamic RGB channel shifts).
 * - Exact scanline-clamped convex planar N-gon & quad rasterization (zero exterior pixel waste).
 * - Per-face pre-packed uint32_t shaded+fogged texture cache (single 32-bit store per pixel!).
 * - 8-pixel perspective-correct UV span interpolation (exact 1/z at span endpoints, 8x fewer divisions).
 * - Full RGBA alpha-cutout & translucency blending (Water, Glass, Foliage).
 */

#include <stdint.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
  #define PF3D_EXPORT __declspec(dllexport)
#else
  #define PF3D_EXPORT __attribute__((visibility("default")))
#endif

static inline int clamp_i(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline uint8_t clamp_u8(float v) {
    if (v <= 0.0f) return 0;
    if (v >= 255.0f) return 255;
    return (uint8_t)(v + 0.5f);
}

static inline uint32_t pack_pixel(uint8_t r, uint8_t g, uint8_t b, int r_sh, int g_sh, int b_sh) {
    return ((uint32_t)r << r_sh) | ((uint32_t)g << g_sh) | ((uint32_t)b << b_sh);
}

PF3D_EXPORT void pf3d_clear_zbuf(float *z_buf, int count) {
    if (count > 0) {
        memset(z_buf, 0, (size_t)count * sizeof(float));
    }
}

/*
 * Ultra-fast scanline-clamped rasterizer for a batch of convex canonical-UV planar quads.
 * Canonical quad UVs: v0=(0,1), v1=(0,0), v2=(1,0), v3=(1,1).
 */
PF3D_EXPORT void pf3d_rasterize_quads(
    uint8_t *fb_bgra,
    int pitch_bytes,
    float *z_buf,
    int width,
    int height,
    int r_off,
    int g_off,
    int b_off,
    const float *sx,
    const float *sy,
    const float *iz,
    const int32_t *tex_ids,
    const float *shades,
    const float *fog_f,
    float fog_r,
    float fog_g,
    float fog_b,
    const uint8_t *solid_rgb,
    const uint8_t *alphas,
    const uint8_t *atlas_rgb,
    const uint8_t *atlas_alpha,
    const int32_t *tex_offsets,
    const int32_t *tex_widths,
    const int32_t *tex_heights,
    const uint8_t *tex_cutout,
    const int32_t *order,
    int num_draw
) {
    const int r_sh = r_off * 8;
    const int g_sh = g_off * 8;
    const int b_sh = b_off * 8;
    uint32_t shaded_lut[1024]; /* Stack-allocated fast LUT for textures up to 32x32 */

    for (int oi = 0; oi < num_draw; ++oi) {
        const int i = order[oi];
        const int base4 = i * 4;

        const float x0 = sx[base4 + 0], y0 = sy[base4 + 0], iz0 = iz[base4 + 0];
        const float x1 = sx[base4 + 1], y1 = sy[base4 + 1], iz1 = iz[base4 + 1];
        const float x2 = sx[base4 + 2], y2 = sy[base4 + 2], iz2 = iz[base4 + 2];
        const float x3 = sx[base4 + 3], y3 = sy[base4 + 3], iz3 = iz[base4 + 3];

        float min_xf = x0, max_xf = x0;
        if (x1 < min_xf) min_xf = x1; else if (x1 > max_xf) max_xf = x1;
        if (x2 < min_xf) min_xf = x2; else if (x2 > max_xf) max_xf = x2;
        if (x3 < min_xf) min_xf = x3; else if (x3 > max_xf) max_xf = x3;

        float min_yf = y0, max_yf = y0;
        if (y1 < min_yf) min_yf = y1; else if (y1 > max_yf) max_yf = y1;
        if (y2 < min_yf) min_yf = y2; else if (y2 > max_yf) max_yf = y2;
        if (y3 < min_yf) min_yf = y3; else if (y3 > max_yf) max_yf = y3;

        const int min_x = min_xf < 0.0f ? 0 : (int)min_xf;
        const int max_x = max_xf >= (float)width ? width - 1 : (int)max_xf;
        const int min_y = min_yf < 0.0f ? 0 : (int)min_yf;
        const int max_y = max_yf >= (float)height ? height - 1 : (int)max_yf;
        if (min_x > max_x || min_y > max_y) continue;

        const float denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
        if (fabsf(denom) < 1e-3f) continue;

        const float orient = denom > 0.0f ? 1.0f : -1.0f;
        const float inv_denom = 1.0f / denom;

        const float a0 = (y1 - y2) * inv_denom;
        const float b0 = (x2 - x1) * inv_denom;
        const float c0 = -(a0 * x2 + b0 * y2);

        const float a1 = (y2 - y0) * inv_denom;
        const float b1 = (x0 - x2) * inv_denom;
        const float c1 = -(a1 * x2 + b1 * y2);

        const float a2 = -a0 - a1;
        const float b2 = -b0 - b1;
        const float c2 = 1.0f - c0 - c1;

        const float az = a0 * iz0 + a1 * iz1 + a2 * iz2;
        const float bz = b0 * iz0 + b1 * iz1 + b2 * iz2;
        const float cz = c0 * iz0 + c1 * iz1 + c2 * iz2;

        float ex[4], ey[4], ec[4], inv_ex[4];
        ex[0] = (y0 - y1) * orient; ey[0] = (x1 - x0) * orient; ec[0] = -(ex[0] * x0 + ey[0] * y0);
        ex[1] = (y1 - y2) * orient; ey[1] = (x2 - x1) * orient; ec[1] = -(ex[1] * x1 + ey[1] * y1);
        ex[2] = (y2 - y3) * orient; ey[2] = (x3 - x2) * orient; ec[2] = -(ex[2] * x2 + ey[2] * y2);
        ex[3] = (y3 - y0) * orient; ey[3] = (x0 - x3) * orient; ec[3] = -(ex[3] * x3 + ey[3] * y3);
        for (int e = 0; e < 4; ++e) {
            inv_ex[e] = (fabsf(ex[e]) > 1e-6f) ? (1.0f / ex[e]) : 0.0f;
        }

        const int tid = tex_ids[i];
        const uint8_t alp = alphas[i];
        const int has_cut = (tid >= 0) ? (int)tex_cutout[tid] : 0;
        const int bw = max_x - min_x + 1;
        const int bh = max_y - min_y + 1;
        const int use_tex = (tid >= 0) && ((bw > 3) || (bh > 3) || has_cut);

        const uint8_t sr = solid_rgb[i * 3 + 0];
        const uint8_t sg = solid_rgb[i * 3 + 1];
        const uint8_t sb = solid_rgb[i * 3 + 2];
        const uint32_t solid_packed = pack_pixel(sr, sg, sb, r_sh, g_sh, b_sh);

        const float sh = shades[i];
        const float ff = fog_f[i];
        const float sh_inv_ff = sh * (1.0f - ff);
        const float fr_add = fog_r * ff;
        const float fg_add = fog_g * ff;
        const float fb_add = fog_b * ff;

        int t_off = 0, tw = 1, th = 1, n_texels = 1;
        float tw_f = 1.0f, th_f = 1.0f;
        int use_lut = 0;
        const uint8_t *t_rgb = atlas_rgb;
        const uint8_t *t_alp = atlas_alpha;

        if (use_tex) {
            t_off = tex_offsets[tid];
            tw = tex_widths[tid];
            th = tex_heights[tid];
            n_texels = tw * th;
            tw_f = (float)tw;
            th_f = (float)th;
            t_rgb = atlas_rgb + (size_t)t_off * 3;
            t_alp = atlas_alpha + (size_t)t_off;

            /* Pre-pack entire shaded+fogged texture into uint32_t LUT when face area justifies it! */
            if (alp == 255 && !has_cut && n_texels <= 1024 && (bw * bh) >= 96) {
                for (int k = 0; k < n_texels; ++k) {
                    const uint8_t *tc = t_rgb + k * 3;
                    shaded_lut[k] = pack_pixel(
                        clamp_u8((float)tc[0] * sh_inv_ff + fr_add),
                        clamp_u8((float)tc[1] * sh_inv_ff + fg_add),
                        clamp_u8((float)tc[2] * sh_inv_ff + fb_add),
                        r_sh, g_sh, b_sh
                    );
                }
                use_lut = 1;
            }
        }

        /* Pre-multiply UV plane gradients by texture dimensions */
        const float au = a2 * iz2 * tw_f, bu = b2 * iz2 * tw_f, cu = c2 * iz2 * tw_f;
        const float av = a0 * iz0 * th_f, bv = b0 * iz0 * th_f, cv = c0 * iz0 * th_f;
        const int tw_m1 = tw - 1;
        const int th_m1 = th - 1;

        for (int py = min_y; py <= max_y; ++py) {
            const float yf = (float)py + 0.5f;
            float xl = (float)min_x + 0.5f;
            float xr = (float)max_x + 0.5f;
            int valid_row = 1;

            for (int e = 0; e < 4; ++e) {
                const float y_term = ey[e] * yf + ec[e];
                if (ex[e] > 1e-6f) {
                    const float bound = (-1e-3f - y_term) * inv_ex[e];
                    if (bound > xl) xl = bound;
                } else if (ex[e] < -1e-6f) {
                    const float bound = (-1e-3f - y_term) * inv_ex[e];
                    if (bound < xr) xr = bound;
                } else if (y_term < -1e-3f) {
                    valid_row = 0;
                    break;
                }
            }
            if (!valid_row || xl > xr) continue;

            int x_start = (int)ceilf(xl - 0.5f);
            int x_end = (int)floorf(xr - 0.5f);
            if (x_start < min_x) x_start = min_x;
            if (x_end > max_x) x_end = max_x;
            if (x_start > x_end) continue;

            float *z_row = z_buf + (size_t)py * width;
            uint8_t *fb_row = fb_bgra + (size_t)py * pitch_bytes;
            uint32_t *fb_u32 = (uint32_t *)fb_row;
            const float xf0 = (float)x_start + 0.5f;
            float rz = az * xf0 + bz * yf + cz;

            if (!use_tex) {
                if (alp == 255) {
                    for (int px = x_start; px <= x_end; ++px) {
                        if (rz > z_row[px]) {
                            fb_u32[px] = solid_packed;
                            z_row[px] = rz;
                        }
                        rz += az;
                    }
                } else {
                    const float pa = (float)alp * (1.0f / 255.0f);
                    const float ipa = 1.0f - pa;
                    const float sr_p = (float)sr * pa;
                    const float sg_p = (float)sg * pa;
                    const float sb_p = (float)sb * pa;
                    for (int px = x_start; px <= x_end; ++px) {
                        if (rz > z_row[px]) {
                            uint8_t *p = fb_row + px * 4;
                            p[r_off] = (uint8_t)((float)p[r_off] * ipa + sr_p);
                            p[g_off] = (uint8_t)((float)p[g_off] * ipa + sg_p);
                            p[b_off] = (uint8_t)((float)p[b_off] * ipa + sb_p);
                        }
                        rz += az;
                    }
                }
            } else if (use_lut) {
                /* Ultra-fast Hot Path A: Opaque Textured with uint32_t LUT + 8px Perspective Spans! */
                float ru = au * xf0 + bu * yf + cu;
                float rv = av * xf0 + bv * yf + cv;
                int px = x_start;
                while (px <= x_end) {
                    int span = x_end - px + 1;
                    if (span > 8) span = 8;
                    const float inv0 = 1.0f / rz;
                    float u_cur = ru * inv0;
                    float v_cur = rv * inv0;
                    float du = 0.0f, dv = 0.0f;
                    if (span > 1) {
                        const float span_m1 = (float)(span - 1);
                        const float inv1 = 1.0f / (rz + az * span_m1);
                        const float u1 = (ru + au * span_m1) * inv1;
                        const float v1 = (rv + av * span_m1) * inv1;
                        const float inv_s = 1.0f / span_m1;
                        du = (u1 - u_cur) * inv_s;
                        dv = (v1 - v_cur) * inv_s;
                    }
                    for (int s = 0; s < span; ++s, ++px) {
                        if (rz > z_row[px]) {
                            const int tx = clamp_i((int)u_cur, 0, tw_m1);
                            const int ty = clamp_i((int)v_cur, 0, th_m1);
                            fb_u32[px] = shaded_lut[ty * tw + tx];
                            z_row[px] = rz;
                        }
                        rz += az;
                        u_cur += du;
                        v_cur += dv;
                    }
                    ru += au * (float)span;
                    rv += av * (float)span;
                }
            } else if (alp == 255 && !has_cut) {
                /* Fast Hot Path B: Small Opaque Textured Quad + 8px Perspective Spans */
                float ru = au * xf0 + bu * yf + cu;
                float rv = av * xf0 + bv * yf + cv;
                int px = x_start;
                while (px <= x_end) {
                    int span = x_end - px + 1;
                    if (span > 8) span = 8;
                    const float inv0 = 1.0f / rz;
                    float u_cur = ru * inv0;
                    float v_cur = rv * inv0;
                    float du = 0.0f, dv = 0.0f;
                    if (span > 1) {
                        const float span_m1 = (float)(span - 1);
                        const float inv1 = 1.0f / (rz + az * span_m1);
                        const float inv_s = 1.0f / span_m1;
                        du = ((ru + au * span_m1) * inv1 - u_cur) * inv_s;
                        dv = ((rv + av * span_m1) * inv1 - v_cur) * inv_s;
                    }
                    for (int s = 0; s < span; ++s, ++px) {
                        if (rz > z_row[px]) {
                            const int tx = clamp_i((int)u_cur, 0, tw_m1);
                            const int ty = clamp_i((int)v_cur, 0, th_m1);
                            const uint8_t *tc = t_rgb + (ty * tw + tx) * 3;
                            fb_u32[px] = pack_pixel(
                                clamp_u8((float)tc[0] * sh_inv_ff + fr_add),
                                clamp_u8((float)tc[1] * sh_inv_ff + fg_add),
                                clamp_u8((float)tc[2] * sh_inv_ff + fb_add),
                                r_sh, g_sh, b_sh
                            );
                            z_row[px] = rz;
                        }
                        rz += az;
                        u_cur += du;
                        v_cur += dv;
                    }
                    ru += au * (float)span;
                    rv += av * (float)span;
                }
            } else {
                /* Path C: Translucent & Alpha-Cutout Textured Quad (Water, Glass, Cutouts) */
                float ru = au * xf0 + bu * yf + cu;
                float rv = av * xf0 + bv * yf + cv;
                const float alp_scale = (float)alp * (1.0f / 65025.0f);
                const float flat_pa = (float)alp * (1.0f / 255.0f);
                for (int px = x_start; px <= x_end; ++px) {
                    if (rz > z_row[px]) {
                        const float inv_iz = 1.0f / rz;
                        const int tx = clamp_i((int)(ru * inv_iz), 0, tw_m1);
                        const int ty = clamp_i((int)(rv * inv_iz), 0, th_m1);
                        const int tidx = ty * tw + tx;
                        float pa = flat_pa;
                        if (has_cut) {
                            const uint8_t ta = t_alp[tidx];
                            if (ta < 32) {
                                rz += az; ru += au; rv += av;
                                continue;
                            }
                            pa = (float)ta * alp_scale;
                        }
                        const uint8_t *tc = t_rgb + tidx * 3;
                        const float cr = (float)tc[0] * sh_inv_ff + fr_add;
                        const float cg = (float)tc[1] * sh_inv_ff + fg_add;
                        const float cb = (float)tc[2] * sh_inv_ff + fb_add;
                        uint8_t *p = fb_row + px * 4;
                        if (pa >= 0.96f) {
                            p[r_off] = clamp_u8(cr);
                            p[g_off] = clamp_u8(cg);
                            p[b_off] = clamp_u8(cb);
                            z_row[px] = rz;
                        } else {
                            const float ipa = 1.0f - pa;
                            p[r_off] = clamp_u8((float)p[r_off] * ipa + cr * pa);
                            p[g_off] = clamp_u8((float)p[g_off] * ipa + cg * pa);
                            p[b_off] = clamp_u8((float)p[b_off] * ipa + cb * pa);
                        }
                    }
                    rz += az;
                    ru += au;
                    rv += av;
                }
            }
        }
    }
}

/*
 * General convex planar N-gon rasterizer (3..8 vertices, arbitrary UVs)
 * Used for near-plane clipped voxel faces AND arbitrary Mesh3D polygons.
 */
PF3D_EXPORT void pf3d_rasterize_polygon(
    uint8_t *fb_bgra,
    int pitch_bytes,
    float *z_buf,
    int width,
    int height,
    int r_off,
    int g_off,
    int b_off,
    const float *px_arr,
    const float *py_arr,
    const float *iz_arr,
    const float *u_arr,
    const float *v_arr,
    int n_verts,
    int tid,
    float shade,
    float fog_f,
    float fog_r,
    float fog_g,
    float fog_b,
    uint8_t sr,
    uint8_t sg,
    uint8_t sb,
    uint8_t alp,
    const uint8_t *atlas_rgb,
    const uint8_t *atlas_alpha,
    const int32_t *tex_offsets,
    const int32_t *tex_widths,
    const int32_t *tex_heights,
    const uint8_t *tex_cutout
) {
    if (n_verts < 3 || n_verts > 8) return;
    const int r_sh = r_off * 8;
    const int g_sh = g_off * 8;
    const int b_sh = b_off * 8;

    float min_xf = px_arr[0], max_xf = px_arr[0];
    float min_yf = py_arr[0], max_yf = py_arr[0];
    for (int k = 1; k < n_verts; ++k) {
        if (px_arr[k] < min_xf) min_xf = px_arr[k];
        else if (px_arr[k] > max_xf) max_xf = px_arr[k];
        if (py_arr[k] < min_yf) min_yf = py_arr[k];
        else if (py_arr[k] > max_yf) max_yf = py_arr[k];
    }

    const int min_x = min_xf < 0.0f ? 0 : (int)min_xf;
    const int max_x = max_xf >= (float)width ? width - 1 : (int)max_xf;
    const int min_y = min_yf < 0.0f ? 0 : (int)min_yf;
    const int max_y = max_yf >= (float)height ? height - 1 : (int)max_yf;
    if (min_x > max_x || min_y > max_y) return;

    const float x0 = px_arr[0], y0 = py_arr[0], iz0 = iz_arr[0];
    int i1 = 1, i2 = 2;
    float denom = (py_arr[1] - py_arr[2]) * (x0 - px_arr[2]) + (px_arr[2] - px_arr[1]) * (y0 - py_arr[2]);
    if (fabsf(denom) < 1e-3f && n_verts > 3) {
        for (int k = 2; k < n_verts - 1; ++k) {
            float td = (py_arr[k] - py_arr[k + 1]) * (x0 - px_arr[k + 1]) + (px_arr[k + 1] - px_arr[k]) * (y0 - py_arr[k + 1]);
            if (fabsf(td) > fabsf(denom)) {
                denom = td; i1 = k; i2 = k + 1;
            }
        }
    }
    if (fabsf(denom) < 1e-3f) return;

    const float x1 = px_arr[i1], y1 = py_arr[i1], iz1 = iz_arr[i1];
    const float x2 = px_arr[i2], y2 = py_arr[i2], iz2 = iz_arr[i2];
    const float orient = denom > 0.0f ? 1.0f : -1.0f;
    const float inv_denom = 1.0f / denom;

    const float a0 = (y1 - y2) * inv_denom;
    const float b0 = (x2 - x1) * inv_denom;
    const float c0 = -(a0 * x2 + b0 * y2);

    const float a1 = (y2 - y0) * inv_denom;
    const float b1 = (x0 - x2) * inv_denom;
    const float c1 = -(a1 * x2 + b1 * y2);

    const float a2 = -a0 - a1;
    const float b2 = -b0 - b1;
    const float c2 = 1.0f - c0 - c1;

    const float az = a0 * iz0 + a1 * iz1 + a2 * iz2;
    const float bz = b0 * iz0 + b1 * iz1 + b2 * iz2;
    const float cz = c0 * iz0 + c1 * iz1 + c2 * iz2;

    float au = 0.0f, bu = 0.0f, cu = 0.0f;
    float av = 0.0f, bv = 0.0f, cv = 0.0f;
    const int has_cut = (tid >= 0) ? (int)tex_cutout[tid] : 0;
    const int use_tex = (tid >= 0) && (u_arr != NULL) && (v_arr != NULL);
    int t_off = 0, tw = 1, th = 1;
    float tw_f = 1.0f, th_f = 1.0f;
    if (use_tex) {
        t_off = tex_offsets[tid];
        tw = tex_widths[tid];
        th = tex_heights[tid];
        tw_f = (float)tw;
        th_f = (float)th;
        const float u0z = u_arr[0] * iz0 * tw_f, u1z = u_arr[i1] * iz1 * tw_f, u2z = u_arr[i2] * iz2 * tw_f;
        const float v0z = v_arr[0] * iz0 * th_f, v1z = v_arr[i1] * iz1 * th_f, v2z = v_arr[i2] * iz2 * th_f;
        au = a0 * u0z + a1 * u1z + a2 * u2z;
        bu = b0 * u0z + b1 * u1z + b2 * u2z;
        cu = c0 * u0z + c1 * u1z + c2 * u2z;
        av = a0 * v0z + a1 * v1z + a2 * v2z;
        bv = b0 * v0z + b1 * v1z + b2 * v2z;
        cv = c0 * v0z + c1 * v1z + c2 * v2z;
    }

    float ex[8], ey[8], ec[8], inv_ex[8];
    for (int e = 0; e < n_verts; ++e) {
        int en = (e + 1) % n_verts;
        ex[e] = (py_arr[e] - py_arr[en]) * orient;
        ey[e] = (px_arr[en] - px_arr[e]) * orient;
        ec[e] = -(ex[e] * px_arr[e] + ey[e] * py_arr[e]);
        inv_ex[e] = (fabsf(ex[e]) > 1e-6f) ? (1.0f / ex[e]) : 0.0f;
    }

    const float sh_inv_ff = shade * (1.0f - fog_f);
    const float fr_add = fog_r * fog_f;
    const float fg_add = fog_g * fog_f;
    const float fb_add = fog_b * fog_f;
    const uint32_t solid_packed = pack_pixel(sr, sg, sb, r_sh, g_sh, b_sh);
    const uint8_t *t_rgb = atlas_rgb + (size_t)t_off * 3;
    const uint8_t *t_alp = atlas_alpha + (size_t)t_off;
    const int tw_m1 = tw - 1, th_m1 = th - 1;

    for (int py = min_y; py <= max_y; ++py) {
        const float yf = (float)py + 0.5f;
        float xl = (float)min_x + 0.5f;
        float xr = (float)max_x + 0.5f;
        int valid_row = 1;
        for (int e = 0; e < n_verts; ++e) {
            const float y_term = ey[e] * yf + ec[e];
            if (ex[e] > 1e-6f) {
                const float bound = (-1e-3f - y_term) * inv_ex[e];
                if (bound > xl) xl = bound;
            } else if (ex[e] < -1e-6f) {
                const float bound = (-1e-3f - y_term) * inv_ex[e];
                if (bound < xr) xr = bound;
            } else if (y_term < -1e-3f) {
                valid_row = 0;
                break;
            }
        }
        if (!valid_row || xl > xr) continue;

        int x_start = (int)ceilf(xl - 0.5f);
        int x_end = (int)floorf(xr - 0.5f);
        if (x_start < min_x) x_start = min_x;
        if (x_end > max_x) x_end = max_x;
        if (x_start > x_end) continue;

        float *z_row = z_buf + (size_t)py * width;
        uint8_t *fb_row = fb_bgra + (size_t)py * pitch_bytes;
        uint32_t *fb_u32 = (uint32_t *)fb_row;
        const float xf0 = (float)x_start + 0.5f;
        float rz = az * xf0 + bz * yf + cz;

        if (!use_tex) {
            if (alp == 255) {
                for (int px = x_start; px <= x_end; ++px) {
                    if (rz > z_row[px]) {
                        fb_u32[px] = solid_packed;
                        z_row[px] = rz;
                    }
                    rz += az;
                }
            } else {
                const float pa = (float)alp * (1.0f / 255.0f);
                const float ipa = 1.0f - pa;
                for (int px = x_start; px <= x_end; ++px) {
                    if (rz > z_row[px]) {
                        uint8_t *p = fb_row + px * 4;
                        p[r_off] = clamp_u8((float)p[r_off] * ipa + (float)sr * pa);
                        p[g_off] = clamp_u8((float)p[g_off] * ipa + (float)sg * pa);
                        p[b_off] = clamp_u8((float)p[b_off] * ipa + (float)sb * pa);
                    }
                    rz += az;
                }
            }
        } else {
            float ru = au * xf0 + bu * yf + cu;
            float rv = av * xf0 + bv * yf + cv;
            const float alp_scale = (float)alp * (1.0f / 65025.0f);
            const float flat_pa = (float)alp * (1.0f / 255.0f);
            for (int px = x_start; px <= x_end; ++px) {
                if (rz > z_row[px]) {
                    const float inv_iz = 1.0f / rz;
                    const int tx = clamp_i((int)(ru * inv_iz), 0, tw_m1);
                    const int ty = clamp_i((int)(rv * inv_iz), 0, th_m1);
                    const int tidx = ty * tw + tx;
                    float pa = flat_pa;
                    if (has_cut) {
                        const uint8_t ta = t_alp[tidx];
                        if (ta < 32) {
                            rz += az; ru += au; rv += av;
                            continue;
                        }
                        pa = (float)ta * alp_scale;
                    }
                    const uint8_t *tc = t_rgb + tidx * 3;
                    const float cr = (float)tc[0] * sh_inv_ff + fr_add;
                    const float cg = (float)tc[1] * sh_inv_ff + fg_add;
                    const float cb = (float)tc[2] * sh_inv_ff + fb_add;
                    uint8_t *p = fb_row + px * 4;
                    if (pa >= 0.96f) {
                        fb_u32[px] = pack_pixel(clamp_u8(cr), clamp_u8(cg), clamp_u8(cb), r_sh, g_sh, b_sh);
                        z_row[px] = rz;
                    } else {
                        const float ipa = 1.0f - pa;
                        p[r_off] = clamp_u8((float)p[r_off] * ipa + cr * pa);
                        p[g_off] = clamp_u8((float)p[g_off] * ipa + cg * pa);
                        p[b_off] = clamp_u8((float)p[b_off] * ipa + cb * pa);
                    }
                }
                rz += az;
                ru += au;
                rv += av;
            }
        }
    }
}
