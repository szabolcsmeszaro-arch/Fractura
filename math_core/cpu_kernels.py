import math
import warnings
import numpy as np
from numba import njit, prange
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

INV_LN2_F32 = np.float32(1.4426950408889634)
LN_BAILOUT_LOG = np.float32(math.log(0.5 * math.log(65536.0)))


# --- CPU Color Palette Shaders & Sampling ---
@njit(fastmath=True, inline='always')
def cpu_palette_norm(s, scheme_id, palette_offset=0.0, color_density=1.0, color_contrast=1.0):
    dens = 1.0 if color_density <= 0.0 else color_density
    cont = 1.0 if color_contrast <= 0.0 else color_contrast

    if scheme_id == 0 or scheme_id > 10:
        phase = (math.sqrt(s) * (0.125 * dens) + palette_offset * 2.0) % 2.0
        tri = 1.0 - math.fabs((phase + 2.0 if phase < 0.0 else phase) - 1.0)
        return math.pow(tri, cont) if cont != 1.0 else tri
    elif scheme_id == 1:
        phase = (math.log(s * cont + 1.0) * (0.45 * dens) + palette_offset * 2.0) % 2.0
        return 1.0 - math.fabs((phase + 2.0 if phase < 0.0 else phase) - 1.0)
    elif scheme_id == 2:
        phase = (math.sqrt(s) * (0.125 * dens) + palette_offset * 2.0) % 2.0
        tri = 1.0 - math.fabs((phase + 2.0 if phase < 0.0 else phase) - 1.0)
        k = 4.0 * cont
        tanh_denom = math.tanh(0.5 * k)
        scale = (0.5 / tanh_denom) if math.fabs(tanh_denom) > 1e-7 else 0.5186574
        return 0.5 + math.tanh((tri - 0.5) * k) * scale
    elif scheme_id == 3:
        phase = (math.sqrt(s) * (0.125 * dens) + palette_offset * 2.0) % 2.0
        tri = 1.0 - math.fabs((phase + 2.0 if phase < 0.0 else phase) - 1.0)
        k = 3.0 * cont
        exp_denom = math.exp(k) - 1.0
        scale = (1.0 / exp_denom) if math.fabs(exp_denom) > 1e-7 else 0.0523957
        return (math.exp(k * tri) - 1.0) * scale
    elif scheme_id == 4:
        phase = (math.sqrt(s) * (0.5 * dens) + palette_offset * 2.0) % 2.0
        tri = 1.0 - math.fabs((phase + 2.0 if phase < 0.0 else phase) - 1.0)
        return math.pow(tri, cont) if cont != 1.0 else tri
    elif scheme_id == 5:
        ang = s * (0.05 * dens) + palette_offset * 6.28318530718
        ratio = 1.61803398875 * cont
        return 0.5 + 0.25 * (math.sin(ang) + math.sin(ang * ratio))
    elif scheme_id == 6:
        ang = s * (0.08 * dens) + palette_offset * 6.28318530718
        val = 0.5 * (1.0 - math.cos(ang))
        return math.pow(val, cont) if cont != 1.0 else val
    elif scheme_id == 7:
        phase = s * (0.05 * dens) + palette_offset * 6.28318530718
        b = 0.5 * cont
        wave = (
            math.sin(phase) +
            b * math.sin(phase * 2.0 + 1.2) +
            (b * b) * math.sin(phase * 4.0 + 2.4) +
            (b * b * b) * math.sin(phase * 8.0 + 3.6)
        )
        norm_sum = 1.0 + b + b * b + b * b * b
        scale = (0.5 / norm_sum) if norm_sum > 1e-7 else 0.26666666666666666
        return 0.5 + scale * wave
    elif scheme_id == 8:
        phase = (math.sqrt(s) * (0.1 * dens) + palette_offset * 2.0) % 2.0
        tri = 1.0 - math.fabs((phase + 2.0 if phase < 0.0 else phase) - 1.0)
        if cont != 1.0:
            tri = math.pow(tri, cont)
        t2 = tri * tri
        return t2 * tri * (tri * (tri * 6.0 - 15.0) + 10.0)
    elif scheme_id == 9:
        angle = s * (0.04 * dens) + (s * s) * (0.00008 * dens * cont) + palette_offset * 6.28318530718
        return 0.5 + 0.5 * math.sin(angle)
    elif scheme_id == 10:
        raw = (1.0 - math.exp(-0.06 * cont * math.sqrt(s * dens)) + palette_offset) % 1.0
        return raw if raw >= 0.0 else raw + 1.0
    return 0.0


@njit(fastmath=True, inline='always')
def cpu_sample_lut_lerp(lut, norm):
    n = 0.0 if norm < 0.0 else (1.0 if norm > 1.0 else norm)
    pos = n * 2047.0
    i0 = int(pos)
    i1 = 2047 if i0 >= 2047 else (i0 + 1)
    frac = pos - float(i0)
    inv_frac = 1.0 - frac
    r = lut[i0, 0] * inv_frac + lut[i1, 0] * frac
    g = lut[i0, 1] * inv_frac + lut[i1, 1] * frac
    b = lut[i0, 2] * inv_frac + lut[i1, 2] * frac
    return r, g, b


# --- CPU Double-Double (DD) Low-Level Operations ---
@njit(inline='always')
def cpu_two_sum(a, b):
    s = a + b
    v = s - a
    e = (a - (s - v)) + (b - v)
    return s, e


@njit(inline='always')
def cpu_two_prod(a, b):
    c = 134217729.0 * a
    a_hi = c - (c - a)
    a_lo = a - a_hi
    c = 134217729.0 * b
    b_hi = c - (c - b)
    b_lo = b - b_hi
    p = a * b
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, e


@njit(inline='always')
def cpu_two_sqr(a):
    return cpu_two_prod(a, a)


@njit(inline='always')
def cpu_dd_abs(a_hi, a_lo):
    return (-a_hi, -a_lo) if (a_hi < 0.0 or (a_hi == 0.0 and a_lo < 0.0)) else (a_hi, a_lo)


@njit(inline='always')
def cpu_dd_add(a_hi, a_lo, b_hi, b_lo):
    s, e = cpu_two_sum(a_hi, b_hi)
    return cpu_two_sum(s, e + a_lo + b_lo)


@njit(inline='always')
def cpu_dd_sub(a_hi, a_lo, b_hi, b_lo):
    s, e = cpu_two_sum(a_hi, -b_hi)
    return cpu_two_sum(s, e + a_lo - b_lo)


@njit(inline='always')
def cpu_dd_mul(a_hi, a_lo, b_hi, b_lo):
    p, e = cpu_two_prod(a_hi, b_hi)
    return cpu_two_sum(p, e + a_hi * b_lo + a_lo * b_hi)


@njit(inline='always')
def cpu_dd_mul_f64(a_hi, a_lo, b):
    p, e = cpu_two_prod(a_hi, b)
    return cpu_two_sum(p, e + a_lo * b)


@njit(inline='always')
def cpu_dd_sqr(a_hi, a_lo):
    p, e = cpu_two_sqr(a_hi)
    return cpu_two_sum(p, e + 2.0 * a_hi * a_lo)


@njit(inline='always')
def cpu_c_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@njit(fastmath=True, inline='always')
def cpu_pow2_f64(e):
    if e >= 1023:
        return 1.7976931348623157e+308
    elif e <= -1022:
        return 0.0
    return math.ldexp(1.0, int(e))


@njit(fastmath=True, inline='always')
def cpu_check_glitch_cancellation_floatexp(rx, ry, dzx, dzy, eff_exp):
    r_max = max(abs(rx), abs(ry))
    if r_max < 1e-4:
        return 0
    dz_max = max(abs(dzx), abs(dzy))
    if dz_max <= 0.0:
        return 0

    r_exp = int(math.log(r_max) * 1.4426950408889634)
    dz_exp = int(math.log(dz_max) * 1.4426950408889634)
    diff_exp = r_exp - (eff_exp + dz_exp)
    if diff_exp > 10 or diff_exp < -10:
        return 0

    scale_r = cpu_pow2_f64(-r_exp)
    scale_dz = cpu_pow2_f64(eff_exp - r_exp)

    rx_sc = rx * scale_r
    ry_sc = ry * scale_r
    dzx_sc = dzx * scale_dz
    dzy_sc = dzy * scale_dz

    zx_sc = rx_sc + dzx_sc
    zy_sc = ry_sc + dzy_sc

    mag2_sc = zx_sc * zx_sc + zy_sc * zy_sc
    r_mag2_sc = rx_sc * rx_sc + ry_sc * ry_sc

    if mag2_sc < 1e-6 * r_mag2_sc:
        return 1
    return 0



@njit(inline='always')
def cpu_eval_bbsa_4(dc_r, dc_i, a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i):
    t3_r, t3_i = cpu_c_mul(dc_r, dc_i, d_r, d_i)
    t2_r, t2_i = cpu_c_mul(dc_r, dc_i, c_r + t3_r, c_i + t3_i)
    t1_r, t1_i = cpu_c_mul(dc_r, dc_i, b_r + t2_r, b_i + t2_i)
    return cpu_c_mul(dc_r, dc_i, a_r + t1_r, a_i + t1_i)


@njit(inline='always')
def cpu_eval_bbsa_8(dc_r, dc_i, a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i,
                    e_r, e_i, f_r, f_i, g_r, g_i, h_r, h_i):
    t7_r, t7_i = cpu_c_mul(dc_r, dc_i, h_r, h_i)
    t6_r, t6_i = cpu_c_mul(dc_r, dc_i, g_r + t7_r, g_i + t7_i)
    t5_r, t5_i = cpu_c_mul(dc_r, dc_i, f_r + t6_r, f_i + t6_i)
    t4_r, t4_i = cpu_c_mul(dc_r, dc_i, e_r + t5_r, e_i + t5_i)
    t3_r, t3_i = cpu_c_mul(dc_r, dc_i, d_r + t4_r, d_i + t4_i)
    t2_r, t2_i = cpu_c_mul(dc_r, dc_i, c_r + t3_r, c_i + t3_i)
    t1_r, t1_i = cpu_c_mul(dc_r, dc_i, b_r + t2_r, b_i + t2_i)
    return cpu_c_mul(dc_r, dc_i, a_r + t1_r, a_i + t1_i)


# --- CPU FP64 Sample Arithmetic Evaluator ---
@njit(fastmath=True)
def cpu_compute_fp64_sample(x, y, max_iter, fractal_type, julia_cx, julia_cy, gen_n, gen_kr, gen_ki, inv_ln2, ln_bailout_log):
    if fractal_type == 0:
        x_m_fourth = x - 0.25
        y2 = y * y
        q = x_m_fourth * x_m_fourth + y2
        if (q * (q + x_m_fourth) <= 0.25 * y2) or ((x + 1.0) * (x + 1.0) + y2 <= 0.0625):
            return np.float32(-1.0)

    if fractal_type in (2, 3):
        zx, zy, cx, cy = x, y, julia_cx, julia_cy
    else:
        zx, zy, cx, cy = 0.0, 0.0, x, y

    inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if fractal_type == 4 and gen_n > 1 else inv_ln2

    for k in range(max_iter):
        zx2, zy2 = zx * zx, zy * zy
        mag2 = zx2 + zy2
        if mag2 > 65536.0:
            log_mag = math.log(mag2)
            if log_mag > 1.0:
                return np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln_n)
            return np.float32(k)

        if fractal_type in (0, 2):
            zy = 2.0 * zx * zy + cy
            zx = zx2 - zy2 + cx
        elif fractal_type in (1, 3):
            zy = 2.0 * math.fabs(zx) * math.fabs(zy) - cy
            zx = zx2 - zy2 + cx
        elif fractal_type == 4:
            zn_r, zn_i = zx, zy
            for _ in range(gen_n - 1):
                zn_r, zn_i = zn_r * zx - zn_i * zy, zn_r * zy + zn_i * zx
            kz_r = gen_kr * zx - gen_ki * zy
            kz_i = gen_kr * zy + gen_ki * zx
            zx = zn_r + kz_r + cx
            zy = zn_i + kz_i + cy
    return np.float32(-1.0)


# --- CPU Double-Double (DD) Sample Evaluator ---
@njit
def cpu_compute_dd_sample(cx_hi, cx_lo, cy_hi, cy_lo, max_iter, fractal_type,
                          julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo, gen_n, gen_kr, gen_ki,
                          inv_ln2, ln_bailout_log):
    if fractal_type == 0:
        x_m_fourth = cx_hi - 0.25
        y2 = cy_hi * cy_hi
        q = x_m_fourth * x_m_fourth + y2
        if (q * (q + x_m_fourth) <= 0.25 * y2) or ((cx_hi + 1.0) * (cx_hi + 1.0) + y2 <= 0.0625):
            return np.float32(-1.0)

    if fractal_type in (0, 2):
        if fractal_type == 2:
            zx_hi, zx_lo, zy_hi, zy_lo = cx_hi, cx_lo, cy_hi, cy_lo
            c_xh, c_xl, c_yh, c_yl = julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo
        else:
            zx_hi, zx_lo, zy_hi, zy_lo = 0.0, 0.0, 0.0, 0.0
            c_xh, c_xl, c_yh, c_yl = cx_hi, cx_lo, cy_hi, cy_lo

        for k in range(max_iter):
            zx2_hi, zx2_lo = cpu_dd_sqr(zx_hi, zx_lo)
            zy2_hi, zy2_lo = cpu_dd_sqr(zy_hi, zy_lo)
            mag2_hi = zx2_hi + zy2_hi
            if mag2_hi > 65536.0:
                log_mag = math.log(mag2_hi)
                return np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln2) if log_mag > 1.0 else np.float32(k)

            sub_h, sub_l = cpu_dd_sub(zx2_hi, zx2_lo, zy2_hi, zy2_lo)
            zxy_hi, zxy_lo = cpu_dd_mul(zx_hi, zx_lo, zy_hi, zy_lo)
            zx_hi, zx_lo = cpu_dd_add(sub_h, sub_l, c_xh, c_xl)
            zy_hi, zy_lo = cpu_dd_add(zxy_hi * 2.0, zxy_lo * 2.0, c_yh, c_yl)
        return np.float32(-1.0)

    elif fractal_type in (1, 3):
        if fractal_type == 3:
            zx_hi, zx_lo, zy_hi, zy_lo = cx_hi, cx_lo, cy_hi, cy_lo
            c_xh, c_xl, c_yh, c_yl = julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo
        else:
            zx_hi, zx_lo, zy_hi, zy_lo = 0.0, 0.0, 0.0, 0.0
            c_xh, c_xl, c_yh, c_yl = cx_hi, cx_lo, cy_hi, cy_lo

        for k in range(max_iter):
            zx2_hi, zx2_lo = cpu_dd_sqr(zx_hi, zx_lo)
            zy2_hi, zy2_lo = cpu_dd_sqr(zy_hi, zy_lo)
            mag2_hi = zx2_hi + zy2_hi
            if mag2_hi > 65536.0:
                log_mag = math.log(mag2_hi)
                return np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln2) if log_mag > 1.0 else np.float32(k)

            sub_h, sub_l = cpu_dd_sub(zx2_hi, zx2_lo, zy2_hi, zy2_lo)
            ax_hi, ax_lo = cpu_dd_abs(zx_hi, zx_lo)
            ay_hi, ay_lo = cpu_dd_abs(zy_hi, zy_lo)
            zxy_hi, zxy_lo = cpu_dd_mul(ax_hi, ax_lo, ay_hi, ay_lo)
            zx_hi, zx_lo = cpu_dd_add(sub_h, sub_l, c_xh, c_xl)
            zy_hi, zy_lo = cpu_dd_sub(zxy_hi * 2.0, zxy_lo * 2.0, c_yh, c_yl)
        return np.float32(-1.0)

    elif fractal_type == 4:
        zx_hi, zx_lo, zy_hi, zy_lo = 0.0, 0.0, 0.0, 0.0
        c_xh, c_xl, c_yh, c_yl = cx_hi, cx_lo, cy_hi, cy_lo
        inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else inv_ln2

        for k in range(max_iter):
            zx2_hi, zx2_lo = cpu_dd_sqr(zx_hi, zx_lo)
            zy2_hi, zy2_lo = cpu_dd_sqr(zy_hi, zy_lo)
            mag2_hi = zx2_hi + zy2_hi
            if mag2_hi > 65536.0:
                log_mag = math.log(mag2_hi)
                return np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln_n) if log_mag > 1.0 else np.float32(k)

            cur_r_h, cur_r_l = zx_hi, zx_lo
            cur_i_h, cur_i_l = zy_hi, zy_lo
            for _ in range(gen_n - 1):
                r1_h, r1_l = cpu_dd_mul(cur_r_h, cur_r_l, zx_hi, zx_lo)
                r2_h, r2_l = cpu_dd_mul(cur_i_h, cur_i_l, zy_hi, zy_lo)
                re_h, re_l = cpu_dd_sub(r1_h, r1_l, r2_h, r2_l)

                i1_h, i1_l = cpu_dd_mul(cur_r_h, cur_r_l, zy_hi, zy_lo)
                i2_h, i2_l = cpu_dd_mul(cur_i_h, cur_i_l, zx_hi, zx_lo)
                im_h, im_l = cpu_dd_add(i1_h, i1_l, i2_h, i2_l)

                cur_r_h, cur_r_l = re_h, re_l
                cur_i_h, cur_i_l = im_h, im_l

            kr_zx_h, kr_zx_l = cpu_dd_mul_f64(zx_hi, zx_lo, gen_kr)
            ki_zy_h, ki_zy_l = cpu_dd_mul_f64(zy_hi, zy_lo, gen_ki)
            kz_r_h, kz_r_l = cpu_dd_sub(kr_zx_h, kr_zx_l, ki_zy_h, ki_zy_l)

            kr_zy_h, kr_zy_l = cpu_dd_mul_f64(zy_hi, zy_lo, gen_kr)
            ki_zx_h, ki_zx_l = cpu_dd_mul_f64(zx_hi, zx_lo, gen_ki)
            kz_i_h, kz_i_l = cpu_dd_add(kr_zy_h, kr_zy_l, ki_zx_h, ki_zx_l)

            tot_r_h, tot_r_l = cpu_dd_add(cur_r_h, cur_r_l, kz_r_h, kz_r_l)
            tot_i_h, tot_i_l = cpu_dd_add(cur_i_h, cur_i_l, kz_i_h, kz_i_l)

            zx_hi, zx_lo = cpu_dd_add(tot_r_h, tot_r_l, c_xh, c_xl)
            zy_hi, zy_lo = cpu_dd_add(tot_i_h, tot_i_l, c_yh, c_yl)
        return np.float32(-1.0)

    return np.float32(-1.0)


# --- CPU Perturbation Sample Evaluator ---
@njit
def cpu_compute_perturbation_sample(ref_re, ref_im, ref_len, max_iter,
                                    ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub, dy_sub,
                                    ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                                    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                                    gen_n, gen_kr, gen_ki,
                                    strict_glitch, inv_ln2, ln_bailout_log,
                                    bla_table=np.zeros((1, 1, 1), dtype=np.float64)):
    is_julia = (fractal_type == 2 or fractal_type == 3)
    dcy = (ref_py_sub - float(sub_j)) * dy_sub
    dcx = (float(sub_i) - ref_px_sub) * dx_sub

    if bbsa_skip > 0 and fractal_type in (0, 2, 4):
        ux = dcx * bbsa_inv_r
        uy = dcy * bbsa_inv_r
        if ux * ux + uy * uy <= 1.0:
            if (bbsa_er != 0.0 or bbsa_ei != 0.0 or bbsa_fr != 0.0 or bbsa_fi != 0.0 or
                bbsa_gr != 0.0 or bbsa_gi != 0.0 or bbsa_hr != 0.0 or bbsa_hi != 0.0):
                dzx, dzy = cpu_eval_bbsa_8(ux, uy, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                           bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                           bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                           bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi)
            else:
                dzx, dzy = cpu_eval_bbsa_4(ux, uy, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                           bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di)
            k = bbsa_skip
        else:
            dzx, dzy = (dcx, dcy) if is_julia else (0.0, 0.0)
            k = 0
    else:
        dzx, dzy = (dcx, dcy) if is_julia else (0.0, 0.0)
        k = 0

    dc_param_x = 0.0 if is_julia else dcx
    dc_param_y = 0.0 if is_julia else dcy
    is_glitch = 0

    has_bla = (bla_table.shape[0] > 1)
    bla_levels = bla_table.shape[0] if has_bla else 0

    # --- Stage 1: Reference Orbit Perturbation ---
    if fractal_type in (0, 2):
        while k < ref_len:
            rx, ry = ref_re[k], ref_im[k]
            zx, zy = rx + dzx, ry + dzy
            mag2 = zx * zx + zy * zy
            if mag2 > 65536.0:
                if strict_glitch > 0 and k > 15 and (rx * rx + ry * ry) < 4.0:
                    is_glitch = 1
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln2) if log_mag > 1.0 else np.float32(k)), is_glitch

            if strict_glitch > 0 and k > 15:
                r_mag2 = rx * rx + ry * ry
                if mag2 < 1e-6 * r_mag2 and r_mag2 > 1e-8:
                    is_glitch = 1

            if has_bla and k < ref_len - 1:
                dz_mag2 = dzx * dzx + dzy * dzy
                if dz_mag2 < bla_table[0, k, 4]:
                    best_d = 0
                    max_d = bla_levels - 1
                    if dz_mag2 == 0.0 and (dzx != 0.0 or dzy != 0.0):
                        max_d = 8 if 8 < max_d else max_d
                    for d in range(max_d, 0, -1):
                        step_d = 1 << d
                        if k + step_d < ref_len and dz_mag2 < bla_table[d, k, 4]:
                            best_d = d
                            break
                    step_d = 1 << best_d
                    a_r = bla_table[best_d, k, 0]
                    a_i = bla_table[best_d, k, 1]
                    b_r = bla_table[best_d, k, 2]
                    b_i = bla_table[best_d, k, 3]
                    new_dzx = (a_r * dzx - a_i * dzy) + (b_r * dc_param_x - b_i * dc_param_y)
                    dzy = (a_r * dzy + a_i * dzx) + (b_r * dc_param_y + b_i * dc_param_x)
                    dzx = new_dzx
                    if math.isnan(dzx) or math.isnan(dzy) or math.isinf(dzx) or math.isinf(dzy):
                        return np.float32(k), 1
                    k += step_d
                    continue

            dzx_new = (2.0 * rx + dzx) * dzx - (2.0 * ry + dzy) * dzy + dc_param_x
            dzy = 2.0 * ((rx + dzx) * dzy + ry * dzx) + dc_param_y
            dzx = dzx_new
            k += 1

    elif fractal_type in (1, 3):
        while k < ref_len:
            rx, ry = ref_re[k], ref_im[k]
            zx, zy = rx + dzx, ry + dzy
            mag2 = zx * zx + zy * zy
            if mag2 > 65536.0:
                if strict_glitch > 0 and k > 15 and (rx * rx + ry * ry) < 4.0:
                    is_glitch = 1
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln2) if log_mag > 1.0 else np.float32(k)), is_glitch

            if strict_glitch > 0 and k > 15:
                r_mag2 = rx * rx + ry * ry
                if mag2 < 1e-6 * r_mag2 and r_mag2 > 1e-8:
                    is_glitch = 1

            dzx_new = (2.0 * rx + dzx) * dzx - (2.0 * ry + dzy) * dzy + dc_param_x
            sx_sign = 1.0 if rx >= 0.0 else -1.0
            sy_sign = 1.0 if ry >= 0.0 else -1.0
            delta_x = sx_sign * dzx if (rx + dzx) * sx_sign >= 0.0 else -sx_sign * (2.0 * rx + dzx)
            delta_y = sy_sign * dzy if (ry + dzy) * sy_sign >= 0.0 else -sy_sign * (2.0 * ry + dzy)
            dzy = 2.0 * (math.fabs(rx) * delta_y + math.fabs(ry) * delta_x + delta_x * delta_y) - dc_param_y
            dzx = dzx_new
            k += 1

    elif fractal_type == 4:
        inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else inv_ln2
        while k < ref_len:
            rx, ry = ref_re[k], ref_im[k]
            zx, zy = rx + dzx, ry + dzy
            mag2 = zx * zx + zy * zy
            if mag2 > 65536.0:
                if strict_glitch > 0 and k > 15 and (rx * rx + ry * ry) < 4.0:
                    is_glitch = 1
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln_n) if log_mag > 1.0 else np.float32(k)), is_glitch

            if strict_glitch > 0 and k > 15:
                r_mag2 = rx * rx + ry * ry
                if mag2 < 1e-6 * r_mag2 and r_mag2 > 1e-8:
                    is_glitch = 1

            if has_bla and k < ref_len - 1:
                dz_mag2 = dzx * dzx + dzy * dzy
                if dz_mag2 < bla_table[0, k, 4]:
                    best_d = 0
                    max_d = bla_levels - 1
                    if dz_mag2 == 0.0 and (dzx != 0.0 or dzy != 0.0):
                        max_d = 8 if 8 < max_d else max_d
                    for d in range(max_d, 0, -1):
                        step_d = 1 << d
                        if k + step_d < ref_len and dz_mag2 < bla_table[d, k, 4]:
                            best_d = d
                            break
                    step_d = 1 << best_d
                    a_r = bla_table[best_d, k, 0]
                    a_i = bla_table[best_d, k, 1]
                    b_r = bla_table[best_d, k, 2]
                    b_i = bla_table[best_d, k, 3]
                    new_dzx = (a_r * dzx - a_i * dzy) + (b_r * dc_param_x - b_i * dc_param_y)
                    dzy = (a_r * dzy + a_i * dzx) + (b_r * dc_param_y + b_i * dc_param_x)
                    dzx = new_dzx
                    if math.isnan(dzx) or math.isnan(dzy) or math.isinf(dzx) or math.isinf(dzy):
                        return np.float32(k), 1
                    k += step_d
                    continue

            p_r, p_i = dzx, dzy
            q_r, q_i = rx, ry
            for _ in range(gen_n - 1):
                t1_r, t1_i = cpu_c_mul(p_r, p_i, rx, ry)
                pq_r, pq_i = p_r + q_r, p_i + q_i
                t2_r, t2_i = cpu_c_mul(pq_r, pq_i, dzx, dzy)
                p_r = t1_r + t2_r
                p_i = t1_i + t2_i
                q_r, q_i = cpu_c_mul(q_r, q_i, rx, ry)

            k_dz_r, k_dz_i = cpu_c_mul(gen_kr, gen_ki, dzx, dzy)
            dzx = p_r + k_dz_r + dc_param_x
            dzy = p_i + k_dz_i + dc_param_y
            k += 1

    # --- Stage 2: Fallback Loop (Escaped Reference) ---
    zx = ref_re[ref_len] + dzx
    zy = ref_im[ref_len] + dzy
    cx = julia_cx if is_julia else ref_cx + dcx
    cy = julia_cy if is_julia else ref_cy + dcy

    if ref_len < max_iter:
        is_glitch = 1

    if fractal_type in (0, 2):
        while k < max_iter:
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if math.isnan(mag2) or math.isinf(mag2):
                return np.float32(k), 1
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln2) if log_mag > 1.0 else np.float32(k)), is_glitch
            zy = 2.0 * zx * zy + cy
            zx = zx2 - zy2 + cx
            k += 1
        return np.float32(-1.0), is_glitch

    elif fractal_type in (1, 3):
        while k < max_iter:
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if math.isnan(mag2) or math.isinf(mag2):
                return np.float32(k), 1
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln2) if log_mag > 1.0 else np.float32(k)), is_glitch
            zy = 2.0 * math.fabs(zx) * math.fabs(zy) - cy
            zx = zx2 - zy2 + cx
            k += 1
        return np.float32(-1.0), is_glitch

    elif fractal_type == 4:
        inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else inv_ln2
        while k < max_iter:
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if math.isnan(mag2) or math.isinf(mag2):
                return np.float32(k), 1
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln_n) if log_mag > 1.0 else np.float32(k)), is_glitch
            zn_r, zn_i = zx, zy
            for _ in range(gen_n - 1):
                zn_r, zn_i = zn_r * zx - zn_i * zy, zn_r * zy + zn_i * zx
            kz_r = gen_kr * zx - gen_ki * zy
            kz_i = gen_kr * zy + gen_ki * zx
            zx = zn_r + kz_r + cx
            zy = zn_i + kz_i + cy
            k += 1
        return np.float32(-1.0), is_glitch

    return np.float32(-1.0), is_glitch


@njit(fastmath=True, inline='always')
def cpu_compute_perturbation_sample_floatexp(
    ref_re, ref_im, ref_len, max_iter,
    ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub_norm, dy_sub_norm,
    E_scale, ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
    gen_n, gen_kr, gen_ki, strict_glitch,
    inv_ln2, ln_bailout_log,
    bla_table=np.zeros((1, 1, 1), dtype=np.float64)
):
    is_julia = (fractal_type == 2 or fractal_type == 3)
    dcy = (ref_py_sub - float(sub_j)) * dy_sub_norm
    dcx = (float(sub_i) - ref_px_sub) * dx_sub_norm

    if bbsa_skip > 0 and fractal_type in (0, 2, 4):
        ux = dcx * bbsa_inv_r
        uy = dcy * bbsa_inv_r
        if ux * ux + uy * uy <= 1.0:
            if (bbsa_hr != 0.0 or bbsa_hi != 0.0 or bbsa_gr != 0.0 or bbsa_gi != 0.0 or bbsa_fr != 0.0 or bbsa_fi != 0.0):
                # 7th-order Horner polynomial on (ux, uy)
                dzx, dzy = bbsa_hr, bbsa_hi
                dzx, dzy = cpu_c_mul(dzx, dzy, ux, uy); dzx += bbsa_gr; dzy += bbsa_gi
                dzx, dzy = cpu_c_mul(dzx, dzy, ux, uy); dzx += bbsa_fr; dzy += bbsa_fi
                dzx, dzy = cpu_c_mul(dzx, dzy, ux, uy); dzx += bbsa_er; dzy += bbsa_ei
                dzx, dzy = cpu_c_mul(dzx, dzy, ux, uy); dzx += bbsa_dr; dzy += bbsa_di
                dzx, dzy = cpu_c_mul(dzx, dzy, ux, uy); dzx += bbsa_cr; dzy += bbsa_ci
                dzx, dzy = cpu_c_mul(dzx, dzy, ux, uy); dzx += bbsa_ar; dzy += bbsa_ai
                dzx, dzy = cpu_c_mul(dzx, dzy, ux, uy)
            else:
                # 4th-order Horner polynomial on (ux, uy)
                dzx, dzy = bbsa_er, bbsa_ei
                dzx, dzy = cpu_c_mul(dzx, dzy, ux, uy); dzx += bbsa_dr; dzy += bbsa_di
                dzx, dzy = cpu_c_mul(dzx, dzy, ux, uy); dzx += bbsa_cr; dzy += bbsa_ci
                dzx, dzy = cpu_c_mul(dzx, dzy, ux, uy); dzx += bbsa_ar; dzy += bbsa_ai
                dzx, dzy = cpu_c_mul(dzx, dzy, ux, uy)

            z_exp = int(bbsa_br)
            mag = max(abs(dzx), abs(dzy))
            while mag > 65536.0:
                dzx *= 1.52587890625e-05
                dzy *= 1.52587890625e-05
                z_exp += 16
                mag = max(abs(dzx), abs(dzy))
            while mag < 1.52587890625e-05 and mag > 0.0:
                dzx *= 65536.0
                dzy *= 65536.0
                z_exp -= 16
                mag = max(abs(dzx), abs(dzy))
            k = bbsa_skip
        else:
            dzx = dcx if is_julia else 0.0
            dzy = dcy if is_julia else 0.0
            z_exp = 0
            k = 0
    else:
        dzx = dcx if is_julia else 0.0
        dzy = dcy if is_julia else 0.0
        z_exp = 0
        k = 0

    dc_param_x = 0.0 if is_julia else dcx
    dc_param_y = 0.0 if is_julia else dcy
    is_glitch = 0

    has_bla = (bla_table.shape[0] > 1)
    bla_levels = bla_table.shape[0] if has_bla else 0

    if fractal_type in (0, 2):
        while k < ref_len:
            rx = ref_re[k]
            ry = ref_im[k]

            # 1. Bailout check on (Z_k + delta_z_k)
            eff_exp = z_exp - E_scale
            if eff_exp >= -50:
                scale_esc = cpu_pow2_f64(eff_exp)
                zx = rx + dzx * scale_esc
                zy = ry + dzy * scale_esc
                mag2 = zx * zx + zy * zy
                if mag2 > 65536.0:
                    log_mag = math.log(mag2)
                    return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln2) if log_mag > 1.0 else np.float32(k)), is_glitch

            if strict_glitch > 0 and k > 15:
                if cpu_check_glitch_cancellation_floatexp(rx, ry, dzx, dzy, eff_exp):
                    is_glitch = 1

            if has_bla and k < ref_len - 1:
                shift_r = z_exp - E_scale
                if shift_r >= -510:
                    sc_r = cpu_pow2_f64(shift_r)
                    dz_mag2 = (dzx * dzx + dzy * dzy) * (sc_r * sc_r)
                    can_step_0 = (dz_mag2 < bla_table[0, k, 4])
                else:
                    dz_mag = math.sqrt(dzx * dzx + dzy * dzy)
                    r0_sq = bla_table[0, k, 4]
                    can_step_0 = (r0_sq > 0.0) and (dz_mag * cpu_pow2_f64(max(-1020, shift_r)) < math.sqrt(r0_sq))

                if can_step_0:
                    best_d = 0
                    max_d = bla_levels - 1
                    for d in range(max_d, 0, -1):
                        step_d = 1 << d
                        if k + step_d < ref_len:
                            rd_sq = bla_table[d, k, 4]
                            if shift_r >= -510:
                                can_step_d = (dz_mag2 < rd_sq)
                            else:
                                can_step_d = (rd_sq > 0.0) and (dz_mag * cpu_pow2_f64(max(-1020, shift_r)) < math.sqrt(rd_sq))
                            if can_step_d:
                                best_d = d
                                break
                    step_d = 1 << best_d
                    a_r = bla_table[best_d, k, 0]
                    a_i = bla_table[best_d, k, 1]
                    b_r = bla_table[best_d, k, 2]
                    b_i = bla_table[best_d, k, 3]
                    scale_c = cpu_pow2_f64(-z_exp) if (-1020 <= z_exp <= 1020) else 0.0
                    dc_x_scaled = dc_param_x * scale_c
                    dc_y_scaled = dc_param_y * scale_c
                    new_dzx = (a_r * dzx - a_i * dzy) + (b_r * dc_x_scaled - b_i * dc_y_scaled)
                    dzy = (a_r * dzy + a_i * dzx) + (b_r * dc_y_scaled + b_i * dc_x_scaled)
                    dzx = new_dzx
                    if math.isnan(dzx) or math.isnan(dzy) or math.isinf(dzx) or math.isinf(dzy):
                        return np.float32(k), 1
                    mag = max(abs(dzx), abs(dzy))
                    while mag > 65536.0 and math.isfinite(mag):
                        dzx *= 1.52587890625e-05
                        dzy *= 1.52587890625e-05
                        z_exp += 16
                        mag = max(abs(dzx), abs(dzy))
                    while mag < 1.52587890625e-05 and mag > 0.0 and math.isfinite(mag):
                        dzx *= 65536.0
                        dzy *= 65536.0
                        z_exp -= 16
                        mag = max(abs(dzx), abs(dzy))
                    k += step_d
                    continue

            # 2. Linear derivative: 2 * Z * delta_z
            lx = 2.0 * (rx * dzx - ry * dzy)
            ly = 2.0 * (rx * dzy + ry * dzx)

            # 3. Non-linear term: delta_z^2 * 2^-E_scale
            shift_nl = z_exp - E_scale
            if shift_nl >= -1020:
                scale_nl = cpu_pow2_f64(shift_nl)
                lx += (dzx * dzx - dzy * dzy) * scale_nl
                ly += (2.0 * dzx * dzy) * scale_nl

            # 4. Delta c term: delta_c has exponent 0, so shifted by -z_exp
            if -1020 <= z_exp <= 1020:
                scale_c = cpu_pow2_f64(-z_exp)
                lx += dc_param_x * scale_c
                ly += dc_param_y * scale_c

            dzx = lx
            dzy = ly

            # 5. Exponent normalization: keep |dz| within [2^-16, 2^16]
            mag = max(abs(dzx), abs(dzy))
            while mag > 65536.0 and math.isfinite(mag):
                dzx *= 1.52587890625e-05
                dzy *= 1.52587890625e-05
                z_exp += 16
                mag = max(abs(dzx), abs(dzy))
            while mag < 1.52587890625e-05 and mag > 0.0 and math.isfinite(mag):
                dzx *= 65536.0
                dzy *= 65536.0
                z_exp -= 16
                mag = max(abs(dzx), abs(dzy))

            k += 1

    elif fractal_type in (1, 3):
        while k < ref_len:
            rx = ref_re[k]
            ry = ref_im[k]

            # 1. Bailout check on (Z_k + delta_z_k)
            eff_exp = z_exp - E_scale
            if eff_exp >= -50:
                scale_esc = cpu_pow2_f64(eff_exp)
                zx = rx + dzx * scale_esc
                zy = ry + dzy * scale_esc
                mag2 = zx * zx + zy * zy
                if mag2 > 65536.0:
                    log_mag = math.log(mag2)
                    return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln2) if log_mag > 1.0 else np.float32(k)), is_glitch

            if strict_glitch > 0 and k > 15:
                if cpu_check_glitch_cancellation_floatexp(rx, ry, dzx, dzy, eff_exp):
                    is_glitch = 1

            shift_nl = z_exp - E_scale
            scale_nl = cpu_pow2_f64(shift_nl) if shift_nl >= -1020 else 0.0
            scale_c = cpu_pow2_f64(-z_exp) if (-1020 <= z_exp <= 1020) else 0.0

            sx_sign = 1.0 if rx >= 0.0 else -1.0
            sy_sign = 1.0 if ry >= 0.0 else -1.0

            scale_esc = cpu_pow2_f64(eff_exp) if eff_exp >= -1020 else 0.0
            actual_dzx = dzx * scale_esc
            actual_dzy = dzy * scale_esc

            if (rx + actual_dzx) * sx_sign >= 0.0:
                factor_x = sx_sign
                extra_x = 0.0
            else:
                factor_x = -sx_sign
                extra_x = -sx_sign * 2.0 * rx * cpu_pow2_f64(-eff_exp) if eff_exp <= 1020 else 0.0

            if (ry + actual_dzy) * sy_sign >= 0.0:
                factor_y = sy_sign
                extra_y = 0.0
            else:
                factor_y = -sy_sign
                extra_y = -sy_sign * 2.0 * ry * cpu_pow2_f64(-eff_exp) if eff_exp <= 1020 else 0.0

            lx = (2.0 * rx + dzx * scale_nl) * dzx - (2.0 * ry + dzy * scale_nl) * dzy + dc_param_x * scale_c
            dx_norm = factor_x * dzx + extra_x
            dy_norm = factor_y * dzy + extra_y
            ly = 2.0 * (math.fabs(rx) * dy_norm + math.fabs(ry) * dx_norm + dx_norm * dy_norm * scale_nl) - dc_param_y * scale_c

            dzx = lx
            dzy = ly

            mag = max(abs(dzx), abs(dzy))
            while mag > 65536.0:
                dzx *= 1.52587890625e-05
                dzy *= 1.52587890625e-05
                z_exp += 16
                mag = max(abs(dzx), abs(dzy))
            while mag < 1.52587890625e-05 and mag > 0.0:
                dzx *= 65536.0
                dzy *= 65536.0
                z_exp -= 16
                mag = max(abs(dzx), abs(dzy))

            k += 1

    elif fractal_type == 4:
        inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else inv_ln2
        while k < ref_len:
            rx = ref_re[k]
            ry = ref_im[k]

            # 1. Bailout check on (Z_k + delta_z_k)
            eff_exp = z_exp - E_scale
            if eff_exp >= -50:
                scale_esc = cpu_pow2_f64(eff_exp)
                zx = rx + dzx * scale_esc
                zy = ry + dzy * scale_esc
                mag2 = zx * zx + zy * zy
                if mag2 > 65536.0:
                    log_mag = math.log(mag2)
                    return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln_n) if log_mag > 1.0 else np.float32(k)), is_glitch

            if strict_glitch > 0 and k > 15:
                if cpu_check_glitch_cancellation_floatexp(rx, ry, dzx, dzy, eff_exp):
                    is_glitch = 1

            if has_bla and k < ref_len - 1:
                shift_r = z_exp - E_scale
                if shift_r >= -510:
                    sc_r = cpu_pow2_f64(shift_r)
                    dz_mag2 = (dzx * dzx + dzy * dzy) * (sc_r * sc_r)
                    can_step_0 = (dz_mag2 < bla_table[0, k, 4])
                else:
                    dz_mag = math.sqrt(dzx * dzx + dzy * dzy)
                    r0_sq = bla_table[0, k, 4]
                    can_step_0 = (r0_sq > 0.0) and (dz_mag * cpu_pow2_f64(max(-1020, shift_r)) < math.sqrt(r0_sq))

                if can_step_0:
                    best_d = 0
                    max_d = bla_levels - 1
                    for d in range(max_d, 0, -1):
                        step_d = 1 << d
                        if k + step_d < ref_len:
                            rd_sq = bla_table[d, k, 4]
                            if shift_r >= -510:
                                can_step_d = (dz_mag2 < rd_sq)
                            else:
                                can_step_d = (rd_sq > 0.0) and (dz_mag * cpu_pow2_f64(max(-1020, shift_r)) < math.sqrt(rd_sq))
                            if can_step_d:
                                best_d = d
                                break
                    step_d = 1 << best_d
                    a_r = bla_table[best_d, k, 0]
                    a_i = bla_table[best_d, k, 1]
                    b_r = bla_table[best_d, k, 2]
                    b_i = bla_table[best_d, k, 3]
                    scale_c = cpu_pow2_f64(-z_exp) if (-1020 <= z_exp <= 1020) else 0.0
                    dc_x_scaled = dc_param_x * scale_c
                    dc_y_scaled = dc_param_y * scale_c
                    new_dzx = (a_r * dzx - a_i * dzy) + (b_r * dc_x_scaled - b_i * dc_y_scaled)
                    dzy = (a_r * dzy + a_i * dzx) + (b_r * dc_y_scaled + b_i * dc_x_scaled)
                    dzx = new_dzx
                    if math.isnan(dzx) or math.isnan(dzy) or math.isinf(dzx) or math.isinf(dzy):
                        return np.float32(k), 1
                    mag = max(abs(dzx), abs(dzy))
                    while mag > 65536.0 and math.isfinite(mag):
                        dzx *= 1.52587890625e-05
                        dzy *= 1.52587890625e-05
                        z_exp += 16
                        mag = max(abs(dzx), abs(dzy))
                    while mag < 1.52587890625e-05 and mag > 0.0 and math.isfinite(mag):
                        dzx *= 65536.0
                        dzy *= 65536.0
                        z_exp -= 16
                        mag = max(abs(dzx), abs(dzy))
                    k += step_d
                    continue

            p_r, p_i = dzx, dzy
            q_r, q_i = rx, ry
            shift_nl = z_exp - E_scale
            scale_nl = cpu_pow2_f64(shift_nl) if shift_nl >= -1020 else 0.0

            for _ in range(gen_n - 1):
                t1_r, t1_i = cpu_c_mul(p_r, p_i, rx, ry)
                t2_r, t2_i = cpu_c_mul(q_r, q_i, dzx, dzy)
                t3_r, t3_i = cpu_c_mul(p_r, p_i, dzx, dzy)
                p_r = t1_r + t2_r + t3_r * scale_nl
                p_i = t1_i + t2_i + t3_i * scale_nl
                q_r, q_i = cpu_c_mul(q_r, q_i, rx, ry)

            k_dz_r, k_dz_i = cpu_c_mul(gen_kr, gen_ki, dzx, dzy)
            scale_c = cpu_pow2_f64(-z_exp) if (-1020 <= z_exp <= 1020) else 0.0

            dzx = p_r + k_dz_r + dc_param_x * scale_c
            dzy = p_i + k_dz_i + dc_param_y * scale_c

            mag = max(abs(dzx), abs(dzy))
            while mag > 65536.0 and math.isfinite(mag):
                dzx *= 1.52587890625e-05
                dzy *= 1.52587890625e-05
                z_exp += 16
                mag = max(abs(dzx), abs(dzy))
            while mag < 1.52587890625e-05 and mag > 0.0 and math.isfinite(mag):
                dzx *= 65536.0
                dzy *= 65536.0
                z_exp -= 16
                mag = max(abs(dzx), abs(dzy))

            k += 1

    eff_exp = z_exp - E_scale
    scale_fb = cpu_pow2_f64(eff_exp) if eff_exp >= -60 else 0.0
    zx = ref_re[ref_len] + dzx * scale_fb
    zy = ref_im[ref_len] + dzy * scale_fb
    scale_c_fb = cpu_pow2_f64(-E_scale) if E_scale <= 1020 else 0.0
    cx = julia_cx if is_julia else (ref_cx + dc_param_x * scale_c_fb)
    cy = julia_cy if is_julia else (ref_cy + dc_param_y * scale_c_fb)

    # Check if pixel bailed out cleanly alongside reference orbit at k = ref_len
    mag2 = zx * zx + zy * zy
    if mag2 > 65536.0:
        log_mag = math.log(mag2)
        inv_ln = (np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else inv_ln2) if fractal_type == 4 else inv_ln2
        return (np.float32(ref_len + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln) if log_mag > 1.0 else np.float32(ref_len)), is_glitch

    if ref_len < max_iter:
        is_glitch = 1

    if fractal_type in (0, 2):
        while k < max_iter:
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if math.isnan(mag2) or math.isinf(mag2):
                return np.float32(k), 1
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln2) if log_mag > 1.0 else np.float32(k)), is_glitch
            zy = 2.0 * zx * zy + cy
            zx = zx2 - zy2 + cx
            k += 1
        return np.float32(-1.0), is_glitch

    elif fractal_type in (1, 3):
        while k < max_iter:
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if math.isnan(mag2) or math.isinf(mag2):
                return np.float32(k), 1
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln2) if log_mag > 1.0 else np.float32(k)), is_glitch
            zy = 2.0 * math.fabs(zx) * math.fabs(zy) - cy
            zx = zx2 - zy2 + cx
            k += 1
        return np.float32(-1.0), is_glitch

    elif fractal_type == 4:
        inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else inv_ln2
        while k < max_iter:
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if math.isnan(mag2) or math.isinf(mag2):
                return np.float32(k), 1
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - ln_bailout_log) * inv_ln_n) if log_mag > 1.0 else np.float32(k)), is_glitch
            zn_r, zn_i = zx, zy
            for _ in range(gen_n - 1):
                zn_r, zn_i = zn_r * zx - zn_i * zy, zn_r * zy + zn_i * zx
            kz_r = gen_kr * zx - gen_ki * zy
            kz_i = gen_kr * zy + gen_ki * zx
            zx = zn_r + kz_r + cx
            zy = zn_i + kz_i + cy
            k += 1
        return np.float32(-1.0), is_glitch

    return np.float32(-1.0), is_glitch


# --- Multi-Core Parallel Iteration Probing ---
@njit(parallel=True, fastmath=True)
def cpu_probe_iter_kernel(out_iters, target_w, target_h, max_iter,
                          x_min, y_max, dx, dy,
                          fractal_type, julia_cx, julia_cy, gen_n, gen_kr, gen_ki,
                          inv_ln2, ln_bailout_log):
    for j in prange(target_h):
        y = y_max - j * dy
        for i in range(target_w):
            x = x_min + i * dx
            out_iters[j, i] = cpu_compute_fp64_sample(
                x, y, max_iter, fractal_type, julia_cx, julia_cy,
                gen_n, gen_kr, gen_ki, inv_ln2, ln_bailout_log
            )


@njit(parallel=True)
def cpu_probe_iter_dd_kernel(out_iters, target_w, target_h, max_iter,
                             x_min_hi, x_min_lo, dx_hi, dx_lo,
                             y_max_hi, y_max_lo, dy_hi, dy_lo,
                             fractal_type, julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo,
                             gen_n, gen_kr, gen_ki, inv_ln2, ln_bailout_log):
    for j in prange(target_h):
        jdy_hi, jdy_lo = cpu_dd_mul_f64(dy_hi, dy_lo, float(j))
        cy_hi, cy_lo = cpu_dd_sub(y_max_hi, y_max_lo, jdy_hi, jdy_lo)
        for i in range(target_w):
            idx_hi, idx_lo = cpu_dd_mul_f64(dx_hi, dx_lo, float(i))
            cx_hi, cx_lo = cpu_dd_add(x_min_hi, x_min_lo, idx_hi, idx_lo)
            out_iters[j, i] = cpu_compute_dd_sample(
                cx_hi, cx_lo, cy_hi, cy_lo, max_iter, fractal_type,
                julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo,
                gen_n, gen_kr, gen_ki, inv_ln2, ln_bailout_log
            )


@njit(parallel=True)
def cpu_probe_iter_perturbation_kernel(out_iters, target_w, target_h, max_iter,
                                       ref_re, ref_im, ref_len,
                                       ref_px, ref_py, dx, dy,
                                       ref_cx, ref_cy, fractal_type,
                                       julia_cx, julia_cy,
                                       bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                       bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                       bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                       bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                                       gen_n, gen_kr, gen_ki, inv_ln2, ln_bailout_log):
    for j in prange(target_h):
        for i in range(target_w):
            s, _ = cpu_compute_perturbation_sample(
                ref_re, ref_im, ref_len, max_iter,
                ref_px, ref_py, float(i), float(j), dx, dy,
                ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                gen_n, gen_kr, gen_ki, 0, inv_ln2, ln_bailout_log
            )
            out_iters[j, i] = s


@njit(parallel=True, fastmath=True)
def cpu_probe_iter_floatexp_kernel(out_iters, target_w, target_h, max_iter,
                                   ref_re, ref_im, ref_len,
                                   ref_px, ref_py, dx_norm, dy_norm, E_scale,
                                   ref_cx, ref_cy, fractal_type,
                                   julia_cx, julia_cy,
                                   bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                   bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                   bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                   bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                                   gen_n, gen_kr, gen_ki, inv_ln2, ln_bailout_log):
    for j in prange(target_h):
        for i in range(target_w):
            s, _ = cpu_compute_perturbation_sample_floatexp(
                ref_re, ref_im, ref_len, max_iter,
                ref_px, ref_py, float(i), float(j), dx_norm, dy_norm,
                E_scale, ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                gen_n, gen_kr, gen_ki, 0, inv_ln2, ln_bailout_log
            )
            out_iters[j, i] = s



# --- Multi-Core Multi-Pass SSAA CPU FP64 Render Kernel ---
@njit(parallel=True, fastmath=True)
def cpu_render_fp64_kernel(out_rgb, iter_map, lut, target_w, target_h, max_iter,
                           x_min, y_max, dx_sub, dy_sub, factor, scheme_id,
                           fractal_type, julia_cx, julia_cy, gen_n, gen_kr, gen_ki,
                           edge_threshold, inv_ln2, ln_bailout_log, palette_offset=0.0,
                           color_density=1.0, color_contrast=1.0):
    if factor == 1:
        for j in prange(target_h):
            y = y_max - j * dy_sub
            for i in range(target_w):
                x = x_min + i * dx_sub
                s = cpu_compute_fp64_sample(x, y, max_iter, fractal_type, julia_cx, julia_cy, gen_n, gen_kr, gen_ki, inv_ln2, ln_bailout_log)
                iter_map[j, i] = s
                if s >= 0.0:
                    norm = cpu_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
                    sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                    out_rgb[j, i, 0] = int(sr + 0.5)
                    out_rgb[j, i, 1] = int(sg + 0.5)
                    out_rgb[j, i, 2] = int(sb + 0.5)
                else:
                    out_rgb[j, i, 0] = 0
                    out_rgb[j, i, 1] = 0
                    out_rgb[j, i, 2] = 0
    else:
        sub_offset = (float(factor) - 1.0) * 0.5
        for j in prange(target_h):
            y = y_max - (float(j * factor) + sub_offset) * dy_sub
            for i in range(target_w):
                x = x_min + (float(i * factor) + sub_offset) * dx_sub
                s = cpu_compute_fp64_sample(x, y, max_iter, fractal_type, julia_cx, julia_cy, gen_n, gen_kr, gen_ki, inv_ln2, ln_bailout_log)
                iter_map[j, i] = s
                if s >= 0.0:
                    norm = cpu_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
                    sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                    out_rgb[j, i, 0] = int(sr + 0.5)
                    out_rgb[j, i, 1] = int(sg + 0.5)
                    out_rgb[j, i, 2] = int(sb + 0.5)
                else:
                    out_rgb[j, i, 0] = 0
                    out_rgb[j, i, 1] = 0
                    out_rgb[j, i, 2] = 0

        num_samples = float(factor * factor)
        inv_samples = 1.0 / num_samples

        for j in prange(target_h):
            base_j = j * factor
            for i in range(target_w):
                s = iter_map[j, i]
                is_edge = False
                if 0 < j < target_h - 1 and 0 < i < target_w - 1:
                    s_r = iter_map[j, i + 1]
                    s_l = iter_map[j, i - 1]
                    s_d = iter_map[j + 1, i]
                    s_u = iter_map[j - 1, i]
                    if (s < 0.0) != (s_r < 0.0) or (s >= 0.0 and math.fabs(s - s_r) > edge_threshold) or \
                       (s < 0.0) != (s_l < 0.0) or (s >= 0.0 and math.fabs(s - s_l) > edge_threshold) or \
                       (s < 0.0) != (s_d < 0.0) or (s >= 0.0 and math.fabs(s - s_d) > edge_threshold) or \
                       (s < 0.0) != (s_u < 0.0) or (s >= 0.0 and math.fabs(s - s_u) > edge_threshold):
                        is_edge = True
                    else:
                        s_dr = iter_map[j + 1, i + 1]
                        s_dl = iter_map[j + 1, i - 1]
                        s_ur = iter_map[j - 1, i + 1]
                        s_ul = iter_map[j - 1, i - 1]
                        if (s < 0.0) != (s_dr < 0.0) or (s >= 0.0 and math.fabs(s - s_dr) > edge_threshold) or \
                           (s < 0.0) != (s_dl < 0.0) or (s >= 0.0 and math.fabs(s - s_dl) > edge_threshold) or \
                           (s < 0.0) != (s_ur < 0.0) or (s >= 0.0 and math.fabs(s - s_ur) > edge_threshold) or \
                           (s < 0.0) != (s_ul < 0.0) or (s >= 0.0 and math.fabs(s - s_ul) > edge_threshold):
                            is_edge = True
                else:
                    for dj in range(-1, 2):
                        nj = j + dj
                        if nj < 0 or nj >= target_h:
                            continue
                        for di in range(-1, 2):
                            ni = i + di
                            if ni < 0 or ni >= target_w or (di == 0 and dj == 0):
                                continue
                            s_neighbor = iter_map[nj, ni]
                            if (s < 0.0) != (s_neighbor < 0.0) or (s >= 0.0 and math.fabs(s - s_neighbor) > edge_threshold):
                                is_edge = True
                                break
                        if is_edge:
                            break

                if is_edge:
                    r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
                    base_i = i * factor
                    for sy in range(factor):
                        y = y_max - float(base_j + sy) * dy_sub
                        for sx in range(factor):
                            x = x_min + float(base_i + sx) * dx_sub
                            s_sample = cpu_compute_fp64_sample(x, y, max_iter, fractal_type, julia_cx, julia_cy, gen_n, gen_kr, gen_ki, inv_ln2, ln_bailout_log)
                            if s_sample >= 0.0:
                                norm = cpu_palette_norm(s_sample, scheme_id, palette_offset, color_density, color_contrast)
                                sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                                r_acc += sr
                                g_acc += sg
                                b_acc += sb
                    out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
                    out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
                    out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)


# --- Multi-Core Multi-Pass SSAA CPU DD Render Kernel ---
@njit(parallel=True)
def cpu_render_dd_kernel(out_rgb, iter_map, lut, target_w, target_h, max_iter,
                         x_min_hi, x_min_lo, dx_hi, dx_lo,
                         y_max_hi, y_max_lo, dy_hi, dy_lo,
                         factor, scheme_id, fractal_type,
                         julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo,
                         gen_n, gen_kr, gen_ki, edge_threshold,
                         inv_ln2, ln_bailout_log, palette_offset=0.0,
                         color_density=1.0, color_contrast=1.0):
    if factor == 1:
        for j in prange(target_h):
            jdy_hi, jdy_lo = cpu_dd_mul_f64(dy_hi, dy_lo, float(j))
            cy_hi, cy_lo = cpu_dd_sub(y_max_hi, y_max_lo, jdy_hi, jdy_lo)
            for i in range(target_w):
                idx_hi, idx_lo = cpu_dd_mul_f64(dx_hi, dx_lo, float(i))
                cx_hi, cx_lo = cpu_dd_add(x_min_hi, x_min_lo, idx_hi, idx_lo)

                s = cpu_compute_dd_sample(
                    cx_hi, cx_lo, cy_hi, cy_lo, max_iter, fractal_type,
                    julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo,
                    gen_n, gen_kr, gen_ki, inv_ln2, ln_bailout_log
                )
                iter_map[j, i] = s
                if s >= 0.0:
                    norm = cpu_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
                    sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                    out_rgb[j, i, 0] = int(sr + 0.5)
                    out_rgb[j, i, 1] = int(sg + 0.5)
                    out_rgb[j, i, 2] = int(sb + 0.5)
                else:
                    out_rgb[j, i, 0] = 0
                    out_rgb[j, i, 1] = 0
                    out_rgb[j, i, 2] = 0
    else:
        sub_offset = (float(factor) - 1.0) * 0.5
        for j in prange(target_h):
            sub_j = float(j * factor) + sub_offset
            jdy_hi, jdy_lo = cpu_dd_mul_f64(dy_hi, dy_lo, sub_j)
            cy_hi, cy_lo = cpu_dd_sub(y_max_hi, y_max_lo, jdy_hi, jdy_lo)
            for i in range(target_w):
                sub_i = float(i * factor) + sub_offset
                idx_hi, idx_lo = cpu_dd_mul_f64(dx_hi, dx_lo, sub_i)
                cx_hi, cx_lo = cpu_dd_add(x_min_hi, x_min_lo, idx_hi, idx_lo)

                s = cpu_compute_dd_sample(
                    cx_hi, cx_lo, cy_hi, cy_lo, max_iter, fractal_type,
                    julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo,
                    gen_n, gen_kr, gen_ki, inv_ln2, ln_bailout_log
                )
                iter_map[j, i] = s
                if s >= 0.0:
                    norm = cpu_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
                    sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                    out_rgb[j, i, 0] = int(sr + 0.5)
                    out_rgb[j, i, 1] = int(sg + 0.5)
                    out_rgb[j, i, 2] = int(sb + 0.5)
                else:
                    out_rgb[j, i, 0] = 0
                    out_rgb[j, i, 1] = 0
                    out_rgb[j, i, 2] = 0

        num_samples = float(factor * factor)
        inv_samples = 1.0 / num_samples

        for j in prange(target_h):
            base_j = float(j * factor)
            for i in range(target_w):
                s = iter_map[j, i]
                is_edge = False
                if 0 < j < target_h - 1 and 0 < i < target_w - 1:
                    s_r = iter_map[j, i + 1]
                    s_l = iter_map[j, i - 1]
                    s_d = iter_map[j + 1, i]
                    s_u = iter_map[j - 1, i]
                    if (s < 0.0) != (s_r < 0.0) or (s >= 0.0 and math.fabs(s - s_r) > edge_threshold) or \
                       (s < 0.0) != (s_l < 0.0) or (s >= 0.0 and math.fabs(s - s_l) > edge_threshold) or \
                       (s < 0.0) != (s_d < 0.0) or (s >= 0.0 and math.fabs(s - s_d) > edge_threshold) or \
                       (s < 0.0) != (s_u < 0.0) or (s >= 0.0 and math.fabs(s - s_u) > edge_threshold):
                        is_edge = True
                    else:
                        s_dr = iter_map[j + 1, i + 1]
                        s_dl = iter_map[j + 1, i - 1]
                        s_ur = iter_map[j - 1, i + 1]
                        s_ul = iter_map[j - 1, i - 1]
                        if (s < 0.0) != (s_dr < 0.0) or (s >= 0.0 and math.fabs(s - s_dr) > edge_threshold) or \
                           (s < 0.0) != (s_dl < 0.0) or (s >= 0.0 and math.fabs(s - s_dl) > edge_threshold) or \
                           (s < 0.0) != (s_ur < 0.0) or (s >= 0.0 and math.fabs(s - s_ur) > edge_threshold) or \
                           (s < 0.0) != (s_ul < 0.0) or (s >= 0.0 and math.fabs(s - s_ul) > edge_threshold):
                            is_edge = True
                else:
                    for dj in range(-1, 2):
                        nj = j + dj
                        if nj < 0 or nj >= target_h:
                            continue
                        for di in range(-1, 2):
                            ni = i + di
                            if ni < 0 or ni >= target_w or (di == 0 and dj == 0):
                                continue
                            s_neighbor = iter_map[nj, ni]
                            if (s < 0.0) != (s_neighbor < 0.0) or (s >= 0.0 and math.fabs(s - s_neighbor) > edge_threshold):
                                is_edge = True
                                break
                        if is_edge:
                            break

                if is_edge:
                    r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
                    base_i = float(i * factor)
                    for sy in range(factor):
                        sub_j = base_j + float(sy)
                        jdy_hi, jdy_lo = cpu_dd_mul_f64(dy_hi, dy_lo, sub_j)
                        cy_hi, cy_lo = cpu_dd_sub(y_max_hi, y_max_lo, jdy_hi, jdy_lo)
                        for sx in range(factor):
                            sub_i = base_i + float(sx)
                            idx_hi, idx_lo = cpu_dd_mul_f64(dx_hi, dx_lo, sub_i)
                            cx_hi, cx_lo = cpu_dd_add(x_min_hi, x_min_lo, idx_hi, idx_lo)

                            s_sample = cpu_compute_dd_sample(
                                cx_hi, cx_lo, cy_hi, cy_lo, max_iter, fractal_type,
                                julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo,
                                gen_n, gen_kr, gen_ki, inv_ln2, ln_bailout_log
                            )
                            if s_sample >= 0.0:
                                norm = cpu_palette_norm(s_sample, scheme_id, palette_offset, color_density, color_contrast)
                                sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                                r_acc += sr
                                g_acc += sg
                                b_acc += sb
                    out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
                    out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
                    out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)


# --- Multi-Core Multi-Pass SSAA CPU Perturbation Render Kernel ---
@njit(parallel=True)
def cpu_render_perturbation_kernel(out_rgb, out_glitch, iter_map, lut, target_w, target_h, max_iter,
                                   ref_re, ref_im, ref_len,
                                   ref_px_sub, ref_py_sub, dx_sub, dy_sub,
                                   ref_cx, ref_cy, factor, scheme_id, fractal_type,
                                   julia_cx, julia_cy,
                                   bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                   bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                   bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                   bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                                   rebase_pass, in_glitch_mask,
                                   gen_n, gen_kr, gen_ki, edge_threshold, strict_glitch,
                                   inv_ln2, ln_bailout_log, palette_offset=0.0,
                                   color_density=1.0, color_contrast=1.0,
                                   bla_table=np.zeros((1, 1, 1), dtype=np.float64)):
    if factor == 1:
        for j in prange(target_h):
            for i in range(target_w):
                if rebase_pass > 0 and in_glitch_mask[j, i] == 0:
                    continue

                s, is_gl = cpu_compute_perturbation_sample(
                    ref_re, ref_im, ref_len, max_iter,
                    ref_px_sub, ref_py_sub, float(i), float(j), dx_sub, dy_sub,
                    ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                    gen_n, gen_kr, gen_ki, strict_glitch,
                    inv_ln2, ln_bailout_log,
                    bla_table=bla_table
                )

                iter_map[j, i] = s
                out_glitch[j, i] = 1 if is_gl > 0 else 0

                if s >= 0.0:
                    norm = cpu_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
                    sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                    out_rgb[j, i, 0] = int(sr + 0.5)
                    out_rgb[j, i, 1] = int(sg + 0.5)
                    out_rgb[j, i, 2] = int(sb + 0.5)
                else:
                    out_rgb[j, i, 0] = 0
                    out_rgb[j, i, 1] = 0
                    out_rgb[j, i, 2] = 0
    else:
        sub_offset = (float(factor) - 1.0) * 0.5
        for j in prange(target_h):
            sub_j = float(j * factor) + sub_offset
            for i in range(target_w):
                if rebase_pass > 0 and in_glitch_mask[j, i] == 0:
                    continue

                sub_i = float(i * factor) + sub_offset
                s, is_gl = cpu_compute_perturbation_sample(
                    ref_re, ref_im, ref_len, max_iter,
                    ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub, dy_sub,
                    ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                    gen_n, gen_kr, gen_ki, strict_glitch,
                    inv_ln2, ln_bailout_log,
                    bla_table=bla_table
                )

                iter_map[j, i] = s
                out_glitch[j, i] = 1 if is_gl > 0 else 0

                if s >= 0.0:
                    norm = cpu_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
                    sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                    out_rgb[j, i, 0] = int(sr + 0.5)
                    out_rgb[j, i, 1] = int(sg + 0.5)
                    out_rgb[j, i, 2] = int(sb + 0.5)
                else:
                    out_rgb[j, i, 0] = 0
                    out_rgb[j, i, 1] = 0
                    out_rgb[j, i, 2] = 0

        num_samples = float(factor * factor)
        inv_samples = 1.0 / num_samples

        for j in prange(target_h):
            base_j = float(j * factor)
            for i in range(target_w):
                if rebase_pass > 0 and in_glitch_mask[j, i] == 0:
                    continue

                s = iter_map[j, i]
                is_edge = False
                if rebase_pass > 0:
                    is_edge = True
                elif 0 < j < target_h - 1 and 0 < i < target_w - 1:
                    s_r = iter_map[j, i + 1]
                    s_l = iter_map[j, i - 1]
                    s_d = iter_map[j + 1, i]
                    s_u = iter_map[j - 1, i]
                    if (s < 0.0) != (s_r < 0.0) or (s >= 0.0 and math.fabs(s - s_r) > edge_threshold) or \
                       (s < 0.0) != (s_l < 0.0) or (s >= 0.0 and math.fabs(s - s_l) > edge_threshold) or \
                       (s < 0.0) != (s_d < 0.0) or (s >= 0.0 and math.fabs(s - s_d) > edge_threshold) or \
                       (s < 0.0) != (s_u < 0.0) or (s >= 0.0 and math.fabs(s - s_u) > edge_threshold):
                        is_edge = True
                    else:
                        s_dr = iter_map[j + 1, i + 1]
                        s_dl = iter_map[j + 1, i - 1]
                        s_ur = iter_map[j - 1, i + 1]
                        s_ul = iter_map[j - 1, i - 1]
                        if (s < 0.0) != (s_dr < 0.0) or (s >= 0.0 and math.fabs(s - s_dr) > edge_threshold) or \
                           (s < 0.0) != (s_dl < 0.0) or (s >= 0.0 and math.fabs(s - s_dl) > edge_threshold) or \
                           (s < 0.0) != (s_ur < 0.0) or (s >= 0.0 and math.fabs(s - s_ur) > edge_threshold) or \
                           (s < 0.0) != (s_ul < 0.0) or (s >= 0.0 and math.fabs(s - s_ul) > edge_threshold):
                            is_edge = True
                else:
                    for dj in range(-1, 2):
                        nj = j + dj
                        if nj < 0 or nj >= target_h:
                            continue
                        for di in range(-1, 2):
                            ni = i + di
                            if ni < 0 or ni >= target_w or (di == 0 and dj == 0):
                                continue
                            s_neighbor = iter_map[nj, ni]
                            if (s < 0.0) != (s_neighbor < 0.0) or (s >= 0.0 and math.fabs(s - s_neighbor) > edge_threshold):
                                is_edge = True
                                break
                        if is_edge:
                            break

                if is_edge:
                    r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
                    glitch_count = 0
                    base_i = float(i * factor)
                    for sy in range(factor):
                        sub_j = base_j + float(sy)
                        for sx in range(factor):
                            sub_i = base_i + float(sx)
                            s_sample, is_gl = cpu_compute_perturbation_sample(
                                ref_re, ref_im, ref_len, max_iter,
                                ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub, dy_sub,
                                ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                                bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                                gen_n, gen_kr, gen_ki, strict_glitch,
                                inv_ln2, ln_bailout_log,
                                bla_table=bla_table
                            )
                            if is_gl > 0:
                                glitch_count += 1
                            if s_sample >= 0.0:
                                norm = cpu_palette_norm(s_sample, scheme_id, palette_offset, color_density, color_contrast)
                                sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                                r_acc += sr
                                g_acc += sg
                                b_acc += sb

                    out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
                    out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
                    out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)
                    out_glitch[j, i] = 1 if glitch_count > 0 else 0


@njit(parallel=True, fastmath=True)
def cpu_render_floatexp_kernel(
    out_rgb, out_glitch, iter_map, lut, target_w, target_h, max_iter,
    ref_re, ref_im, ref_len,
    ref_px_sub, ref_py_sub, dx_sub_norm, dy_sub_norm, E_scale,
    ref_cx, ref_cy, factor, scheme_id, fractal_type,
    julia_cx, julia_cy,
    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
    rebase_pass, in_glitch_mask, gen_n, gen_kr, gen_ki, strict_glitch,
    edge_threshold, inv_ln2, ln_bailout_log, palette_offset=0.0,
    color_density=1.0, color_contrast=1.0,
    bla_table=np.zeros((1, 1, 1), dtype=np.float64)
):
    sub_offset = (float(factor) - 1.0) * 0.5
    if factor == 1:
        for j in prange(target_h):
            for i in range(target_w):
                if rebase_pass > 0 and in_glitch_mask[j, i] == 0:
                    continue
                sub_j = float(j) + sub_offset
                sub_i = float(i) + sub_offset
                s, is_gl = cpu_compute_perturbation_sample_floatexp(
                    ref_re, ref_im, ref_len, max_iter,
                    ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub_norm, dy_sub_norm,
                    E_scale, ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                    gen_n, gen_kr, gen_ki, strict_glitch,
                    inv_ln2, ln_bailout_log,
                    bla_table=bla_table
                )
                iter_map[j, i] = s
                out_glitch[j, i] = 1 if is_gl > 0 else 0
                if s >= 0.0:
                    norm = cpu_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
                    sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                    out_rgb[j, i, 0] = int(sr + 0.5)
                    out_rgb[j, i, 1] = int(sg + 0.5)
                    out_rgb[j, i, 2] = int(sb + 0.5)
                else:
                    out_rgb[j, i, 0] = 0
                    out_rgb[j, i, 1] = 0
                    out_rgb[j, i, 2] = 0
    else:
        # Pass 1: Base coarse grid
        for j in prange(target_h):
            for i in range(target_w):
                if rebase_pass > 0 and in_glitch_mask[j, i] == 0:
                    continue
                sub_j = float(j * factor) + sub_offset
                sub_i = float(i * factor) + sub_offset
                s, is_gl = cpu_compute_perturbation_sample_floatexp(
                    ref_re, ref_im, ref_len, max_iter,
                    ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub_norm, dy_sub_norm,
                    E_scale, ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                    gen_n, gen_kr, gen_ki, strict_glitch,
                    inv_ln2, ln_bailout_log,
                    bla_table=bla_table
                )
                iter_map[j, i] = s
                out_glitch[j, i] = 1 if is_gl > 0 else 0
                if s >= 0.0:
                    norm = cpu_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
                    sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                    out_rgb[j, i, 0] = int(sr + 0.5)
                    out_rgb[j, i, 1] = int(sg + 0.5)
                    out_rgb[j, i, 2] = int(sb + 0.5)
                else:
                    out_rgb[j, i, 0] = 0
                    out_rgb[j, i, 1] = 0
                    out_rgb[j, i, 2] = 0

        # Pass 2: Edge refinement
        inv_samples = 1.0 / float(factor * factor)
        for j in prange(target_h):
            base_j = float(j * factor)
            for i in range(target_w):
                if rebase_pass > 0 and in_glitch_mask[j, i] == 0:
                    continue

                s = iter_map[j, i]
                is_edge = False
                if rebase_pass > 0:
                    is_edge = True
                elif 0 < j < target_h - 1 and 0 < i < target_w - 1:
                    s_r = iter_map[j, i + 1]
                    s_l = iter_map[j, i - 1]
                    s_d = iter_map[j + 1, i]
                    s_u = iter_map[j - 1, i]
                    if (s < 0.0) != (s_r < 0.0) or (s >= 0.0 and math.fabs(s - s_r) > edge_threshold) or \
                       (s < 0.0) != (s_l < 0.0) or (s >= 0.0 and math.fabs(s - s_l) > edge_threshold) or \
                       (s < 0.0) != (s_d < 0.0) or (s >= 0.0 and math.fabs(s - s_d) > edge_threshold) or \
                       (s < 0.0) != (s_u < 0.0) or (s >= 0.0 and math.fabs(s - s_u) > edge_threshold):
                        is_edge = True
                    else:
                        s_dr = iter_map[j + 1, i + 1]
                        s_dl = iter_map[j + 1, i - 1]
                        s_ur = iter_map[j - 1, i + 1]
                        s_ul = iter_map[j - 1, i - 1]
                        if (s < 0.0) != (s_dr < 0.0) or (s >= 0.0 and math.fabs(s - s_dr) > edge_threshold) or \
                           (s < 0.0) != (s_dl < 0.0) or (s >= 0.0 and math.fabs(s - s_dl) > edge_threshold) or \
                           (s < 0.0) != (s_ur < 0.0) or (s >= 0.0 and math.fabs(s - s_ur) > edge_threshold) or \
                           (s < 0.0) != (s_ul < 0.0) or (s >= 0.0 and math.fabs(s - s_ul) > edge_threshold):
                            is_edge = True
                else:
                    for dj in range(-1, 2):
                        nj = j + dj
                        if nj < 0 or nj >= target_h:
                            continue
                        for di in range(-1, 2):
                            ni = i + di
                            if ni < 0 or ni >= target_w or (di == 0 and dj == 0):
                                continue
                            s_neighbor = iter_map[nj, ni]
                            if (s < 0.0) != (s_neighbor < 0.0) or (s >= 0.0 and math.fabs(s - s_neighbor) > edge_threshold):
                                is_edge = True
                                break
                        if is_edge:
                            break

                if is_edge:
                    r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
                    glitch_count = 0
                    base_i = float(i * factor)
                    for sy in range(factor):
                        sub_j = base_j + float(sy)
                        for sx in range(factor):
                            sub_i = base_i + float(sx)
                            s_sample, is_gl = cpu_compute_perturbation_sample_floatexp(
                                ref_re, ref_im, ref_len, max_iter,
                                ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub_norm, dy_sub_norm,
                                E_scale, ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                                bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                                gen_n, gen_kr, gen_ki, strict_glitch,
                                inv_ln2, ln_bailout_log,
                                bla_table=bla_table
                            )
                            if is_gl > 0:
                                glitch_count += 1
                            if s_sample >= 0.0:
                                norm = cpu_palette_norm(s_sample, scheme_id, palette_offset, color_density, color_contrast)
                                sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                                r_acc += sr
                                g_acc += sg
                                b_acc += sb

                    out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
                    out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
                    out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)
                    out_glitch[j, i] = 1 if glitch_count > 0 else 0
# --- CPU Histogram Equalization Shaders & Kernels ---
@njit(fastmath=True, inline='always')
def cpu_sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset=0.0, color_density=1.0, color_contrast=1.0):
    if s < 0.0:
        return -1.0
    inv_max = 1.0 / max(1.0, float(max_iter))
    pos = (s * inv_max) * float(num_bins - 1)
    k = int(pos)
    if k >= num_bins - 1:
        cdf_val = cdf_lut[num_bins - 1]
    elif k < 0:
        cdf_val = cdf_lut[0]
    else:
        frac = pos - float(k)
        c0 = cdf_lut[k]
        c1 = cdf_lut[k + 1]
        cdf_val = c0 + frac * (c1 - c0)

    if color_contrast != 1.0 and color_contrast > 0.0:
        cdf_val = math.pow(cdf_val, color_contrast)

    dens = 1.0 if color_density <= 0.0 else color_density
    phase = (cdf_val * (2.0 * dens) + palette_offset * 2.0) % 2.0
    tri = 1.0 - math.fabs((phase + 2.0 if phase < 0.0 else phase) - 1.0)
    return tri


@njit(parallel=True)
def cpu_build_histogram(iter_map, hist, target_w, target_h, max_iter, num_bins):
    for j in range(target_h):
        for i in range(target_w):
            s = iter_map[j, i]
            if s >= 0.0:
                inv_max = 1.0 / max(1.0, float(max_iter))
                pos = (s * inv_max) * float(num_bins - 1)
                bin_idx = int(pos)
                if bin_idx >= num_bins:
                    bin_idx = num_bins - 1
                elif bin_idx < 0:
                    bin_idx = 0
                hist[bin_idx] += 1


@njit(parallel=True, fastmath=True)
def cpu_render_fp64_hist_kernel(out_rgb, iter_map, cdf_lut, lut, target_w, target_h, max_iter,
                                x_min, y_max, dx_sub, dy_sub, factor, fractal_type,
                                julia_cx, julia_cy, gen_n, gen_kr, gen_ki,
                                edge_threshold, inv_ln2, ln_bailout_log, num_bins, palette_offset=0.0,
                                color_density=1.0, color_contrast=1.0):
    if factor == 1:
        for j in prange(target_h):
            for i in range(target_w):
                s = iter_map[j, i]
                if s >= 0.0:
                    norm = cpu_sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                    sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                    out_rgb[j, i, 0] = int(sr + 0.5)
                    out_rgb[j, i, 1] = int(sg + 0.5)
                    out_rgb[j, i, 2] = int(sb + 0.5)
                else:
                    out_rgb[j, i, 0] = 0
                    out_rgb[j, i, 1] = 0
                    out_rgb[j, i, 2] = 0
    else:
        for j in prange(target_h):
            for i in range(target_w):
                s = iter_map[j, i]
                if s >= 0.0:
                    norm = cpu_sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                    sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                    out_rgb[j, i, 0] = int(sr + 0.5)
                    out_rgb[j, i, 1] = int(sg + 0.5)
                    out_rgb[j, i, 2] = int(sb + 0.5)
                else:
                    out_rgb[j, i, 0] = 0
                    out_rgb[j, i, 1] = 0
                    out_rgb[j, i, 2] = 0

        num_samples = float(factor * factor)
        inv_samples = 1.0 / num_samples

        for j in prange(target_h):
            for i in range(target_w):
                s = iter_map[j, i]
                is_edge = False
                for dj in range(-1, 2):
                    nj = j + dj
                    if nj < 0 or nj >= target_h:
                        continue
                    for di in range(-1, 2):
                        ni = i + di
                        if ni < 0 or ni >= target_w or (di == 0 and dj == 0):
                            continue
                        s_neighbor = iter_map[nj, ni]
                        if (s < 0.0) != (s_neighbor < 0.0) or (s >= 0.0 and math.fabs(s - s_neighbor) > edge_threshold):
                            is_edge = True
                            break
                    if is_edge:
                        break

                if is_edge:
                    r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
                    for sy in range(factor):
                        y = y_max - (j * factor + sy) * dy_sub
                        for sx in range(factor):
                            x = x_min + (i * factor + sx) * dx_sub
                            s_sample = cpu_compute_fp64_sample(x, y, max_iter, fractal_type, julia_cx, julia_cy, gen_n, gen_kr, gen_ki, inv_ln2, ln_bailout_log)
                            if s_sample >= 0.0:
                                norm = cpu_sample_cdf_norm(s_sample, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                                sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                                r_acc += sr
                                g_acc += sg
                                b_acc += sb
                    out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
                    out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
                    out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)


@njit(parallel=True)
def cpu_render_dd_hist_kernel(out_rgb, iter_map, cdf_lut, lut, target_w, target_h, max_iter,
                              x_min_hi, x_min_lo, dx_hi, dx_lo,
                              y_max_hi, y_max_lo, dy_hi, dy_lo,
                              factor, fractal_type,
                              julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo,
                              gen_n, gen_kr, gen_ki, edge_threshold,
                              inv_ln2, ln_bailout_log, num_bins, palette_offset=0.0,
                              color_density=1.0, color_contrast=1.0):
    if factor == 1:
        for j in prange(target_h):
            for i in range(target_w):
                s = iter_map[j, i]
                if s >= 0.0:
                    norm = cpu_sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                    sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                    out_rgb[j, i, 0] = int(sr + 0.5)
                    out_rgb[j, i, 1] = int(sg + 0.5)
                    out_rgb[j, i, 2] = int(sb + 0.5)
                else:
                    out_rgb[j, i, 0] = 0
                    out_rgb[j, i, 1] = 0
                    out_rgb[j, i, 2] = 0
    else:
        for j in prange(target_h):
            for i in range(target_w):
                s = iter_map[j, i]
                if s >= 0.0:
                    norm = cpu_sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                    sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                    out_rgb[j, i, 0] = int(sr + 0.5)
                    out_rgb[j, i, 1] = int(sg + 0.5)
                    out_rgb[j, i, 2] = int(sb + 0.5)
                else:
                    out_rgb[j, i, 0] = 0
                    out_rgb[j, i, 1] = 0
                    out_rgb[j, i, 2] = 0

        num_samples = float(factor * factor)
        inv_samples = 1.0 / num_samples

        for j in prange(target_h):
            for i in range(target_w):
                s = iter_map[j, i]
                is_edge = False
                for dj in range(-1, 2):
                    nj = j + dj
                    if nj < 0 or nj >= target_h:
                        continue
                    for di in range(-1, 2):
                        ni = i + di
                        if ni < 0 or ni >= target_w or (di == 0 and dj == 0):
                            continue
                        s_neighbor = iter_map[nj, ni]
                        if (s < 0.0) != (s_neighbor < 0.0) or (s >= 0.0 and math.fabs(s - s_neighbor) > edge_threshold):
                            is_edge = True
                            break
                    if is_edge:
                        break

                if is_edge:
                    r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
                    for sy in range(factor):
                        sub_j = j * factor + sy
                        jdy_hi, jdy_lo = cpu_dd_mul_f64(dy_hi, dy_lo, float(sub_j))
                        cy_hi, cy_lo = cpu_dd_sub(y_max_hi, y_max_lo, jdy_hi, jdy_lo)
                        for sx in range(factor):
                            sub_i = i * factor + sx
                            idx_hi, idx_lo = cpu_dd_mul_f64(dx_hi, dx_lo, float(sub_i))
                            cx_hi, cx_lo = cpu_dd_add(x_min_hi, x_min_lo, idx_hi, idx_lo)

                            s_sample = cpu_compute_dd_sample(
                                cx_hi, cx_lo, cy_hi, cy_lo, max_iter, fractal_type,
                                julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo,
                                gen_n, gen_kr, gen_ki, inv_ln2, ln_bailout_log
                            )
                            if s_sample >= 0.0:
                                norm = cpu_sample_cdf_norm(s_sample, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                                sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                                r_acc += sr
                                g_acc += sg
                                b_acc += sb
                    out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
                    out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
                    out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)


@njit(parallel=True, fastmath=True)
def cpu_render_perturbation_hist_kernel(out_rgb, out_glitch, iter_map, cdf_lut, lut, target_w, target_h, max_iter,
                                        ref_re, ref_im, ref_len,
                                        ref_px_sub, ref_py_sub, dx_sub, dy_sub,
                                        ref_cx, ref_cy, factor, fractal_type,
                                        julia_cx, julia_cy,
                                        bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                        bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                        bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                        bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                                        rebase_pass, in_glitch_mask, gen_n, gen_kr, gen_ki, strict_glitch,
                                        edge_threshold, inv_ln2, ln_bailout_log, num_bins, palette_offset=0.0,
                                        color_density=1.0, color_contrast=1.0):
    if factor == 1:
        for j in prange(target_h):
            for i in range(target_w):
                if rebase_pass > 0:
                    if in_glitch_mask[j, i] == 0:
                        continue
                    s, is_gl = cpu_compute_perturbation_sample(
                        ref_re, ref_im, ref_len, max_iter,
                        ref_px_sub, ref_py_sub, float(i), float(j), dx_sub, dy_sub,
                        ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                        bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                        bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                        bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                        bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                        gen_n, gen_kr, gen_ki, strict_glitch,
                        inv_ln2, ln_bailout_log
                    )
                    out_glitch[j, i] = 1 if is_gl > 0 else 0
                    iter_map[j, i] = s
                else:
                    s = iter_map[j, i]
                if s >= 0.0:
                    norm = cpu_sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                    sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                    out_rgb[j, i, 0] = int(sr + 0.5)
                    out_rgb[j, i, 1] = int(sg + 0.5)
                    out_rgb[j, i, 2] = int(sb + 0.5)
                else:
                    out_rgb[j, i, 0] = 0
                    out_rgb[j, i, 1] = 0
                    out_rgb[j, i, 2] = 0
    else:
        if rebase_pass == 0:
            for j in prange(target_h):
                for i in range(target_w):
                    s = iter_map[j, i]
                    if s >= 0.0:
                        norm = cpu_sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                        sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                        out_rgb[j, i, 0] = int(sr + 0.5)
                        out_rgb[j, i, 1] = int(sg + 0.5)
                        out_rgb[j, i, 2] = int(sb + 0.5)
                    else:
                        out_rgb[j, i, 0] = 0
                        out_rgb[j, i, 1] = 0
                        out_rgb[j, i, 2] = 0

        num_samples = float(factor * factor)
        inv_samples = 1.0 / num_samples

        for j in prange(target_h):
            for i in range(target_w):
                if rebase_pass > 0 and in_glitch_mask[j, i] == 0:
                    continue

                s = iter_map[j, i]
                is_edge = False
                if rebase_pass > 0:
                    is_edge = True
                else:
                    for dj in range(-1, 2):
                        nj = j + dj
                        if nj < 0 or nj >= target_h:
                            continue
                        for di in range(-1, 2):
                            ni = i + di
                            if ni < 0 or ni >= target_w or (di == 0 and dj == 0):
                                continue
                            s_neighbor = iter_map[nj, ni]
                            if (s < 0.0) != (s_neighbor < 0.0) or (s >= 0.0 and math.fabs(s - s_neighbor) > edge_threshold):
                                is_edge = True
                                break
                        if is_edge:
                            break

                if is_edge:
                    r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
                    glitch_count = 0
                    for sy in range(factor):
                        sub_j = j * factor + sy
                        for sx in range(factor):
                            sub_i = i * factor + sx
                            s_sample, is_gl = cpu_compute_perturbation_sample(
                                ref_re, ref_im, ref_len, max_iter,
                                ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub, dy_sub,
                                ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                                bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                                gen_n, gen_kr, gen_ki, strict_glitch,
                                inv_ln2, ln_bailout_log
                            )
                            if is_gl > 0:
                                glitch_count += 1
                            if s_sample >= 0.0:
                                norm = cpu_sample_cdf_norm(s_sample, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                                sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                                r_acc += sr
                                g_acc += sg
                                b_acc += sb

                    out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
                    out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
                    out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)
                    out_glitch[j, i] = 1 if glitch_count > 0 else 0
                    if rebase_pass > 0:
                        iter_map[j, i] = s_sample


@njit(parallel=True, fastmath=True)
def cpu_render_floatexp_hist_kernel(
    out_rgb, out_glitch, iter_map, cdf_lut, lut, target_w, target_h, max_iter,
    ref_re, ref_im, ref_len,
    ref_px_sub, ref_py_sub, dx_sub_norm, dy_sub_norm, E_scale,
    ref_cx, ref_cy, factor, fractal_type,
    julia_cx, julia_cy,
    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
    rebase_pass, in_glitch_mask, gen_n, gen_kr, gen_ki, strict_glitch,
    edge_threshold, inv_ln2, ln_bailout_log, num_bins, palette_offset=0.0,
    color_density=1.0, color_contrast=1.0
):
    sub_offset = (float(factor) - 1.0) * 0.5
    if factor == 1:
        for j in prange(target_h):
            for i in range(target_w):
                if rebase_pass > 0:
                    if in_glitch_mask[j, i] == 0:
                        continue
                    s, is_gl = cpu_compute_perturbation_sample_floatexp(
                        ref_re, ref_im, ref_len, max_iter,
                        ref_px_sub, ref_py_sub, float(i), float(j), dx_sub_norm, dy_sub_norm,
                        E_scale, ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                        bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                        bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                        bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                        bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                        gen_n, gen_kr, gen_ki, strict_glitch,
                        inv_ln2, ln_bailout_log
                    )
                    out_glitch[j, i] = 1 if is_gl > 0 else 0
                    iter_map[j, i] = s
                else:
                    s = iter_map[j, i]
                if s >= 0.0:
                    norm = cpu_sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                    sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                    out_rgb[j, i, 0] = int(sr + 0.5)
                    out_rgb[j, i, 1] = int(sg + 0.5)
                    out_rgb[j, i, 2] = int(sb + 0.5)
                else:
                    out_rgb[j, i, 0] = 0
                    out_rgb[j, i, 1] = 0
                    out_rgb[j, i, 2] = 0
    else:
        if rebase_pass == 0:
            for j in prange(target_h):
                for i in range(target_w):
                    s = iter_map[j, i]
                    if s >= 0.0:
                        norm = cpu_sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                        sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                        out_rgb[j, i, 0] = int(sr + 0.5)
                        out_rgb[j, i, 1] = int(sg + 0.5)
                        out_rgb[j, i, 2] = int(sb + 0.5)
                    else:
                        out_rgb[j, i, 0] = 0
                        out_rgb[j, i, 1] = 0
                        out_rgb[j, i, 2] = 0

        inv_samples = 1.0 / float(factor * factor)
        for j in prange(target_h):
            for i in range(target_w):
                if rebase_pass > 0 and in_glitch_mask[j, i] == 0:
                    continue

                s = iter_map[j, i]
                is_edge = False
                if rebase_pass > 0:
                    is_edge = True
                else:
                    for dj in range(-1, 2):
                        nj = j + dj
                        if nj < 0 or nj >= target_h:
                            continue
                        for di in range(-1, 2):
                            ni = i + di
                            if ni < 0 or ni >= target_w or (di == 0 and dj == 0):
                                continue
                            s_neighbor = iter_map[nj, ni]
                            if (s < 0.0) != (s_neighbor < 0.0) or (s >= 0.0 and math.fabs(s - s_neighbor) > edge_threshold):
                                is_edge = True
                                break
                        if is_edge:
                            break

                if is_edge:
                    r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
                    glitch_count = 0
                    for sy in range(factor):
                        sub_j = float(j * factor + sy)
                        for sx in range(factor):
                            sub_i = float(i * factor + sx)
                            s_sample, is_gl = cpu_compute_perturbation_sample_floatexp(
                                ref_re, ref_im, ref_len, max_iter,
                                ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub_norm, dy_sub_norm,
                                E_scale, ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                                bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                                gen_n, gen_kr, gen_ki, strict_glitch,
                                inv_ln2, ln_bailout_log
                            )
                            if is_gl > 0:
                                glitch_count += 1
                            if s_sample >= 0.0:
                                norm = cpu_sample_cdf_norm(s_sample, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                                sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                                r_acc += sr
                                g_acc += sg
                                b_acc += sb

                    out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
                    out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
                    out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)
                    out_glitch[j, i] = 1 if glitch_count > 0 else 0
                    if rebase_pass > 0:
                        iter_map[j, i] = s_sample


@njit(parallel=True, fastmath=True)
def cpu_recolor_iter_map_kernel(out_rgb, iter_map, lut, target_w, target_h,
                                scheme_id, palette_offset=0.0, color_density=1.0, color_contrast=1.0):
    """Ultra-fast CPU parallel kernel to re-shade already-computed iteration maps with new palette settings."""
    for j in prange(target_h):
        for i in range(target_w):
            s = iter_map[j, i]
            if s >= 0.0:
                norm = cpu_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
                sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                out_rgb[j, i, 0] = int(sr + 0.5)
                out_rgb[j, i, 1] = int(sg + 0.5)
                out_rgb[j, i, 2] = int(sb + 0.5)
            else:
                out_rgb[j, i, 0] = 0
                out_rgb[j, i, 1] = 0
                out_rgb[j, i, 2] = 0


@njit(parallel=True, fastmath=True)
def cpu_recolor_hist_kernel(out_rgb, iter_map, cdf_lut, lut, target_w, target_h,
                            max_iter, num_bins, palette_offset=0.0, color_density=1.0, color_contrast=1.0):
    """Ultra-fast CPU parallel kernel to re-shade already-computed iteration maps with histogram equalization."""
    for j in prange(target_h):
        for i in range(target_w):
            s = iter_map[j, i]
            if s >= 0.0:
                norm = cpu_sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                sr, sg, sb = cpu_sample_lut_lerp(lut, norm)
                out_rgb[j, i, 0] = int(sr + 0.5)
                out_rgb[j, i, 1] = int(sg + 0.5)
                out_rgb[j, i, 2] = int(sb + 0.5)
            else:
                out_rgb[j, i, 0] = 0
                out_rgb[j, i, 1] = 0
                out_rgb[j, i, 2] = 0