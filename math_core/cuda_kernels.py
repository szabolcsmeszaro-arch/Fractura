import math
from decimal import Decimal
import numpy as np
from numba import cuda, float32, float64

from config import HAS_GMPY2, INV_LN2_F32, LN_BAILOUT_LOG

# --- CUDA Hardware State & Initialization ---
HAS_CUDA = False

def init_primary_cuda_device():
    global HAS_CUDA
    try:
        if cuda.is_available():
            cuda.select_device(0)
            ctx = cuda.current_context()
            dev = cuda.get_current_device()
            dev_name = dev.name.decode('utf-8') if isinstance(dev.name, bytes) else dev.name
            free_mem_mb = ctx.get_memory_info().free / (1024**2)
            total_mem_mb = ctx.get_memory_info().total / (1024**2)
            print(f"GPU Initialized: {dev_name} | VRAM: {free_mem_mb:.0f} MB / {total_mem_mb:.0f} MB | GMPY2: {HAS_GMPY2}")
            HAS_CUDA = True
        else:
            print("CUDA is not available. Defaulting compute to CPU.")
            HAS_CUDA = False
    except Exception as e:
        print(f"CUDA Device Error: {e}. Defaulting compute to CPU.")
        HAS_CUDA = False

init_primary_cuda_device()


# --- CUDA Device Math Operators ---
@cuda.jit(device=True, inline=True)
def stage_lut_to_shared(lut, s_lut):
    tid = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    threads_per_block = cuda.blockDim.x * cuda.blockDim.y
    for idx in range(tid, 2048, threads_per_block):
        s_lut[idx, 0] = lut[idx, 0]
        s_lut[idx, 1] = lut[idx, 1]
        s_lut[idx, 2] = lut[idx, 2]
    cuda.syncthreads()

@cuda.jit(device=True, inline=True)
def two_sum(a, b):
    s = a + b
    v = s - a
    e = (a - (s - v)) + (b - v)
    return s, e

@cuda.jit(device=True, inline=True)
def two_prod(a, b):
    p = a * b
    e = cuda.fma(a, b, -p)
    return p, e

@cuda.jit(device=True, inline=True)
def two_sqr(a):
    p = a * a
    e = cuda.fma(a, a, -p)
    return p, e

@cuda.jit(device=True, inline=True)
def dd_abs(a_hi, a_lo):
    return (-a_hi, -a_lo) if (a_hi < 0.0 or (a_hi == 0.0 and a_lo < 0.0)) else (a_hi, a_lo)

@cuda.jit(device=True, inline=True)
def dd_add(a_hi, a_lo, b_hi, b_lo):
    s, e = two_sum(a_hi, b_hi)
    return two_sum(s, e + a_lo + b_lo)

@cuda.jit(device=True, inline=True)
def dd_sub(a_hi, a_lo, b_hi, b_lo):
    s, e = two_sum(a_hi, -b_hi)
    return two_sum(s, e + a_lo - b_lo)

@cuda.jit(device=True, inline=True)
def dd_mul(a_hi, a_lo, b_hi, b_lo):
    p, e = two_prod(a_hi, b_hi)
    return two_sum(p, e + a_hi * b_lo + a_lo * b_hi)

@cuda.jit(device=True, inline=True)
def dd_mul_f64(a_hi, a_lo, b):
    p, e = two_prod(a_hi, b)
    return two_sum(p, e + a_lo * b)

@cuda.jit(device=True, inline=True)
def dd_sqr(a_hi, a_lo):
    p, e = two_sqr(a_hi)
    return two_sum(p, e + 2.0 * a_hi * a_lo)

@cuda.jit(device=True, inline=True)
def c_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br

@cuda.jit(device=True, inline=True)
def pow2_f64(e):
    if e >= 1023:
        return 1.7976931348623157e+308
    elif e <= -1022:
        return 0.0
    return math.ldexp(1.0, int(e))

@cuda.jit(device=True, inline=True)
def check_glitch_cancellation_floatexp(rx, ry, dzx, dzy, eff_exp):
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

    scale_r = pow2_f64(-r_exp)
    scale_dz = pow2_f64(eff_exp - r_exp)

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

@cuda.jit(device=True, inline=True)
def eval_bbsa_4(dc_r, dc_i, a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i):
    t3_r, t3_i = c_mul(dc_r, dc_i, d_r, d_i)
    t2_r, t2_i = c_mul(dc_r, dc_i, c_r + t3_r, c_i + t3_i)
    t1_r, t1_i = c_mul(dc_r, dc_i, b_r + t2_r, b_i + t2_i)
    return c_mul(dc_r, dc_i, a_r + t1_r, a_i + t1_i)

@cuda.jit(device=True, inline=True)
def eval_bbsa_8(dc_r, dc_i, a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i,
                e_r, e_i, f_r, f_i, g_r, g_i, h_r, h_i):
    t7_r, t7_i = c_mul(dc_r, dc_i, h_r, h_i)
    t6_r, t6_i = c_mul(dc_r, dc_i, g_r + t7_r, g_i + t7_i)
    t5_r, t5_i = c_mul(dc_r, dc_i, f_r + t6_r, f_i + t6_i)
    t4_r, t4_i = c_mul(dc_r, dc_i, e_r + t5_r, e_i + t5_i)
    t3_r, t3_i = c_mul(dc_r, dc_i, d_r + t4_r, d_i + t4_i)
    t2_r, t2_i = c_mul(dc_r, dc_i, c_r + t3_r, c_i + t3_i)
    t1_r, t1_i = c_mul(dc_r, dc_i, b_r + t2_r, b_i + t2_i)
    return c_mul(dc_r, dc_i, a_r + t1_r, a_i + t1_i)

@cuda.jit(device=True, inline=True)
def sample_lut_lerp(lut, norm):
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

@cuda.jit(device=True, inline=True)
def compute_palette_norm(s, scheme_id, palette_offset=0.0, color_density=1.0, color_contrast=1.0):
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


@cuda.jit(device=True, inline=True)
def sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset=0.0, color_density=1.0, color_contrast=1.0):
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


# --- Sample Computation Device Functions ---
@cuda.jit(device=True, inline=True)
def compute_fp64_sample(x, y, max_iter, fractal_type, julia_cx, julia_cy, gen_n, gen_kr, gen_ki):
    if fractal_type == 0:
        x_m_fourth = x - 0.25
        y2 = y * y
        q = x_m_fourth * x_m_fourth + y2
        if (q * (q + x_m_fourth) <= 0.25 * y2) or ((x + 1.0) * (x + 1.0) + y2 <= 0.0625):
            return -1.0

    if fractal_type in (0, 2):
        zx, zy = (x, y) if fractal_type == 2 else (0.0, 0.0)
        cx, cy = (julia_cx, julia_cy) if fractal_type == 2 else (x, y)
        for k in range(max_iter):
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)
            zy = 2.0 * zx * zy + cy
            zx = zx2 - zy2 + cx
        return -1.0

    elif fractal_type in (1, 3):
        zx, zy = (x, y) if fractal_type == 3 else (0.0, 0.0)
        cx, cy = (julia_cx, julia_cy) if fractal_type == 3 else (x, y)
        for k in range(max_iter):
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)
            zy = 2.0 * math.fabs(zx) * math.fabs(zy) - cy
            zx = zx2 - zy2 + cx
        return -1.0

    elif fractal_type == 4:
        zx, zy = 0.0, 0.0
        cx, cy = x, y
        inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else INV_LN2_F32
        for k in range(max_iter):
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * inv_ln_n) if log_mag > 1.0 else np.float32(k)
            zn_r, zn_i = zx, zy
            for _ in range(gen_n - 1):
                zn_r, zn_i = zn_r * zx - zn_i * zy, zn_r * zy + zn_i * zx
            kz_r = gen_kr * zx - gen_ki * zy
            kz_i = gen_kr * zy + gen_ki * zx
            zx = zn_r + kz_r + cx
            zy = zn_i + kz_i + cy
        return -1.0

    return -1.0


@cuda.jit(device=True, inline=True)
def compute_dd_sample(cx_hi, cx_lo, cy_hi, cy_lo, max_iter, fractal_type,
                      julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo, gen_n, gen_kr, gen_ki):
    if fractal_type == 0:
        x_m_fourth = cx_hi - 0.25
        y2 = cy_hi * cy_hi
        q = x_m_fourth * x_m_fourth + y2
        if (q * (q + x_m_fourth) <= 0.25 * y2) or ((cx_hi + 1.0) * (cx_hi + 1.0) + y2 <= 0.0625):
            return -1.0

    if fractal_type in (0, 2):
        if fractal_type == 2:
            zx_hi, zx_lo, zy_hi, zy_lo = cx_hi, cx_lo, cy_hi, cy_lo
            c_xh, c_xl, c_yh, c_yl = julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo
        else:
            zx_hi, zx_lo, zy_hi, zy_lo = 0.0, 0.0, 0.0, 0.0
            c_xh, c_xl, c_yh, c_yl = cx_hi, cx_lo, cy_hi, cy_lo

        for k in range(max_iter):
            zx2_hi, zx2_lo = dd_sqr(zx_hi, zx_lo)
            zy2_hi, zy2_lo = dd_sqr(zy_hi, zy_lo)
            mag2_hi = zx2_hi + zy2_hi
            if mag2_hi > 65536.0:
                log_mag = math.log(mag2_hi)
                return np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)

            sub_h, sub_l = dd_sub(zx2_hi, zx2_lo, zy2_hi, zy2_lo)
            zxy_hi, zxy_lo = dd_mul(zx_hi, zx_lo, zy_hi, zy_lo)
            zx_hi, zx_lo = dd_add(sub_h, sub_l, c_xh, c_xl)
            zy_hi, zy_lo = dd_add(zxy_hi * 2.0, zxy_lo * 2.0, c_yh, c_yl)
        return -1.0

    elif fractal_type in (1, 3):
        if fractal_type == 3:
            zx_hi, zx_lo, zy_hi, zy_lo = cx_hi, cx_lo, cy_hi, cy_lo
            c_xh, c_xl, c_yh, c_yl = julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo
        else:
            zx_hi, zx_lo, zy_hi, zy_lo = 0.0, 0.0, 0.0, 0.0
            c_xh, c_xl, c_yh, c_yl = cx_hi, cx_lo, cy_hi, cy_lo

        for k in range(max_iter):
            zx2_hi, zx2_lo = dd_sqr(zx_hi, zx_lo)
            zy2_hi, zy2_lo = dd_sqr(zy_hi, zy_lo)
            mag2_hi = zx2_hi + zy2_hi
            if mag2_hi > 65536.0:
                log_mag = math.log(mag2_hi)
                return np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)

            sub_h, sub_l = dd_sub(zx2_hi, zx2_lo, zy2_hi, zy2_lo)
            ax_hi, ax_lo = dd_abs(zx_hi, zx_lo)
            ay_hi, ay_lo = dd_abs(zy_hi, zy_lo)
            zxy_hi, zxy_lo = dd_mul(ax_hi, ax_lo, ay_hi, ay_lo)
            zx_hi, zx_lo = dd_add(sub_h, sub_l, c_xh, c_xl)
            zy_hi, zy_lo = dd_sub(zxy_hi * 2.0, zxy_lo * 2.0, c_yh, c_yl)
        return -1.0

    elif fractal_type == 4:
        zx_hi, zx_lo, zy_hi, zy_lo = 0.0, 0.0, 0.0, 0.0
        c_xh, c_xl, c_yh, c_yl = cx_hi, cx_lo, cy_hi, cy_lo
        inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else INV_LN2_F32

        for k in range(max_iter):
            zx2_hi, zx2_lo = dd_sqr(zx_hi, zx_lo)
            zy2_hi, zy2_lo = dd_sqr(zy_hi, zy_lo)
            mag2_hi = zx2_hi + zy2_hi
            if mag2_hi > 65536.0:
                log_mag = math.log(mag2_hi)
                return np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * inv_ln_n) if log_mag > 1.0 else np.float32(k)

            cur_r_h, cur_r_l = zx_hi, zx_lo
            cur_i_h, cur_i_l = zy_hi, zy_lo
            for _ in range(gen_n - 1):
                r1_h, r1_l = dd_mul(cur_r_h, cur_r_l, zx_hi, zx_lo)
                r2_h, r2_l = dd_mul(cur_i_h, cur_i_l, zy_hi, zy_lo)
                re_h, re_l = dd_sub(r1_h, r1_l, r2_h, r2_l)

                i1_h, i1_l = dd_mul(cur_r_h, cur_r_l, zy_hi, zy_lo)
                i2_h, i2_l = dd_mul(cur_i_h, cur_i_l, zx_hi, zx_lo)
                im_h, im_l = dd_add(i1_h, i1_l, i2_h, i2_l)

                cur_r_h, cur_r_l = re_h, re_l
                cur_i_h, cur_i_l = im_h, im_l

            kr_zx_h, kr_zx_l = dd_mul_f64(zx_hi, zx_lo, gen_kr)
            ki_zy_h, ki_zy_l = dd_mul_f64(zy_hi, zy_lo, gen_ki)
            kz_r_h, kz_r_l = dd_sub(kr_zx_h, kr_zx_l, ki_zy_h, ki_zy_l)

            kr_zy_h, kr_zy_l = dd_mul_f64(zy_hi, zy_lo, gen_kr)
            ki_zx_h, ki_zx_l = dd_mul_f64(zx_hi, zx_lo, gen_ki)
            kz_i_h, kz_i_l = dd_add(kr_zy_h, kr_zy_l, ki_zx_h, ki_zx_l)

            tot_r_h, tot_r_l = dd_add(cur_r_h, cur_r_l, kz_r_h, kz_r_l)
            tot_i_h, tot_i_l = dd_add(cur_i_h, cur_i_l, kz_i_h, kz_i_l)

            zx_hi, zx_lo = dd_add(tot_r_h, tot_r_l, c_xh, c_xl)
            zy_hi, zy_lo = dd_add(tot_i_h, tot_i_l, c_yh, c_yl)
        return -1.0

    return -1.0


@cuda.jit(device=True, inline=True)
def compute_perturbation_sample(ref_re, ref_im, ref_len, max_iter,
                                ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub, dy_sub,
                                ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                                bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                                gen_n, gen_kr, gen_ki,
                                strict_glitch=0):
    is_julia = (fractal_type == 2 or fractal_type == 3)
    dcy = (ref_py_sub - float(sub_j)) * dy_sub
    dcx = (float(sub_i) - ref_px_sub) * dx_sub

    if bbsa_skip > 0 and fractal_type in (0, 2, 4):
        ux = dcx * bbsa_inv_r
        uy = dcy * bbsa_inv_r
        if ux * ux + uy * uy <= 1.0:
            if (bbsa_er != 0.0 or bbsa_ei != 0.0 or bbsa_fr != 0.0 or bbsa_fi != 0.0 or
                bbsa_gr != 0.0 or bbsa_gi != 0.0 or bbsa_hr != 0.0 or bbsa_hi != 0.0):
                dzx, dzy = eval_bbsa_8(ux, uy, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                       bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                       bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                       bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi)
            else:
                dzx, dzy = eval_bbsa_4(ux, uy, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
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

    # --- Specialized Perturbation Loop (Stage 1: Reference Orbit) ---
    if fractal_type in (0, 2):
        while k < ref_len:
            rx, ry = ref_re[k], ref_im[k]
            zx, zy = rx + dzx, ry + dzy
            mag2 = zx * zx + zy * zy
            if mag2 > 65536.0:
                if strict_glitch > 0 and k > 15 and (rx * rx + ry * ry) < 4.0:
                    is_glitch = 1
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch

            if strict_glitch > 0 and k > 15:
                r_mag2 = rx * rx + ry * ry
                if mag2 < 1e-6 * r_mag2 and r_mag2 > 1e-8:
                    is_glitch = 1

            dzx_new = cuda.fma(2.0 * rx + dzx, dzx, cuda.fma(-(2.0 * ry + dzy), dzy, dc_param_x))
            dzy = cuda.fma(2.0, cuda.fma(rx + dzx, dzy, ry * dzx), dc_param_y)
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
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch

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
        inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else INV_LN2_F32
        while k < ref_len:
            rx, ry = ref_re[k], ref_im[k]
            zx, zy = rx + dzx, ry + dzy
            mag2 = zx * zx + zy * zy
            if mag2 > 65536.0:
                if strict_glitch > 0 and k > 15 and (rx * rx + ry * ry) < 4.0:
                    is_glitch = 1
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * inv_ln_n) if log_mag > 1.0 else np.float32(k)), is_glitch

            if strict_glitch > 0 and k > 15:
                r_mag2 = rx * rx + ry * ry
                if mag2 < 1e-6 * r_mag2 and r_mag2 > 1e-8:
                    is_glitch = 1

            p_r, p_i = dzx, dzy
            q_r, q_i = rx, ry
            for _ in range(gen_n - 1):
                t1_r, t1_i = c_mul(p_r, p_i, rx, ry)
                pq_r, pq_i = p_r + q_r, p_i + q_i
                t2_r, t2_i = c_mul(pq_r, pq_i, dzx, dzy)
                p_r = t1_r + t2_r
                p_i = t1_i + t2_i
                q_r, q_i = c_mul(q_r, q_i, rx, ry)

            k_dz_r, k_dz_i = c_mul(gen_kr, gen_ki, dzx, dzy)
            dzx = p_r + k_dz_r + dc_param_x
            dzy = p_i + k_dz_i + dc_param_y
            k += 1

    # --- Specialized Fallback Loop (Stage 2: Escaped Reference) ---
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
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch
            zy = 2.0 * zx * zy + cy
            zx = zx2 - zy2 + cx
            k += 1
        return -1.0, is_glitch

    elif fractal_type in (1, 3):
        while k < max_iter:
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch
            zy = 2.0 * math.fabs(zx) * math.fabs(zy) - cy
            zx = zx2 - zy2 + cx
            k += 1
        return -1.0, is_glitch

    elif fractal_type == 4:
        inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else INV_LN2_F32
        while k < max_iter:
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * inv_ln_n) if log_mag > 1.0 else np.float32(k)), is_glitch
            zn_r, zn_i = zx, zy
            for _ in range(gen_n - 1):
                zn_r, zn_i = zn_r * zx - zn_i * zy, zn_r * zy + zn_i * zx
            kz_r = gen_kr * zx - gen_ki * zy
            kz_i = gen_kr * zy + gen_ki * zx
            zx = zn_r + kz_r + cx
            zy = zn_i + kz_i + cy
            k += 1
        return -1.0, is_glitch

    return -1.0, is_glitch


@cuda.jit(device=True, inline=True)
def compute_perturbation_sample_bla(ref_re, ref_im, ref_len, max_iter,
                                    ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub, dy_sub,
                                    ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                                    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                                    gen_n, gen_kr, gen_ki,
                                    strict_glitch, bla_table):
    is_julia = (fractal_type == 2 or fractal_type == 3)
    dcy = (ref_py_sub - float(sub_j)) * dy_sub
    dcx = (float(sub_i) - ref_px_sub) * dx_sub

    if bbsa_skip > 0 and fractal_type in (0, 2, 4):
        ux = dcx * bbsa_inv_r
        uy = dcy * bbsa_inv_r
        if ux * ux + uy * uy <= 1.0:
            if (bbsa_er != 0.0 or bbsa_ei != 0.0 or bbsa_fr != 0.0 or bbsa_fi != 0.0 or
                bbsa_gr != 0.0 or bbsa_gi != 0.0 or bbsa_hr != 0.0 or bbsa_hi != 0.0):
                dzx, dzy = eval_bbsa_8(ux, uy, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                       bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                       bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                       bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi)
            else:
                dzx, dzy = eval_bbsa_4(ux, uy, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
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
    bla_levels = bla_table.shape[0]

    # --- Specialized Perturbation Loop (Stage 1: Reference Orbit) ---
    if fractal_type in (0, 2):
        while k < ref_len:
            rx, ry = ref_re[k], ref_im[k]
            zx, zy = rx + dzx, ry + dzy
            mag2 = zx * zx + zy * zy
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch

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
                    new_dzx = cuda.fma(a_r, dzx, cuda.fma(-a_i, dzy, cuda.fma(b_r, dc_param_x, -b_i * dc_param_y)))
                    dzy = cuda.fma(a_r, dzy, cuda.fma(a_i, dzx, cuda.fma(b_r, dc_param_y, b_i * dc_param_x)))
                    dzx = new_dzx
                    if math.isnan(dzx) or math.isnan(dzy) or math.isinf(dzx) or math.isinf(dzy):
                        return np.float32(k), 1
                    k += step_d
                    continue

            dzx_new = cuda.fma(2.0 * rx + dzx, dzx, cuda.fma(-(2.0 * ry + dzy), dzy, dc_param_x))
            dzy = cuda.fma(2.0, cuda.fma(rx + dzx, dzy, ry * dzx), dc_param_y)
            dzx = dzx_new
            k += 1

    elif fractal_type in (1, 3):
        while k < ref_len:
            rx, ry = ref_re[k], ref_im[k]
            zx, zy = rx + dzx, ry + dzy
            mag2 = zx * zx + zy * zy
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch

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
        inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else INV_LN2_F32
        while k < ref_len:
            rx, ry = ref_re[k], ref_im[k]
            zx, zy = rx + dzx, ry + dzy
            mag2 = zx * zx + zy * zy
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * inv_ln_n) if log_mag > 1.0 else np.float32(k)), is_glitch

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
                    new_dzx = cuda.fma(a_r, dzx, cuda.fma(-a_i, dzy, cuda.fma(b_r, dc_param_x, -b_i * dc_param_y)))
                    dzy = cuda.fma(a_r, dzy, cuda.fma(a_i, dzx, cuda.fma(b_r, dc_param_y, b_i * dc_param_x)))
                    dzx = new_dzx
                    if math.isnan(dzx) or math.isnan(dzy) or math.isinf(dzx) or math.isinf(dzy):
                        return np.float32(k), 1
                    k += step_d
                    continue

            p_r, p_i = dzx, dzy
            q_r, q_i = rx, ry
            for _ in range(gen_n - 1):
                t1_r, t1_i = c_mul(p_r, p_i, rx, ry)
                pq_r, pq_i = p_r + q_r, p_i + q_i
                t2_r, t2_i = c_mul(pq_r, pq_i, dzx, dzy)
                p_r = t1_r + t2_r
                p_i = t1_i + t2_i
                q_r, q_i = c_mul(q_r, q_i, rx, ry)

            k_dz_r, k_dz_i = c_mul(gen_kr, gen_ki, dzx, dzy)
            dzx = p_r + k_dz_r + dc_param_x
            dzy = p_i + k_dz_i + dc_param_y
            k += 1

    # --- Specialized Fallback Loop (Stage 2: Escaped Reference) ---
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
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch
            zy = 2.0 * zx * zy + cy
            zx = zx2 - zy2 + cx
            k += 1
        return -1.0, is_glitch

    elif fractal_type in (1, 3):
        while k < max_iter:
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if math.isnan(mag2) or math.isinf(mag2):
                return np.float32(k), 1
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch
            zy = 2.0 * math.fabs(zx) * math.fabs(zy) - cy
            zx = zx2 - zy2 + cx
            k += 1
        return -1.0, is_glitch

    elif fractal_type == 4:
        inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else INV_LN2_F32
        while k < max_iter:
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if math.isnan(mag2) or math.isinf(mag2):
                return np.float32(k), 1
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * inv_ln_n) if log_mag > 1.0 else np.float32(k)), is_glitch
            zn_r, zn_i = zx, zy
            for _ in range(gen_n - 1):
                zn_r, zn_i = zn_r * zx - zn_i * zy, zn_r * zy + zn_i * zx
            kz_r = gen_kr * zx - gen_ki * zy
            kz_i = gen_kr * zy + gen_ki * zx
            zx = zn_r + kz_r + cx
            zy = zn_i + kz_i + cy
            k += 1
        return -1.0, is_glitch

    return -1.0, is_glitch


@cuda.jit(device=True, inline=True)
def compute_perturbation_sample_floatexp(
    ref_re, ref_im, ref_len, max_iter,
    ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub_norm, dy_sub_norm,
    E_scale, ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
    gen_n, gen_kr, gen_ki, strict_glitch
):
    is_julia = (fractal_type == 2 or fractal_type == 3)
    dcy = (ref_py_sub - float(sub_j)) * dy_sub_norm
    dcx = (float(sub_i) - ref_px_sub) * dx_sub_norm

    if bbsa_skip > 0 and fractal_type in (0, 2, 4):
        ux = dcx * bbsa_inv_r
        uy = dcy * bbsa_inv_r
        if ux * ux + uy * uy <= 1.0:
            if (bbsa_hr != 0.0 or bbsa_hi != 0.0 or bbsa_gr != 0.0 or bbsa_gi != 0.0 or bbsa_fr != 0.0 or bbsa_fi != 0.0):
                dzx, dzy = bbsa_hr, bbsa_hi
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_gr; dzy += bbsa_gi
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_fr; dzy += bbsa_fi
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_er; dzy += bbsa_ei
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_dr; dzy += bbsa_di
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_cr; dzy += bbsa_ci
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_ar; dzy += bbsa_ai
                dzx, dzy = c_mul(dzx, dzy, ux, uy)
            else:
                dzx, dzy = bbsa_er, bbsa_ei
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_dr; dzy += bbsa_di
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_cr; dzy += bbsa_ci
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_ar; dzy += bbsa_ai
                dzx, dzy = c_mul(dzx, dzy, ux, uy)

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

    if fractal_type in (0, 2):
        while k < ref_len:
            rx = ref_re[k]
            ry = ref_im[k]

            eff_exp = z_exp - E_scale
            if eff_exp >= -50:
                scale_esc = pow2_f64(eff_exp)
                zx = rx + dzx * scale_esc
                zy = ry + dzy * scale_esc
                mag2 = zx * zx + zy * zy
                if mag2 > 65536.0:
                    log_mag = math.log(mag2)
                    return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch

            if strict_glitch > 0 and k > 15:
                if check_glitch_cancellation_floatexp(rx, ry, dzx, dzy, eff_exp):
                    is_glitch = 1

            lx = 2.0 * (rx * dzx - ry * dzy)
            ly = 2.0 * (rx * dzy + ry * dzx)

            shift_nl = z_exp - E_scale
            if shift_nl >= -1020:
                scale_nl = pow2_f64(shift_nl)
                lx += (dzx * dzx - dzy * dzy) * scale_nl
                ly += (2.0 * dzx * dzy) * scale_nl

            scale_c = pow2_f64(-z_exp) if (-1020 <= z_exp <= 1020) else 0.0
            dzx = lx + dc_param_x * scale_c
            dzy = ly + dc_param_y * scale_c

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

    elif fractal_type in (1, 3):
        while k < ref_len:
            rx = ref_re[k]
            ry = ref_im[k]

            eff_exp = z_exp - E_scale
            if eff_exp >= -50:
                scale_esc = pow2_f64(eff_exp)
                zx = rx + dzx * scale_esc
                zy = ry + dzy * scale_esc
                mag2 = zx * zx + zy * zy
                if mag2 > 65536.0:
                    log_mag = math.log(mag2)
                    return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch

            if strict_glitch > 0 and k > 15:
                if check_glitch_cancellation_floatexp(rx, ry, dzx, dzy, eff_exp):
                    is_glitch = 1

            eff_exp_delta = z_exp - E_scale
            scale_delta = pow2_f64(eff_exp_delta) if eff_exp_delta >= -60 else 0.0
            zx = rx + dzx * scale_delta
            zy = ry + dzy * scale_delta

            sx_sign = 1.0 if rx >= 0.0 else -1.0
            sy_sign = 1.0 if ry >= 0.0 else -1.0
            delta_x = sx_sign * dzx if (rx + dzx * scale_delta) * sx_sign >= 0.0 else -sx_sign * (2.0 * rx * pow2_f64(-eff_exp_delta) + dzx)
            delta_y = sy_sign * dzy if (ry + dzy * scale_delta) * sy_sign >= 0.0 else -sy_sign * (2.0 * ry * pow2_f64(-eff_exp_delta) + dzy)

            lx = 2.0 * (rx * dzx - ry * dzy)
            shift_nl = z_exp - E_scale
            if shift_nl >= -1020:
                scale_nl = pow2_f64(shift_nl)
                lx += (dzx * dzx - dzy * dzy) * scale_nl

            ly = 2.0 * (math.fabs(rx) * delta_y + math.fabs(ry) * delta_x)
            if shift_nl >= -1020:
                ly += (2.0 * delta_x * delta_y) * scale_nl

            scale_c = pow2_f64(-z_exp) if (-1020 <= z_exp <= 1020) else 0.0
            dzx = lx + dc_param_x * scale_c
            dzy = ly - dc_param_y * scale_c

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
        inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else INV_LN2_F32
        while k < ref_len:
            rx = ref_re[k]
            ry = ref_im[k]

            eff_exp = z_exp - E_scale
            if eff_exp >= -50:
                scale_esc = pow2_f64(eff_exp)
                zx = rx + dzx * scale_esc
                zy = ry + dzy * scale_esc
                mag2 = zx * zx + zy * zy
                if mag2 > 65536.0:
                    log_mag = math.log(mag2)
                    return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * inv_ln_n) if log_mag > 1.0 else np.float32(k)), is_glitch

            if strict_glitch > 0 and k > 15:
                if check_glitch_cancellation_floatexp(rx, ry, dzx, dzy, eff_exp):
                    is_glitch = 1

            p_r, p_i = dzx, dzy
            q_r, q_i = rx, ry
            shift_nl = z_exp - E_scale
            scale_nl = pow2_f64(shift_nl) if shift_nl >= -1020 else 0.0

            for _ in range(gen_n - 1):
                t1_r, t1_i = c_mul(p_r, p_i, rx, ry)
                t2_r, t2_i = c_mul(q_r, q_i, dzx, dzy)
                t3_r, t3_i = c_mul(p_r, p_i, dzx, dzy)
                p_r = t1_r + t2_r + t3_r * scale_nl
                p_i = t1_i + t2_i + t3_i * scale_nl
                q_r, q_i = c_mul(q_r, q_i, rx, ry)

            k_dz_r, k_dz_i = c_mul(gen_kr, gen_ki, dzx, dzy)
            scale_c = pow2_f64(-z_exp) if (-1020 <= z_exp <= 1020) else 0.0

            dzx = p_r + k_dz_r + dc_param_x * scale_c
            dzy = p_i + k_dz_i + dc_param_y * scale_c

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

    # --- Stage 2 Fallback if reference orbit escaped before pixel ---
    eff_exp = z_exp - E_scale
    scale_fb = pow2_f64(eff_exp) if eff_exp >= -60 else 0.0
    zx = ref_re[ref_len] + dzx * scale_fb
    zy = ref_im[ref_len] + dzy * scale_fb
    scale_c_fb = pow2_f64(-E_scale) if E_scale <= 1020 else 0.0
    cx = julia_cx if is_julia else (ref_cx + dc_param_x * scale_c_fb)
    cy = julia_cy if is_julia else (ref_cy + dc_param_y * scale_c_fb)

    # Check if pixel bailed out cleanly alongside reference orbit at k = ref_len
    mag2 = zx * zx + zy * zy
    if mag2 > 65536.0:
        log_mag = math.log(mag2)
        inv_ln = (np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else INV_LN2_F32) if fractal_type == 4 else INV_LN2_F32
        return (np.float32(ref_len + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * inv_ln) if log_mag > 1.0 else np.float32(ref_len)), is_glitch

    if ref_len < max_iter:
        is_glitch = 1

    if fractal_type in (0, 2):
        while k < max_iter:
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch
            zy = 2.0 * zx * zy + cy
            zx = zx2 - zy2 + cx
            k += 1
        return -1.0, is_glitch

    elif fractal_type in (1, 3):
        while k < max_iter:
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch
            zy = 2.0 * math.fabs(zx) * math.fabs(zy) - cy
            zx = zx2 - zy2 + cx
            k += 1
        return -1.0, is_glitch

    elif fractal_type == 4:
        inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else INV_LN2_F32
        while k < max_iter:
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * inv_ln_n) if log_mag > 1.0 else np.float32(k)), is_glitch
            zn_r, zn_i = zx, zy
            for _ in range(gen_n - 1):
                zn_r, zn_i = zn_r * zx - zn_i * zy, zn_r * zy + zn_i * zx
            kz_r = gen_kr * zx - gen_ki * zy
            kz_i = gen_kr * zy + gen_ki * zx
            zx = zn_r + kz_r + cx
            zy = zn_i + kz_i + cy
            k += 1
        return -1.0, is_glitch

    return -1.0, is_glitch


@cuda.jit(device=True, inline=True)
def compute_perturbation_sample_floatexp_bla(
    ref_re, ref_im, ref_len, max_iter,
    ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub_norm, dy_sub_norm,
    E_scale, ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
    gen_n, gen_kr, gen_ki, strict_glitch,
    bla_table
):
    is_julia = (fractal_type == 2 or fractal_type == 3)
    dcy = (ref_py_sub - float(sub_j)) * dy_sub_norm
    dcx = (float(sub_i) - ref_px_sub) * dx_sub_norm

    if bbsa_skip > 0 and fractal_type in (0, 2, 4):
        ux = dcx * bbsa_inv_r
        uy = dcy * bbsa_inv_r
        if ux * ux + uy * uy <= 1.0:
            if (bbsa_hr != 0.0 or bbsa_hi != 0.0 or bbsa_gr != 0.0 or bbsa_gi != 0.0 or bbsa_fr != 0.0 or bbsa_fi != 0.0):
                dzx, dzy = bbsa_hr, bbsa_hi
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_gr; dzy += bbsa_gi
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_fr; dzy += bbsa_fi
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_er; dzy += bbsa_ei
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_dr; dzy += bbsa_di
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_cr; dzy += bbsa_ci
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_ar; dzy += bbsa_ai
                dzx, dzy = c_mul(dzx, dzy, ux, uy)
            else:
                dzx, dzy = bbsa_er, bbsa_ei
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_dr; dzy += bbsa_di
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_cr; dzy += bbsa_ci
                dzx, dzy = c_mul(dzx, dzy, ux, uy); dzx += bbsa_ar; dzy += bbsa_ai
                dzx, dzy = c_mul(dzx, dzy, ux, uy)

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
    bla_levels = bla_table.shape[0]

    if fractal_type in (0, 2):
        while k < ref_len:
            rx = ref_re[k]
            ry = ref_im[k]

            eff_exp = z_exp - E_scale
            if eff_exp >= -50:
                scale_esc = pow2_f64(eff_exp)
                zx = rx + dzx * scale_esc
                zy = ry + dzy * scale_esc
                mag2 = zx * zx + zy * zy
                if mag2 > 65536.0:
                    log_mag = math.log(mag2)
                    return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch

            if strict_glitch > 0 and k > 15:
                if check_glitch_cancellation_floatexp(rx, ry, dzx, dzy, eff_exp):
                    is_glitch = 1

            if has_bla and k < ref_len - 1:
                shift_r = z_exp - E_scale
                if shift_r >= -510:
                    sc_r = pow2_f64(shift_r)
                    dz_mag2 = (dzx * dzx + dzy * dzy) * (sc_r * sc_r)
                    can_step_0 = (dz_mag2 < bla_table[0, k, 4])
                else:
                    dz_mag = math.sqrt(dzx * dzx + dzy * dzy)
                    r0_sq = bla_table[0, k, 4]
                    can_step_0 = (r0_sq > 0.0) and (dz_mag * pow2_f64(max(-1020, shift_r)) < math.sqrt(r0_sq))

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
                                can_step_d = (rd_sq > 0.0) and (dz_mag * pow2_f64(max(-1020, shift_r)) < math.sqrt(rd_sq))
                            if can_step_d:
                                best_d = d
                                break
                    step_d = 1 << best_d
                    a_r = bla_table[best_d, k, 0]
                    a_i = bla_table[best_d, k, 1]
                    b_r = bla_table[best_d, k, 2]
                    b_i = bla_table[best_d, k, 3]
                    scale_c = pow2_f64(-z_exp) if (-1020 <= z_exp <= 1020) else 0.0
                    dc_x_scaled = dc_param_x * scale_c
                    dc_y_scaled = dc_param_y * scale_c
                    new_dzx = cuda.fma(a_r, dzx, cuda.fma(-a_i, dzy, cuda.fma(b_r, dc_x_scaled, -b_i * dc_y_scaled)))
                    dzy = cuda.fma(a_r, dzy, cuda.fma(a_i, dzx, cuda.fma(b_r, dc_y_scaled, b_i * dc_x_scaled)))
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
                scale_nl = pow2_f64(shift_nl)
                lx += (dzx * dzx - dzy * dzy) * scale_nl
                ly += (2.0 * dzx * dzy) * scale_nl

            # 4. Add scaled dc: dc * 2^-z_exp
            scale_c = pow2_f64(-z_exp) if (-1020 <= z_exp <= 1020) else 0.0
            dzx = lx + dc_param_x * scale_c
            dzy = ly + dc_param_y * scale_c

            # 5. Base-2 Exponent Normalization
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

            eff_exp = z_exp - E_scale
            if eff_exp >= -50:
                scale_esc = pow2_f64(eff_exp)
                zx = rx + dzx * scale_esc
                zy = ry + dzy * scale_esc
                mag2 = zx * zx + zy * zy
                if mag2 > 65536.0:
                    log_mag = math.log(mag2)
                    return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch

            if strict_glitch > 0 and k > 15:
                if check_glitch_cancellation_floatexp(rx, ry, dzx, dzy, eff_exp):
                    is_glitch = 1

            eff_exp_delta = z_exp - E_scale
            scale_delta = pow2_f64(eff_exp_delta) if eff_exp_delta >= -60 else 0.0
            zx = rx + dzx * scale_delta
            zy = ry + dzy * scale_delta

            sx_sign = 1.0 if rx >= 0.0 else -1.0
            sy_sign = 1.0 if ry >= 0.0 else -1.0
            delta_x = sx_sign * dzx if (rx + dzx * scale_delta) * sx_sign >= 0.0 else -sx_sign * (2.0 * rx * pow2_f64(-eff_exp_delta) + dzx)
            delta_y = sy_sign * dzy if (ry + dzy * scale_delta) * sy_sign >= 0.0 else -sy_sign * (2.0 * ry * pow2_f64(-eff_exp_delta) + dzy)

            lx = 2.0 * (rx * dzx - ry * dzy)
            shift_nl = z_exp - E_scale
            if shift_nl >= -1020:
                scale_nl = pow2_f64(shift_nl)
                lx += (dzx * dzx - dzy * dzy) * scale_nl

            ly = 2.0 * (math.fabs(rx) * delta_y + math.fabs(ry) * delta_x)
            if shift_nl >= -1020:
                ly += (2.0 * delta_x * delta_y) * scale_nl

            scale_c = pow2_f64(-z_exp) if (-1020 <= z_exp <= 1020) else 0.0
            dzx = lx + dc_param_x * scale_c
            dzy = ly - dc_param_y * scale_c

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
        inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else INV_LN2_F32
        while k < ref_len:
            rx = ref_re[k]
            ry = ref_im[k]

            eff_exp = z_exp - E_scale
            if eff_exp >= -50:
                scale_esc = pow2_f64(eff_exp)
                zx = rx + dzx * scale_esc
                zy = ry + dzy * scale_esc
                mag2 = zx * zx + zy * zy
                if mag2 > 65536.0:
                    log_mag = math.log(mag2)
                    return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * inv_ln_n) if log_mag > 1.0 else np.float32(k)), is_glitch

            if strict_glitch > 0 and k > 15:
                if check_glitch_cancellation_floatexp(rx, ry, dzx, dzy, eff_exp):
                    is_glitch = 1

            if has_bla and k < ref_len - 1:
                shift_r = z_exp - E_scale
                if shift_r >= -510:
                    sc_r = pow2_f64(shift_r)
                    dz_mag2 = (dzx * dzx + dzy * dzy) * (sc_r * sc_r)
                    can_step_0 = (dz_mag2 < bla_table[0, k, 4])
                else:
                    dz_mag = math.sqrt(dzx * dzx + dzy * dzy)
                    r0_sq = bla_table[0, k, 4]
                    can_step_0 = (r0_sq > 0.0) and (dz_mag * pow2_f64(max(-1020, shift_r)) < math.sqrt(r0_sq))

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
                                can_step_d = (rd_sq > 0.0) and (dz_mag * pow2_f64(max(-1020, shift_r)) < math.sqrt(rd_sq))
                            if can_step_d:
                                best_d = d
                                break
                    step_d = 1 << best_d
                    a_r = bla_table[best_d, k, 0]
                    a_i = bla_table[best_d, k, 1]
                    b_r = bla_table[best_d, k, 2]
                    b_i = bla_table[best_d, k, 3]
                    scale_c = pow2_f64(-z_exp) if (-1020 <= z_exp <= 1020) else 0.0
                    dc_x_scaled = dc_param_x * scale_c
                    dc_y_scaled = dc_param_y * scale_c
                    new_dzx = cuda.fma(a_r, dzx, cuda.fma(-a_i, dzy, cuda.fma(b_r, dc_x_scaled, -b_i * dc_y_scaled)))
                    dzy = cuda.fma(a_r, dzy, cuda.fma(a_i, dzx, cuda.fma(b_r, dc_y_scaled, b_i * dc_x_scaled)))
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
            scale_nl = pow2_f64(shift_nl) if shift_nl >= -1020 else 0.0

            for _ in range(gen_n - 1):
                t1_r, t1_i = c_mul(p_r, p_i, rx, ry)
                t2_r, t2_i = c_mul(q_r, q_i, dzx, dzy)
                t3_r, t3_i = c_mul(p_r, p_i, dzx, dzy)
                p_r = t1_r + t2_r + t3_r * scale_nl
                p_i = t1_i + t2_i + t3_i * scale_nl
                q_r, q_i = c_mul(q_r, q_i, rx, ry)

            k_dz_r, k_dz_i = c_mul(gen_kr, gen_ki, dzx, dzy)
            scale_c = pow2_f64(-z_exp) if (-1020 <= z_exp <= 1020) else 0.0

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

    # --- Stage 2 Fallback if reference orbit escaped before pixel ---
    eff_exp = z_exp - E_scale
    scale_fb = pow2_f64(eff_exp) if eff_exp >= -60 else 0.0
    zx = ref_re[ref_len] + dzx * scale_fb
    zy = ref_im[ref_len] + dzy * scale_fb
    scale_c_fb = pow2_f64(-E_scale) if E_scale <= 1020 else 0.0
    cx = julia_cx if is_julia else (ref_cx + dc_param_x * scale_c_fb)
    cy = julia_cy if is_julia else (ref_cy + dc_param_y * scale_c_fb)

    # Check if pixel bailed out cleanly alongside reference orbit at k = ref_len
    mag2 = zx * zx + zy * zy
    if mag2 > 65536.0:
        log_mag = math.log(mag2)
        inv_ln = (np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else INV_LN2_F32) if fractal_type == 4 else INV_LN2_F32
        return (np.float32(ref_len + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * inv_ln) if log_mag > 1.0 else np.float32(ref_len)), is_glitch

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
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch
            zy = 2.0 * zx * zy + cy
            zx = zx2 - zy2 + cx
            k += 1
        return -1.0, is_glitch

    elif fractal_type in (1, 3):
        while k < max_iter:
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if math.isnan(mag2) or math.isinf(mag2):
                return np.float32(k), 1
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)), is_glitch
            zy = 2.0 * math.fabs(zx) * math.fabs(zy) - cy
            zx = zx2 - zy2 + cx
            k += 1
        return -1.0, is_glitch

    elif fractal_type == 4:
        inv_ln_n = np.float32(1.0 / math.log(float(gen_n))) if gen_n > 1 else INV_LN2_F32
        while k < max_iter:
            zx2, zy2 = zx * zx, zy * zy
            mag2 = zx2 + zy2
            if math.isnan(mag2) or math.isinf(mag2):
                return np.float32(k), 1
            if mag2 > 65536.0:
                log_mag = math.log(mag2)
                return (np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * inv_ln_n) if log_mag > 1.0 else np.float32(k)), is_glitch
            zn_r, zn_i = zx, zy
            for _ in range(gen_n - 1):
                zn_r, zn_i = zn_r * zx - zn_i * zy, zn_r * zy + zn_i * zx
            kz_r = gen_kr * zx - gen_ki * zy
            kz_i = gen_kr * zy + gen_ki * zx
            zx = zn_r + kz_r + cx
            zy = zn_i + kz_i + cy
            k += 1
        return -1.0, is_glitch

    return -1.0, is_glitch


# --- Adaptive Iteration Probe Kernels ---
@cuda.jit
def probe_iter_fp64_kernel(out_iters, target_w, target_h, max_iter, x_min, y_max, dx, dy,
                           fractal_type, julia_cx, julia_cy, gen_n, gen_kr, gen_ki):
    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return
    x = x_min + i * dx
    y = y_max - j * dy
    out_iters[j, i] = compute_fp64_sample(x, y, max_iter, fractal_type, julia_cx, julia_cy, gen_n, gen_kr, gen_ki)

@cuda.jit
def probe_iter_dd_kernel(out_iters, target_w, target_h, max_iter,
                         x_min_hi, x_min_lo, dx_hi, dx_lo,
                         y_max_hi, y_max_lo, dy_hi, dy_lo,
                         fractal_type, julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo,
                         gen_n, gen_kr, gen_ki):
    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return
    jdy_hi, jdy_lo = dd_mul_f64(dy_hi, dy_lo, float(j))
    cy_hi, cy_lo = dd_sub(y_max_hi, y_max_lo, jdy_hi, jdy_lo)
    idx_hi, idx_lo = dd_mul_f64(dx_hi, dx_lo, float(i))
    cx_hi, cx_lo = dd_add(x_min_hi, x_min_lo, idx_hi, idx_lo)
    out_iters[j, i] = compute_dd_sample(cx_hi, cx_lo, cy_hi, cy_lo, max_iter, fractal_type,
                                        julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo,
                                        gen_n, gen_kr, gen_ki)

@cuda.jit
def probe_iter_perturbation_kernel(out_iters, target_w, target_h, max_iter,
                                   ref_re, ref_im, ref_len,
                                   ref_px, ref_py, dx, dy,
                                   ref_cx, ref_cy, fractal_type,
                                   julia_cx, julia_cy,
                                   bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                   bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                   bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                   bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                                   gen_n, gen_kr, gen_ki):
    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return
    s, _ = compute_perturbation_sample(
        ref_re, ref_im, ref_len, max_iter,
        ref_px, ref_py, i, j, dx, dy,
        ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
        bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
        bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
        bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
        bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
        gen_n, gen_kr, gen_ki, 0
    )
    out_iters[j, i] = s


@cuda.jit
def probe_iter_floatexp_kernel(
    out_iters, target_w, target_h, max_iter,
    ref_re, ref_im, ref_len,
    ref_px, ref_py, dx_norm, dy_norm, E_scale,
    ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
    gen_n, gen_kr, gen_ki
):
    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return
    s, _ = compute_perturbation_sample_floatexp(
        ref_re, ref_im, ref_len, max_iter,
        ref_px, ref_py, float(i), float(j), dx_norm, dy_norm,
        E_scale, ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
        bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
        bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
        bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
        bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
        gen_n, gen_kr, gen_ki, 0
    )
    out_iters[j, i] = s



# --- Edge Detection Operator ---
@cuda.jit
def edge_detect_kernel(iter_map, edge_mask, target_w, target_h, threshold):
    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return

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
            if (s < 0.0) != (s_neighbor < 0.0) or (s >= 0.0 and math.fabs(s - s_neighbor) > threshold):
                is_edge = True
                break
        if is_edge:
            break

    edge_mask[j, i] = 1 if is_edge else 0


# --- Unified Multi-Pass SSAA GPU Kernels ---
@cuda.jit(fastmath=True)
def mandelbrot_unified_fp64_kernel(out_rgb, out_iter, edge_mask, lut, target_w, target_h, max_iter,
                                   x_min, y_max, dx_sub, dy_sub, factor, scheme_id, fractal_type,
                                   julia_cx, julia_cy, gen_n, gen_kr, gen_ki, mode, palette_offset=0.0,
                                   color_density=1.0, color_contrast=1.0):
    s_lut = cuda.shared.array(shape=(2048, 3), dtype=float32)
    stage_lut_to_shared(lut, s_lut)

    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return
    if mode == 2 and edge_mask[j, i] == 0:
        return

    sub_offset = (float(factor) - 1.0) * 0.5
    if mode == 0:
        y = y_max - (float(j * factor) + sub_offset) * dy_sub
        x = x_min + (float(i * factor) + sub_offset) * dx_sub
        s = compute_fp64_sample(x, y, max_iter, fractal_type, julia_cx, julia_cy, gen_n, gen_kr, gen_ki)
        out_iter[j, i] = s

        if s >= 0.0:
            norm = compute_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
            sr, sg, sb = sample_lut_lerp(s_lut, norm)
            out_rgb[j, i, 0] = int(sr + 0.5)
            out_rgb[j, i, 1] = int(sg + 0.5)
            out_rgb[j, i, 2] = int(sb + 0.5)
        else:
            out_rgb[j, i, 0] = 0
            out_rgb[j, i, 1] = 0
            out_rgb[j, i, 2] = 0
    else:
        r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
        num_samples = float(factor * factor)
        for sy in range(factor):
            y = y_max - (j * factor + sy) * dy_sub
            for sx in range(factor):
                x = x_min + (i * factor + sx) * dx_sub
                s = compute_fp64_sample(x, y, max_iter, fractal_type, julia_cx, julia_cy, gen_n, gen_kr, gen_ki)
                if s >= 0.0:
                    norm = compute_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
                    sr, sg, sb = sample_lut_lerp(s_lut, norm)
                    r_acc += sr
                    g_acc += sg
                    b_acc += sb

        inv_samples = 1.0 / num_samples
        out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
        out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
        out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)

@cuda.jit
def mandelbrot_unified_dd_kernel(out_rgb, out_iter, edge_mask, lut, target_w, target_h, max_iter,
                                 x_min_hi, x_min_lo, dx_hi, dx_lo,
                                 y_max_hi, y_max_lo, dy_hi, dy_lo,
                                 factor, scheme_id, fractal_type,
                                 julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo,
                                 gen_n, gen_kr, gen_ki, mode, palette_offset=0.0,
                                 color_density=1.0, color_contrast=1.0):
    s_lut = cuda.shared.array(shape=(2048, 3), dtype=float32)
    stage_lut_to_shared(lut, s_lut)

    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return
    if mode == 2 and edge_mask[j, i] == 0:
        return

    sub_offset = (float(factor) - 1.0) * 0.5
    if mode == 0:
        sub_j = float(j * factor) + sub_offset
        jdy_hi, jdy_lo = dd_mul_f64(dy_hi, dy_lo, sub_j)
        cy_hi, cy_lo = dd_sub(y_max_hi, y_max_lo, jdy_hi, jdy_lo)

        sub_i = float(i * factor) + sub_offset
        idx_hi, idx_lo = dd_mul_f64(dx_hi, dx_lo, sub_i)
        cx_hi, cx_lo = dd_add(x_min_hi, x_min_lo, idx_hi, idx_lo)

        s = compute_dd_sample(cx_hi, cx_lo, cy_hi, cy_lo, max_iter, fractal_type,
                              julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo,
                              gen_n, gen_kr, gen_ki)
        out_iter[j, i] = s

        if s >= 0.0:
            norm = compute_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
            sr, sg, sb = sample_lut_lerp(s_lut, norm)
            out_rgb[j, i, 0] = int(sr + 0.5)
            out_rgb[j, i, 1] = int(sg + 0.5)
            out_rgb[j, i, 2] = int(sb + 0.5)
        else:
            out_rgb[j, i, 0] = 0
            out_rgb[j, i, 1] = 0
            out_rgb[j, i, 2] = 0
    else:
        r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
        num_samples = float(factor * factor)
        for sy in range(factor):
            sub_j = j * factor + sy
            jdy_hi, jdy_lo = dd_mul_f64(dy_hi, dy_lo, float(sub_j))
            cy_hi, cy_lo = dd_sub(y_max_hi, y_max_lo, jdy_hi, jdy_lo)

            for sx in range(factor):
                sub_i = i * factor + sx
                idx_hi, idx_lo = dd_mul_f64(dx_hi, dx_lo, float(sub_i))
                cx_hi, cx_lo = dd_add(x_min_hi, x_min_lo, idx_hi, idx_lo)

                s = compute_dd_sample(cx_hi, cx_lo, cy_hi, cy_lo, max_iter, fractal_type,
                                      julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo,
                                      gen_n, gen_kr, gen_ki)
                if s >= 0.0:
                    norm = compute_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
                    sr, sg, sb = sample_lut_lerp(s_lut, norm)
                    r_acc += sr
                    g_acc += sg
                    b_acc += sb

        inv_samples = 1.0 / num_samples
        out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
        out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
        out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)

@cuda.jit(fastmath=True)
def mandelbrot_unified_perturbation_kernel(out_rgb, out_glitch, out_iter, edge_mask, lut, target_w, target_h, max_iter,
                                          ref_re, ref_im, ref_len,
                                          ref_px_sub, ref_py_sub, dx_sub, dy_sub,
                                          ref_cx, ref_cy, factor, scheme_id, fractal_type,
                                          julia_cx, julia_cy,
                                          bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                          bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                          bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                          bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                                          rebase_pass, in_glitch_mask, gen_n, gen_kr, gen_ki, mode, strict_glitch,
                                          palette_offset, color_density, color_contrast,
                                          bla_table):
    s_lut = cuda.shared.array(shape=(2048, 3), dtype=float32)
    stage_lut_to_shared(lut, s_lut)

    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return
    if rebase_pass > 0 and in_glitch_mask[j, i] == 0:
        return
    if mode == 2 and edge_mask[j, i] == 0:
        return

    sub_offset = (float(factor) - 1.0) * 0.5
    if mode == 0:
        sub_j = float(j * factor) + sub_offset
        sub_i = float(i * factor) + sub_offset

        s, is_gl = compute_perturbation_sample_bla(
            ref_re, ref_im, ref_len, max_iter,
            ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub, dy_sub,
            ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
            bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
            bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
            bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
            bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
            gen_n, gen_kr, gen_ki, strict_glitch,
            bla_table
        )

        out_iter[j, i] = s
        out_glitch[j, i] = 1 if is_gl > 0 else 0

        if s >= 0.0:
            norm = compute_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
            sr, sg, sb = sample_lut_lerp(s_lut, norm)
            out_rgb[j, i, 0] = int(sr + 0.5)
            out_rgb[j, i, 1] = int(sg + 0.5)
            out_rgb[j, i, 2] = int(sb + 0.5)
        else:
            out_rgb[j, i, 0] = 0
            out_rgb[j, i, 1] = 0
            out_rgb[j, i, 2] = 0
    else:
        r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
        num_samples = float(factor * factor)
        glitch_count = 0

        for sy in range(factor):
            sub_j = j * factor + sy
            for sx in range(factor):
                sub_i = i * factor + sx
                s, is_gl = compute_perturbation_sample_bla(
                    ref_re, ref_im, ref_len, max_iter,
                    ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub, dy_sub,
                    ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                    gen_n, gen_kr, gen_ki, strict_glitch,
                    bla_table
                )
                if is_gl > 0:
                    glitch_count += 1
                if s >= 0.0:
                    norm = compute_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
                    sr, sg, sb = sample_lut_lerp(s_lut, norm)
                    r_acc += sr
                    g_acc += sg
                    b_acc += sb

        inv_samples = 1.0 / num_samples
        out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
        out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
        out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)
        out_glitch[j, i] = 1 if glitch_count > 0 else 0
        out_iter[j, i] = s


@cuda.jit(fastmath=True)
def mandelbrot_unified_floatexp_kernel(
    out_rgb, out_glitch, out_iter, edge_mask, lut, target_w, target_h, max_iter,
    ref_re, ref_im, ref_len,
    ref_px_sub, ref_py_sub, dx_sub_norm, dy_sub_norm, E_scale,
    ref_cx, ref_cy, factor, scheme_id, fractal_type,
    julia_cx, julia_cy,
    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
    rebase_pass, in_glitch_mask, gen_n, gen_kr, gen_ki, mode, strict_glitch,
    palette_offset, color_density, color_contrast,
    bla_table
):
    s_lut = cuda.shared.array(shape=(2048, 3), dtype=float32)
    stage_lut_to_shared(lut, s_lut)

    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return
    if rebase_pass > 0 and in_glitch_mask[j, i] == 0:
        return
    if mode == 2 and edge_mask[j, i] == 0:
        return

    sub_offset = (float(factor) - 1.0) * 0.5
    if mode == 0:
        sub_j = float(j * factor) + sub_offset
        sub_i = float(i * factor) + sub_offset

        s, is_gl = compute_perturbation_sample_floatexp_bla(
            ref_re, ref_im, ref_len, max_iter,
            ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub_norm, dy_sub_norm,
            E_scale, ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
            bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
            bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
            bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
            bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
            gen_n, gen_kr, gen_ki, strict_glitch,
            bla_table
        )

        out_iter[j, i] = s
        out_glitch[j, i] = 1 if is_gl > 0 else 0

        if s >= 0.0:
            norm = compute_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
            sr, sg, sb = sample_lut_lerp(s_lut, norm)
            out_rgb[j, i, 0] = int(sr + 0.5)
            out_rgb[j, i, 1] = int(sg + 0.5)
            out_rgb[j, i, 2] = int(sb + 0.5)
        else:
            out_rgb[j, i, 0] = 0
            out_rgb[j, i, 1] = 0
            out_rgb[j, i, 2] = 0
    else:
        r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
        num_samples = float(factor * factor)
        glitch_count = 0

        for sy in range(factor):
            sub_j = float(j * factor + sy)
            for sx in range(factor):
                sub_i = float(i * factor + sx)
                s, is_gl = compute_perturbation_sample_floatexp_bla(
                    ref_re, ref_im, ref_len, max_iter,
                    ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub_norm, dy_sub_norm,
                    E_scale, ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                    gen_n, gen_kr, gen_ki, strict_glitch,
                    bla_table
                )
                if is_gl > 0:
                    glitch_count += 1
                if s >= 0.0:
                    norm = compute_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
                    sr, sg, sb = sample_lut_lerp(s_lut, norm)
                    r_acc += sr
                    g_acc += sg
                    b_acc += sb

        inv_samples = 1.0 / num_samples
        out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
        out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
        out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)
        out_glitch[j, i] = 1 if glitch_count > 0 else 0
        out_iter[j, i] = s



# --- GPU Histogram Equalization Rendering & Scan Kernels ---
@cuda.jit
def zero_histogram_cuda_kernel(d_hist, num_bins):
    idx = cuda.grid(1)
    if idx < num_bins:
        d_hist[idx] = 0


@cuda.jit
def scan_blocks_cuda_kernel(d_in, d_block_sums, d_out):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    idx = bid * 512 + tid

    sdata = cuda.shared.array(512, dtype=float64)
    sdata[tid] = float64(d_in[idx])
    cuda.syncthreads()

    stride = 1
    while stride < 512:
        val = 0.0
        if tid >= stride:
            val = sdata[tid - stride]
        cuda.syncthreads()
        if tid >= stride:
            sdata[tid] += val
        cuda.syncthreads()
        stride *= 2

    d_out[idx] = sdata[tid]
    if tid == 511:
        d_block_sums[bid] = sdata[511]


@cuda.jit
def scan_single_block_cuda_kernel(d_block_sums, num_blocks):
    tid = cuda.threadIdx.x
    sdata = cuda.shared.array(128, dtype=float64)
    sdata[tid] = d_block_sums[tid] if tid < num_blocks else 0.0
    cuda.syncthreads()

    stride = 1
    while stride < 128:
        val = 0.0
        if tid >= stride:
            val = sdata[tid - stride]
        cuda.syncthreads()
        if tid >= stride:
            sdata[tid] += val
        cuda.syncthreads()
        stride *= 2

    if tid < num_blocks:
        d_block_sums[tid] = sdata[tid]


@cuda.jit
def finalize_normalized_cdf_cuda_kernel(d_unnorm, d_block_sums, d_cdf_lut, num_bins):
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    idx = bid * 512 + tid
    if idx >= num_bins:
        return

    offset = 0.0
    if bid > 0:
        offset = d_block_sums[bid - 1]

    val = d_unnorm[idx] + offset
    num_blocks = (num_bins + 511) // 512
    total_esc = d_block_sums[num_blocks - 1]

    if total_esc > 0.0:
        d_cdf_lut[idx] = float32(val / total_esc)
    else:
        d_cdf_lut[idx] = float32(idx / float(num_bins - 1))


@cuda.jit
def build_histogram_kernel(iter_map, hist, target_w, target_h, max_iter, num_bins):
    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return
    s = iter_map[j, i]
    if s >= 0.0:
        inv_max = 1.0 / max(1.0, float(max_iter))
        pos = (s * inv_max) * float(num_bins - 1)
        bin_idx = int(pos)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        elif bin_idx < 0:
            bin_idx = 0
        cuda.atomic.add(hist, bin_idx, 1)


@cuda.jit(fastmath=True)
def apply_histogram_equalization_kernel(out_rgb, iter_map, cdf_lut, lut, target_w, target_h, max_iter, num_bins, palette_offset=0.0, color_density=1.0, color_contrast=1.0):
    s_lut = cuda.shared.array(shape=(2048, 3), dtype=float32)
    stage_lut_to_shared(lut, s_lut)

    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return

    s = iter_map[j, i]
    if s >= 0.0:
        norm = sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
        sr, sg, sb = sample_lut_lerp(s_lut, norm)
        out_rgb[j, i, 0] = int(sr + 0.5)
        out_rgb[j, i, 1] = int(sg + 0.5)
        out_rgb[j, i, 2] = int(sb + 0.5)
    else:
        out_rgb[j, i, 0] = 0
        out_rgb[j, i, 1] = 0
        out_rgb[j, i, 2] = 0


@cuda.jit(fastmath=True)
def ssaa_resolve_hist_fp64_kernel(out_rgb, iter_map, edge_mask, cdf_lut, lut, target_w, target_h, max_iter,
                                  x_min, y_max, dx_sub, dy_sub, factor, fractal_type,
                                  julia_cx, julia_cy, gen_n, gen_kr, gen_ki, num_bins, palette_offset=0.0,
                                  color_density=1.0, color_contrast=1.0):
    s_lut = cuda.shared.array(shape=(2048, 3), dtype=float32)
    stage_lut_to_shared(lut, s_lut)

    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return
    if edge_mask[j, i] == 0:
        return

    r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
    num_samples = float(factor * factor)
    for sy in range(factor):
        y = y_max - (j * factor + sy) * dy_sub
        for sx in range(factor):
            x = x_min + (i * factor + sx) * dx_sub
            s = compute_fp64_sample(x, y, max_iter, fractal_type, julia_cx, julia_cy, gen_n, gen_kr, gen_ki)
            if s >= 0.0:
                norm = sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                sr, sg, sb = sample_lut_lerp(s_lut, norm)
                r_acc += sr
                g_acc += sg
                b_acc += sb

    inv_samples = 1.0 / num_samples
    out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
    out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
    out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)


@cuda.jit
def ssaa_resolve_hist_dd_kernel(out_rgb, iter_map, edge_mask, cdf_lut, lut, target_w, target_h, max_iter,
                                x_min_hi, x_min_lo, dx_hi, dx_lo,
                                y_max_hi, y_max_lo, dy_hi, dy_lo,
                                factor, fractal_type,
                                julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo,
                                gen_n, gen_kr, gen_ki, num_bins, palette_offset=0.0,
                                color_density=1.0, color_contrast=1.0):
    s_lut = cuda.shared.array(shape=(2048, 3), dtype=float32)
    stage_lut_to_shared(lut, s_lut)

    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return
    if edge_mask[j, i] == 0:
        return

    r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
    num_samples = float(factor * factor)
    for sy in range(factor):
        sub_j = j * factor + sy
        jdy_hi, jdy_lo = dd_mul_f64(dy_hi, dy_lo, float(sub_j))
        cy_hi, cy_lo = dd_sub(y_max_hi, y_max_lo, jdy_hi, jdy_lo)

        for sx in range(factor):
            sub_i = i * factor + sx
            idx_hi, idx_lo = dd_mul_f64(dx_hi, dx_lo, float(sub_i))
            cx_hi, cx_lo = dd_add(x_min_hi, x_min_lo, idx_hi, idx_lo)

            s = compute_dd_sample(cx_hi, cx_lo, cy_hi, cy_lo, max_iter, fractal_type,
                                  julia_cx_hi, julia_cx_lo, julia_cy_hi, julia_cy_lo,
                                  gen_n, gen_kr, gen_ki)
            if s >= 0.0:
                norm = sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                sr, sg, sb = sample_lut_lerp(s_lut, norm)
                r_acc += sr
                g_acc += sg
                b_acc += sb

    inv_samples = 1.0 / num_samples
    out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
    out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
    out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)


@cuda.jit(fastmath=True)
def ssaa_resolve_hist_perturbation_kernel(out_rgb, out_glitch, iter_map, edge_mask, cdf_lut, lut, target_w, target_h, max_iter,
                                         ref_re, ref_im, ref_len,
                                         ref_px_sub, ref_py_sub, dx_sub, dy_sub,
                                         ref_cx, ref_cy, factor, fractal_type,
                                         julia_cx, julia_cy,
                                         bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                                         bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                                         bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                                         bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                                         rebase_pass, in_glitch_mask, gen_n, gen_kr, gen_ki, strict_glitch,
                                         num_bins, palette_offset=0.0, color_density=1.0, color_contrast=1.0):
    s_lut = cuda.shared.array(shape=(2048, 3), dtype=float32)
    stage_lut_to_shared(lut, s_lut)

    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return
    if rebase_pass > 0 and in_glitch_mask[j, i] == 0:
        return
    if rebase_pass == 0 and edge_mask[j, i] == 0:
        return

    r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
    num_samples = float(factor * factor)
    glitch_count = 0

    for sy in range(factor):
        sub_j = j * factor + sy
        for sx in range(factor):
            sub_i = i * factor + sx
            s, is_gl = compute_perturbation_sample(
                ref_re, ref_im, ref_len, max_iter,
                ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub, dy_sub,
                ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                gen_n, gen_kr, gen_ki, strict_glitch
            )
            if is_gl > 0:
                glitch_count += 1
            if s >= 0.0:
                norm = sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                sr, sg, sb = sample_lut_lerp(s_lut, norm)
                r_acc += sr
                g_acc += sg
                b_acc += sb

    inv_samples = 1.0 / num_samples
    out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
    out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
    out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)
    out_glitch[j, i] = 1 if glitch_count > 0 else 0
    if rebase_pass > 0:
        iter_map[j, i] = s


@cuda.jit(fastmath=True)
def ssaa_resolve_hist_floatexp_kernel(
    out_rgb, out_glitch, iter_map, edge_mask, cdf_lut, lut, target_w, target_h, max_iter,
    ref_re, ref_im, ref_len,
    ref_px_sub, ref_py_sub, dx_sub_norm, dy_sub_norm, E_scale,
    ref_cx, ref_cy, factor, fractal_type,
    julia_cx, julia_cy,
    bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
    bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
    bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
    bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
    rebase_pass, in_glitch_mask, gen_n, gen_kr, gen_ki, strict_glitch,
    num_bins, palette_offset=0.0, color_density=1.0, color_contrast=1.0
):
    s_lut = cuda.shared.array(shape=(2048, 3), dtype=float32)
    stage_lut_to_shared(lut, s_lut)

    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return
    if rebase_pass > 0 and in_glitch_mask[j, i] == 0:
        return
    if rebase_pass == 0 and edge_mask[j, i] == 0:
        return

    r_acc, g_acc, b_acc = 0.0, 0.0, 0.0
    num_samples = float(factor * factor)
    glitch_count = 0

    for sy in range(factor):
        sub_j = float(j * factor + sy)
        for sx in range(factor):
            sub_i = float(i * factor + sx)
            s, is_gl = compute_perturbation_sample_floatexp(
                ref_re, ref_im, ref_len, max_iter,
                ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub_norm, dy_sub_norm,
                E_scale, ref_cx, ref_cy, fractal_type, julia_cx, julia_cy,
                bbsa_skip, bbsa_inv_r, bbsa_ar, bbsa_ai, bbsa_br, bbsa_bi,
                bbsa_cr, bbsa_ci, bbsa_dr, bbsa_di,
                bbsa_er, bbsa_ei, bbsa_fr, bbsa_fi,
                bbsa_gr, bbsa_gi, bbsa_hr, bbsa_hi,
                gen_n, gen_kr, gen_ki, strict_glitch
            )
            if is_gl > 0:
                glitch_count += 1
            if s >= 0.0:
                norm = sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                sr, sg, sb = sample_lut_lerp(s_lut, norm)
                r_acc += sr
                g_acc += sg
                b_acc += sb

    inv_samples = 1.0 / num_samples
    out_rgb[j, i, 0] = int(r_acc * inv_samples + 0.5)
    out_rgb[j, i, 1] = int(g_acc * inv_samples + 0.5)
    out_rgb[j, i, 2] = int(b_acc * inv_samples + 0.5)
    out_glitch[j, i] = 1 if glitch_count > 0 else 0
    if rebase_pass > 0:
        iter_map[j, i] = s



# --- Multi-Limb High-Precision Device Arithmetic (Signed-Magnitude 34 Limbs, 1056 bits) ---
ML_LIMBS = 34


def decimal_to_limbs(d, num_limbs=ML_LIMBS):
    """Converts a Decimal value to signed-magnitude uint32 limbs representation for CUDA."""
    if HAS_GMPY2:
        try:
            import gmpy2
            m = d if isinstance(d, gmpy2.mpfr) else gmpy2.mpfr(str(d))
            sign = 1 if m < 0 else 0
            val = abs(m)
            int_part = int(val)
            limbs = np.zeros(num_limbs, dtype=np.uint32)
            limbs[0] = np.uint32(int_part)
            frac = val - int_part
            for i in range(1, num_limbs):
                frac = gmpy2.mul_2exp(frac, 32)
                limb_val = int(frac)
                limbs[i] = np.uint32(limb_val & 0xFFFFFFFF)
                frac = frac - limb_val
            return sign, limbs
        except Exception:
            pass

    d = Decimal(str(d))
    sign = 1 if d < 0 else 0
    val = abs(d)
    int_part = int(val)
    limbs = np.zeros(num_limbs, dtype=np.uint32)
    limbs[0] = np.uint32(int_part)
    frac = val - Decimal(int_part)
    for i in range(1, num_limbs):
        frac = frac * Decimal(4294967296)
        limb_val = int(frac)
        limbs[i] = np.uint32(limb_val & 0xFFFFFFFF)
        frac = frac - Decimal(limb_val)
    return sign, limbs


@cuda.jit(device=True, inline=True)
def ml_set_zero(out):
    for i in range(ML_LIMBS):
        out[i] = np.uint32(0)


@cuda.jit(device=True, inline=True)
def ml_copy(src, dst):
    for i in range(ML_LIMBS):
        dst[i] = src[i]


@cuda.jit(device=True, inline=True)
def ml_cmp_mag(a, b):
    for i in range(ML_LIMBS):
        if a[i] > b[i]:
            return 1
        elif a[i] < b[i]:
            return -1
    return 0


@cuda.jit(device=True, inline=True)
def ml_add_mag(a, b, out):
    carry = np.uint64(0)
    for i in range(ML_LIMBS - 1, -1, -1):
        s = np.uint64(a[i]) + np.uint64(b[i]) + carry
        out[i] = np.uint32(s & np.uint64(0xFFFFFFFF))
        carry = s >> np.uint64(32)
    return np.uint32(carry)


@cuda.jit(device=True, inline=True)
def ml_sub_mag(a, b, out):
    # Assumes a >= b
    borrow = np.int64(0)
    for i in range(ML_LIMBS - 1, -1, -1):
        diff = np.int64(a[i]) - np.int64(b[i]) - borrow
        if diff < np.int64(0):
            diff += np.int64(4294967296)
            borrow = np.int64(1)
        else:
            borrow = np.int64(0)
        out[i] = np.uint32(diff & np.int64(0xFFFFFFFF))


@cuda.jit(device=True, inline=True)
def ml_add_signed(a_sign, a_limbs, b_sign, b_limbs, out_limbs):
    if a_sign == b_sign:
        ml_add_mag(a_limbs, b_limbs, out_limbs)
        return a_sign
    else:
        cmp_res = ml_cmp_mag(a_limbs, b_limbs)
        if cmp_res >= 0:
            ml_sub_mag(a_limbs, b_limbs, out_limbs)
            return a_sign
        else:
            ml_sub_mag(b_limbs, a_limbs, out_limbs)
            return b_sign


@cuda.jit(device=True, inline=True)
def ml_sub_signed(a_sign, a_limbs, b_sign, b_limbs, out_limbs):
    inv_b_sign = 1 - b_sign if b_sign == 0 else 0
    return ml_add_signed(a_sign, a_limbs, inv_b_sign, b_limbs, out_limbs)


@cuda.jit(device=True, inline=True)
def ml_mul_mag(a, b, out):
    carry = np.uint64(0)
    for k in range(ML_LIMBS - 1, -1, -1):
        sum_k = carry
        for i in range(k + 1):
            j = k - i
            p = np.uint64(a[i]) * np.uint64(b[j])
            sum_k += (p & np.uint64(0xFFFFFFFF))

        k_plus = k + 1
        if k_plus < ML_LIMBS:
            for i in range(k_plus + 1):
                j = k_plus - i
                p = np.uint64(a[i]) * np.uint64(b[j])
                sum_k += (p >> np.uint64(32))
        elif k_plus == ML_LIMBS:
            for i in range(1, ML_LIMBS):
                j = ML_LIMBS - i
                p = np.uint64(a[i]) * np.uint64(b[j])
                sum_k += (p >> np.uint64(32))

        out[k] = np.uint32(sum_k & np.uint64(0xFFFFFFFF))
        carry = sum_k >> np.uint64(32)


@cuda.jit(device=True, inline=True)
def compute_ml_sample(cx_sign, cx_limbs, cy_sign, cy_limbs, max_iter, fractal_type,
                      julia_cx_sign, julia_cx_limbs, julia_cy_sign, julia_cy_limbs):
    zx = cuda.local.array(ML_LIMBS, dtype=np.uint32)
    zy = cuda.local.array(ML_LIMBS, dtype=np.uint32)
    zx2 = cuda.local.array(ML_LIMBS, dtype=np.uint32)
    zy2 = cuda.local.array(ML_LIMBS, dtype=np.uint32)
    zxy = cuda.local.array(ML_LIMBS, dtype=np.uint32)
    two_zxy = cuda.local.array(ML_LIMBS, dtype=np.uint32)
    sub_xy = cuda.local.array(ML_LIMBS, dtype=np.uint32)
    tmp_z = cuda.local.array(ML_LIMBS, dtype=np.uint32)
    c_x_limbs = cuda.local.array(ML_LIMBS, dtype=np.uint32)
    c_y_limbs = cuda.local.array(ML_LIMBS, dtype=np.uint32)

    ml_set_zero(zx)
    ml_set_zero(zy)
    zx_sign = 0
    zy_sign = 0

    if fractal_type == 4:
        return np.float32(-1.0)

    if fractal_type in (2, 3):
        ml_copy(cx_limbs, zx)
        ml_copy(cy_limbs, zy)
        zx_sign = cx_sign
        zy_sign = cy_sign
        c_x_sign = julia_cx_sign
        c_y_sign = julia_cy_sign
        for l_idx in range(ML_LIMBS):
            c_x_limbs[l_idx] = julia_cx_limbs[l_idx]
            c_y_limbs[l_idx] = julia_cy_limbs[l_idx]
    else:
        c_x_sign = cx_sign
        c_y_sign = cy_sign
        ml_copy(cx_limbs, c_x_limbs)
        ml_copy(cy_limbs, c_y_limbs)

    for k in range(max_iter):
        if fractal_type in (1, 3):
            zx_sign = 0
            zy_sign = 0

        # zx2 = zx^2
        ml_mul_mag(zx, zx, zx2)
        # zy2 = zy^2
        ml_mul_mag(zy, zy, zy2)

        # Bailout check: |Z|^2 >= 65536.0 (R=256)
        if zx2[0] >= 65536 or zy2[0] >= 65536 or (zx2[0] + zy2[0]) >= 65536:
            mag2 = float(zx2[0]) + float(zy2[0]) + (float(zx2[1]) + float(zy2[1])) * 2.3283064365386963e-10
            log_mag = math.log(mag2)
            return np.float32(k + 1.0) - np.float32((math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * INV_LN2_F32) if log_mag > 1.0 else np.float32(k)

        # zxy = zx * zy
        zxy_sign = 0 if (zx_sign == zy_sign) else 1
        ml_mul_mag(zx, zy, zxy)

        # two_zxy = 2 * zxy
        ml_add_mag(zxy, zxy, two_zxy)

        # sub_xy = zx2 - zy2
        sub_sign = ml_sub_signed(0, zx2, 0, zy2, sub_xy)

        # zx_next = (zx2 - zy2) + cx
        zx_sign = ml_add_signed(sub_sign, sub_xy, c_x_sign, c_x_limbs, tmp_z)
        ml_copy(tmp_z, zx)

        # zy_next = 2 * zx * zy + cy (or - cy for Burning Ship)
        if fractal_type in (1, 3):
            zy_sign = ml_sub_signed(0, two_zxy, c_y_sign, c_y_limbs, tmp_z)
        else:
            zy_sign = ml_add_signed(zxy_sign, two_zxy, c_y_sign, c_y_limbs, tmp_z)
        ml_copy(tmp_z, zy)

    return np.float32(-1.0)


# --- GPU Exhaustive Fallback Kernels for Strict Metric Mode ---
@cuda.jit(fastmath=True)
def eval_glitched_pixels_cuda_kernel(
    out_rgb, glitch_out, iter_map, lut, coords, num_glitches,
    max_iter, fractal_type, scheme_id, palette_offset,
    precision_tier,
    x_min_hi, x_min_lo, y_max_hi, y_max_lo,
    dx_hi, dx_lo, dy_hi, dy_lo,
    j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo,
    gen_n, gen_kr, gen_ki,
    cx_signs, cx_limbs, cy_signs, cy_limbs,
    j_cx_sign, j_cx_limbs, j_cy_sign, j_cy_limbs,
    color_density=1.0, color_contrast=1.0
):
    s_lut = cuda.shared.array(shape=(2048, 3), dtype=float32)
    stage_lut_to_shared(lut, s_lut)

    idx = cuda.grid(1)
    if idx >= num_glitches:
        return

    r = coords[idx, 0]
    c = coords[idx, 1]

    s = np.float32(-1.0)
    if precision_tier == 0:
        x = x_min_hi + float(c) * dx_hi
        y = y_max_hi - float(r) * dy_hi
        s = compute_fp64_sample(x, y, max_iter, fractal_type, j_cx_hi, j_cy_hi, gen_n, gen_kr, gen_ki)
    elif precision_tier == 1:
        cdx_h, cdx_l = dd_mul_f64(dx_hi, dx_lo, float(c))
        cx_h, cx_l = dd_add(x_min_hi, x_min_lo, cdx_h, cdx_l)
        rdy_h, rdy_l = dd_mul_f64(dy_hi, dy_lo, float(r))
        cy_h, cy_l = dd_sub(y_max_hi, y_max_lo, rdy_h, rdy_l)
        s = compute_dd_sample(cx_h, cx_l, cy_h, cy_l, max_iter, fractal_type, j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo, gen_n, gen_kr, gen_ki)
    else:
        cur_cx_limbs = cuda.local.array(ML_LIMBS, dtype=np.uint32)
        cur_cy_limbs = cuda.local.array(ML_LIMBS, dtype=np.uint32)
        for l_idx in range(ML_LIMBS):
            cur_cx_limbs[l_idx] = cx_limbs[idx, l_idx]
            cur_cy_limbs[l_idx] = cy_limbs[idx, l_idx]
        s = compute_ml_sample(
            cx_signs[idx], cur_cx_limbs, cy_signs[idx], cur_cy_limbs,
            max_iter, fractal_type,
            j_cx_sign, j_cx_limbs, j_cy_sign, j_cy_limbs
        )

    iter_map[r, c] = s
    if s >= 0.0:
        norm = compute_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
        sr, sg, sb = sample_lut_lerp(s_lut, norm)
        out_rgb[r, c, 0] = int(sr + 0.5)
        out_rgb[r, c, 1] = int(sg + 0.5)
        out_rgb[r, c, 2] = int(sb + 0.5)
    else:
        out_rgb[r, c, 0] = 0
        out_rgb[r, c, 1] = 0
        out_rgb[r, c, 2] = 0

    glitch_out[r, c] = 0


@cuda.jit(fastmath=True)
def eval_glitched_pixels_hist_cuda_kernel(
    out_rgb, glitch_out, iter_map, lut, cdf_lut, num_bins, coords, num_glitches,
    max_iter, fractal_type, palette_offset,
    precision_tier,
    x_min_hi, x_min_lo, y_max_hi, y_max_lo,
    dx_hi, dx_lo, dy_hi, dy_lo,
    j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo,
    gen_n, gen_kr, gen_ki,
    cx_signs, cx_limbs, cy_signs, cy_limbs,
    j_cx_sign, j_cx_limbs, j_cy_sign, j_cy_limbs,
    color_density=1.0, color_contrast=1.0
):
    s_lut = cuda.shared.array(shape=(2048, 3), dtype=float32)
    stage_lut_to_shared(lut, s_lut)

    idx = cuda.grid(1)
    if idx >= num_glitches:
        return

    r = coords[idx, 0]
    c = coords[idx, 1]

    s = np.float32(-1.0)
    if precision_tier == 0:
        x = x_min_hi + float(c) * dx_hi
        y = y_max_hi - float(r) * dy_hi
        s = compute_fp64_sample(x, y, max_iter, fractal_type, j_cx_hi, j_cy_hi, gen_n, gen_kr, gen_ki)
    elif precision_tier == 1:
        cdx_h, cdx_l = dd_mul_f64(dx_hi, dx_lo, float(c))
        cx_h, cx_l = dd_add(x_min_hi, x_min_lo, cdx_h, cdx_l)
        rdy_h, rdy_l = dd_mul_f64(dy_hi, dy_lo, float(r))
        cy_h, cy_l = dd_sub(y_max_hi, y_max_lo, rdy_h, rdy_l)
        s = compute_dd_sample(cx_h, cx_l, cy_h, cy_l, max_iter, fractal_type, j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo, gen_n, gen_kr, gen_ki)
    else:
        cur_cx_limbs = cuda.local.array(ML_LIMBS, dtype=np.uint32)
        cur_cy_limbs = cuda.local.array(ML_LIMBS, dtype=np.uint32)
        for l_idx in range(ML_LIMBS):
            cur_cx_limbs[l_idx] = cx_limbs[idx, l_idx]
            cur_cy_limbs[l_idx] = cy_limbs[idx, l_idx]
        s = compute_ml_sample(
            cx_signs[idx], cur_cx_limbs, cy_signs[idx], cur_cy_limbs,
            max_iter, fractal_type,
            j_cx_sign, j_cx_limbs, j_cy_sign, j_cy_limbs
        )

    iter_map[r, c] = s
    if s >= 0.0:
        norm = sample_cdf_norm(s, max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
        sr, sg, sb = sample_lut_lerp(s_lut, norm)
        out_rgb[r, c, 0] = int(sr + 0.5)
        out_rgb[r, c, 1] = int(sg + 0.5)
        out_rgb[r, c, 2] = int(sb + 0.5)
    else:
        out_rgb[r, c, 0] = 0
        out_rgb[r, c, 1] = 0
        out_rgb[r, c, 2] = 0

    glitch_out[r, c] = 0


# --- GPU BBSA Series Coefficient & Skip Calculation Kernel ---
@cuda.jit(device=True, inline=True)
def _term_magnitude_cuda(coeff_r, coeff_i, r, m):
    val = math.sqrt(coeff_r * coeff_r + coeff_i * coeff_i)
    if val == 0.0:
        return 0.0
    for _ in range(m):
        val *= r
        if val == 0.0:
            return 0.0
    return val


@cuda.jit(device=True, inline=True)
def _cuda_comb(n, k):
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    if k > n // 2:
        k = n - k
    res = 1.0
    for i in range(1, k + 1):
        res = res * float(n - (k - i)) / float(i)
    return res


@cuda.jit(device=True, inline=True)
def _cuda_term_mag(val, r, m):
    if val == 0.0:
        return 0.0
    for _ in range(m):
        val *= r
        if val == 0.0:
            return 0.0
    return val


@cuda.jit
def compute_bbsa_cuda_kernel(
    ref_re, ref_im, ref_len, max_r, fractal_type,
    bbsa_tol, bbsa_order, gen_n, gen_kr, gen_ki,
    E_scale,
    out_bbsa_res
):
    """Computes radius-normalized Taylor series perturbation coefficients (A..H) directly on the GPU."""
    if cuda.grid(1) != 0:
        return

    for idx in range(17):
        out_bbsa_res[idx] = 0.0

    if fractal_type not in (0, 2, 4) or bbsa_tol <= 0.0 or bbsa_order <= 0 or ref_len < 2 or max_r <= 0.0:
        return

    if bbsa_order == 32:
        return

    r = float(max_r)
    if math.isnan(r) or math.isinf(r) or r <= 0.0:
        return

    order = 4 if bbsa_order == 4 else 8
    tol_threshold = bbsa_tol
    linear_bound = 0.5

    # FloatExp polynomial BBSA is disabled in favor of chained BLA
    if E_scale > 0:
        for idx in range(17):
            out_bbsa_res[idx] = 0.0
        return

    # Standard Perturbation BBSA Branch (E_scale == 0)
    c_const_r = r if fractal_type in (0, 4) else 0.0
    c_const_i = 0.0
    scale_nl = 1.0
    best_skip = 0

    if fractal_type in (0, 2):
        if order == 8:
            a1_r = 0.0 if fractal_type == 0 else r
            a1_i = 0.0
            a2_r = 0.0; a2_i = 0.0
            a3_r = 0.0; a3_i = 0.0
            a4_r = 0.0; a4_i = 0.0
            a5_r = 0.0; a5_i = 0.0
            a6_r = 0.0; a6_i = 0.0
            a7_r = 0.0; a7_i = 0.0
            a8_r = 0.0; a8_i = 0.0

            best_a1_r = a1_r; best_a1_i = a1_i
            best_a2_r = 0.0; best_a2_i = 0.0
            best_a3_r = 0.0; best_a3_i = 0.0
            best_a4_r = 0.0; best_a4_i = 0.0
            best_a5_r = 0.0; best_a5_i = 0.0
            best_a6_r = 0.0; best_a6_i = 0.0
            best_a7_r = 0.0; best_a7_i = 0.0
            best_a8_r = 0.0; best_a8_i = 0.0

            for k in range(ref_len):
                c1_r = 2.0 * ref_re[k]
                c1_i = 2.0 * ref_im[k]

                # q2 = a1 * a1 * scale_nl
                t_r, t_i = c_mul(a1_r, a1_i, a1_r, a1_i)
                q2_r = t_r * scale_nl; q2_i = t_i * scale_nl

                # q3 = 2.0 * (a1 * a2) * scale_nl
                t_r, t_i = c_mul(a1_r, a1_i, a2_r, a2_i)
                q3_r = 2.0 * t_r * scale_nl; q3_i = 2.0 * t_i * scale_nl

                # q4 = (2.0 * (a1 * a3) + a2 * a2) * scale_nl
                t_r, t_i = c_mul(a1_r, a1_i, a3_r, a3_i)
                s_r, s_i = c_mul(a2_r, a2_i, a2_r, a2_i)
                q4_r = (2.0 * t_r + s_r) * scale_nl; q4_i = (2.0 * t_i + s_i) * scale_nl

                # q5 = 2.0 * (a1 * a4 + a2 * a3) * scale_nl
                t1_r, t1_i = c_mul(a1_r, a1_i, a4_r, a4_i)
                t2_r, t2_i = c_mul(a2_r, a2_i, a3_r, a3_i)
                q5_r = 2.0 * (t1_r + t2_r) * scale_nl; q5_i = 2.0 * (t1_i + t2_i) * scale_nl

                # q6 = (2.0 * (a1 * a5 + a2 * a4) + a3 * a3) * scale_nl
                t1_r, t1_i = c_mul(a1_r, a1_i, a5_r, a5_i)
                t2_r, t2_i = c_mul(a2_r, a2_i, a4_r, a4_i)
                s_r, s_i = c_mul(a3_r, a3_i, a3_r, a3_i)
                q6_r = (2.0 * (t1_r + t2_r) + s_r) * scale_nl; q6_i = (2.0 * (t1_i + t2_i) + s_i) * scale_nl

                # q7 = 2.0 * (a1 * a6 + a2 * a5 + a3 * a4) * scale_nl
                t1_r, t1_i = c_mul(a1_r, a1_i, a6_r, a6_i)
                t2_r, t2_i = c_mul(a2_r, a2_i, a5_r, a5_i)
                t3_r, t3_i = c_mul(a3_r, a3_i, a4_r, a4_i)
                q7_r = 2.0 * (t1_r + t2_r + t3_r) * scale_nl; q7_i = 2.0 * (t1_i + t2_i + t3_i) * scale_nl

                # q8 = (2.0 * (a1 * a7 + a2 * a6 + a3 * a5) + a4 * a4) * scale_nl
                t1_r, t1_i = c_mul(a1_r, a1_i, a7_r, a7_i)
                t2_r, t2_i = c_mul(a2_r, a2_i, a6_r, a6_i)
                t3_r, t3_i = c_mul(a3_r, a3_i, a5_r, a5_i)
                s_r, s_i = c_mul(a4_r, a4_i, a4_r, a4_i)
                q8_r = (2.0 * (t1_r + t2_r + t3_r) + s_r) * scale_nl; q8_i = (2.0 * (t1_i + t2_i + t3_i) + s_i) * scale_nl

                # q9 = 2.0 * (a1 * a8 + a2 * a7 + a3 * a6 + a4 * a5) * scale_nl
                t1_r, t1_i = c_mul(a1_r, a1_i, a8_r, a8_i)
                t2_r, t2_i = c_mul(a2_r, a2_i, a7_r, a7_i)
                t3_r, t3_i = c_mul(a3_r, a3_i, a6_r, a6_i)
                t4_r, t4_i = c_mul(a4_r, a4_i, a5_r, a5_i)
                q9_r = 2.0 * (t1_r + t2_r + t3_r + t4_r) * scale_nl; q9_i = 2.0 * (t1_i + t2_i + t3_i + t4_i) * scale_nl

                # n1 = c1 * a1 + c_const
                t_r, t_i = c_mul(c1_r, c1_i, a1_r, a1_i)
                n1_r = t_r + c_const_r; n1_i = t_i + c_const_i

                # n2 = c1 * a2 + q2
                t_r, t_i = c_mul(c1_r, c1_i, a2_r, a2_i)
                n2_r = t_r + q2_r; n2_i = t_i + q2_i

                # n3 = c1 * a3 + q3
                t_r, t_i = c_mul(c1_r, c1_i, a3_r, a3_i)
                n3_r = t_r + q3_r; n3_i = t_i + q3_i

                # n4 = c1 * a4 + q4
                t_r, t_i = c_mul(c1_r, c1_i, a4_r, a4_i)
                n4_r = t_r + q4_r; n4_i = t_i + q4_i

                # n5 = c1 * a5 + q5
                t_r, t_i = c_mul(c1_r, c1_i, a5_r, a5_i)
                n5_r = t_r + q5_r; n5_i = t_i + q5_i

                # n6 = c1 * a6 + q6
                t_r, t_i = c_mul(c1_r, c1_i, a6_r, a6_i)
                n6_r = t_r + q6_r; n6_i = t_i + q6_i

                # n7 = c1 * a7 + q7
                t_r, t_i = c_mul(c1_r, c1_i, a7_r, a7_i)
                n7_r = t_r + q7_r; n7_i = t_i + q7_i

                # n8 = c1 * a8 + q8
                t_r, t_i = c_mul(c1_r, c1_i, a8_r, a8_i)
                n8_r = t_r + q8_r; n8_i = t_i + q8_i

                mag_A = math.sqrt(n1_r * n1_r + n1_i * n1_i)
                mag_H = math.sqrt(n8_r * n8_r + n8_i * n8_i)
                if math.isnan(mag_A) or math.isinf(mag_A) or math.isnan(mag_H) or math.isinf(mag_H):
                    break

                linear_term = mag_A
                error_term = math.sqrt(q9_r * q9_r + q9_i * q9_i) + mag_H * 0.1

                if linear_term > 0.0:
                    if (error_term / linear_term) > tol_threshold or linear_term > linear_bound:
                        break

                a1_r = n1_r; a1_i = n1_i
                a2_r = n2_r; a2_i = n2_i
                a3_r = n3_r; a3_i = n3_i
                a4_r = n4_r; a4_i = n4_i
                a5_r = n5_r; a5_i = n5_i
                a6_r = n6_r; a6_i = n6_i
                a7_r = n7_r; a7_i = n7_i
                a8_r = n8_r; a8_i = n8_i

                best_skip = k + 1
                best_a1_r = a1_r; best_a1_i = a1_i
                best_a2_r = a2_r; best_a2_i = a2_i
                best_a3_r = a3_r; best_a3_i = a3_i
                best_a4_r = a4_r; best_a4_i = a4_i
                best_a5_r = a5_r; best_a5_i = a5_i
                best_a6_r = a6_r; best_a6_i = a6_i
                best_a7_r = a7_r; best_a7_i = a7_i
                best_a8_r = a8_r; best_a8_i = a8_i

            best_skip = min(best_skip, max(0, ref_len - 2))
            out_bbsa_res[0] = float(best_skip)
            out_bbsa_res[1] = best_a1_r; out_bbsa_res[2] = best_a1_i
            out_bbsa_res[3] = best_a2_r; out_bbsa_res[4] = best_a2_i
            out_bbsa_res[5] = best_a3_r; out_bbsa_res[6] = best_a3_i
            out_bbsa_res[7] = best_a4_r; out_bbsa_res[8] = best_a4_i
            out_bbsa_res[9] = best_a5_r; out_bbsa_res[10] = best_a5_i
            out_bbsa_res[11] = best_a6_r; out_bbsa_res[12] = best_a6_i
            out_bbsa_res[13] = best_a7_r; out_bbsa_res[14] = best_a7_i
            out_bbsa_res[15] = best_a8_r; out_bbsa_res[16] = best_a8_i
            return
        else:
            # 4th order unrolled
            a1_r = 0.0 if fractal_type == 0 else r
            a1_i = 0.0
            a2_r = 0.0; a2_i = 0.0
            a3_r = 0.0; a3_i = 0.0
            a4_r = 0.0; a4_i = 0.0

            best_a1_r = a1_r; best_a1_i = a1_i
            best_a2_r = 0.0; best_a2_i = 0.0
            best_a3_r = 0.0; best_a3_i = 0.0
            best_a4_r = 0.0; best_a4_i = 0.0

            for k in range(ref_len):
                c1_r = 2.0 * ref_re[k]
                c1_i = 2.0 * ref_im[k]

                t_r, t_i = c_mul(a1_r, a1_i, a1_r, a1_i)
                q2_r = t_r * scale_nl; q2_i = t_i * scale_nl

                t_r, t_i = c_mul(a1_r, a1_i, a2_r, a2_i)
                q3_r = 2.0 * t_r * scale_nl; q3_i = 2.0 * t_i * scale_nl

                t_r, t_i = c_mul(a1_r, a1_i, a3_r, a3_i)
                s_r, s_i = c_mul(a2_r, a2_i, a2_r, a2_i)
                q4_r = (2.0 * t_r + s_r) * scale_nl; q4_i = (2.0 * t_i + s_i) * scale_nl

                t1_r, t1_i = c_mul(a1_r, a1_i, a4_r, a4_i)
                t2_r, t2_i = c_mul(a2_r, a2_i, a3_r, a3_i)
                q5_r = 2.0 * (t1_r + t2_r) * scale_nl; q5_i = 2.0 * (t1_i + t2_i) * scale_nl

                t_r, t_i = c_mul(c1_r, c1_i, a1_r, a1_i)
                n1_r = t_r + c_const_r; n1_i = t_i + c_const_i

                t_r, t_i = c_mul(c1_r, c1_i, a2_r, a2_i)
                n2_r = t_r + q2_r; n2_i = t_i + q2_i

                t_r, t_i = c_mul(c1_r, c1_i, a3_r, a3_i)
                n3_r = t_r + q3_r; n3_i = t_i + q3_i

                t_r, t_i = c_mul(c1_r, c1_i, a4_r, a4_i)
                n4_r = t_r + q4_r; n4_i = t_i + q4_i

                mag_A = math.sqrt(n1_r * n1_r + n1_i * n1_i)
                mag_D = math.sqrt(n4_r * n4_r + n4_i * n4_i)
                if math.isnan(mag_A) or math.isinf(mag_A) or math.isnan(mag_D) or math.isinf(mag_D):
                    break

                linear_term = mag_A
                error_term = math.sqrt(q5_r * q5_r + q5_i * q5_i) + mag_D * 0.1

                if linear_term > 0.0:
                    if (error_term / linear_term) > bbsa_tol or linear_term > linear_bound:
                        break

                a1_r = n1_r; a1_i = n1_i
                a2_r = n2_r; a2_i = n2_i
                a3_r = n3_r; a3_i = n3_i
                a4_r = n4_r; a4_i = n4_i

                best_skip = k + 1
                best_a1_r = a1_r; best_a1_i = a1_i
                best_a2_r = a2_r; best_a2_i = a2_i
                best_a3_r = a3_r; best_a3_i = a3_i
                best_a4_r = a4_r; best_a4_i = a4_i

            best_skip = min(best_skip, max(0, ref_len - 2))
            out_bbsa_res[0] = float(best_skip)
            out_bbsa_res[1] = best_a1_r; out_bbsa_res[2] = best_a1_i
            out_bbsa_res[3] = best_a2_r; out_bbsa_res[4] = best_a2_i
            out_bbsa_res[5] = best_a3_r; out_bbsa_res[6] = best_a3_i
            out_bbsa_res[7] = best_a4_r; out_bbsa_res[8] = best_a4_i
            return

    elif fractal_type == 4 and gen_n == 3:
        if order == 8:
            a1_r = 0.0; a1_i = 0.0
            a2_r = 0.0; a2_i = 0.0
            a3_r = 0.0; a3_i = 0.0
            a4_r = 0.0; a4_i = 0.0
            a5_r = 0.0; a5_i = 0.0
            a6_r = 0.0; a6_i = 0.0
            a7_r = 0.0; a7_i = 0.0
            a8_r = 0.0; a8_i = 0.0

            best_a1_r = a1_r; best_a1_i = a1_i
            best_a2_r = 0.0; best_a2_i = 0.0
            best_a3_r = 0.0; best_a3_i = 0.0
            best_a4_r = 0.0; best_a4_i = 0.0
            best_a5_r = 0.0; best_a5_i = 0.0
            best_a6_r = 0.0; best_a6_i = 0.0
            best_a7_r = 0.0; best_a7_i = 0.0
            best_a8_r = 0.0; best_a8_i = 0.0

            for k in range(ref_len):
                xr = ref_re[k]; xi = ref_im[k]
                x2_r, x2_i = c_mul(xr, xi, xr, xi)
                c1_r = 3.0 * x2_r + gen_kr
                c1_i = 3.0 * x2_i + gen_ki
                c2_r = 3.0 * xr
                c2_i = 3.0 * xi

                t_r, t_i = c_mul(a1_r, a1_i, a1_r, a1_i)
                q2_r = t_r; q2_i = t_i

                t_r, t_i = c_mul(a1_r, a1_i, a2_r, a2_i)
                q3_r = 2.0 * t_r; q3_i = 2.0 * t_i

                t_r, t_i = c_mul(a1_r, a1_i, a3_r, a3_i)
                s_r, s_i = c_mul(a2_r, a2_i, a2_r, a2_i)
                q4_r = 2.0 * t_r + s_r; q4_i = 2.0 * t_i + s_i

                t1_r, t1_i = c_mul(a1_r, a1_i, a4_r, a4_i)
                t2_r, t2_i = c_mul(a2_r, a2_i, a3_r, a3_i)
                q5_r = 2.0 * (t1_r + t2_r); q5_i = 2.0 * (t1_i + t2_i)

                t1_r, t1_i = c_mul(a1_r, a1_i, a5_r, a5_i)
                t2_r, t2_i = c_mul(a2_r, a2_i, a4_r, a4_i)
                s_r, s_i = c_mul(a3_r, a3_i, a3_r, a3_i)
                q6_r = 2.0 * (t1_r + t2_r) + s_r; q6_i = 2.0 * (t1_i + t2_i) + s_i

                t1_r, t1_i = c_mul(a1_r, a1_i, a6_r, a6_i)
                t2_r, t2_i = c_mul(a2_r, a2_i, a5_r, a5_i)
                t3_r, t3_i = c_mul(a3_r, a3_i, a4_r, a4_i)
                q7_r = 2.0 * (t1_r + t2_r + t3_r); q7_i = 2.0 * (t1_i + t2_i + t3_i)

                t1_r, t1_i = c_mul(a1_r, a1_i, a7_r, a7_i)
                t2_r, t2_i = c_mul(a2_r, a2_i, a6_r, a6_i)
                t3_r, t3_i = c_mul(a3_r, a3_i, a5_r, a5_i)
                s_r, s_i = c_mul(a4_r, a4_i, a4_r, a4_i)
                q8_r = 2.0 * (t1_r + t2_r + t3_r) + s_r; q8_i = 2.0 * (t1_i + t2_i + t3_i) + s_i

                w3_r, w3_i = c_mul(q2_r, q2_i, a1_r, a1_i)

                t1_r, t1_i = c_mul(q2_r, q2_i, a2_r, a2_i)
                t2_r, t2_i = c_mul(q3_r, q3_i, a1_r, a1_i)
                w4_r = t1_r + t2_r; w4_i = t1_i + t2_i

                t1_r, t1_i = c_mul(q2_r, q2_i, a3_r, a3_i)
                t2_r, t2_i = c_mul(q3_r, q3_i, a2_r, a2_i)
                t3_r, t3_i = c_mul(q4_r, q4_i, a1_r, a1_i)
                w5_r = t1_r + t2_r + t3_r; w5_i = t1_i + t2_i + t3_i

                t1_r, t1_i = c_mul(q2_r, q2_i, a4_r, a4_i)
                t2_r, t2_i = c_mul(q3_r, q3_i, a3_r, a3_i)
                t3_r, t3_i = c_mul(q4_r, q4_i, a2_r, a2_i)
                t4_r, t4_i = c_mul(q5_r, q5_i, a1_r, a1_i)
                w6_r = t1_r + t2_r + t3_r + t4_r; w6_i = t1_i + t2_i + t3_i + t4_i

                t1_r, t1_i = c_mul(q2_r, q2_i, a5_r, a5_i)
                t2_r, t2_i = c_mul(q3_r, q3_i, a4_r, a4_i)
                t3_r, t3_i = c_mul(q4_r, q4_i, a3_r, a3_i)
                t4_r, t4_i = c_mul(q5_r, q5_i, a2_r, a2_i)
                t5_r, t5_i = c_mul(q6_r, q6_i, a1_r, a1_i)
                w7_r = t1_r + t2_r + t3_r + t4_r + t5_r; w7_i = t1_i + t2_i + t3_i + t4_i + t5_i

                t1_r, t1_i = c_mul(q2_r, q2_i, a6_r, a6_i)
                t2_r, t2_i = c_mul(q3_r, q3_i, a5_r, a5_i)
                t3_r, t3_i = c_mul(q4_r, q4_i, a4_r, a4_i)
                t4_r, t4_i = c_mul(q5_r, q5_i, a3_r, a3_i)
                t5_r, t5_i = c_mul(q6_r, q6_i, a2_r, a2_i)
                t6_r, t6_i = c_mul(q7_r, q7_i, a1_r, a1_i)
                w8_r = t1_r + t2_r + t3_r + t4_r + t5_r + t6_r; w8_i = t1_i + t2_i + t3_i + t4_i + t5_i + t6_i

                t1_r, t1_i = c_mul(c1_r, c1_i, a1_r, a1_i)
                n1_r = t1_r + c_const_r; n1_i = t1_i + c_const_i

                t1_r, t1_i = c_mul(c1_r, c1_i, a2_r, a2_i)
                t2_r, t2_i = c_mul(c2_r, c2_i, q2_r, q2_i)
                n2_r = t1_r + t2_r; n2_i = t1_i + t2_i

                t1_r, t1_i = c_mul(c1_r, c1_i, a3_r, a3_i)
                t2_r, t2_i = c_mul(c2_r, c2_i, q3_r, q3_i)
                n3_r = t1_r + t2_r + w3_r; n3_i = t1_i + t2_i + w3_i

                t1_r, t1_i = c_mul(c1_r, c1_i, a4_r, a4_i)
                t2_r, t2_i = c_mul(c2_r, c2_i, q4_r, q4_i)
                n4_r = t1_r + t2_r + w4_r; n4_i = t1_i + t2_i + w4_i

                t1_r, t1_i = c_mul(c1_r, c1_i, a5_r, a5_i)
                t2_r, t2_i = c_mul(c2_r, c2_i, q5_r, q5_i)
                n5_r = t1_r + t2_r + w5_r; n5_i = t1_i + t2_i + w5_i

                t1_r, t1_i = c_mul(c1_r, c1_i, a6_r, a6_i)
                t2_r, t2_i = c_mul(c2_r, c2_i, q6_r, q6_i)
                n6_r = t1_r + t2_r + w6_r; n6_i = t1_i + t2_i + w6_i

                t1_r, t1_i = c_mul(c1_r, c1_i, a7_r, a7_i)
                t2_r, t2_i = c_mul(c2_r, c2_i, q7_r, q7_i)
                n7_r = t1_r + t2_r + w7_r; n7_i = t1_i + t2_i + w7_i

                t1_r, t1_i = c_mul(c1_r, c1_i, a8_r, a8_i)
                t2_r, t2_i = c_mul(c2_r, c2_i, q8_r, q8_i)
                n8_r = t1_r + t2_r + w8_r; n8_i = t1_i + t2_i + w8_i

                mag_A = math.sqrt(n1_r * n1_r + n1_i * n1_i)
                mag_H = math.sqrt(n8_r * n8_r + n8_i * n8_i)
                if math.isnan(mag_A) or math.isinf(mag_A) or math.isnan(mag_H) or math.isinf(mag_H):
                    break

                linear_term = mag_A
                error_term = mag_H * 0.1

                if linear_term > 0.0:
                    if (error_term / linear_term) > tol_threshold or linear_term > linear_bound:
                        break

                a1_r = n1_r; a1_i = n1_i
                a2_r = n2_r; a2_i = n2_i
                a3_r = n3_r; a3_i = n3_i
                a4_r = n4_r; a4_i = n4_i
                a5_r = n5_r; a5_i = n5_i
                a6_r = n6_r; a6_i = n6_i
                a7_r = n7_r; a7_i = n7_i
                a8_r = n8_r; a8_i = n8_i

                best_skip = k + 1
                best_a1_r = a1_r; best_a1_i = a1_i
                best_a2_r = a2_r; best_a2_i = a2_i
                best_a3_r = a3_r; best_a3_i = a3_i
                best_a4_r = a4_r; best_a4_i = a4_i
                best_a5_r = a5_r; best_a5_i = a5_i
                best_a6_r = a6_r; best_a6_i = a6_i
                best_a7_r = a7_r; best_a7_i = a7_i
                best_a8_r = a8_r; best_a8_i = a8_i

            best_skip = min(best_skip, max(0, ref_len - 2))
            out_bbsa_res[0] = float(best_skip)
            out_bbsa_res[1] = best_a1_r; out_bbsa_res[2] = best_a1_i
            out_bbsa_res[3] = best_a2_r; out_bbsa_res[4] = best_a2_i
            out_bbsa_res[5] = best_a3_r; out_bbsa_res[6] = best_a3_i
            out_bbsa_res[7] = best_a4_r; out_bbsa_res[8] = best_a4_i
            out_bbsa_res[9] = best_a5_r; out_bbsa_res[10] = best_a5_i
            out_bbsa_res[11] = best_a6_r; out_bbsa_res[12] = best_a6_i
            out_bbsa_res[13] = best_a7_r; out_bbsa_res[14] = best_a7_i
            out_bbsa_res[15] = best_a8_r; out_bbsa_res[16] = best_a8_i
            return
        else:
            a1_r = 0.0; a1_i = 0.0
            a2_r = 0.0; a2_i = 0.0
            a3_r = 0.0; a3_i = 0.0
            a4_r = 0.0; a4_i = 0.0

            best_a1_r = a1_r; best_a1_i = a1_i
            best_a2_r = 0.0; best_a2_i = 0.0
            best_a3_r = 0.0; best_a3_i = 0.0
            best_a4_r = 0.0; best_a4_i = 0.0

            for k in range(ref_len):
                xr = ref_re[k]; xi = ref_im[k]
                x2_r, x2_i = c_mul(xr, xi, xr, xi)
                c1_r = 3.0 * x2_r + gen_kr
                c1_i = 3.0 * x2_i + gen_ki
                c2_r = 3.0 * xr
                c2_i = 3.0 * xi

                t_r, t_i = c_mul(a1_r, a1_i, a1_r, a1_i)
                q2_r = t_r; q2_i = t_i

                t_r, t_i = c_mul(a1_r, a1_i, a2_r, a2_i)
                q3_r = 2.0 * t_r; q3_i = 2.0 * t_i

                t_r, t_i = c_mul(a1_r, a1_i, a3_r, a3_i)
                s_r, s_i = c_mul(a2_r, a2_i, a2_r, a2_i)
                q4_r = 2.0 * t_r + s_r; q4_i = 2.0 * t_i + s_i

                w3_r, w3_i = c_mul(q2_r, q2_i, a1_r, a1_i)

                t1_r, t1_i = c_mul(q2_r, q2_i, a2_r, a2_i)
                t2_r, t2_i = c_mul(q3_r, q3_i, a1_r, a1_i)
                w4_r = t1_r + t2_r; w4_i = t1_i + t2_i

                t1_r, t1_i = c_mul(c1_r, c1_i, a1_r, a1_i)
                n1_r = t1_r + c_const_r; n1_i = t1_i + c_const_i

                t1_r, t1_i = c_mul(c1_r, c1_i, a2_r, a2_i)
                t2_r, t2_i = c_mul(c2_r, c2_i, q2_r, q2_i)
                n2_r = t1_r + t2_r; n2_i = t1_i + t2_i

                t1_r, t1_i = c_mul(c1_r, c1_i, a3_r, a3_i)
                t2_r, t2_i = c_mul(c2_r, c2_i, q3_r, q3_i)
                n3_r = t1_r + t2_r + w3_r; n3_i = t1_i + t2_i + w3_i

                t1_r, t1_i = c_mul(c1_r, c1_i, a4_r, a4_i)
                t2_r, t2_i = c_mul(c2_r, c2_i, q4_r, q4_i)
                n4_r = t1_r + t2_r + w4_r; n4_i = t1_i + t2_i + w4_i

                mag_A = math.sqrt(n1_r * n1_r + n1_i * n1_i)
                mag_D = math.sqrt(n4_r * n4_r + n4_i * n4_i)
                if math.isnan(mag_A) or math.isinf(mag_A) or math.isnan(mag_D) or math.isinf(mag_D):
                    break

                linear_term = mag_A
                error_term = mag_D * 0.1

                if linear_term > 0.0:
                    if (error_term / linear_term) > bbsa_tol or linear_term > linear_bound:
                        break

                a1_r = n1_r; a1_i = n1_i
                a2_r = n2_r; a2_i = n2_i
                a3_r = n3_r; a3_i = n3_i
                a4_r = n4_r; a4_i = n4_i

                best_skip = k + 1
                best_a1_r = a1_r; best_a1_i = a1_i
                best_a2_r = a2_r; best_a2_i = a2_i
                best_a3_r = a3_r; best_a3_i = a3_i
                best_a4_r = a4_r; best_a4_i = a4_i

            best_skip = min(best_skip, max(0, ref_len - 2))
            out_bbsa_res[0] = float(best_skip)
            out_bbsa_res[1] = best_a1_r; out_bbsa_res[2] = best_a1_i
            out_bbsa_res[3] = best_a2_r; out_bbsa_res[4] = best_a2_i
            out_bbsa_res[5] = best_a3_r; out_bbsa_res[6] = best_a3_i
            out_bbsa_res[7] = best_a4_r; out_bbsa_res[8] = best_a4_i
            return

    elif fractal_type == 4:
        a_r = cuda.local.array(9, dtype=float64)
        a_i = cuda.local.array(9, dtype=float64)
        a_next_r = cuda.local.array(9, dtype=float64)
        a_next_i = cuda.local.array(9, dtype=float64)
        best_a_r = cuda.local.array(9, dtype=float64)
        best_a_i = cuda.local.array(9, dtype=float64)
        c_r = cuda.local.array(9, dtype=float64)
        c_i = cuda.local.array(9, dtype=float64)
        p_r = cuda.local.array((9, 9), dtype=float64)
        p_i = cuda.local.array((9, 9), dtype=float64)

        for m in range(9):
            a_r[m] = 0.0; a_i[m] = 0.0
            best_a_r[m] = 0.0; best_a_i[m] = 0.0

        a_r[1] = 0.0 if fractal_type == 4 else r

        for k in range(ref_len):
            xr = ref_re[k]; xi = ref_im[k]

            p_xr = 1.0; p_xi = 0.0
            for _ in range(gen_n - 1):
                p_xr, p_xi = c_mul(p_xr, p_xi, xr, xi)

            c_r[1] = float(gen_n) * p_xr + gen_kr
            c_i[1] = float(gen_n) * p_xi + gen_ki

            for j in range(2, min(int(gen_n), order) + 1):
                comb_val = _cuda_comb(int(gen_n), j)
                pwr_r = 1.0; pwr_i = 0.0
                for _ in range(gen_n - j):
                    pwr_r, pwr_i = c_mul(pwr_r, pwr_i, xr, xi)
                c_r[j] = comb_val * pwr_r
                c_i[j] = comb_val * pwr_i
            for j in range(min(int(gen_n), order) + 1, order + 1):
                c_r[j] = 0.0; c_i[j] = 0.0

            for m in range(1, order + 1):
                p_r[1, m] = a_r[m]
                p_i[1, m] = a_i[m]

            for j in range(2, order + 1):
                for m in range(j, order + 1):
                    s_r = 0.0; s_i = 0.0
                    for i in range(1, m):
                        tr, ti = c_mul(p_r[j - 1, i], p_i[j - 1, i], a_r[m - i], a_i[m - i])
                        s_r += tr; s_i += ti
                    p_r[j, m] = s_r * scale_nl
                    p_i[j, m] = s_i * scale_nl

            for m in range(1, order + 1):
                term_r = c_const_r if m == 1 else 0.0
                term_i = c_const_i if m == 1 else 0.0
                for j in range(1, m + 1):
                    if c_r[j] != 0.0 or c_i[j] != 0.0:
                        tr, ti = c_mul(c_r[j], c_i[j], p_r[j, m], p_i[j, m])
                        term_r += tr; term_i += ti
                a_next_r[m] = term_r
                a_next_i[m] = term_i

            mag_A = math.sqrt(a_next_r[1] * a_next_r[1] + a_next_i[1] * a_next_i[1])
            mag_last = math.sqrt(a_next_r[order] * a_next_r[order] + a_next_i[order] * a_next_i[order])
            if math.isnan(mag_A) or math.isinf(mag_A) or math.isnan(mag_last) or math.isinf(mag_last):
                break

            linear_term = mag_A
            error_term = mag_last * 0.1

            if linear_term > 0.0:
                if (error_term / linear_term) > tol_threshold or linear_term > linear_bound:
                    break

            for m in range(1, order + 1):
                a_r[m] = a_next_r[m]
                a_i[m] = a_next_i[m]

            best_skip = k + 1
            for m in range(1, order + 1):
                best_a_r[m] = a_r[m]
                best_a_i[m] = a_i[m]

        best_skip = min(best_skip, max(0, ref_len - 2))
        out_bbsa_res[0] = float(best_skip)
        for idx in range(1, min(order + 1, 9)):
            out_bbsa_res[(idx - 1) * 2 + 1] = best_a_r[idx]
            out_bbsa_res[(idx - 1) * 2 + 2] = best_a_i[idx]
        return


@cuda.jit(fastmath=True)
def recolor_iter_map_cuda_kernel(out_rgb, iter_map, lut, target_w, target_h,
                                 scheme_id, palette_offset=0.0, color_density=1.0, color_contrast=1.0):
    """Ultra-fast (~0.08ms) GPU kernel to re-shade already-computed iteration maps with new palette settings."""
    s_lut = cuda.shared.array(shape=(2048, 3), dtype=float32)
    stage_lut_to_shared(lut, s_lut)

    i, j = cuda.grid(2)
    if i >= target_w or j >= target_h:
        return

    s = iter_map[j, i]
    if s >= 0.0:
        norm = compute_palette_norm(s, scheme_id, palette_offset, color_density, color_contrast)
        sr, sg, sb = sample_lut_lerp(s_lut, norm)
        out_rgb[j, i, 0] = int(sr + 0.5)
        out_rgb[j, i, 1] = int(sg + 0.5)
        out_rgb[j, i, 2] = int(sb + 0.5)
    else:
        out_rgb[j, i, 0] = 0
        out_rgb[j, i, 1] = 0
        out_rgb[j, i, 2] = 0