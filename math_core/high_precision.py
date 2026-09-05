import os
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, getcontext
import numpy as np
from numba import njit

from config import HAS_GMPY2, compute_required_prec_bits, ensure_decimal_precision, INV_LN2_F32, LN_BAILOUT_LOG

try:
    import gmpy2
except ImportError:
    gmpy2 = None


def to_dd(dec_val):
    """Converts a Decimal value to a double-double (hi, lo) float64 pair."""
    d = Decimal(dec_val)
    hi = float(d)
    return hi, float(d - Decimal(hi))


def _term_magnitude(coeff, r, m):
    """Underflow-safe computation of |coeff| * r^m at extreme zoom depths."""
    val = abs(coeff)
    if val == 0.0:
        return 0.0
    for _ in range(m):
        val *= r
        if val == 0.0:
            return 0.0
    return val


@njit(inline='always')
def _c_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


@njit(inline='always')
def _pow2_f64(n):
    if n >= 1023:
        return 1.7976931348623157e+308
    if n <= -1074:
        return 0.0
    return math.ldexp(1.0, int(n))


@njit(fastmath=True)
def _jit_compute_bbsa_coeffs(
    ref_re, ref_im, ref_len, max_r, fractal_type,
    bbsa_tol, bbsa_order, gen_n, gen_kr, gen_ki,
    E_scale
):
    out_bbsa_res = np.zeros(17, dtype=np.float64)

    if fractal_type not in (0, 2, 4) or bbsa_tol <= 0.0 or bbsa_order <= 0 or ref_len < 2 or max_r <= 0.0:
        return out_bbsa_res

    if bbsa_order == 32:
        return out_bbsa_res

    r = float(max_r)
    if math.isnan(r) or math.isinf(r) or r <= 0.0:
        return out_bbsa_res

    order = 4 if bbsa_order == 4 else 8
    tol_threshold = bbsa_tol
    linear_bound = 0.5

    # FloatExp BBSA Branch (E_scale > 0): In FloatExp, higher-order polynomial terms
    # underflow in FP64 (2^-E_scale << 1e-16), rendering polynomial Taylor approximation
    # inaccurate over multi-step skips. BBSA is disabled (skip = 0) so FloatExp executes
    # exact scaled recurrence directly from k=0, eliminating circular artifacts and panning glitches.
    if E_scale > 0:
        return out_bbsa_res

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

                t_r, t_i = _c_mul(a1_r, a1_i, a1_r, a1_i)
                q2_r = t_r * scale_nl; q2_i = t_i * scale_nl

                t_r, t_i = _c_mul(a1_r, a1_i, a2_r, a2_i)
                q3_r = 2.0 * t_r * scale_nl; q3_i = 2.0 * t_i * scale_nl

                t_r, t_i = _c_mul(a1_r, a1_i, a3_r, a3_i)
                s_r, s_i = _c_mul(a2_r, a2_i, a2_r, a2_i)
                q4_r = (2.0 * t_r + s_r) * scale_nl; q4_i = (2.0 * t_i + s_i) * scale_nl

                t1_r, t1_i = _c_mul(a1_r, a1_i, a4_r, a4_i)
                t2_r, t2_i = _c_mul(a2_r, a2_i, a3_r, a3_i)
                q5_r = 2.0 * (t1_r + t2_r) * scale_nl; q5_i = 2.0 * (t1_i + t2_i) * scale_nl

                t1_r, t1_i = _c_mul(a1_r, a1_i, a5_r, a5_i)
                t2_r, t2_i = _c_mul(a2_r, a2_i, a4_r, a4_i)
                s_r, s_i = _c_mul(a3_r, a3_i, a3_r, a3_i)
                q6_r = (2.0 * (t1_r + t2_r) + s_r) * scale_nl; q6_i = (2.0 * (t1_i + t2_i) + s_i) * scale_nl

                t1_r, t1_i = _c_mul(a1_r, a1_i, a6_r, a6_i)
                t2_r, t2_i = _c_mul(a2_r, a2_i, a5_r, a5_i)
                t3_r, t3_i = _c_mul(a3_r, a3_i, a4_r, a4_i)
                q7_r = 2.0 * (t1_r + t2_r + t3_r) * scale_nl; q7_i = 2.0 * (t1_i + t2_i + t3_i) * scale_nl

                t1_r, t1_i = _c_mul(a1_r, a1_i, a7_r, a7_i)
                t2_r, t2_i = _c_mul(a2_r, a2_i, a6_r, a6_i)
                t3_r, t3_i = _c_mul(a3_r, a3_i, a5_r, a5_i)
                s_r, s_i = _c_mul(a4_r, a4_i, a4_r, a4_i)
                q8_r = (2.0 * (t1_r + t2_r + t3_r) + s_r) * scale_nl; q8_i = (2.0 * (t1_i + t2_i + t3_i) + s_i) * scale_nl

                t1_r, t1_i = _c_mul(a1_r, a1_i, a8_r, a8_i)
                t2_r, t2_i = _c_mul(a2_r, a2_i, a7_r, a7_i)
                t3_r, t3_i = _c_mul(a3_r, a3_i, a6_r, a6_i)
                t4_r, t4_i = _c_mul(a4_r, a4_i, a5_r, a5_i)
                q9_r = 2.0 * (t1_r + t2_r + t3_r + t4_r) * scale_nl; q9_i = 2.0 * (t1_i + t2_i + t3_i + t4_i) * scale_nl

                t_r, t_i = _c_mul(c1_r, c1_i, a1_r, a1_i)
                n1_r = t_r + c_const_r; n1_i = t_i + c_const_i

                t_r, t_i = _c_mul(c1_r, c1_i, a2_r, a2_i)
                n2_r = t_r + q2_r; n2_i = t_i + q2_i

                t_r, t_i = _c_mul(c1_r, c1_i, a3_r, a3_i)
                n3_r = t_r + q3_r; n3_i = t_i + q3_i

                t_r, t_i = _c_mul(c1_r, c1_i, a4_r, a4_i)
                n4_r = t_r + q4_r; n4_i = t_i + q4_i

                t_r, t_i = _c_mul(c1_r, c1_i, a5_r, a5_i)
                n5_r = t_r + q5_r; n5_i = t_i + q5_i

                t_r, t_i = _c_mul(c1_r, c1_i, a6_r, a6_i)
                n6_r = t_r + q6_r; n6_i = t_i + q6_i

                t_r, t_i = _c_mul(c1_r, c1_i, a7_r, a7_i)
                n7_r = t_r + q7_r; n7_i = t_i + q7_i

                t_r, t_i = _c_mul(c1_r, c1_i, a8_r, a8_i)
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
            return out_bbsa_res
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

                t_r, t_i = _c_mul(a1_r, a1_i, a1_r, a1_i)
                q2_r = t_r * scale_nl; q2_i = t_i * scale_nl

                t_r, t_i = _c_mul(a1_r, a1_i, a2_r, a2_i)
                q3_r = 2.0 * t_r * scale_nl; q3_i = 2.0 * t_i * scale_nl

                t_r, t_i = _c_mul(a1_r, a1_i, a3_r, a3_i)
                s_r, s_i = _c_mul(a2_r, a2_i, a2_r, a2_i)
                q4_r = (2.0 * t_r + s_r) * scale_nl; q4_i = (2.0 * t_i + s_i) * scale_nl

                t1_r, t1_i = _c_mul(a1_r, a1_i, a4_r, a4_i)
                t2_r, t2_i = _c_mul(a2_r, a2_i, a3_r, a3_i)
                q5_r = 2.0 * (t1_r + t2_r) * scale_nl; q5_i = 2.0 * (t1_i + t2_i) * scale_nl

                t_r, t_i = _c_mul(c1_r, c1_i, a1_r, a1_i)
                n1_r = t_r + c_const_r; n1_i = t_i + c_const_i

                t_r, t_i = _c_mul(c1_r, c1_i, a2_r, a2_i)
                n2_r = t_r + q2_r; n2_i = t_i + q2_i

                t_r, t_i = _c_mul(c1_r, c1_i, a3_r, a3_i)
                n3_r = t_r + q3_r; n3_i = t_i + q3_i

                t_r, t_i = _c_mul(c1_r, c1_i, a4_r, a4_i)
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
            return out_bbsa_res

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
                x2_r, x2_i = _c_mul(xr, xi, xr, xi)
                c1_r = 3.0 * x2_r + gen_kr
                c1_i = 3.0 * x2_i + gen_ki
                c2_r = 3.0 * xr
                c2_i = 3.0 * xi

                t_r, t_i = _c_mul(a1_r, a1_i, a1_r, a1_i)
                q2_r = t_r; q2_i = t_i

                t_r, t_i = _c_mul(a1_r, a1_i, a2_r, a2_i)
                q3_r = 2.0 * t_r; q3_i = 2.0 * t_i

                t_r, t_i = _c_mul(a1_r, a1_i, a3_r, a3_i)
                s_r, s_i = _c_mul(a2_r, a2_i, a2_r, a2_i)
                q4_r = 2.0 * t_r + s_r; q4_i = 2.0 * t_i + s_i

                t1_r, t1_i = _c_mul(a1_r, a1_i, a4_r, a4_i)
                t2_r, t2_i = _c_mul(a2_r, a2_i, a3_r, a3_i)
                q5_r = 2.0 * (t1_r + t2_r); q5_i = 2.0 * (t1_i + t2_i)

                t1_r, t1_i = _c_mul(a1_r, a1_i, a5_r, a5_i)
                t2_r, t2_i = _c_mul(a2_r, a2_i, a4_r, a4_i)
                s_r, s_i = _c_mul(a3_r, a3_i, a3_r, a3_i)
                q6_r = 2.0 * (t1_r + t2_r) + s_r; q6_i = 2.0 * (t1_i + t2_i) + s_i

                t1_r, t1_i = _c_mul(a1_r, a1_i, a6_r, a6_i)
                t2_r, t2_i = _c_mul(a2_r, a2_i, a5_r, a5_i)
                t3_r, t3_i = _c_mul(a3_r, a3_i, a4_r, a4_i)
                q7_r = 2.0 * (t1_r + t2_r + t3_r); q7_i = 2.0 * (t1_i + t2_i + t3_i)

                t1_r, t1_i = _c_mul(a1_r, a1_i, a7_r, a7_i)
                t2_r, t2_i = _c_mul(a2_r, a2_i, a6_r, a6_i)
                t3_r, t3_i = _c_mul(a3_r, a3_i, a5_r, a5_i)
                s_r, s_i = _c_mul(a4_r, a4_i, a4_r, a4_i)
                q8_r = 2.0 * (t1_r + t2_r + t3_r) + s_r; q8_i = 2.0 * (t1_i + t2_i + t3_i) + s_i

                w3_r, w3_i = _c_mul(q2_r, q2_i, a1_r, a1_i)

                t1_r, t1_i = _c_mul(q2_r, q2_i, a2_r, a2_i)
                t2_r, t2_i = _c_mul(q3_r, q3_i, a1_r, a1_i)
                w4_r = t1_r + t2_r; w4_i = t1_i + t2_i

                t1_r, t1_i = _c_mul(q2_r, q2_i, a3_r, a3_i)
                t2_r, t2_i = _c_mul(q3_r, q3_i, a2_r, a2_i)
                t3_r, t3_i = _c_mul(q4_r, q4_i, a1_r, a1_i)
                w5_r = t1_r + t2_r + t3_r; w5_i = t1_i + t2_i + t3_i

                t1_r, t1_i = _c_mul(q2_r, q2_i, a4_r, a4_i)
                t2_r, t2_i = _c_mul(q3_r, q3_i, a3_r, a3_i)
                t3_r, t3_i = _c_mul(q4_r, q4_i, a2_r, a2_i)
                t4_r, t4_i = _c_mul(q5_r, q5_i, a1_r, a1_i)
                w6_r = t1_r + t2_r + t3_r + t4_r; w6_i = t1_i + t2_i + t3_i + t4_i

                t1_r, t1_i = _c_mul(q2_r, q2_i, a5_r, a5_i)
                t2_r, t2_i = _c_mul(q3_r, q3_i, a4_r, a4_i)
                t3_r, t3_i = _c_mul(q4_r, q4_i, a3_r, a3_i)
                t4_r, t4_i = _c_mul(q5_r, q5_i, a2_r, a2_i)
                t5_r, t5_i = _c_mul(q6_r, q6_i, a1_r, a1_i)
                w7_r = t1_r + t2_r + t3_r + t4_r + t5_r; w7_i = t1_i + t2_i + t3_i + t4_i + t5_i

                t1_r, t1_i = _c_mul(q2_r, q2_i, a6_r, a6_i)
                t2_r, t2_i = _c_mul(q3_r, q3_i, a5_r, a5_i)
                t3_r, t3_i = _c_mul(q4_r, q4_i, a4_r, a4_i)
                t4_r, t4_i = _c_mul(q5_r, q5_i, a3_r, a3_i)
                t5_r, t5_i = _c_mul(q6_r, q6_i, a2_r, a2_i)
                t6_r, t6_i = _c_mul(q7_r, q7_i, a1_r, a1_i)
                w8_r = t1_r + t2_r + t3_r + t4_r + t5_r + t6_r; w8_i = t1_i + t2_i + t3_i + t4_i + t5_i + t6_i

                t1_r, t1_i = _c_mul(c1_r, c1_i, a1_r, a1_i)
                n1_r = t1_r + c_const_r; n1_i = t1_i + c_const_i

                t1_r, t1_i = _c_mul(c1_r, c1_i, a2_r, a2_i)
                t2_r, t2_i = _c_mul(c2_r, c2_i, q2_r, q2_i)
                n2_r = t1_r + t2_r; n2_i = t1_i + t2_i

                t1_r, t1_i = _c_mul(c1_r, c1_i, a3_r, a3_i)
                t2_r, t2_i = _c_mul(c2_r, c2_i, q3_r, q3_i)
                n3_r = t1_r + t2_r + w3_r; n3_i = t1_i + t2_i + w3_i

                t1_r, t1_i = _c_mul(c1_r, c1_i, a4_r, a4_i)
                t2_r, t2_i = _c_mul(c2_r, c2_i, q4_r, q4_i)
                n4_r = t1_r + t2_r + w4_r; n4_i = t1_i + t2_i + w4_i

                t1_r, t1_i = _c_mul(c1_r, c1_i, a5_r, a5_i)
                t2_r, t2_i = _c_mul(c2_r, c2_i, q5_r, q5_i)
                n5_r = t1_r + t2_r + w5_r; n5_i = t1_i + t2_i + w5_i

                t1_r, t1_i = _c_mul(c1_r, c1_i, a6_r, a6_i)
                t2_r, t2_i = _c_mul(c2_r, c2_i, q6_r, q6_i)
                n6_r = t1_r + t2_r + w6_r; n6_i = t1_i + t2_i + w6_i

                t1_r, t1_i = _c_mul(c1_r, c1_i, a7_r, a7_i)
                t2_r, t2_i = _c_mul(c2_r, c2_i, q7_r, q7_i)
                n7_r = t1_r + t2_r + w7_r; n7_i = t1_i + t2_i + w7_i

                t1_r, t1_i = _c_mul(c1_r, c1_i, a8_r, a8_i)
                t2_r, t2_i = _c_mul(c2_r, c2_i, q8_r, q8_i)
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
            return out_bbsa_res
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
                x2_r, x2_i = _c_mul(xr, xi, xr, xi)
                c1_r = 3.0 * x2_r + gen_kr
                c1_i = 3.0 * x2_i + gen_ki
                c2_r = 3.0 * xr
                c2_i = 3.0 * xi

                t_r, t_i = _c_mul(a1_r, a1_i, a1_r, a1_i)
                q2_r = t_r; q2_i = t_i

                t_r, t_i = _c_mul(a1_r, a1_i, a2_r, a2_i)
                q3_r = 2.0 * t_r; q3_i = 2.0 * t_i

                t_r, t_i = _c_mul(a1_r, a1_i, a3_r, a3_i)
                s_r, s_i = _c_mul(a2_r, a2_i, a2_r, a2_i)
                q4_r = 2.0 * t_r + s_r; q4_i = 2.0 * t_i + s_i

                w3_r, w3_i = _c_mul(q2_r, q2_i, a1_r, a1_i)

                t1_r, t1_i = _c_mul(q2_r, q2_i, a2_r, a2_i)
                t2_r, t2_i = _c_mul(q3_r, q3_i, a1_r, a1_i)
                w4_r = t1_r + t2_r; w4_i = t1_i + t2_i

                t1_r, t1_i = _c_mul(c1_r, c1_i, a1_r, a1_i)
                n1_r = t1_r + c_const_r; n1_i = t1_i + c_const_i

                t1_r, t1_i = _c_mul(c1_r, c1_i, a2_r, a2_i)
                t2_r, t2_i = _c_mul(c2_r, c2_i, q2_r, q2_i)
                n2_r = t1_r + t2_r; n2_i = t1_i + t2_i

                t1_r, t1_i = _c_mul(c1_r, c1_i, a3_r, a3_i)
                t2_r, t2_i = _c_mul(c2_r, c2_i, q3_r, q3_i)
                n3_r = t1_r + t2_r + w3_r; n3_i = t1_i + t2_i + w3_i

                t1_r, t1_i = _c_mul(c1_r, c1_i, a4_r, a4_i)
                t2_r, t2_i = _c_mul(c2_r, c2_i, q4_r, q4_i)
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
            return out_bbsa_res

    return out_bbsa_res


_BBSA_COEFFS_CACHE = {}
_BBSA_CACHE_LOCK = threading.Lock()


def compute_bbsa_coeffs(ref_re, ref_im, ref_len, max_r, fractal_type, bbsa_tol=1e-4, bbsa_order=4,
                        gen_n=3, gen_kr=0.25, gen_ki=1.0, E_scale=0):
    """Computes radius-normalized Taylor series perturbation coefficients (A..H) with zero per-iteration heap allocations."""
    if fractal_type not in (0, 2, 4) or bbsa_tol <= 0.0 or bbsa_order <= 0 or ref_len < 2 or max_r <= 0.0:
        return (0,) + (0.0,) * 16

    try:
        r = float(max_r)
        if math.isnan(r) or math.isinf(r) or r <= 0.0:
            return (0,) + (0.0,) * 16
    except (OverflowError, ZeroDivisionError):
        return (0,) + (0.0,) * 16

    if bbsa_order == 32:
        return (0,) + (0.0,) * 16

    sample_pt = (float(ref_re[0]), float(ref_re[min(5, ref_len-1)]), float(ref_im[min(5, ref_len-1)])) if ref_len > 0 else (0.0, 0.0, 0.0)
    cache_key = (id(ref_re), sample_pt, int(ref_len), float(r), int(fractal_type), float(bbsa_tol), int(bbsa_order), int(gen_n), float(gen_kr), float(gen_ki), int(E_scale))
    with _BBSA_CACHE_LOCK:
        if cache_key in _BBSA_COEFFS_CACHE:
            cached_ref, cached_out = _BBSA_COEFFS_CACHE[cache_key]
            if cached_ref is ref_re:
                return cached_out

    if not isinstance(ref_re, np.ndarray):
        ref_re = np.asarray(ref_re, dtype=np.float64)
    if not isinstance(ref_im, np.ndarray):
        ref_im = np.asarray(ref_im, dtype=np.float64)

    res = _jit_compute_bbsa_coeffs(
        ref_re, ref_im, int(ref_len), float(r), int(fractal_type),
        float(bbsa_tol), int(bbsa_order), int(gen_n), float(gen_kr), float(gen_ki),
        int(E_scale)
    )
    best_skip = int(res[0])
    coeffs = tuple(float(x) for x in res[1:17])
    out = (best_skip,) + coeffs
    with _BBSA_CACHE_LOCK:
        if len(_BBSA_COEFFS_CACHE) > 64:
            _BBSA_COEFFS_CACHE.clear()
        _BBSA_COEFFS_CACHE[cache_key] = (ref_re, out)
    return out


def compute_highprec_escape_scalar(cand_cx, cand_cy, max_iter, fractal_type=0,
                                   julia_cx=0.0, julia_cy=0.0, prec_bits=256,
                                   gen_n=3, gen_kr=0.25, gen_ki=1.0):
    """Calculates continuous smooth escape iteration depth for a single point using GMPY2 or Decimal arbitrary precision."""
    if HAS_GMPY2:
        with gmpy2.local_context(precision=prec_bits):
            bailout = gmpy2.mpfr(65536)
            two = gmpy2.mpfr(2)

            if fractal_type in (2, 3):
                zx = cand_cx if isinstance(cand_cx, gmpy2.mpfr) else gmpy2.mpfr(str(cand_cx))
                zy = cand_cy if isinstance(cand_cy, gmpy2.mpfr) else gmpy2.mpfr(str(cand_cy))
                c_x = julia_cx if isinstance(julia_cx, gmpy2.mpfr) else gmpy2.mpfr(str(julia_cx))
                c_y = julia_cy if isinstance(julia_cy, gmpy2.mpfr) else gmpy2.mpfr(str(julia_cy))
            else:
                zx, zy = gmpy2.mpfr(0), gmpy2.mpfr(0)
                c_x = cand_cx if isinstance(cand_cx, gmpy2.mpfr) else gmpy2.mpfr(str(cand_cx))
                c_y = cand_cy if isinstance(cand_cy, gmpy2.mpfr) else gmpy2.mpfr(str(cand_cy))

            kr_mpfr = gmpy2.mpfr(str(gen_kr)) if fractal_type == 4 else None
            ki_mpfr = gmpy2.mpfr(str(gen_ki)) if fractal_type == 4 else None

            if fractal_type in (0, 2):
                check_step = 8
                saved_k = 0
                z_saved_x, z_saved_y = zx, zy
                for k in range(max_iter):
                    zx2 = gmpy2.square(zx)
                    zy2 = gmpy2.square(zy)
                    mag2_mp = zx2 + zy2
                    if mag2_mp > bailout:
                        mag2 = float(mag2_mp)
                        log_mag = math.log(mag2)
                        return float(k + 1.0) - (math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * float(INV_LN2_F32) if log_mag > 1.0 else float(k)
                    if k == check_step:
                        check_step <<= 1
                        saved_k = k
                        z_saved_x, z_saved_y = zx, zy
                    elif k > saved_k and zx == z_saved_x and zy == z_saved_y:
                        return float(max_iter)
                    zy = gmpy2.fma(gmpy2.mul_2exp(zx, 1), zy, c_y)
                    zx = (zx2 - zy2) + c_x
                return float(max_iter)
            elif fractal_type in (1, 3):
                for k in range(max_iter):
                    zx2 = gmpy2.square(zx)
                    zy2 = gmpy2.square(zy)
                    mag2_mp = zx2 + zy2
                    if mag2_mp > bailout:
                        mag2 = float(mag2_mp)
                        log_mag = math.log(mag2)
                        return float(k + 1.0) - (math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * float(INV_LN2_F32) if log_mag > 1.0 else float(k)
                    zy = gmpy2.mul_2exp(abs(zx) * abs(zy), 1) - c_y
                    zx = (zx2 - zy2) + c_x
                return float(max_iter)
            elif fractal_type == 4:
                inv_ln_n = 1.0 / math.log(float(gen_n)) if gen_n > 1 else float(INV_LN2_F32)
                for k in range(max_iter):
                    zx2 = gmpy2.square(zx)
                    zy2 = gmpy2.square(zy)
                    mag2_mp = zx2 + zy2
                    if mag2_mp > bailout:
                        mag2 = float(mag2_mp)
                        log_mag = math.log(mag2)
                        return float(k + 1.0) - (math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * inv_ln_n if log_mag > 1.0 else float(k)
                    zn_r, zn_i = zx, zy
                    for _ in range(gen_n - 1):
                        next_r = zn_r * zx - zn_i * zy
                        next_i = zn_r * zy + zn_i * zx
                        zn_r, zn_i = next_r, next_i
                    kz_r = kr_mpfr * zx - ki_mpfr * zy
                    kz_i = kr_mpfr * zy + ki_mpfr * zx
                    zx = zn_r + kz_r + c_x
                    zy = zn_i + kz_i + c_y
                return float(max_iter)
            return float(max_iter)
    else:
        orig_prec = getcontext().prec
        getcontext().prec = int(prec_bits / 3.32) + 30
        try:
            bailout = Decimal(65536)
            two = Decimal(2)

            if fractal_type in (2, 3):
                zx = cand_cx if isinstance(cand_cx, Decimal) else Decimal(str(cand_cx))
                zy = cand_cy if isinstance(cand_cy, Decimal) else Decimal(str(cand_cy))
                c_x = julia_cx if isinstance(julia_cx, Decimal) else Decimal(str(julia_cx))
                c_y = julia_cy if isinstance(julia_cy, Decimal) else Decimal(str(julia_cy))
            else:
                zx, zy = Decimal(0), Decimal(0)
                c_x = cand_cx if isinstance(cand_cx, Decimal) else Decimal(str(cand_cx))
                c_y = cand_cy if isinstance(cand_cy, Decimal) else Decimal(str(cand_cy))

            gen_kr_dec = Decimal(str(gen_kr)) if fractal_type == 4 else None
            gen_ki_dec = Decimal(str(gen_ki)) if fractal_type == 4 else None
            inv_n = (1.0 / math.log(float(gen_n))) if (fractal_type == 4 and gen_n > 1) else float(INV_LN2_F32)

            for k in range(max_iter):
                zx2, zy2 = zx * zx, zy * zy
                mag2_dec = zx2 + zy2
                if mag2_dec > bailout:
                    mag2 = float(mag2_dec)
                    log_mag = math.log(mag2)
                    return float(k + 1.0) - (math.log(0.5 * log_mag) - LN_BAILOUT_LOG) * inv_n if log_mag > 1.0 else float(k)
                if fractal_type in (0, 2):
                    zy = two * zx * zy + c_y
                    zx = zx2 - zy2 + c_x
                elif fractal_type in (1, 3):
                    zy = two * abs(zx) * abs(zy) - c_y
                    zx = zx2 - zy2 + c_x
                elif fractal_type == 4:
                    zn_r, zn_i = zx, zy
                    for _ in range(gen_n - 1):
                        next_r = zn_r * zx - zn_i * zy
                        next_i = zn_r * zy + zn_i * zx
                        zn_r, zn_i = next_r, next_i
                    kz_r = gen_kr_dec * zx - gen_ki_dec * zy
                    kz_i = gen_kr_dec * zy + gen_ki_dec * zx
                    zx = zn_r + kz_r + c_x
                    zy = zn_i + kz_i + c_y
            return float(max_iter)
        finally:
            getcontext().prec = orig_prec


def compute_single_highprec_orbit(cand_cx_dec, cand_cy_dec, max_iter, fractal_type=0,
                                  julia_cx=0.0, julia_cy=0.0, prec_bits=256,
                                  gen_n=3, gen_kr=0.25, gen_ki=1.0):
    """Computes a full arbitrary-precision reference orbit array for perturbation rendering."""
    ref_re = np.zeros(max_iter + 1, dtype=np.float64)
    ref_im = np.zeros(max_iter + 1, dtype=np.float64)

    if HAS_GMPY2:
        with gmpy2.local_context(precision=prec_bits):
            bailout = gmpy2.mpfr(65536)
            two = gmpy2.mpfr(2)

            if fractal_type in (2, 3):
                zx = cand_cx_dec if isinstance(cand_cx_dec, gmpy2.mpfr) else gmpy2.mpfr(str(cand_cx_dec))
                zy = cand_cy_dec if isinstance(cand_cy_dec, gmpy2.mpfr) else gmpy2.mpfr(str(cand_cy_dec))
                c_x = julia_cx if isinstance(julia_cx, gmpy2.mpfr) else gmpy2.mpfr(str(julia_cx))
                c_y = julia_cy if isinstance(julia_cy, gmpy2.mpfr) else gmpy2.mpfr(str(julia_cy))
            else:
                zx, zy = gmpy2.mpfr(0), gmpy2.mpfr(0)
                c_x = cand_cx_dec if isinstance(cand_cx_dec, gmpy2.mpfr) else gmpy2.mpfr(str(cand_cx_dec))
                c_y = cand_cy_dec if isinstance(cand_cy_dec, gmpy2.mpfr) else gmpy2.mpfr(str(cand_cy_dec))

            kr_mpfr = gmpy2.mpfr(str(gen_kr)) if fractal_type == 4 else None
            ki_mpfr = gmpy2.mpfr(str(gen_ki)) if fractal_type == 4 else None
            ref_len = max_iter

            if fractal_type in (0, 2):
                check_step = 8
                saved_k = 0
                z_saved_x, z_saved_y = zx, zy
                z_saved_fx, z_saved_fy = 0.0, 0.0
                for k in range(max_iter):
                    rx = float(zx)
                    ry = float(zy)
                    ref_re[k] = rx
                    ref_im[k] = ry
                    if (rx * rx + ry * ry) > 65536.0:
                        ref_len = k
                        break

                    zx2 = gmpy2.square(zx)
                    zy2 = gmpy2.square(zy)

                    if k == check_step:
                        check_step <<= 1
                        saved_k = k
                        z_saved_x, z_saved_y = zx, zy
                        z_saved_fx, z_saved_fy = rx, ry
                    elif k > saved_k and rx == z_saved_fx and ry == z_saved_fy and zx == z_saved_x and zy == z_saved_y:
                        period = k - saved_k
                        if period > 0:
                            for j in range(k, max_iter + 1):
                                ref_re[j] = ref_re[j - period]
                                ref_im[j] = ref_im[j - period]
                            ref_len = max_iter
                            return ref_re, ref_im, ref_len

                    zy = gmpy2.fma(gmpy2.mul_2exp(zx, 1), zy, c_y)
                    zx = (zx2 - zy2) + c_x

                ref_re[ref_len], ref_im[ref_len] = float(zx), float(zy)
            elif fractal_type in (1, 3):
                for k in range(max_iter):
                    rx = float(zx)
                    ry = float(zy)
                    ref_re[k] = rx
                    ref_im[k] = ry
                    if (rx * rx + ry * ry) > 65536.0:
                        ref_len = k
                        break
                    zx2 = gmpy2.square(zx)
                    zy2 = gmpy2.square(zy)
                    zy = gmpy2.mul_2exp(abs(zx) * abs(zy), 1) - c_y
                    zx = (zx2 - zy2) + c_x

                ref_re[ref_len], ref_im[ref_len] = float(zx), float(zy)
            elif fractal_type == 4:
                for k in range(max_iter):
                    rx = float(zx)
                    ry = float(zy)
                    ref_re[k] = rx
                    ref_im[k] = ry
                    if (rx * rx + ry * ry) > 65536.0:
                        ref_len = k
                        break
                    zn_r, zn_i = zx, zy
                    for _ in range(gen_n - 1):
                        next_r = zn_r * zx - zn_i * zy
                        next_i = zn_r * zy + zn_i * zx
                        zn_r, zn_i = next_r, next_i
                    kz_r = kr_mpfr * zx - ki_mpfr * zy
                    kz_i = kr_mpfr * zy + ki_mpfr * zx
                    zx = zn_r + kz_r + c_x
                    zy = zn_i + kz_i + c_y

                ref_re[ref_len], ref_im[ref_len] = float(zx), float(zy)
    else:
        orig_prec = getcontext().prec
        getcontext().prec = int(prec_bits / 3.32) + 30
        try:
            bailout = Decimal(65536)
            two = Decimal(2)

            if fractal_type in (2, 3):
                zx, zy = cand_cx_dec, cand_cy_dec
                c_x, c_y = Decimal(str(julia_cx)), Decimal(str(julia_cy))
            else:
                zx, zy = Decimal(0), Decimal(0)
                c_x, c_y = cand_cx_dec, cand_cy_dec

            gen_kr_dec = Decimal(str(gen_kr))
            gen_ki_dec = Decimal(str(gen_ki))
            ref_len = max_iter

            if fractal_type in (0, 2):
                check_step = 8
                saved_k = 0
                z_saved_x, z_saved_y = zx, zy
                for k in range(max_iter):
                    ref_re[k], ref_im[k] = float(zx), float(zy)
                    zx2, zy2 = zx * zx, zy * zy
                    if (zx2 + zy2) > bailout:
                        ref_len = k
                        break
                    if k == check_step:
                        check_step <<= 1
                        saved_k = k
                        z_saved_x, z_saved_y = zx, zy
                    elif k > saved_k and zx == z_saved_x and zy == z_saved_y:
                        period = k - saved_k
                        if period > 0:
                            for j in range(k, max_iter + 1):
                                ref_re[j] = ref_re[j - period]
                                ref_im[j] = ref_im[j - period]
                            ref_len = max_iter
                            return ref_re, ref_im, ref_len
                    zy = two * zx * zy + c_y
                    zx = zx2 - zy2 + c_x
                ref_re[ref_len], ref_im[ref_len] = float(zx), float(zy)
            elif fractal_type in (1, 3):
                for k in range(max_iter):
                    ref_re[k], ref_im[k] = float(zx), float(zy)
                    zx2, zy2 = zx * zx, zy * zy
                    if (zx2 + zy2) > bailout:
                        ref_len = k
                        break
                    zy = two * abs(zx) * abs(zy) - c_y
                    zx = zx2 - zy2 + c_x
                ref_re[ref_len], ref_im[ref_len] = float(zx), float(zy)
            elif fractal_type == 4:
                for k in range(max_iter):
                    ref_re[k], ref_im[k] = float(zx), float(zy)
                    zx2, zy2 = zx * zx, zy * zy
                    if (zx2 + zy2) > bailout:
                        ref_len = k
                        break
                    zn_r, zn_i = zx, zy
                    for _ in range(gen_n - 1):
                        next_r = zn_r * zx - zn_i * zy
                        next_i = zn_r * zy + zn_i * zx
                        zn_r, zn_i = next_r, next_i
                    kz_r = gen_kr_dec * zx - gen_ki_dec * zy
                    kz_i = gen_kr_dec * zy + gen_ki_dec * zx
                    zx = zn_r + kz_r + c_x
                    zy = zn_i + kz_i + c_y
                ref_re[ref_len], ref_im[ref_len] = float(zx), float(zy)
        finally:
            getcontext().prec = orig_prec

    return ref_re, ref_im, ref_len


_REFERENCE_ORBIT_CACHE = []
_ORBIT_CACHE_LOCK = threading.Lock()


def clear_reference_orbit_cache():
    """Clears all cached reference orbits, BBSA coefficients, and BLA tables."""
    global _REFERENCE_ORBIT_CACHE, _BBSA_COEFFS_CACHE, _BLA_CACHE
    with _ORBIT_CACHE_LOCK:
        _REFERENCE_ORBIT_CACHE.clear()
    with _BBSA_CACHE_LOCK:
        _BBSA_COEFFS_CACHE.clear()
    with _BLA_CACHE_LOCK:
        _BLA_CACHE.clear()


def cache_reference_orbit(cx_dec, cy_dec, pw_dec, max_iter, ref_re, ref_im, ref_len,
                          fractal_type=0, julia_cx=0.0, julia_cy=0.0,
                          gen_n=3, gen_kr=0.25, gen_ki=1.0, is_floatexp=False):
    """Registers a reference orbit into the persistent LRU cache for cross-frame reuse."""
    global _REFERENCE_ORBIT_CACHE
    with _ORBIT_CACHE_LOCK:
        _REFERENCE_ORBIT_CACHE.insert(0, {
            'is_floatexp': is_floatexp,
            'fractal_type': fractal_type,
            'julia_cx': julia_cx,
            'julia_cy': julia_cy,
            'gen_n': gen_n,
            'gen_kr': gen_kr,
            'gen_ki': gen_ki,
            'pw_dec': pw_dec,
            'cx_dec': cx_dec,
            'cy_dec': cy_dec,
            'ref_len': ref_len,
            'max_iter': max_iter,
            'ref_re': ref_re,
            'ref_im': ref_im
        })
        if len(_REFERENCE_ORBIT_CACHE) > 16:
            _REFERENCE_ORBIT_CACHE.pop()


def find_optimal_reference_orbit(cx_dec, cy_dec, pw_dec, ph_dec, max_iter, screen_w, screen_h,
                                 fractal_type=0, julia_cx=0.0, julia_cy=0.0,
                                 gen_n=3, gen_kr=0.25, gen_ki=1.0, is_floatexp=False):
    """Probes candidate screen locations to find the longest-surviving reference orbit,
    with intelligent persistent caching across viewport pans and translations.
    """
    global _REFERENCE_ORBIT_CACHE
    ensure_decimal_precision(pw_dec)

    x_min_dec = cx_dec - (pw_dec / Decimal(2))
    y_max_dec = cy_dec + (ph_dec / Decimal(2))

    # 1. Check persistent reference orbit cache
    best_entry = None
    best_score = -1

    with _ORBIT_CACHE_LOCK:
        for entry in _REFERENCE_ORBIT_CACHE:
            if entry.get('is_floatexp', False) != is_floatexp:
                continue

            if (entry['fractal_type'] == fractal_type and
                entry['julia_cx'] == julia_cx and
                entry['julia_cy'] == julia_cy and
                entry['gen_n'] == gen_n and
                entry['gen_kr'] == gen_kr and
                entry['gen_ki'] == gen_ki):

                scale_ratio = pw_dec / entry['pw_dec']
                if Decimal("0.25") <= scale_ratio <= Decimal("4.0"):
                    cand_u = float((entry['cx_dec'] - x_min_dec) / pw_dec)
                    cand_v = float((y_max_dec - entry['cy_dec']) / ph_dec)
                    if 0.10 <= cand_u <= 0.90 and 0.10 <= cand_v <= 0.90:
                        if entry['ref_len'] > best_score:
                            best_score = entry['ref_len']
                            best_entry = entry
                            if entry['ref_len'] >= max_iter:
                                break

        can_reuse = False
        if best_entry is not None:
            entry_max_iter = best_entry.get('max_iter', best_entry['ref_len'])
            is_natural_escape = (best_entry['ref_len'] < entry_max_iter)
            scale_ratio = pw_dec / best_entry['pw_dec']
            is_panning = (Decimal("0.75") <= scale_ratio <= Decimal("1.33"))

            if best_score >= max_iter:
                can_reuse = True
            elif is_panning and best_score >= 100 and (best_entry['ref_len'] >= max_iter or is_natural_escape):
                # During panning at constant/near-constant zoom, retain the cached orbit within viewport bounds
                can_reuse = True
            elif is_natural_escape and best_score >= int(max_iter * 0.35):
                can_reuse = True
            elif best_entry['cx_dec'] == cx_dec and best_entry['cy_dec'] == cy_dec and best_entry['pw_dec'] == pw_dec and (best_entry['ref_len'] >= max_iter or is_natural_escape):
                can_reuse = True

        if can_reuse and best_entry is not None:
            best_u = float((best_entry['cx_dec'] - x_min_dec) / pw_dec)
            best_v = float((y_max_dec - best_entry['cy_dec']) / ph_dec)
            return (best_entry['cx_dec'], best_entry['cy_dec'], best_u, best_v,
                    best_entry['ref_re'], best_entry['ref_im'], min(best_entry['ref_len'], max_iter))

    bits_needed = compute_required_prec_bits(pw_dec)

    # 2. Compute reference orbit at viewport center (0.5, 0.5)
    ref_re, ref_im, ref_len = compute_single_highprec_orbit(
        cx_dec, cy_dec, max_iter, fractal_type, julia_cx, julia_cy,
        bits_needed, gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
    )

    best_cx_dec = cx_dec
    best_cy_dec = cy_dec
    best_u, best_v = 0.5, 0.5

    # 3. If center already has >= 60% survival, it is optimal (BBSA achieves near-maximal skip).
    if ref_len < int(max_iter * 0.6) and (pw_dec >= Decimal("1e-10000")):
        candidates_uv = [
            (0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65),
            (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5),
            (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8),
            (0.5, 0.35), (0.5, 0.65), (0.35, 0.5), (0.65, 0.5)
        ]

        if HAS_GMPY2:
            with gmpy2.local_context(precision=bits_needed):
                x_min_mpfr = gmpy2.mpfr(str(x_min_dec))
                y_max_mpfr = gmpy2.mpfr(str(y_max_dec))
                pw_mpfr = gmpy2.mpfr(str(pw_dec))
                ph_mpfr = gmpy2.mpfr(str(ph_dec))

            def _probe_worker(uv):
                u, v = uv
                with gmpy2.local_context(precision=bits_needed):
                    cand_x_mpfr = gmpy2.fma(gmpy2.mpfr(u), pw_mpfr, x_min_mpfr)
                    cand_y_mpfr = y_max_mpfr - gmpy2.mpfr(v) * ph_mpfr
                cand_x = Decimal(str(cand_x_mpfr))
                cand_y = Decimal(str(cand_y_mpfr))
                esc = compute_highprec_escape_scalar(
                    cand_x_mpfr, cand_y_mpfr, max_iter, fractal_type=fractal_type,
                    julia_cx=julia_cx, julia_cy=julia_cy, prec_bits=bits_needed,
                    gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
                )
                return esc, cand_x, cand_y, u, v
        else:
            def _probe_worker(uv):
                u, v = uv
                ensure_decimal_precision(pw_dec)
                cand_x = x_min_dec + Decimal(str(u)) * pw_dec
                cand_y = y_max_dec - Decimal(str(v)) * ph_dec
                esc = compute_highprec_escape_scalar(
                    cand_x, cand_y, max_iter, fractal_type=fractal_type,
                    julia_cx=julia_cx, julia_cy=julia_cy, prec_bits=bits_needed,
                    gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
                )
                return esc, cand_x, cand_y, u, v

        max_workers = min(len(candidates_uv), os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            probe_results = list(executor.map(_probe_worker, candidates_uv))

        # Require candidate survival to strictly exceed center survival by at least 20%
        # to justify moving away from the geometric center of the viewport.
        best_esc = max(int(ref_len * 1.2), ref_len + 32)
        best_cand_cx = cx_dec
        best_cand_cy = cy_dec
        best_cand_u = 0.5
        best_cand_v = 0.5

        for esc, cand_x, cand_y, u, v in probe_results:
            if esc > best_esc:
                best_esc = esc
                best_cand_cx = cand_x
                best_cand_cy = cand_y
                best_cand_u = u
                best_cand_v = v

        if best_cand_cx != cx_dec or best_cand_cy != cy_dec:
            cand_re, cand_im, cand_len = compute_single_highprec_orbit(
                best_cand_cx, best_cand_cy, max_iter, fractal_type, julia_cx, julia_cy,
                bits_needed, gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
            )
            # Only adopt the candidate if its actual full orbit survival exceeds the center's survival
            if cand_len > ref_len:
                ref_re, ref_im, ref_len = cand_re, cand_im, cand_len
                best_cx_dec = best_cand_cx
                best_cy_dec = best_cand_cy
                best_u = best_cand_u
                best_v = best_cand_v

    # Fallback safeguard: If candidate probes did not exceed best_entry's survival, retain best_entry
    # ONLY if best_entry is eligible for reuse (matching regime, valid scale, and inside viewport)
    if can_reuse and best_entry is not None and best_entry['ref_len'] > ref_len:
        best_u = float((best_entry['cx_dec'] - x_min_dec) / pw_dec)
        best_v = float((y_max_dec - best_entry['cy_dec']) / ph_dec)
        return (best_entry['cx_dec'], best_entry['cy_dec'], best_u, best_v,
                best_entry['ref_re'], best_entry['ref_im'], min(best_entry['ref_len'], max_iter))

    # Cache orbit (LRU with capacity 16)
    with _ORBIT_CACHE_LOCK:
        _REFERENCE_ORBIT_CACHE.insert(0, {
            'is_floatexp': is_floatexp,
            'fractal_type': fractal_type,
            'julia_cx': julia_cx,
            'julia_cy': julia_cy,
            'gen_n': gen_n,
            'gen_kr': gen_kr,
            'gen_ki': gen_ki,
            'cx_dec': best_cx_dec,
            'cy_dec': best_cy_dec,
            'pw_dec': pw_dec,
            'max_iter': max_iter,
            'ref_re': ref_re,
            'ref_im': ref_im,
            'ref_len': ref_len
        })
        if len(_REFERENCE_ORBIT_CACHE) > 16:
            _REFERENCE_ORBIT_CACHE.pop()

    return best_cx_dec, best_cy_dec, best_u, best_v, ref_re, ref_im, ref_len


@njit(fastmath=True)
def _jit_build_bla_table(ref_re, ref_im, ref_len, fractal_type, tol, max_levels,
                         gen_n=3, gen_kr=0.25, gen_ki=1.0):
    bla_table = np.zeros((max_levels, ref_len, 5), dtype=np.float64)

    if fractal_type not in (0, 2, 4) or ref_len < 4 or tol <= 0.0:
        return bla_table

    tol_sq_4 = 4.0 * tol * tol

    # Level 0: single-step recurrence
    for k in range(ref_len - 1):
        rx = ref_re[k]
        ry = ref_im[k]
        r_mag2 = rx * rx + ry * ry
        if fractal_type in (0, 2):
            bla_table[0, k, 0] = 2.0 * rx
            bla_table[0, k, 1] = 2.0 * ry
            bla_table[0, k, 2] = 1.0
            bla_table[0, k, 3] = 0.0
            r_sq = tol_sq_4 * r_mag2
            if r_sq > 0.25:
                r_sq = 0.25
            if k == 0 and r_mag2 < 1e-14:
                r_sq = 0.0
            bla_table[0, k, 4] = r_sq
        elif fractal_type == 4:
            # Complex power zn = R_k^(gen_n - 1)
            zn_r = 1.0
            zn_i = 0.0
            for _ in range(gen_n - 1):
                nr = zn_r * rx - zn_i * ry
                ni = zn_r * ry + zn_i * rx
                zn_r = nr
                zn_i = ni

            # Linear derivative A_k = n * R_k^(n-1) + K
            ar = float(gen_n) * zn_r + gen_kr
            ai = float(gen_n) * zn_i + gen_ki
            bla_table[0, k, 0] = ar
            bla_table[0, k, 1] = ai
            bla_table[0, k, 2] = 1.0
            bla_table[0, k, 3] = 0.0

            a_mag = math.sqrt(ar * ar + ai * ai)
            tol_a = tol * a_mag
            if tol_a <= 0.0 or math.isnan(tol_a) or math.isinf(tol_a):
                bla_table[0, k, 4] = 0.0
            else:
                r_mag = math.sqrt(r_mag2)
                if gen_n == 2:
                    r_pow = 1.0
                elif gen_n == 3:
                    r_pow = r_mag
                elif gen_n == 4:
                    r_pow = r_mag2
                else:
                    r_pow = r_mag ** float(gen_n - 2)

                b = 0.25 * float(gen_n * (gen_n - 1)) * r_pow
                denom = b + math.sqrt(b * b + tol_a)
                x = (tol_a / denom) if denom > 1e-300 else 0.0

                if gen_n > 3 and tol_a > 0.0:
                    x_max = tol_a ** (1.0 / float(gen_n - 1))
                    if x > x_max:
                        x = x_max

                if x > 0.5:
                    x = 0.5

                r_sq = x * x
                if k == 0 and r_mag2 < 1e-14:
                    r_sq = 0.0

                bla_table[0, k, 4] = r_sq

    # Levels 1 to max_levels - 1: dyadic merging
    step = 1
    for d in range(1, max_levels):
        step_curr = step
        step_next = step * 2
        for k in range(ref_len - step_next):
            k2 = k + step_curr
            r1_sq = bla_table[d - 1, k, 4]
            r2_sq = bla_table[d - 1, k2, 4]
            if r1_sq > 0.0 and r2_sq > 0.0:
                a1_r = bla_table[d - 1, k, 0]
                a1_i = bla_table[d - 1, k, 1]
                b1_r = bla_table[d - 1, k, 2]
                b1_i = bla_table[d - 1, k, 3]

                a2_r = bla_table[d - 1, k2, 0]
                a2_i = bla_table[d - 1, k2, 1]
                b2_r = bla_table[d - 1, k2, 2]
                b2_i = bla_table[d - 1, k2, 3]

                # A = A2 * A1
                ar = a2_r * a1_r - a2_i * a1_i
                ai = a2_r * a1_i + a2_i * a1_r

                # B = A2 * B1 + B2
                br = (a2_r * b1_r - a2_i * b1_i) + b2_r
                bi = (a2_r * b1_i + a2_i * b1_r) + b2_i

                # Check for FP64 overflow or non-finite numbers in A or B
                a_mag2 = ar * ar + ai * ai
                b_mag2 = br * br + bi * bi
                if math.isnan(a_mag2) or math.isinf(a_mag2) or math.isnan(b_mag2) or math.isinf(b_mag2) or a_mag2 > 1e150 or b_mag2 > 1e150:
                    bla_table[d, k, 0] = 0.0
                    bla_table[d, k, 1] = 0.0
                    bla_table[d, k, 2] = 0.0
                    bla_table[d, k, 3] = 0.0
                    bla_table[d, k, 4] = 0.0
                    continue

                bla_table[d, k, 0] = ar
                bla_table[d, k, 1] = ai
                bla_table[d, k, 2] = br
                bla_table[d, k, 3] = bi

                # r_sq = min(r1_sq, 0.25 * r2_sq / |A1|^2) to account for B1*dc triangle inequality
                a1_mag2 = a1_r * a1_r + a1_i * a1_i
                if a1_mag2 > 1e-300:
                    r2_scaled_sq = (0.25 * r2_sq) / a1_mag2
                    r_cand = r1_sq if r1_sq < r2_scaled_sq else r2_scaled_sq
                else:
                    r_cand = r1_sq

                if math.isnan(r_cand) or math.isinf(r_cand) or r_cand <= 0.0:
                    bla_table[d, k, 4] = 0.0
                elif r_cand > 0.25:
                    bla_table[d, k, 4] = 0.25
                else:
                    bla_table[d, k, 4] = r_cand

        step = step_next

    return bla_table


_BLA_CACHE = {}
_BLA_CACHE_LOCK = threading.Lock()


def build_bla_table(ref_re, ref_im, ref_len, fractal_type=0, tol=1e-6, max_levels=16,
                    gen_n=3, gen_kr=0.25, gen_ki=1.0):
    """Precomputes dyadic Bilinear Approximation (BLA) leap tables for perturbation acceleration."""
    if fractal_type not in (0, 2, 4) or ref_len < 4 or tol <= 0.0:
        return np.zeros((1, 1, 1), dtype=np.float64)
    sample_pt = (float(ref_re[0]), float(ref_re[min(5, ref_len-1)]), float(ref_im[min(5, ref_len-1)])) if ref_len > 0 else (0.0, 0.0, 0.0)
    cache_key = (id(ref_re), sample_pt, int(ref_len), int(fractal_type), float(tol), int(max_levels), int(gen_n), float(gen_kr), float(gen_ki))
    with _BLA_CACHE_LOCK:
        if cache_key in _BLA_CACHE:
            cached_ref, cached_table = _BLA_CACHE[cache_key]
            if cached_ref is ref_re:
                return cached_table

    table = _jit_build_bla_table(
        ref_re, ref_im, int(ref_len), int(fractal_type), float(tol), int(max_levels),
        int(gen_n), float(gen_kr), float(gen_ki)
    )
    with _BLA_CACHE_LOCK:
        if len(_BLA_CACHE) > 16:
            _BLA_CACHE.clear()
        _BLA_CACHE[cache_key] = (ref_re, table)
    return table