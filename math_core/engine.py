from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, getcontext
import math
import os
from numba import cuda, set_num_threads
import numpy as np

from color_engine import get_cpu_cmap_lut, get_gpu_cmap_lut
from config import (
    GLOBAL_MAX_ITER,
    HAS_GMPY2,
    INV_LN2_F32,
    LN_BAILOUT_LOG,
    MAX_SYSTEM_CORES,
    RenderParameters,
    compute_floatexp_scale,
    compute_required_prec_bits,
    ensure_decimal_precision,
)
from math_core.cpu_kernels import (
    cpu_build_histogram,
    cpu_compute_dd_sample,
    cpu_compute_fp64_sample,
    cpu_compute_perturbation_sample,
    cpu_compute_perturbation_sample_floatexp,
    cpu_dd_add,
    cpu_dd_mul_f64,
    cpu_dd_sub,
    cpu_palette_norm,
    cpu_probe_iter_dd_kernel,
    cpu_probe_iter_floatexp_kernel,
    cpu_probe_iter_kernel,
    cpu_probe_iter_perturbation_kernel,
    cpu_render_dd_hist_kernel,
    cpu_render_dd_kernel,
    cpu_render_floatexp_hist_kernel,
    cpu_render_floatexp_kernel,
    cpu_render_fp64_hist_kernel,
    cpu_render_fp64_kernel,
    cpu_render_perturbation_hist_kernel,
    cpu_render_perturbation_kernel,
    cpu_recolor_hist_kernel,
    cpu_recolor_iter_map_kernel,
    cpu_sample_cdf_norm,
    cpu_sample_lut_lerp,
)
from math_core.cuda_kernels import (
    HAS_CUDA,
    ML_LIMBS,
    apply_histogram_equalization_kernel,
    recolor_iter_map_cuda_kernel,
    build_histogram_kernel,
    compute_bbsa_cuda_kernel,
    compute_perturbation_sample_floatexp,
    decimal_to_limbs,
    edge_detect_kernel,
    eval_glitched_pixels_cuda_kernel,
    eval_glitched_pixels_hist_cuda_kernel,
    finalize_normalized_cdf_cuda_kernel,
    mandelbrot_unified_dd_kernel,
    mandelbrot_unified_floatexp_kernel,
    mandelbrot_unified_fp64_kernel,
    mandelbrot_unified_perturbation_kernel,
    probe_iter_dd_kernel,
    probe_iter_floatexp_kernel,
    probe_iter_fp64_kernel,
    probe_iter_perturbation_kernel,
    scan_blocks_cuda_kernel,
    scan_single_block_cuda_kernel,
    ssaa_resolve_hist_dd_kernel,
    ssaa_resolve_hist_floatexp_kernel,
    ssaa_resolve_hist_fp64_kernel,
    ssaa_resolve_hist_perturbation_kernel,
    zero_histogram_cuda_kernel,
)
from math_core.high_precision import (
    build_bla_table,
    cache_reference_orbit,
    compute_bbsa_coeffs,
    compute_highprec_escape_scalar,
    compute_single_highprec_orbit,
    find_optimal_reference_orbit,
    to_dd,
)

import threading

try:
    import gmpy2
except ImportError:
    gmpy2 = None


# --- GPU Scratch Buffer Pool ---
_SCRATCH_POOL = {}
_POOL_LOCK = threading.Lock()
_DUMMY_EDGE = None
_DUMMY_ITER = None

# --- Stream Slot Registry ---
_STREAM_SLOT_MAP = {}
_STREAM_SLOT_LOCK = threading.Lock()
_NEXT_STREAM_SLOT = 1


def get_stream_slot(stream, explicit_slot=None):
    """Resolves a thread-safe, distinct integer slot ID for per-stream scratch arrays."""
    if explicit_slot is not None and explicit_slot >= 0:
        return int(explicit_slot)
    if stream == 0 or stream is None:
        return 0
    with _STREAM_SLOT_LOCK:
        slot = _STREAM_SLOT_MAP.get(id(stream))
        if slot is None:
            global _NEXT_STREAM_SLOT
            slot = _NEXT_STREAM_SLOT
            _NEXT_STREAM_SLOT = (_NEXT_STREAM_SLOT % 16) + 1
            _STREAM_SLOT_MAP[id(stream)] = slot
        return slot



def get_scratch_device_array(shape, dtype, tag="scratch"):
    """Thread-safe retrieval of pooled GPU scratch arrays to eliminate per-frame cudaMalloc overhead."""
    if not HAS_CUDA:
        return None
    shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
    key = (tag, shape_tuple, np.dtype(dtype).str)
    with _POOL_LOCK:
        buf = _SCRATCH_POOL.get(key)
        if buf is None:
            buf = cuda.device_array(shape_tuple, dtype=dtype)
            _SCRATCH_POOL[key] = buf
        return buf


def get_scratch_device_array_1d(min_size, dtype, tag="scratch_1d"):
    """Thread-safe retrieval of a pooled 1D GPU scratch array with capacity >= min_size to eliminate per-frame cudaMalloc overhead."""
    if not HAS_CUDA:
        return None
    key = (tag, np.dtype(dtype).str)
    with _POOL_LOCK:
        buf = _SCRATCH_POOL.get(key)
        if buf is None or buf.size < min_size:
            alloc_size = max(int(min_size * 1.5), 65536)
            buf = cuda.device_array((alloc_size,), dtype=dtype)
            _SCRATCH_POOL[key] = buf
        return buf


def get_dummy_edge_buffer():
    """Global singleton (1, 1) uint8 dummy edge array for CUDA kernels."""
    global _DUMMY_EDGE
    if _DUMMY_EDGE is None and HAS_CUDA:
        with _POOL_LOCK:
            if _DUMMY_EDGE is None:
                _DUMMY_EDGE = cuda.device_array((1, 1), dtype=np.uint8)
    return _DUMMY_EDGE


def get_dummy_iter_buffer():
    """Global singleton (1, 1) float32 dummy iter array for CUDA kernels."""
    global _DUMMY_ITER
    if _DUMMY_ITER is None and HAS_CUDA:
        with _POOL_LOCK:
            if _DUMMY_ITER is None:
                _DUMMY_ITER = cuda.device_array((1, 1), dtype=np.float32)
    return _DUMMY_ITER


def clear_scratch_buffer_pool():
    """Clears all cached GPU scratch buffers to free VRAM when needed."""
    global _DUMMY_EDGE, _DUMMY_ITER, _NEXT_STREAM_SLOT, _LAST_COPIED_BLA_MAP
    with _POOL_LOCK:
        _SCRATCH_POOL.clear()
        _DUMMY_EDGE = None
        _DUMMY_ITER = None
    with _STREAM_SLOT_LOCK:
        _STREAM_SLOT_MAP.clear()
        _NEXT_STREAM_SLOT = 1
    _LAST_COPIED_BLA_MAP.clear()
    if HAS_CUDA:
        try:
            cuda.synchronize()
        except Exception:
            pass
    import gc
    gc.collect()


def eval_glitched_pixels_fallback(out_ary, glitched_indices, target_w, target_h, max_iter,
                                  center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
                                  cmap_name, color_scheme_id, fractal_type, julia_cx, julia_cy,
                                  gen_n=3, gen_kr=0.25, gen_ki=1.0, is_hdr=False, palette_offset=0.0,
                                  color_density=1.0, color_contrast=1.0,
                                  cdf_lut=None, num_bins=2048, iter_map=None):
    """Shared multi-threaded arbitrary-precision fallback evaluator for glitched pixels in Strict Metric mode."""
    if len(glitched_indices) == 0:
        return out_ary

    ensure_decimal_precision(plot_width_dec)
    lut_cpu = get_cpu_cmap_lut(cmap_name, lut_size=2048, is_hdr=is_hdr)
    bits_needed = compute_required_prec_bits(plot_width_dec)
    x_min_dec = center_x_dec - (plot_width_dec / Decimal(2))
    y_max_dec = center_y_dec + (plot_height_dec / Decimal(2))
    inv_w = Decimal(1) / Decimal(max(1, target_w - 1))
    inv_h = Decimal(1) / Decimal(max(1, target_h - 1))

    if HAS_GMPY2:
        with gmpy2.local_context(precision=bits_needed):
            x_min_mp = gmpy2.mpfr(str(x_min_dec))
            y_max_mp = gmpy2.mpfr(str(y_max_dec))
            dx_mp = gmpy2.mpfr(str(plot_width_dec * inv_w))
            dy_mp = gmpy2.mpfr(str(plot_height_dec * inv_h))
            j_cx_mp = gmpy2.mpfr(str(julia_cx))
            j_cy_mp = gmpy2.mpfr(str(julia_cy))

        def eval_px(coord):
            r, c = coord
            with gmpy2.local_context(precision=bits_needed):
                px_cx = x_min_mp + gmpy2.mpfr(int(c)) * dx_mp
                px_cy = y_max_mp - gmpy2.mpfr(int(r)) * dy_mp
                surv = compute_highprec_escape_scalar(
                    px_cx, px_cy, max_iter, fractal_type, j_cx_mp, j_cy_mp, bits_needed,
                    gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
                )
                return r, c, surv
    else:
        dx_dec = plot_width_dec * inv_w
        dy_dec = plot_height_dec * inv_h

        def eval_px(coord):
            r, c = coord
            orig_prec = getcontext().prec
            getcontext().prec = int(bits_needed / 3.32) + 30
            try:
                px_cx = x_min_dec + Decimal(int(c)) * dx_dec
                px_cy = y_max_dec - Decimal(int(r)) * dy_dec
                surv = compute_highprec_escape_scalar(
                    px_cx, px_cy, max_iter, fractal_type, julia_cx, julia_cy, bits_needed,
                    gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
                )
                return r, c, surv
            finally:
                getcontext().prec = orig_prec

    workers = min(MAX_SYSTEM_CORES, max(1, len(glitched_indices)))
    chunksize = max(1, len(glitched_indices) // (workers * 4))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(eval_px, glitched_indices, chunksize=chunksize)
        for r, c, surv in results:
            if iter_map is not None:
                iter_map[r, c] = float(surv)
            if surv >= max_iter or surv < 0:
                out_ary[r, c, 0] = 0
                out_ary[r, c, 1] = 0
                out_ary[r, c, 2] = 0
            else:
                if cdf_lut is not None and color_scheme_id == 11:
                    norm = cpu_sample_cdf_norm(float(surv), max_iter, cdf_lut, num_bins, palette_offset, color_density, color_contrast)
                else:
                    norm = cpu_palette_norm(float(surv), color_scheme_id, palette_offset, color_density, color_contrast)
                sr, sg, sb = cpu_sample_lut_lerp(lut_cpu, norm)
                out_ary[r, c, 0] = int(sr + 0.5)
                out_ary[r, c, 1] = int(sg + 0.5)
                out_ary[r, c, 2] = int(sb + 0.5)
    return out_ary


def eval_glitched_pixels_fallback_cuda(
    d_out_rgb, d_glitch_out, d_lut, glitched_indices,
    target_w, target_h, max_iter,
    center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
    color_scheme_id, fractal_type, julia_cx, julia_cy,
    gen_n=3, gen_kr=0.25, gen_ki=1.0, palette_offset=0.0,
    color_density=1.0, color_contrast=1.0,
    d_cdf_lut=None, num_bins=2048, cmap_name="inferno", is_hdr=False, stream=0,
    d_iter_map=None, stream_slot=0
):
    """GPU-accelerated exhaustive fallback evaluator for glitched pixels in Strict Metric mode."""
    num_glitches = len(glitched_indices)
    if num_glitches == 0:
        return

    if plot_width_dec >= Decimal("1e-12"):
        precision_tier = 0
    elif plot_width_dec >= Decimal("1e-30"):
        precision_tier = 1
    elif plot_width_dec >= Decimal("1e-315"):
        precision_tier = 2
    else:
        # Multi-limb GPU kernel supports up to 34 limbs (1088 bits ~ 317.8 decimal digits).
        # For zoom depths beyond 315 digits (including FloatExp perturbation zoom depths),
        # fall back to multi-threaded arbitrary precision CPU evaluator.
        precision_tier = 3

    x_min_dec = center_x_dec - (plot_width_dec / Decimal(2))
    y_max_dec = center_y_dec + (plot_height_dec / Decimal(2))
    dx_dec = plot_width_dec / Decimal(max(1, target_w - 1))
    dy_dec = plot_height_dec / Decimal(max(1, target_h - 1))

    x_min_hi, x_min_lo = to_dd(x_min_dec)
    y_max_hi, y_max_lo = to_dd(y_max_dec)
    dx_hi, dx_lo = to_dd(dx_dec)
    dy_hi, dy_lo = to_dd(dy_dec)
    j_cx_hi, j_cx_lo = to_dd(julia_cx)
    j_cy_hi, j_cy_lo = to_dd(julia_cy)

    h_coords = np.ascontiguousarray(glitched_indices.astype(np.int32))
    d_coords = get_scratch_device_array((num_glitches, 2), np.int32, f"fallback_coords_{stream_slot}")
    d_coords.copy_to_device(h_coords, stream=stream)

    if precision_tier == 3 or (precision_tier == 2 and fractal_type == 4):
        # CPU fallback for General Mandelbrot or ultra-deep zooms > 250 digits
        h_out_rgb = d_out_rgb.copy_to_host(stream=stream)
        h_cdf_lut = d_cdf_lut.copy_to_host(stream=stream) if d_cdf_lut is not None else None
        h_iter_map = d_iter_map.copy_to_host(stream=stream) if d_iter_map is not None else None
        _cuda_sync_stream(stream)
        eval_glitched_pixels_fallback(
            h_out_rgb, glitched_indices, target_w, target_h, max_iter,
            center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
            cmap_name, color_scheme_id, fractal_type, julia_cx, julia_cy,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_hdr=is_hdr, palette_offset=palette_offset,
            color_density=color_density, color_contrast=color_contrast,
            cdf_lut=h_cdf_lut,
            num_bins=num_bins,
            iter_map=h_iter_map
        )
        d_out_rgb.copy_to_device(h_out_rgb, stream=stream)
        if d_iter_map is not None and h_iter_map is not None:
            d_iter_map.copy_to_device(h_iter_map, stream=stream)
        if d_glitch_out is not None:
            d_glitch_out.copy_to_device(np.zeros((target_h, target_w), dtype=np.uint8), stream=stream)
        _cuda_sync_stream(stream)
        return

    if precision_tier == 2:

        h_cx_signs = np.zeros(num_glitches, dtype=np.int32)
        h_cx_limbs = np.zeros((num_glitches, ML_LIMBS), dtype=np.uint32)
        h_cy_signs = np.zeros(num_glitches, dtype=np.int32)
        h_cy_limbs = np.zeros((num_glitches, ML_LIMBS), dtype=np.uint32)

        for idx in range(num_glitches):
            r, c = glitched_indices[idx]
            px_cx = x_min_dec + Decimal(int(c)) * dx_dec
            px_cy = y_max_dec - Decimal(int(r)) * dy_dec
            s_x, l_x = decimal_to_limbs(px_cx, ML_LIMBS)
            s_y, l_y = decimal_to_limbs(px_cy, ML_LIMBS)
            h_cx_signs[idx] = s_x
            h_cx_limbs[idx] = l_x
            h_cy_signs[idx] = s_y
            h_cy_limbs[idx] = l_y

        d_cx_signs = get_scratch_device_array((num_glitches,), np.int32, f"fallback_cx_signs_{stream_slot}")
        d_cx_signs.copy_to_device(h_cx_signs, stream=stream)
        d_cx_limbs = get_scratch_device_array((num_glitches, ML_LIMBS), np.uint32, f"fallback_cx_limbs_{stream_slot}")
        d_cx_limbs.copy_to_device(h_cx_limbs, stream=stream)
        d_cy_signs = get_scratch_device_array((num_glitches,), np.int32, f"fallback_cy_signs_{stream_slot}")
        d_cy_signs.copy_to_device(h_cy_signs, stream=stream)
        d_cy_limbs = get_scratch_device_array((num_glitches, ML_LIMBS), np.uint32, f"fallback_cy_limbs_{stream_slot}")
        d_cy_limbs.copy_to_device(h_cy_limbs, stream=stream)

        j_cx_sign, j_cx_limbs = decimal_to_limbs(Decimal(str(julia_cx)), ML_LIMBS)
        j_cy_sign, j_cy_limbs = decimal_to_limbs(Decimal(str(julia_cy)), ML_LIMBS)
        d_j_cx_limbs = get_scratch_device_array((ML_LIMBS,), np.uint32, f"fallback_j_cx_limbs_{stream_slot}")
        d_j_cx_limbs.copy_to_device(j_cx_limbs, stream=stream)
        d_j_cy_limbs = get_scratch_device_array((ML_LIMBS,), np.uint32, f"fallback_j_cy_limbs_{stream_slot}")
        d_j_cy_limbs.copy_to_device(j_cy_limbs, stream=stream)
    else:
        d_cx_signs = get_scratch_device_array((1,), np.int32, f"dummy_cx_signs_{stream_slot}")
        d_cx_limbs = get_scratch_device_array((1, ML_LIMBS), np.uint32, f"dummy_cx_limbs_{stream_slot}")
        d_cy_signs = get_scratch_device_array((1,), np.int32, f"dummy_cy_signs_{stream_slot}")
        d_cy_limbs = get_scratch_device_array((1, ML_LIMBS), np.uint32, f"dummy_cy_limbs_{stream_slot}")
        j_cx_sign = 0
        j_cy_sign = 0
        d_j_cx_limbs = get_scratch_device_array((ML_LIMBS,), np.uint32, f"dummy_j_cx_limbs_{stream_slot}")
        d_j_cy_limbs = get_scratch_device_array((ML_LIMBS,), np.uint32, f"dummy_j_cy_limbs_{stream_slot}")

    threads_per_block = min(256, max(32, num_glitches))
    blocks = (num_glitches + threads_per_block - 1) // threads_per_block
    d_active_fallback_iter = d_iter_map if d_iter_map is not None else get_scratch_device_array((target_h, target_w), np.float32, f"fallback_dummy_iter_{stream_slot}")

    if d_cdf_lut is not None:
        eval_glitched_pixels_hist_cuda_kernel[blocks, threads_per_block, stream](
            d_out_rgb, d_glitch_out, d_active_fallback_iter, d_lut, d_cdf_lut, num_bins,
            d_coords, num_glitches,
            max_iter, fractal_type, float(palette_offset),
            precision_tier,
            x_min_hi, x_min_lo, y_max_hi, y_max_lo,
            dx_hi, dx_lo, dy_hi, dy_lo,
            j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo,
            int(gen_n), float(gen_kr), float(gen_ki),
            d_cx_signs, d_cx_limbs, d_cy_signs, d_cy_limbs,
            j_cx_sign, d_j_cx_limbs, j_cy_sign, d_j_cy_limbs,
            float(color_density), float(color_contrast)
        )
    else:
        eval_glitched_pixels_cuda_kernel[blocks, threads_per_block, stream](
            d_out_rgb, d_glitch_out, d_active_fallback_iter, d_lut, d_coords, num_glitches,
            max_iter, fractal_type, color_scheme_id, float(palette_offset),
            precision_tier,
            x_min_hi, x_min_lo, y_max_hi, y_max_lo,
            dx_hi, dx_lo, dy_hi, dy_lo,
            j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo,
            int(gen_n), float(gen_kr), float(gen_ki),
            d_cx_signs, d_cx_limbs, d_cy_signs, d_cy_limbs,
            j_cx_sign, d_j_cx_limbs, j_cy_sign, d_j_cy_limbs,
            float(color_density), float(color_contrast)
        )


_LAST_COPIED_BLA_MAP = {}


def upload_bla_table_cuda(bla_table, stream_slot, stream=0):
    """Uploads BLA table to device memory, skipping redundant PCIe transfers if the table in RAM is unchanged."""
    d_bla_table = get_scratch_device_array(bla_table.shape, np.float64, f"bla_table_{stream_slot}")
    cached = _LAST_COPIED_BLA_MAP.get(stream_slot)
    if cached is None or cached[0] is not bla_table or cached[1] != bla_table.shape:
        d_bla_table.copy_to_device(bla_table, stream=stream)
        _LAST_COPIED_BLA_MAP[stream_slot] = (bla_table, bla_table.shape)
    return d_bla_table


def compute_bbsa_coeffs_cuda(
    d_ref_re, d_ref_im, ref_len, max_r, fractal_type,
    bbsa_tol=1e-4, bbsa_order=4, gen_n=3, gen_kr=0.25, gen_ki=1.0, E_scale=0, stream=0,
    h_ref_re=None, h_ref_im=None
):
    """Computes BBSA Taylor series recurrence coefficients and skip count on CPU or GPU."""
    if fractal_type not in (0, 2, 4) or bbsa_tol <= 0.0 or bbsa_order <= 0 or ref_len < 2 or max_r <= 0.0:
        return (0,) + (0.0,) * 16

    if bbsa_order == 32:
        return (0,) + (0.0,) * 16

    if h_ref_re is not None and h_ref_im is not None:
        return compute_bbsa_coeffs(
            h_ref_re, h_ref_im, ref_len, max_r, fractal_type,
            bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki,
            E_scale=E_scale
        )

    try:
        h_re = d_ref_re.copy_to_host(stream=stream)
        h_im = d_ref_im.copy_to_host(stream=stream)
        _cuda_sync_stream(stream)
        return compute_bbsa_coeffs(
            h_re, h_im, ref_len, max_r, fractal_type,
            bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki,
            E_scale=E_scale
        )
    except Exception:
        d_bbsa_res = get_scratch_device_array((17,), np.float64, "bbsa_res")
        compute_bbsa_cuda_kernel[1, 1, stream](
            d_ref_re, d_ref_im, int(ref_len), float(max_r), int(fractal_type),
            float(bbsa_tol), int(bbsa_order), int(gen_n), float(gen_kr), float(gen_ki),
            int(E_scale),
            d_bbsa_res
        )
        h_bbsa_res = d_bbsa_res.copy_to_host(stream=stream)
        _cuda_sync_stream(stream)
        best_skip = int(h_bbsa_res[0])
        bbsa_coeffs = tuple(float(x) for x in h_bbsa_res[1:17])
        return (best_skip,) + bbsa_coeffs


def render_frame_cpu(target_w, target_h=None, max_iter=None, center_x_dec=None, center_y_dec=None, plot_width_dec=None,
                     cmap_name=None, ssaa_factor=1, edge_threshold=0.35, color_scheme_id=0,
                     precision_mode="1e-300 (Perturbation)", fractal_type=0, julia_cx=0.0, julia_cy=0.0,
                     bbsa_tol=1e-4, bbsa_order=4, glitch_mode="Off (Legacy Single-Ref)",
                     gen_n=3, gen_kr=0.25, gen_ki=1.0,
                     h_rgb_buf=None, is_hdr=False, cpu_threads=2, palette_offset=0.0,
                     color_density=1.0, color_contrast=1.0, h_iter_buf=None):
    """Renders a single frame using the multi-core CPU Numba engine supporting FP64, DD, and Perturbation."""
    if isinstance(target_w, RenderParameters):
        p = target_w
        return render_frame_cpu(
            p.target_w, p.target_h, p.max_iter, p.center_x_dec, p.center_y_dec, p.plot_width_dec,
            p.cmap_name, ssaa_factor=p.ssaa_factor, edge_threshold=p.edge_threshold, color_scheme_id=p.color_scheme_id,
            precision_mode=p.precision_mode, fractal_type=p.fractal_type, julia_cx=p.julia_cx, julia_cy=p.julia_cy,
            bbsa_tol=p.bbsa_tol, bbsa_order=p.bbsa_order, glitch_mode=p.glitch_mode,
            gen_n=p.gen_n, gen_kr=p.gen_kr, gen_ki=p.gen_ki,
            h_rgb_buf=p.h_rgb_buf if h_rgb_buf is None else h_rgb_buf,
            is_hdr=p.is_hdr, cpu_threads=p.cpu_threads, palette_offset=p.palette_offset,
            color_density=p.color_density, color_contrast=p.color_contrast,
            h_iter_buf=getattr(p, 'h_iter_buf', h_iter_buf)
        )
    set_num_threads(max(1, min(MAX_SYSTEM_CORES, int(cpu_threads))))
    factor = max(1, int(ssaa_factor))
    aspect_ratio = Decimal(target_h) / Decimal(target_w)
    plot_height_dec = plot_width_dec * aspect_ratio

    dtype_val = np.uint16 if is_hdr else np.uint8
    if h_rgb_buf is not None and h_rgb_buf.shape == (target_h, target_w, 3) and h_rgb_buf.dtype == dtype_val:
        out_rgb = h_rgb_buf
    else:
        out_rgb = np.empty((target_h, target_w, 3), dtype=dtype_val)
    if h_iter_buf is not None and h_iter_buf.shape == (target_h, target_w) and h_iter_buf.dtype == np.float32:
        active_iter = h_iter_buf
    else:
        active_iter = np.empty((target_h, target_w), dtype=np.float32)
    lut = get_cpu_cmap_lut(cmap_name, lut_size=2048, is_hdr=is_hdr)

    dx_f = float(plot_width_dec / Decimal(max(1, target_w - 1)))
    dy_f = float(plot_height_dec / Decimal(max(1, target_h - 1)))
    dx_sub_f = dx_f / float(factor)
    dy_sub_f = dy_f / float(factor)

    ensure_decimal_precision(plot_width_dec)
    strict_glitch_flag = 1 if ("Auto Rebasing" in glitch_mode or "Strict" in glitch_mode or "Pauldelbrot" in glitch_mode) else 0
    use_floatexp = (plot_width_dec < Decimal("1e-300"))
    use_fp64 = (plot_width_dec >= Decimal("1e-12")) and (not use_floatexp)
    use_dd = (plot_width_dec >= Decimal("1e-24") and "dd" in str(precision_mode).lower()) and (not use_floatexp)
    if use_dd:
        use_fp64 = False
    use_perturbation = (not use_fp64) and (not use_dd) and (not use_floatexp)

    if use_floatexp:
        E_scale = compute_floatexp_scale(plot_width_dec)
        scale_factor_dec = Decimal(2) ** E_scale
        norm_width_dec = plot_width_dec * scale_factor_dec
        norm_height_dec = plot_height_dec * scale_factor_dec
        w_norm = float(norm_width_dec)
        dx_norm_f = float(norm_width_dec / Decimal(max(1, target_w - 1)))
        dy_norm_f = float(norm_height_dec / Decimal(max(1, target_h - 1)))
        dx_sub_norm_f = dx_norm_f / float(factor)
        dy_sub_norm_f = dy_norm_f / float(factor)

    cdf_lut = None
    num_bins = 2048

    if color_scheme_id == 11:
        num_bins = 65536
        iter_map = active_iter
        out_glitch = np.zeros((target_h, target_w), dtype=np.uint8)
        in_glitch = np.zeros((target_h, target_w), dtype=np.uint8)
        sub_offset = (float(factor) - 1.0) * 0.5
        if use_fp64:
            w_f, h_f = float(plot_width_dec), float(plot_height_dec)
            cx, cy = float(center_x_dec), float(center_y_dec)
            x_min = cx - (w_f * 0.5)
            y_max = cy + (h_f * 0.5)
            dx_sub = dx_sub_f
            dy_sub = dy_sub_f
            for j in range(target_h):
                y = y_max - (float(j * factor) + sub_offset) * dy_sub
                for i in range(target_w):
                    x = x_min + (float(i * factor) + sub_offset) * dx_sub
                    iter_map[j, i] = cpu_compute_fp64_sample(x, y, max_iter, fractal_type, float(julia_cx), float(julia_cy), int(gen_n), float(gen_kr), float(gen_ki), INV_LN2_F32, LN_BAILOUT_LOG)
        elif use_dd:
            x_min_hi, x_min_lo = to_dd(center_x_dec - (plot_width_dec / Decimal(2)))
            y_max_hi, y_max_lo = to_dd(center_y_dec + (plot_height_dec / Decimal(2)))
            dx_hi, dx_lo = to_dd((plot_width_dec / Decimal(max(1, target_w - 1))) / Decimal(factor))
            dy_hi, dy_lo = to_dd((plot_height_dec / Decimal(max(1, target_h - 1))) / Decimal(factor))
            j_cx_hi, j_cx_lo = to_dd(julia_cx)
            j_cy_hi, j_cy_lo = to_dd(julia_cy)
            for j in range(target_h):
                sub_j = float(j * factor) + sub_offset
                jdy_hi, jdy_lo = cpu_dd_mul_f64(dy_hi, dy_lo, sub_j)
                cy_hi, cy_lo = cpu_dd_sub(y_max_hi, y_max_lo, jdy_hi, jdy_lo)
                for i in range(target_w):
                    sub_i = float(i * factor) + sub_offset
                    idx_hi, idx_lo = cpu_dd_mul_f64(dx_hi, dx_lo, sub_i)
                    cx_hi, cx_lo = cpu_dd_add(x_min_hi, x_min_lo, idx_hi, idx_lo)
                    iter_map[j, i] = cpu_compute_dd_sample(cx_hi, cx_lo, cy_hi, cy_lo, max_iter, fractal_type, j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo, int(gen_n), float(gen_kr), float(gen_ki), INV_LN2_F32, LN_BAILOUT_LOG)
        elif use_floatexp:
            best_cx, best_cy, best_u, best_v, ref_re, ref_im, ref_len = find_optimal_reference_orbit(
                center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
                max_iter, target_w, target_h, fractal_type=fractal_type, julia_cx=julia_cx, julia_cy=julia_cy,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_floatexp=True
            )
            max_r_norm = float(w_norm) * 0.75
            bbsa_inv_r = 1.0 / max_r_norm if max_r_norm > 0.0 else 1.0
            bbsa_res = compute_bbsa_coeffs(
                ref_re, ref_im, ref_len, max_r_norm, fractal_type,
                bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki,
                E_scale=E_scale
            )
            bbsa_skip = bbsa_res[0]
            bbsa_coeffs = bbsa_res[1:17]
            ref_px_sub = float(target_w - 1) * float(best_u) * float(factor)
            ref_py_sub = float(target_h - 1) * float(best_v) * float(factor)
            ref_cx_f = float(best_cx)
            ref_cy_f = float(best_cy)
            for j in range(target_h):
                sub_j = float(j * factor) + sub_offset
                for i in range(target_w):
                    sub_i = float(i * factor) + sub_offset
                    s, is_gl = cpu_compute_perturbation_sample_floatexp(
                        ref_re, ref_im, ref_len, max_iter,
                        ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub_norm_f, dy_sub_norm_f,
                        E_scale, ref_cx_f, ref_cy_f, fractal_type, float(julia_cx), float(julia_cy),
                        bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                        int(gen_n), float(gen_kr), float(gen_ki), strict_glitch_flag,
                        INV_LN2_F32, LN_BAILOUT_LOG
                    )
                    iter_map[j, i] = s
                    out_glitch[j, i] = 1 if is_gl > 0 else 0
        else:
            best_cx, best_cy, best_u, best_v, ref_re, ref_im, ref_len = find_optimal_reference_orbit(
                center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
                max_iter, target_w, target_h, fractal_type=fractal_type, julia_cx=julia_cx, julia_cy=julia_cy,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_floatexp=False
            )
            max_r = float(plot_width_dec) * 0.75
            bbsa_inv_r = 1.0 / max_r if max_r > 0.0 else 1.0
            bbsa_res = compute_bbsa_coeffs(
                ref_re, ref_im, ref_len, max_r, fractal_type,
                bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki,
                E_scale=0
            )
            bbsa_skip = bbsa_res[0]
            bbsa_coeffs = bbsa_res[1:17]
            ref_px_sub = float(target_w - 1) * float(best_u) * float(factor)
            ref_py_sub = float(target_h - 1) * float(best_v) * float(factor)
            ref_cx_f = float(best_cx)
            ref_cy_f = float(best_cy)
            for j in range(target_h):
                sub_j = float(j * factor) + sub_offset
                for i in range(target_w):
                    sub_i = float(i * factor) + sub_offset
                    s, is_gl = cpu_compute_perturbation_sample(
                        ref_re, ref_im, ref_len, max_iter,
                        ref_px_sub, ref_py_sub, sub_i, sub_j, dx_sub_f, dy_sub_f,
                        ref_cx_f, ref_cy_f, fractal_type, float(julia_cx), float(julia_cy),
                        bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                        int(gen_n), float(gen_kr), float(gen_ki), strict_glitch_flag,
                        INV_LN2_F32, LN_BAILOUT_LOG
                    )
                    iter_map[j, i] = s
                    out_glitch[j, i] = 1 if is_gl > 0 else 0

        hist = np.zeros(num_bins, dtype=np.int64)
        cpu_build_histogram(iter_map, hist, target_w, target_h, max_iter, num_bins)
        cdf = np.cumsum(hist, dtype=np.float64)
        total_esc = cdf[-1]
        cdf_lut = (cdf / total_esc).astype(np.float32) if total_esc > 0 else np.linspace(0.0, 1.0, num_bins, dtype=np.float32)

        if use_fp64:
            cpu_render_fp64_hist_kernel(
                out_rgb, iter_map, cdf_lut, lut, target_w, target_h, max_iter,
                x_min, y_max, dx_sub, dy_sub, factor, fractal_type,
                float(julia_cx), float(julia_cy), int(gen_n), float(gen_kr), float(gen_ki),
                float(edge_threshold), INV_LN2_F32, LN_BAILOUT_LOG, num_bins, float(palette_offset),
                float(color_density), float(color_contrast)
            )
        elif use_dd:
            cpu_render_dd_hist_kernel(
                out_rgb, iter_map, cdf_lut, lut, target_w, target_h, max_iter,
                x_min_hi, x_min_lo, dx_hi, dx_lo,
                y_max_hi, y_max_lo, dy_hi, dy_lo,
                factor, fractal_type,
                j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo,
                int(gen_n), float(gen_kr), float(gen_ki), float(edge_threshold),
                INV_LN2_F32, LN_BAILOUT_LOG, num_bins, float(palette_offset),
                float(color_density), float(color_contrast)
            )
        elif use_floatexp:
            cpu_render_floatexp_hist_kernel(
                out_rgb, out_glitch, iter_map, cdf_lut, lut, target_w, target_h, max_iter,
                ref_re, ref_im, ref_len,
                ref_px_sub, ref_py_sub, dx_sub_norm_f, dy_sub_norm_f, E_scale,
                ref_cx_f, ref_cy_f, factor, fractal_type,
                float(julia_cx), float(julia_cy),
                bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                0, in_glitch, int(gen_n), float(gen_kr), float(gen_ki), strict_glitch_flag,
                float(edge_threshold), INV_LN2_F32, LN_BAILOUT_LOG, num_bins, float(palette_offset),
                float(color_density), float(color_contrast)
            )
        else:
            strict_glitch_flag = 1 if ("Auto Rebasing" in glitch_mode or "Strict" in glitch_mode or "Pauldelbrot" in glitch_mode) else 0
            cpu_render_perturbation_hist_kernel(
                out_rgb, out_glitch, iter_map, cdf_lut, lut, target_w, target_h, max_iter,
                ref_re, ref_im, ref_len,
                ref_px_sub, ref_py_sub, dx_sub_f, dy_sub_f,
                ref_cx_f, ref_cy_f, factor, fractal_type,
                float(julia_cx), float(julia_cy),
                bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                0, in_glitch, int(gen_n), float(gen_kr), float(gen_ki), strict_glitch_flag,
                float(edge_threshold), INV_LN2_F32, LN_BAILOUT_LOG, num_bins, float(palette_offset),
                float(color_density), float(color_contrast)
            )
        allow_rebasing = ("Auto Rebasing" in glitch_mode or "Strict" in glitch_mode or "Pauldelbrot" in glitch_mode)
        if not (allow_rebasing and (use_floatexp or use_perturbation) and (plot_width_dec < Decimal("1e-12"))):
            return out_rgb
    elif use_fp64:
        w_f, h_f = float(plot_width_dec), float(plot_height_dec)
        cx, cy = float(center_x_dec), float(center_y_dec)
        x_min = cx - (w_f * 0.5)
        y_max = cy + (h_f * 0.5)
        dx_sub = dx_sub_f
        dy_sub = dy_sub_f

        cpu_render_fp64_kernel(
            out_rgb, active_iter, lut, target_w, target_h, max_iter,
            x_min, y_max, dx_sub, dy_sub, factor, color_scheme_id,
            fractal_type, float(julia_cx), float(julia_cy), int(gen_n), float(gen_kr), float(gen_ki),
            float(edge_threshold), INV_LN2_F32, LN_BAILOUT_LOG, float(palette_offset),
            float(color_density), float(color_contrast)
        )
    elif use_dd:
        x_min_hi, x_min_lo = to_dd(center_x_dec - (plot_width_dec / Decimal(2)))
        y_max_hi, y_max_lo = to_dd(center_y_dec + (plot_height_dec / Decimal(2)))
        dx_hi, dx_lo = to_dd((plot_width_dec / Decimal(max(1, target_w - 1))) / Decimal(factor))
        dy_hi, dy_lo = to_dd((plot_height_dec / Decimal(max(1, target_h - 1))) / Decimal(factor))
        j_cx_hi, j_cx_lo = to_dd(julia_cx)
        j_cy_hi, j_cy_lo = to_dd(julia_cy)

        cpu_render_dd_kernel(
            out_rgb, active_iter, lut, target_w, target_h, max_iter,
            x_min_hi, x_min_lo, dx_hi, dx_lo,
            y_max_hi, y_max_lo, dy_hi, dy_lo,
            factor, color_scheme_id, fractal_type,
            j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo,
            int(gen_n), float(gen_kr), float(gen_ki),
            float(edge_threshold), INV_LN2_F32, LN_BAILOUT_LOG, float(palette_offset),
            float(color_density), float(color_contrast)
        )
    elif use_floatexp:
        if bbsa_order in (4, 8):
            bbsa_order = 32
            if bbsa_tol > 1e-6:
                bbsa_tol = 1e-6
        best_cx, best_cy, best_u, best_v, ref_re, ref_im, ref_len = find_optimal_reference_orbit(
            center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
            max_iter, target_w, target_h, fractal_type=fractal_type, julia_cx=julia_cx, julia_cy=julia_cy,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_floatexp=True
        )
        max_r_norm = float(w_norm) * 0.75
        bbsa_inv_r = 1.0 / max_r_norm if max_r_norm > 0.0 else 1.0
        bbsa_res = compute_bbsa_coeffs(
            ref_re, ref_im, ref_len, max_r_norm, fractal_type,
            bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki,
            E_scale=E_scale
        )
        bbsa_skip = bbsa_res[0]
        bbsa_coeffs = bbsa_res[1:17]

        ref_px_sub = float(target_w - 1) * float(best_u) * float(factor)
        ref_py_sub = float(target_h - 1) * float(best_v) * float(factor)
        ref_cx_f = float(best_cx)
        ref_cy_f = float(best_cy)

        out_glitch = np.zeros((target_h, target_w), dtype=np.uint8)
        in_glitch = np.zeros((target_h, target_w), dtype=np.uint8)
        iter_map = active_iter

        if bbsa_order == 32:
            bla_table = build_bla_table(
                ref_re, ref_im, ref_len, fractal_type=fractal_type, tol=bbsa_tol,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
            )
        else:
            bla_table = np.zeros((1, 1, 1), dtype=np.float64)

        cpu_render_floatexp_kernel(
            out_rgb, out_glitch, active_iter, lut, target_w, target_h, max_iter,
            ref_re, ref_im, ref_len,
            ref_px_sub, ref_py_sub, dx_sub_norm_f, dy_sub_norm_f, E_scale,
            ref_cx_f, ref_cy_f, factor, color_scheme_id, fractal_type,
            float(julia_cx), float(julia_cy),
            bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
            0, in_glitch, int(gen_n), float(gen_kr), float(gen_ki), strict_glitch_flag,
            float(edge_threshold), INV_LN2_F32, LN_BAILOUT_LOG, float(palette_offset),
            float(color_density), float(color_contrast),
            bla_table=bla_table
        )
    else:
        best_cx, best_cy, best_u, best_v, ref_re, ref_im, ref_len = find_optimal_reference_orbit(
            center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
            max_iter, target_w, target_h, fractal_type=fractal_type, julia_cx=julia_cx, julia_cy=julia_cy,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_floatexp=False
        )
        max_r = float(plot_width_dec) * 0.75
        bbsa_inv_r = 1.0 / max_r if max_r > 0.0 else 1.0
        bbsa_res = compute_bbsa_coeffs(
            ref_re, ref_im, ref_len, max_r, fractal_type,
            bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki,
            E_scale=0
        )
        bbsa_skip = bbsa_res[0]
        bbsa_coeffs = bbsa_res[1:17]

        ref_px_sub = float(target_w - 1) * float(best_u) * float(factor)
        ref_py_sub = float(target_h - 1) * float(best_v) * float(factor)
        ref_cx_f = float(best_cx)
        ref_cy_f = float(best_cy)

        out_glitch = np.zeros((target_h, target_w), dtype=np.uint8)
        in_glitch = np.zeros((target_h, target_w), dtype=np.uint8)

        if bbsa_order == 32:
            bla_table = build_bla_table(
                ref_re, ref_im, ref_len, fractal_type=fractal_type, tol=bbsa_tol,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
            )
        else:
            bla_table = np.zeros((1, 1, 1), dtype=np.float64)

        cpu_render_perturbation_kernel(
            out_rgb, out_glitch, active_iter, lut, target_w, target_h, max_iter,
            ref_re, ref_im, ref_len,
            ref_px_sub, ref_py_sub, dx_sub_f, dy_sub_f,
            ref_cx_f, ref_cy_f, factor, color_scheme_id, fractal_type,
            float(julia_cx), float(julia_cy),
            bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
            0, in_glitch, int(gen_n), float(gen_kr), float(gen_ki),
            float(edge_threshold), strict_glitch_flag,
            INV_LN2_F32, LN_BAILOUT_LOG, float(palette_offset),
            float(color_density), float(color_contrast),
            bla_table=bla_table
        )

    is_auto_rebase = ("Auto Rebasing" in glitch_mode or "Strict" in glitch_mode or "Pauldelbrot" in glitch_mode)
    is_exhaustive = ("Strict" in glitch_mode)

    if (use_floatexp or use_perturbation) and is_auto_rebase:
        max_rebase_passes = 4
        curr_pass = 0
        init_r = int(np.clip(ref_py_sub / factor, 0, target_h - 1))
        init_c = int(np.clip(ref_px_sub / factor, 0, target_w - 1))
        visited_rebase_coords = {(init_r, init_c)}
        prev_glitch_count = None
        no_progress_passes = 0

        while curr_pass < max_rebase_passes:
            glitched_indices = np.argwhere(out_glitch > 0)
            num_glitches = len(glitched_indices)
            if num_glitches == 0:
                break
            if is_exhaustive and num_glitches <= 128:
                break
            if prev_glitch_count is not None and num_glitches >= prev_glitch_count:
                no_progress_passes += 1
                if no_progress_passes >= 2:
                    break
            else:
                no_progress_passes = 0
            prev_glitch_count = num_glitches

            curr_pass += 1
            in_glitch[:] = out_glitch[:]
            out_glitch.fill(0)

            # Cluster-targeted rebase point selection
            bin_r = np.clip((glitched_indices[:, 0] * 16) // target_h, 0, 15)
            bin_c = np.clip((glitched_indices[:, 1] * 16) // target_w, 0, 15)
            bin_idx = bin_r * 16 + bin_c
            counts = np.bincount(bin_idx, minlength=256)
            sorted_bins = np.argsort(-counts)

            chosen_pixel = None

            for b in sorted_bins:
                if counts[b] == 0:
                    break
                cluster_pixels = glitched_indices[bin_idx == b]
                cluster_iters = active_iter[cluster_pixels[:, 0], cluster_pixels[:, 1]]
                center_r = np.median(cluster_pixels[:, 0])
                center_c = np.median(cluster_pixels[:, 1])

                # 1. First check for interior/un-escaped pixels in the glitched cluster
                interior_mask = (cluster_iters < 0) | (cluster_iters >= max_iter)
                if np.any(interior_mask):
                    int_pixels = cluster_pixels[interior_mask]
                    dists = (int_pixels[:, 0] - center_r)**2 + (int_pixels[:, 1] - center_c)**2
                    dist_order = np.argsort(dists)
                    for d_idx in dist_order:
                        cand_coord = (int(int_pixels[d_idx, 0]), int(int_pixels[d_idx, 1]))
                        if cand_coord not in visited_rebase_coords:
                            chosen_pixel = cand_coord
                            break

                # 2. If no unvisited interior pixels, select escaped candidate with highest iteration count
                if chosen_pixel is None:
                    escaped_mask = (cluster_iters > 0) & (cluster_iters < max_iter)
                    candidates = cluster_pixels[escaped_mask]
                    if len(candidates) > 0:
                        cand_iters = cluster_iters[escaped_mask]
                        best_cand_order = np.argsort(-cand_iters)
                        for c_idx in best_cand_order:
                            cand_coord = (int(candidates[c_idx, 0]), int(candidates[c_idx, 1]))
                            if cand_coord not in visited_rebase_coords:
                                chosen_pixel = cand_coord
                                break

                # 3. Fallback: closest candidate to cluster center
                if chosen_pixel is None:
                    dists = (cluster_pixels[:, 0] - center_r)**2 + (cluster_pixels[:, 1] - center_c)**2
                    dist_order = np.argsort(dists)
                    for d_idx in dist_order:
                        cand_coord = (int(cluster_pixels[d_idx, 0]), int(cluster_pixels[d_idx, 1]))
                        if cand_coord not in visited_rebase_coords:
                            chosen_pixel = cand_coord
                            break

                if chosen_pixel is not None:
                    break

            if chosen_pixel is None:
                for g_idx in range(len(glitched_indices)):
                    cand = (int(glitched_indices[g_idx, 0]), int(glitched_indices[g_idx, 1]))
                    if cand not in visited_rebase_coords:
                        chosen_pixel = cand
                        break

            if chosen_pixel is None:
                break

            g_row, g_col = chosen_pixel
            visited_rebase_coords.add((g_row, g_col))

            x_min_dec = center_x_dec - (plot_width_dec / Decimal(2))
            y_max_dec = center_y_dec + (plot_height_dec / Decimal(2))
            rebase_cx_dec = x_min_dec + (Decimal(g_col) / Decimal(max(1, target_w - 1))) * plot_width_dec
            rebase_cy_dec = y_max_dec - (Decimal(g_row) / Decimal(max(1, target_h - 1))) * plot_height_dec

            bits_needed = compute_required_prec_bits(plot_width_dec)
            r_re, r_im, r_len = compute_single_highprec_orbit(
                rebase_cx_dec, rebase_cy_dec, max_iter, fractal_type, julia_cx, julia_cy, bits_needed,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
            )
            ref_px_rebase = float(g_col) * float(factor)
            ref_py_rebase = float(g_row) * float(factor)

            if r_len > 100:
                cache_reference_orbit(
                    rebase_cx_dec, rebase_cy_dec, plot_width_dec, max_iter,
                    r_re, r_im, r_len,
                    fractal_type=fractal_type, julia_cx=julia_cx, julia_cy=julia_cy,
                    gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_floatexp=use_floatexp
                )

            if use_floatexp:
                r_bbsa_res = compute_bbsa_coeffs(
                    r_re, r_im, r_len, max_r_norm, fractal_type,
                    bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
                    gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki,
                    E_scale=E_scale
                )
                r_bbsa_skip = r_bbsa_res[0]
                r_bbsa_coeffs = r_bbsa_res[1:17]
                if bbsa_order == 32:
                    r_bla_table = build_bla_table(
                        r_re, r_im, r_len, fractal_type=fractal_type, tol=bbsa_tol,
                        gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
                    )
                else:
                    r_bla_table = np.zeros((1, 1, 1), dtype=np.float64)

                if color_scheme_id == 11 and cdf_lut is not None:
                    cpu_render_floatexp_hist_kernel(
                        out_rgb, out_glitch, iter_map, cdf_lut, lut, target_w, target_h, max_iter,
                        r_re, r_im, r_len,
                        ref_px_rebase, ref_py_rebase,
                        dx_sub_norm_f, dy_sub_norm_f, E_scale,
                        float(rebase_cx_dec), float(rebase_cy_dec), factor, fractal_type,
                        float(julia_cx), float(julia_cy),
                        r_bbsa_skip, bbsa_inv_r, *r_bbsa_coeffs,
                        curr_pass, in_glitch, int(gen_n), float(gen_kr), float(gen_ki), strict_glitch_flag,
                        float(edge_threshold), INV_LN2_F32, LN_BAILOUT_LOG, num_bins, float(palette_offset),
                        float(color_density), float(color_contrast)
                    )
                else:
                    cpu_render_floatexp_kernel(
                        out_rgb, out_glitch, active_iter, lut, target_w, target_h, max_iter,
                        r_re, r_im, r_len,
                        ref_px_rebase, ref_py_rebase,
                        dx_sub_norm_f, dy_sub_norm_f, E_scale,
                        float(rebase_cx_dec), float(rebase_cy_dec), factor, color_scheme_id, fractal_type,
                        float(julia_cx), float(julia_cy),
                        r_bbsa_skip, bbsa_inv_r, *r_bbsa_coeffs,
                        curr_pass, in_glitch, int(gen_n), float(gen_kr), float(gen_ki), strict_glitch_flag,
                        float(edge_threshold), INV_LN2_F32, LN_BAILOUT_LOG, float(palette_offset),
                        float(color_density), float(color_contrast),
                        bla_table=r_bla_table
                    )
            else:
                r_bbsa_res = compute_bbsa_coeffs(
                    r_re, r_im, r_len, max_r, fractal_type,
                    bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
                    gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki,
                    E_scale=0
                )
                r_bbsa_skip = r_bbsa_res[0]
                r_bbsa_coeffs = r_bbsa_res[1:17]

                if bbsa_order == 32:
                    r_bla_table = build_bla_table(
                        r_re, r_im, r_len, fractal_type=fractal_type, tol=bbsa_tol,
                        gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
                    )
                else:
                    r_bla_table = np.zeros((1, 1, 1), dtype=np.float64)

                if color_scheme_id == 11 and cdf_lut is not None:
                    cpu_render_perturbation_hist_kernel(
                        out_rgb, out_glitch, iter_map, cdf_lut, lut, target_w, target_h, max_iter,
                        r_re, r_im, r_len,
                        ref_px_rebase, ref_py_rebase,
                        dx_sub_f, dy_sub_f,
                        float(rebase_cx_dec), float(rebase_cy_dec), factor, fractal_type,
                        float(julia_cx), float(julia_cy),
                        r_bbsa_skip, bbsa_inv_r, *r_bbsa_coeffs,
                        curr_pass, in_glitch, int(gen_n), float(gen_kr), float(gen_ki), strict_glitch_flag,
                        float(edge_threshold), INV_LN2_F32, LN_BAILOUT_LOG, num_bins, float(palette_offset),
                        float(color_density), float(color_contrast)
                    )
                else:
                    cpu_render_perturbation_kernel(
                        out_rgb, out_glitch, active_iter, lut, target_w, target_h, max_iter,
                        r_re, r_im, r_len,
                        ref_px_rebase, ref_py_rebase,
                        dx_sub_f, dy_sub_f,
                        float(rebase_cx_dec), float(rebase_cy_dec), factor, color_scheme_id, fractal_type,
                        float(julia_cx), float(julia_cy),
                        r_bbsa_skip, bbsa_inv_r, *r_bbsa_coeffs,
                        curr_pass, in_glitch, int(gen_n), float(gen_kr), float(gen_ki),
                        float(edge_threshold), strict_glitch_flag,
                        INV_LN2_F32, LN_BAILOUT_LOG, float(palette_offset),
                        float(color_density), float(color_contrast),
                        bla_table=r_bla_table
                    )

        if is_exhaustive:
            glitched_indices = np.argwhere(out_glitch > 0)
            if len(glitched_indices) > 0:
                eval_glitched_pixels_fallback(
                    out_rgb, glitched_indices, target_w, target_h, max_iter,
                    center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
                    cmap_name, color_scheme_id, fractal_type, julia_cx, julia_cy,
                    gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_hdr=is_hdr,
                    palette_offset=float(palette_offset),
                    color_density=float(color_density), color_contrast=float(color_contrast),
                    cdf_lut=cdf_lut if color_scheme_id == 11 else None,
                    num_bins=num_bins if color_scheme_id == 11 else 2048,
                    iter_map=active_iter
                )

    return out_rgb


def get_adaptive_max_iter_from_probe(iters_array, base_iter):
    """Calculates adaptive iteration depth from a low-resolution probe map."""
    escaped = iters_array[iters_array >= 0.0]
    if escaped.size == 0:
        return int(base_iter)
    
    max_escaped = float(np.max(escaped))
    percentile_98 = float(np.percentile(escaped, 98))
    target_iter = int(max(percentile_98 * 1.35, max_escaped * 1.20))
    return min(GLOBAL_MAX_ITER, max(100, target_iter))


def probe_adaptive_max_iter(center_x_dec, center_y_dec, plot_width_dec, base_iter,
                            target_w=128, target_h=72, fractal_type=0,
                            julia_cx=0.0, julia_cy=0.0, gen_n=3, gen_kr=0.25, gen_ki=1.0,
                            precision_mode="1e-300 (Perturbation)", bbsa_tol=1e-4, bbsa_order=4,
                            compute_device="GPU", cpu_threads=2):
    """Executes a lightweight 128x72 probe pass to dynamically determine the optimal iteration depth."""
    probe_search_max = min(GLOBAL_MAX_ITER, max(1000, int(base_iter * 2)))
    aspect_ratio = Decimal(target_h) / Decimal(target_w)
    plot_height_dec = plot_width_dec * aspect_ratio

    ensure_decimal_precision(plot_width_dec)
    use_floatexp = (plot_width_dec < Decimal("1e-300"))
    use_fp64 = (plot_width_dec >= Decimal("1e-12")) and (not use_floatexp)
    use_dd = (plot_width_dec >= Decimal("1e-24") and "dd" in str(precision_mode).lower()) and (not use_floatexp)
    if use_dd:
        use_fp64 = False
    use_perturbation = (not use_fp64) and (not use_dd) and (not use_floatexp)

    if use_floatexp:
        E_scale = compute_floatexp_scale(plot_width_dec)
        scale_factor_dec = Decimal(2) ** E_scale
        norm_width_dec = plot_width_dec * scale_factor_dec
        norm_height_dec = plot_height_dec * scale_factor_dec
        w_norm = float(norm_width_dec)
        dx_norm_f = float(norm_width_dec / Decimal(max(1, target_w - 1)))
        dy_norm_f = float(norm_height_dec / Decimal(max(1, target_h - 1)))

    if compute_device == "CPU" or not HAS_CUDA:
        out_iters = np.empty((target_h, target_w), dtype=np.float32)
        set_num_threads(max(1, min(MAX_SYSTEM_CORES, int(cpu_threads))))

        if use_fp64:
            w_f, h_f = float(plot_width_dec), float(plot_height_dec)
            cx, cy = float(center_x_dec), float(center_y_dec)
            dx = w_f / max(1, target_w - 1)
            dy = h_f / max(1, target_h - 1)
            cpu_probe_iter_kernel(
                out_iters, target_w, target_h, probe_search_max,
                cx - (w_f * 0.5), cy + (h_f * 0.5), dx, dy,
                fractal_type, float(julia_cx), float(julia_cy),
                int(gen_n), float(gen_kr), float(gen_ki),
                INV_LN2_F32, LN_BAILOUT_LOG
            )
        elif use_dd:
            x_min_hi, x_min_lo = to_dd(center_x_dec - (plot_width_dec / Decimal(2)))
            y_max_hi, y_max_lo = to_dd(center_y_dec + (plot_height_dec / Decimal(2)))
            dx_hi, dx_lo = to_dd(plot_width_dec / Decimal(max(1, target_w - 1)))
            dy_hi, dy_lo = to_dd(plot_height_dec / Decimal(max(1, target_h - 1)))
            j_cx_hi, j_cx_lo = to_dd(julia_cx)
            j_cy_hi, j_cy_lo = to_dd(julia_cy)
            cpu_probe_iter_dd_kernel(
                out_iters, target_w, target_h, probe_search_max,
                x_min_hi, x_min_lo, dx_hi, dx_lo,
                y_max_hi, y_max_lo, dy_hi, dy_lo,
                fractal_type, j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo,
                int(gen_n), float(gen_kr), float(gen_ki),
                INV_LN2_F32, LN_BAILOUT_LOG
            )
        elif use_floatexp:
            best_cx, best_cy, best_u, best_v, ref_re, ref_im, ref_len = find_optimal_reference_orbit(
                center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
                probe_search_max, target_w, target_h, fractal_type=fractal_type, julia_cx=julia_cx, julia_cy=julia_cy,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_floatexp=True
            )
            max_r_norm = float(w_norm) * 0.75
            bbsa_inv_r = 1.0 / max_r_norm if max_r_norm > 0.0 else 1.0
            bbsa_res = compute_bbsa_coeffs(
                ref_re, ref_im, ref_len, max_r_norm, fractal_type,
                bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki,
                E_scale=E_scale
            )
            bbsa_skip = bbsa_res[0]
            bbsa_coeffs = bbsa_res[1:17]
            ref_px = float(target_w - 1) * float(best_u)
            ref_py = float(target_h - 1) * float(best_v)
            cpu_probe_iter_floatexp_kernel(
                out_iters, target_w, target_h, probe_search_max,
                ref_re, ref_im, ref_len,
                ref_px, ref_py, dx_norm_f, dy_norm_f, E_scale,
                float(best_cx), float(best_cy), fractal_type,
                float(julia_cx), float(julia_cy),
                bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                int(gen_n), float(gen_kr), float(gen_ki),
                INV_LN2_F32, LN_BAILOUT_LOG
            )
        else:
            best_cx, best_cy, best_u, best_v, ref_re, ref_im, ref_len = find_optimal_reference_orbit(
                center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
                probe_search_max, target_w, target_h, fractal_type=fractal_type, julia_cx=julia_cx, julia_cy=julia_cy,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_floatexp=False
            )
            max_r = float(plot_width_dec) * 0.75
            bbsa_inv_r = 1.0 / max_r if max_r > 0.0 else 1.0
            bbsa_res = compute_bbsa_coeffs(
                ref_re, ref_im, ref_len, max_r, fractal_type,
                bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki,
                E_scale=0
            )
            bbsa_skip = bbsa_res[0]
            bbsa_coeffs = bbsa_res[1:17]
            dx_f = float(plot_width_dec / Decimal(max(1, target_w - 1)))
            dy_f = float(plot_height_dec / Decimal(max(1, target_h - 1)))
            ref_px = float(target_w - 1) * float(best_u)
            ref_py = float(target_h - 1) * float(best_v)

            cpu_probe_iter_perturbation_kernel(
                out_iters, target_w, target_h, probe_search_max,
                ref_re, ref_im, ref_len,
                ref_px, ref_py, dx_f, dy_f,
                float(best_cx), float(best_cy), fractal_type,
                float(julia_cx), float(julia_cy),
                bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                int(gen_n), float(gen_kr), float(gen_ki),
                INV_LN2_F32, LN_BAILOUT_LOG
            )

        return get_adaptive_max_iter_from_probe(out_iters, base_iter)

    d_probe = get_scratch_device_array((target_h, target_w), np.float32, "probe_iter")
    threads = (16, 16)
    blocks = ((target_w + 15) // 16, (target_h + 15) // 16)

    if use_fp64:
        w_f, h_f = float(plot_width_dec), float(plot_height_dec)
        cx, cy = float(center_x_dec), float(center_y_dec)
        dx = w_f / max(1, target_w - 1)
        dy = h_f / max(1, target_h - 1)
        probe_iter_fp64_kernel[blocks, threads](
            d_probe, target_w, target_h, probe_search_max,
            cx - (w_f * 0.5), cy + (h_f * 0.5), dx, dy,
            fractal_type, float(julia_cx), float(julia_cy), int(gen_n), float(gen_kr), float(gen_ki)
        )
    elif use_dd:
        x_min_hi, x_min_lo = to_dd(center_x_dec - (plot_width_dec / Decimal(2)))
        y_max_hi, y_max_lo = to_dd(center_y_dec + (plot_height_dec / Decimal(2)))
        dx_hi, dx_lo = to_dd(plot_width_dec / Decimal(max(1, target_w - 1)))
        dy_hi, dy_lo = to_dd(plot_height_dec / Decimal(max(1, target_h - 1)))
        j_cx_hi, j_cx_lo = to_dd(julia_cx)
        j_cy_hi, j_cy_lo = to_dd(julia_cy)
        probe_iter_dd_kernel[blocks, threads](
            d_probe, target_w, target_h, probe_search_max,
            x_min_hi, x_min_lo, dx_hi, dx_lo,
            y_max_hi, y_max_lo, dy_hi, dy_lo,
            fractal_type, j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo,
            int(gen_n), float(gen_kr), float(gen_ki)
        )
    elif use_floatexp:
        best_cx, best_cy, best_u, best_v, ref_re, ref_im, ref_len = find_optimal_reference_orbit(
            center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
            probe_search_max, target_w, target_h, fractal_type=fractal_type, julia_cx=julia_cx, julia_cy=julia_cy,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_floatexp=True
        )
        max_r_norm = float(w_norm) * 0.75
        bbsa_inv_r = 1.0 / max_r_norm if max_r_norm > 0.0 else 1.0
        d_ref_re_buf = get_scratch_device_array_1d(ref_len + 1, np.float64, "probe_ref_re")
        d_ref_im_buf = get_scratch_device_array_1d(ref_len + 1, np.float64, "probe_ref_im")
        d_ref_re = d_ref_re_buf[:ref_len + 1]
        d_ref_im = d_ref_im_buf[:ref_len + 1]
        d_ref_re.copy_to_device(ref_re[:ref_len + 1])
        d_ref_im.copy_to_device(ref_im[:ref_len + 1])
        bbsa_res = compute_bbsa_coeffs_cuda(
            d_ref_re, d_ref_im, ref_len, max_r_norm, fractal_type,
            bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki,
            E_scale=E_scale, h_ref_re=ref_re, h_ref_im=ref_im
        )
        bbsa_skip = bbsa_res[0]
        bbsa_coeffs = bbsa_res[1:17]
        ref_px = float(target_w - 1) * float(best_u)
        ref_py = float(target_h - 1) * float(best_v)
        probe_iter_floatexp_kernel[blocks, threads](
            d_probe, target_w, target_h, probe_search_max,
            d_ref_re, d_ref_im, ref_len,
            ref_px, ref_py, dx_norm_f, dy_norm_f, E_scale,
            float(best_cx), float(best_cy), fractal_type,
            float(julia_cx), float(julia_cy),
            bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
            int(gen_n), float(gen_kr), float(gen_ki)
        )
    else:
        best_cx, best_cy, best_u, best_v, ref_re, ref_im, ref_len = find_optimal_reference_orbit(
            center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
            probe_search_max, target_w, target_h, fractal_type=fractal_type, julia_cx=julia_cx, julia_cy=julia_cy,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_floatexp=False
        )
        max_r = float(plot_width_dec) * 0.75
        bbsa_inv_r = 1.0 / max_r if max_r > 0.0 else 1.0
        d_ref_re_buf = get_scratch_device_array_1d(ref_len + 1, np.float64, "probe_ref_re")
        d_ref_im_buf = get_scratch_device_array_1d(ref_len + 1, np.float64, "probe_ref_im")
        d_ref_re = d_ref_re_buf[:ref_len + 1]
        d_ref_im = d_ref_im_buf[:ref_len + 1]
        d_ref_re.copy_to_device(ref_re[:ref_len + 1])
        d_ref_im.copy_to_device(ref_im[:ref_len + 1])
        bbsa_res = compute_bbsa_coeffs_cuda(
            d_ref_re, d_ref_im, ref_len, max_r, fractal_type,
            bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki,
            E_scale=0, h_ref_re=ref_re, h_ref_im=ref_im
        )
        bbsa_skip = bbsa_res[0]
        bbsa_coeffs = bbsa_res[1:17]
        ref_px = float(target_w - 1) * float(best_u)
        ref_py = float(target_h - 1) * float(best_v)
        dx_f = float(plot_width_dec / Decimal(max(1, target_w - 1)))
        dy_f = float(plot_height_dec / Decimal(max(1, target_h - 1)))
        probe_iter_perturbation_kernel[blocks, threads](
            d_probe, target_w, target_h, probe_search_max,
            d_ref_re, d_ref_im, ref_len,
            ref_px, ref_py, dx_f, dy_f,
            float(best_cx), float(best_cy), fractal_type,
            float(julia_cx), float(julia_cy),
            bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
            int(gen_n), float(gen_kr), float(gen_ki)
        )

    out_iters = d_probe.copy_to_host()
    return get_adaptive_max_iter_from_probe(out_iters, base_iter)


def compute_dynamic_max_iter(base_max_iter, plot_width_dec, dynamic_mode,
                             center_x_dec=Decimal("0.0"), center_y_dec=Decimal("0.0"),
                             **probe_kwargs):
    """Dispatches dynamic iteration scaling according to selected heuristic."""
    clamped_base = min(GLOBAL_MAX_ITER, max(1, int(base_max_iter)))
    if dynamic_mode == "Off" or not dynamic_mode or base_max_iter <= 0:
        return clamped_base

    mode_str = str(dynamic_mode).lower()

    if "adaptive" in mode_str:
        try:
            return probe_adaptive_max_iter(
                center_x_dec=center_x_dec, center_y_dec=center_y_dec,
                plot_width_dec=plot_width_dec, base_iter=clamped_base,
                **probe_kwargs
            )
        except Exception:
            return clamped_base

    try:
        p = Decimal(str(plot_width_dec))
        if p <= 0 or p >= Decimal("4.0"):
            return clamped_base

        # Continuous smooth logarithmic zoom depth: log10(4.0 / p)
        mantissa_f = float(p.scaleb(-p.adjusted()))
        zoom_depth = max(0.0, float(-p.adjusted()) + math.log10(4.0) - math.log10(max(1e-12, mantissa_f)))

        if "sqrt" in mode_str:
            multiplier = 1.0 + 0.85 * math.sqrt(zoom_depth)
        elif "linear" in mode_str or "log" in mode_str:
            multiplier = 1.0 + 0.35 * zoom_depth
        else:
            multiplier = 1.0

        scaled_iter = max(clamped_base, int(round(clamped_base * multiplier)))
        return min(GLOBAL_MAX_ITER, scaled_iter)
    except Exception:
        return clamped_base


def render_frame(params: RenderParameters):
    """Unified entry point for rendering frames via RenderParameters dataclass."""
    return render_frame_ssaa(params)


def _cuda_sync_stream(stream):
    """Safely synchronizes the given CUDA stream or the global CUDA context."""
    if stream == 0 or stream is None:
        cuda.synchronize()
    else:
        stream.synchronize()


def render_frame_ssaa(target_w, target_h=None, max_iter=None, center_x_dec=None, center_y_dec=None, plot_width_dec=None, cmap_name=None,
                      ssaa_factor=1, edge_threshold=0.35, color_scheme_id=0, precision_mode="1e-300 (Perturbation)",
                      fractal_type=0, julia_cx=0.0, julia_cy=0.0, bbsa_tol=1e-4, bbsa_order=4,
                      glitch_mode="Off (Legacy Single-Ref)", gen_n=3, gen_kr=0.25, gen_ki=1.0,
                      d_rgb_buf=None, h_rgb_buf=None, stream=0, is_hdr=False,
                      compute_device="GPU", cpu_threads=2, palette_offset=0.0,
                      color_density=1.0, color_contrast=1.0, sync_stream=True,
                      d_iter_buf=None, h_iter_buf=None, stream_slot=None):
    """Main rendering dispatcher routing tasks to GPU kernels or CPU multi-threading."""
    if isinstance(target_w, RenderParameters):
        p = target_w
        return render_frame_ssaa(
            p.target_w, p.target_h, p.max_iter, p.center_x_dec, p.center_y_dec, p.plot_width_dec, p.cmap_name,
            ssaa_factor=p.ssaa_factor, edge_threshold=p.edge_threshold, color_scheme_id=p.color_scheme_id,
            precision_mode=p.precision_mode, fractal_type=p.fractal_type, julia_cx=p.julia_cx, julia_cy=p.julia_cy,
            bbsa_tol=p.bbsa_tol, bbsa_order=p.bbsa_order, glitch_mode=p.glitch_mode,
            gen_n=p.gen_n, gen_kr=p.gen_kr, gen_ki=p.gen_ki,
            d_rgb_buf=p.d_rgb_buf if d_rgb_buf is None else d_rgb_buf,
            h_rgb_buf=p.h_rgb_buf if h_rgb_buf is None else h_rgb_buf,
            stream=p.stream if p.stream is not None else stream,
            is_hdr=p.is_hdr, compute_device=p.compute_device, cpu_threads=p.cpu_threads, palette_offset=p.palette_offset,
            color_density=p.color_density, color_contrast=p.color_contrast,
            sync_stream=getattr(p, 'sync_stream', sync_stream),
            d_iter_buf=getattr(p, 'd_iter_buf', d_iter_buf),
            h_iter_buf=getattr(p, 'h_iter_buf', h_iter_buf),
            stream_slot=getattr(p, 'stream_slot', stream_slot)
        )

    if stream is None:
        stream = 0

    stream_slot = get_stream_slot(stream, explicit_slot=stream_slot)

    if compute_device == "CPU" or not HAS_CUDA:
        return render_frame_cpu(
            target_w, target_h, max_iter, center_x_dec, center_y_dec, plot_width_dec,
            cmap_name, ssaa_factor=ssaa_factor, edge_threshold=edge_threshold,
            color_scheme_id=color_scheme_id, precision_mode=precision_mode,
            fractal_type=fractal_type, julia_cx=julia_cx, julia_cy=julia_cy,
            bbsa_tol=bbsa_tol, bbsa_order=bbsa_order, glitch_mode=glitch_mode,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki,
            h_rgb_buf=h_rgb_buf, is_hdr=is_hdr, cpu_threads=cpu_threads,
            palette_offset=palette_offset,
            color_density=color_density, color_contrast=color_contrast,
            h_iter_buf=h_iter_buf
        )

    factor = max(1, int(ssaa_factor))
    aspect_ratio = Decimal(target_h) / Decimal(target_w)
    plot_height_dec = plot_width_dec * aspect_ratio

    dtype_val = np.uint16 if is_hdr else np.uint8
    d_out_rgb = d_rgb_buf if d_rgb_buf is not None else get_scratch_device_array((target_h, target_w, 3), dtype_val, f"out_rgb_{stream_slot}")
    d_lut = get_gpu_cmap_lut(cmap_name, lut_size=2048, is_hdr=is_hdr)

    threads = (16, 16)
    blocks = ((target_w + 15) // 16, (target_h + 15) // 16)

    dx_f = float(plot_width_dec / Decimal(max(1, target_w - 1)))
    dy_f = float(plot_height_dec / Decimal(max(1, target_h - 1)))
    dx_sub_f = dx_f / float(factor)
    dy_sub_f = dy_f / float(factor)

    ensure_decimal_precision(plot_width_dec)
    use_floatexp = (plot_width_dec < Decimal("1e-300"))
    use_fp64 = (plot_width_dec >= Decimal("1e-12")) and (not use_floatexp)
    use_dd = (plot_width_dec >= Decimal("1e-24") and "dd" in str(precision_mode).lower()) and (not use_floatexp)
    if use_dd:
        use_fp64 = False
    use_perturbation = (not use_fp64) and (not use_dd) and (not use_floatexp)
    sub_w, sub_h = target_w * factor, target_h * factor
    edge_thresh_f = np.float32(edge_threshold)

    if use_floatexp:
        E_scale = compute_floatexp_scale(plot_width_dec)
        scale_factor_dec = Decimal(2) ** E_scale
        norm_width_dec = plot_width_dec * scale_factor_dec
        norm_height_dec = plot_height_dec * scale_factor_dec
        w_norm = float(norm_width_dec)
        dx_norm_f = float(norm_width_dec / Decimal(max(1, target_w - 1)))
        dy_norm_f = float(norm_height_dec / Decimal(max(1, target_h - 1)))
        dx_sub_norm_f = dx_norm_f / float(factor)
        dy_sub_norm_f = dy_norm_f / float(factor)

    d_dummy_edge = get_dummy_edge_buffer()
    d_dummy_iter = get_dummy_iter_buffer()
    d_active_iter = d_iter_buf if d_iter_buf is not None else get_scratch_device_array((target_h, target_w), np.float32, f"iter_map_{stream_slot}")

    d_cdf_lut = None
    num_bins = 2048

    if color_scheme_id == 11:
        num_bins = 65536
        d_iter_map = d_active_iter
        d_edge_mask = get_scratch_device_array((target_h, target_w), np.uint8, f"edge_mask_{stream_slot}")
        d_hist = get_scratch_device_array((num_bins,), np.int32, f"hist_bins_{stream_slot}")
        zero_histogram_cuda_kernel[128, 512, stream](d_hist, num_bins)

        if use_fp64:
            w_f, h_f = float(plot_width_dec), float(plot_height_dec)
            cx, cy = float(center_x_dec), float(center_y_dec)
            x_min = cx - (w_f * 0.5)
            y_max = cy + (h_f * 0.5)
            dx_sub = dx_sub_f
            dy_sub = dy_sub_f

            mandelbrot_unified_fp64_kernel[blocks, threads, stream](
                d_out_rgb, d_iter_map, d_dummy_edge, d_lut, target_w, target_h, max_iter,
                x_min, y_max, dx_sub, dy_sub, factor, 0, fractal_type,
                float(julia_cx), float(julia_cy), int(gen_n), float(gen_kr), float(gen_ki), 0,
                float(palette_offset), float(color_density), float(color_contrast)
            )
        elif use_dd:
            x_min_hi, x_min_lo = to_dd(center_x_dec - (plot_width_dec / Decimal(2)))
            y_max_hi, y_max_lo = to_dd(center_y_dec + (plot_height_dec / Decimal(2)))
            dx_hi, dx_lo = to_dd((plot_width_dec / Decimal(max(1, target_w - 1))) / Decimal(factor))
            dy_hi, dy_lo = to_dd((plot_height_dec / Decimal(max(1, target_h - 1))) / Decimal(factor))
            j_cx_hi, j_cx_lo = to_dd(julia_cx)
            j_cy_hi, j_cy_lo = to_dd(julia_cy)

            mandelbrot_unified_dd_kernel[blocks, threads, stream](
                d_out_rgb, d_iter_map, d_dummy_edge, d_lut, target_w, target_h, max_iter,
                x_min_hi, x_min_lo, dx_hi, dx_lo, y_max_hi, y_max_lo, dy_hi, dy_lo,
                factor, 0, fractal_type, j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo,
                int(gen_n), float(gen_kr), float(gen_ki), 0, float(palette_offset),
                float(color_density), float(color_contrast)
            )
        elif use_floatexp:
            if bbsa_order in (4, 8):
                bbsa_order = 32
                if bbsa_tol > 1e-6:
                    bbsa_tol = 1e-6
            strict_glitch_flag = 1 if ("Auto Rebasing" in glitch_mode or "Strict" in glitch_mode or "Pauldelbrot" in glitch_mode) else 0
            d_glitch_out = get_scratch_device_array((target_h, target_w), np.uint8, f"glitch_out_{stream_slot}")
            d_glitch_in = get_scratch_device_array((target_h, target_w), np.uint8, f"glitch_in_{stream_slot}")

            best_cx, best_cy, best_u, best_v, ref_re, ref_im, ref_len = find_optimal_reference_orbit(
                center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
                max_iter, target_w, target_h, fractal_type=fractal_type, julia_cx=julia_cx, julia_cy=julia_cy,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_floatexp=True
            )
            max_r_norm = float(w_norm) * 0.75
            bbsa_inv_r = 1.0 / max_r_norm if max_r_norm > 0.0 else 1.0
            d_ref_re_buf = get_scratch_device_array_1d(ref_len + 1, np.float64, f"ref_re_{stream_slot}")
            d_ref_im_buf = get_scratch_device_array_1d(ref_len + 1, np.float64, f"ref_im_{stream_slot}")
            d_ref_re = d_ref_re_buf[:ref_len + 1]
            d_ref_im = d_ref_im_buf[:ref_len + 1]
            d_ref_re.copy_to_device(ref_re[:ref_len + 1], stream=stream)
            d_ref_im.copy_to_device(ref_im[:ref_len + 1], stream=stream)

            bbsa_res = compute_bbsa_coeffs_cuda(
                d_ref_re, d_ref_im, ref_len, max_r_norm, fractal_type,
                bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, E_scale=E_scale, stream=stream,
                h_ref_re=ref_re, h_ref_im=ref_im
            )
            bbsa_skip = bbsa_res[0]
            bbsa_coeffs = bbsa_res[1:17]

            if bbsa_order == 32 and fractal_type in (0, 2, 4):
                bla_table = build_bla_table(
                    ref_re, ref_im, ref_len, fractal_type=fractal_type, tol=bbsa_tol,
                    gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
                )
                d_bla_table = upload_bla_table_cuda(bla_table, stream_slot, stream=stream)
            else:
                d_bla_table = get_scratch_device_array((1, 1, 1), np.float64, f"bla_dummy_{stream_slot}")

            ref_px_sub = float(target_w - 1) * float(best_u) * float(factor)
            ref_py_sub = float(target_h - 1) * float(best_v) * float(factor)
            ref_cx_f = float(best_cx)
            ref_cy_f = float(best_cy)

            mandelbrot_unified_floatexp_kernel[blocks, threads, stream](
                d_out_rgb, d_glitch_out, d_iter_map, d_dummy_edge, d_lut, target_w, target_h, max_iter,
                d_ref_re, d_ref_im, ref_len,
                ref_px_sub, ref_py_sub, dx_sub_norm_f, dy_sub_norm_f, E_scale,
                ref_cx_f, ref_cy_f, factor, 0, fractal_type,
                float(julia_cx), float(julia_cy),
                bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                0, d_glitch_in, int(gen_n), float(gen_kr), float(gen_ki), 0, strict_glitch_flag,
                float(palette_offset), float(color_density), float(color_contrast),
                d_bla_table
            )
        else:
            strict_glitch_flag = 1 if ("Auto Rebasing" in glitch_mode or "Strict" in glitch_mode or "Pauldelbrot" in glitch_mode) else 0
            d_glitch_out = get_scratch_device_array((target_h, target_w), np.uint8, f"glitch_out_{stream_slot}")
            d_glitch_in = get_scratch_device_array((target_h, target_w), np.uint8, f"glitch_in_{stream_slot}")

            best_cx, best_cy, best_u, best_v, ref_re, ref_im, ref_len = find_optimal_reference_orbit(
                center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
                max_iter, target_w, target_h, fractal_type=fractal_type, julia_cx=julia_cx, julia_cy=julia_cy,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_floatexp=False
            )
            max_r = float(plot_width_dec) * 0.75
            bbsa_inv_r = 1.0 / max_r if max_r > 0.0 else 1.0
            d_ref_re_buf = get_scratch_device_array_1d(ref_len + 1, np.float64, f"ref_re_{stream_slot}")
            d_ref_im_buf = get_scratch_device_array_1d(ref_len + 1, np.float64, f"ref_im_{stream_slot}")
            d_ref_re = d_ref_re_buf[:ref_len + 1]
            d_ref_im = d_ref_im_buf[:ref_len + 1]
            d_ref_re.copy_to_device(ref_re[:ref_len + 1], stream=stream)
            d_ref_im.copy_to_device(ref_im[:ref_len + 1], stream=stream)

            bbsa_res = compute_bbsa_coeffs_cuda(
                d_ref_re, d_ref_im, ref_len, max_r, fractal_type,
                bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, E_scale=0, stream=stream,
                h_ref_re=ref_re, h_ref_im=ref_im
            )
            bbsa_skip = bbsa_res[0]
            bbsa_coeffs = bbsa_res[1:17]

            if bbsa_order == 32 and fractal_type in (0, 2, 4):
                bla_table = build_bla_table(
                    ref_re, ref_im, ref_len, fractal_type=fractal_type, tol=bbsa_tol,
                    gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
                )
                d_bla_table = upload_bla_table_cuda(bla_table, stream_slot, stream=stream)
            else:
                d_bla_table = get_scratch_device_array((1, 1, 1), np.float64, f"bla_dummy_{stream_slot}")

            ref_px_sub = float(target_w - 1) * float(best_u) * float(factor)
            ref_py_sub = float(target_h - 1) * float(best_v) * float(factor)
            ref_cx_f = float(best_cx)
            ref_cy_f = float(best_cy)

            mandelbrot_unified_perturbation_kernel[blocks, threads, stream](
                d_out_rgb, d_glitch_out, d_iter_map, d_dummy_edge, d_lut, target_w, target_h, max_iter,
                d_ref_re, d_ref_im, ref_len,
                ref_px_sub, ref_py_sub, dx_sub_f, dy_sub_f,
                ref_cx_f, ref_cy_f, factor, 0, fractal_type,
                float(julia_cx), float(julia_cy), bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                0, d_glitch_in, int(gen_n), float(gen_kr), float(gen_ki), 0, strict_glitch_flag,
                float(palette_offset), float(color_density), float(color_contrast),
                d_bla_table
            )

        build_histogram_kernel[blocks, threads, stream](
            d_iter_map, d_hist, target_w, target_h, max_iter, num_bins
        )
        d_cdf_lut = get_scratch_device_array((num_bins,), np.float32, f"cdf_lut_{stream_slot}")
        d_block_sums = get_scratch_device_array((128,), np.float64, f"hist_block_sums_{stream_slot}")
        d_unnorm_cdf = get_scratch_device_array((num_bins,), np.float64, f"unnorm_cdf_{stream_slot}")

        scan_blocks_cuda_kernel[128, 512, stream](d_hist, d_block_sums, d_unnorm_cdf)
        scan_single_block_cuda_kernel[1, 128, stream](d_block_sums, 128)
        finalize_normalized_cdf_cuda_kernel[128, 512, stream](d_unnorm_cdf, d_block_sums, d_cdf_lut, num_bins)

        apply_histogram_equalization_kernel[blocks, threads, stream](
            d_out_rgb, d_iter_map, d_cdf_lut, d_lut, target_w, target_h, max_iter, num_bins, float(palette_offset),
            float(color_density), float(color_contrast)
        )

        if factor > 1:
            edge_detect_kernel[blocks, threads, stream](
                d_iter_map, d_edge_mask, target_w, target_h, edge_thresh_f
            )
            if use_fp64:
                ssaa_resolve_hist_fp64_kernel[blocks, threads, stream](
                    d_out_rgb, d_iter_map, d_edge_mask, d_cdf_lut, d_lut, target_w, target_h, max_iter,
                    x_min, y_max, dx_sub, dy_sub, factor, fractal_type,
                    float(julia_cx), float(julia_cy), int(gen_n), float(gen_kr), float(gen_ki), num_bins, float(palette_offset),
                    float(color_density), float(color_contrast)
                )
            elif use_dd:
                ssaa_resolve_hist_dd_kernel[blocks, threads, stream](
                    d_out_rgb, d_iter_map, d_edge_mask, d_cdf_lut, d_lut, target_w, target_h, max_iter,
                    x_min_hi, x_min_lo, dx_hi, dx_lo,
                    y_max_hi, y_max_lo, dy_hi, dy_lo,
                    factor, fractal_type,
                    j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo,
                    int(gen_n), float(gen_kr), float(gen_ki), num_bins, float(palette_offset),
                    float(color_density), float(color_contrast)
                )
            elif use_floatexp:
                ssaa_resolve_hist_floatexp_kernel[blocks, threads, stream](
                    d_out_rgb, d_glitch_out, d_iter_map, d_edge_mask, d_cdf_lut, d_lut, target_w, target_h, max_iter,
                    d_ref_re, d_ref_im, ref_len,
                    ref_px_sub, ref_py_sub, dx_sub_norm_f, dy_sub_norm_f, E_scale,
                    ref_cx_f, ref_cy_f, factor, fractal_type,
                    float(julia_cx), float(julia_cy),
                    bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                    0, d_glitch_in, int(gen_n), float(gen_kr), float(gen_ki), strict_glitch_flag,
                    num_bins, float(palette_offset),
                    float(color_density), float(color_contrast)
                )
            else:
                ssaa_resolve_hist_perturbation_kernel[blocks, threads, stream](
                    d_out_rgb, d_glitch_out, d_iter_map, d_edge_mask, d_cdf_lut, d_lut, target_w, target_h, max_iter,
                    d_ref_re, d_ref_im, ref_len,
                    ref_px_sub, ref_py_sub, dx_sub_f, dy_sub_f,
                    ref_cx_f, ref_cy_f, factor, fractal_type,
                    float(julia_cx), float(julia_cy), bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                    0, d_glitch_in, int(gen_n), float(gen_kr), float(gen_ki), strict_glitch_flag,
                    num_bins, float(palette_offset),
                    float(color_density), float(color_contrast)
                )

        allow_rebasing = ("Auto Rebasing" in glitch_mode or "Strict" in glitch_mode or "Pauldelbrot" in glitch_mode)
        if not (allow_rebasing and (use_floatexp or use_perturbation) and (plot_width_dec < Decimal("1e-12"))):
            out_ary = d_out_rgb.copy_to_host(ary=h_rgb_buf, stream=stream)
            if sync_stream:
                _cuda_sync_stream(stream)
            return out_ary
    elif use_fp64:
        w_f, h_f = float(plot_width_dec), float(plot_height_dec)
        cx, cy = float(center_x_dec), float(center_y_dec)
        x_min = cx - (w_f * 0.5)
        y_max = cy + (h_f * 0.5)
        dx_sub = dx_sub_f
        dy_sub = dy_sub_f

        if factor == 1:
            mandelbrot_unified_fp64_kernel[blocks, threads, stream](
                d_out_rgb, d_active_iter, d_dummy_edge, d_lut, target_w, target_h, max_iter,
                x_min, y_max, dx_sub, dy_sub, factor, color_scheme_id, fractal_type,
                float(julia_cx), float(julia_cy), int(gen_n), float(gen_kr), float(gen_ki), 0,
                float(palette_offset), float(color_density), float(color_contrast)
            )
        else:
            d_iter_map = d_active_iter
            d_edge_mask = get_scratch_device_array((target_h, target_w), np.uint8, f"edge_mask_{stream_slot}")

            mandelbrot_unified_fp64_kernel[blocks, threads, stream](
                d_out_rgb, d_iter_map, d_dummy_edge, d_lut, target_w, target_h, max_iter,
                x_min, y_max, dx_sub, dy_sub, factor, color_scheme_id, fractal_type,
                float(julia_cx), float(julia_cy), int(gen_n), float(gen_kr), float(gen_ki), 0,
                float(palette_offset), float(color_density), float(color_contrast)
            )

            edge_detect_kernel[blocks, threads, stream](
                d_iter_map, d_edge_mask, target_w, target_h, edge_thresh_f
            )

            mandelbrot_unified_fp64_kernel[blocks, threads, stream](
                d_out_rgb, d_iter_map, d_edge_mask, d_lut, target_w, target_h, max_iter,
                x_min, y_max, dx_sub, dy_sub, factor, color_scheme_id, fractal_type,
                float(julia_cx), float(julia_cy), int(gen_n), float(gen_kr), float(gen_ki), 2,
                float(palette_offset), float(color_density), float(color_contrast)
            )
    elif use_dd:
        x_min_hi, x_min_lo = to_dd(center_x_dec - (plot_width_dec / Decimal(2)))
        y_max_hi, y_max_lo = to_dd(center_y_dec + (plot_height_dec / Decimal(2)))
        dx_hi, dx_lo = to_dd((plot_width_dec / Decimal(max(1, target_w - 1))) / Decimal(factor))
        dy_hi, dy_lo = to_dd((plot_height_dec / Decimal(max(1, target_h - 1))) / Decimal(factor))
        j_cx_hi, j_cx_lo = to_dd(julia_cx)
        j_cy_hi, j_cy_lo = to_dd(julia_cy)

        if factor == 1:
            mandelbrot_unified_dd_kernel[blocks, threads, stream](
                d_out_rgb, d_active_iter, d_dummy_edge, d_lut, target_w, target_h, max_iter,
                x_min_hi, x_min_lo, dx_hi, dx_lo, y_max_hi, y_max_lo, dy_hi, dy_lo,
                factor, color_scheme_id, fractal_type, j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo,
                int(gen_n), float(gen_kr), float(gen_ki), 0, float(palette_offset),
                float(color_density), float(color_contrast)
            )
        else:
            d_iter_map = d_active_iter
            d_edge_mask = get_scratch_device_array((target_h, target_w), np.uint8, f"edge_mask_{stream_slot}")

            mandelbrot_unified_dd_kernel[blocks, threads, stream](
                d_out_rgb, d_iter_map, d_dummy_edge, d_lut, target_w, target_h, max_iter,
                x_min_hi, x_min_lo, dx_hi, dx_lo, y_max_hi, y_max_lo, dy_hi, dy_lo,
                factor, color_scheme_id, fractal_type, j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo,
                int(gen_n), float(gen_kr), float(gen_ki), 0, float(palette_offset),
                float(color_density), float(color_contrast)
            )

            edge_detect_kernel[blocks, threads, stream](
                d_iter_map, d_edge_mask, target_w, target_h, edge_thresh_f
            )

            mandelbrot_unified_dd_kernel[blocks, threads, stream](
                d_out_rgb, d_iter_map, d_edge_mask, d_lut, target_w, target_h, max_iter,
                x_min_hi, x_min_lo, dx_hi, dx_lo, y_max_hi, y_max_lo, dy_hi, dy_lo,
                factor, color_scheme_id, fractal_type, j_cx_hi, j_cx_lo, j_cy_hi, j_cy_lo,
                int(gen_n), float(gen_kr), float(gen_ki), 2, float(palette_offset),
                float(color_density), float(color_contrast)
            )
    elif use_floatexp:
        if bbsa_order in (4, 8):
            bbsa_order = 32
            if bbsa_tol > 1e-6:
                bbsa_tol = 1e-6
        strict_glitch_flag = 1 if ("Auto Rebasing" in glitch_mode or "Strict" in glitch_mode or "Pauldelbrot" in glitch_mode) else 0

        d_glitch_out = get_scratch_device_array((target_h, target_w), np.uint8, f"glitch_out_{stream_slot}")
        d_glitch_in = get_scratch_device_array((target_h, target_w), np.uint8, f"glitch_in_{stream_slot}")

        best_cx, best_cy, best_u, best_v, ref_re, ref_im, ref_len = find_optimal_reference_orbit(
            center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
            max_iter, target_w, target_h, fractal_type=fractal_type, julia_cx=julia_cx, julia_cy=julia_cy,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_floatexp=True
        )
        max_r_norm = float(w_norm) * 0.75
        bbsa_inv_r = 1.0 / max_r_norm if max_r_norm > 0.0 else 1.0
        d_ref_re_buf = get_scratch_device_array_1d(ref_len + 1, np.float64, f"ref_re_{stream_slot}")
        d_ref_im_buf = get_scratch_device_array_1d(ref_len + 1, np.float64, f"ref_im_{stream_slot}")
        d_ref_re = d_ref_re_buf[:ref_len + 1]
        d_ref_im = d_ref_im_buf[:ref_len + 1]
        d_ref_re.copy_to_device(ref_re[:ref_len + 1], stream=stream)
        d_ref_im.copy_to_device(ref_im[:ref_len + 1], stream=stream)
        bbsa_res = compute_bbsa_coeffs_cuda(
            d_ref_re, d_ref_im, ref_len, max_r_norm, fractal_type,
            bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, E_scale=E_scale, stream=stream,
            h_ref_re=ref_re, h_ref_im=ref_im
        )
        bbsa_skip = bbsa_res[0]
        bbsa_coeffs = bbsa_res[1:17]

        if bbsa_order == 32:
            bla_table = build_bla_table(
                ref_re, ref_im, ref_len, fractal_type=fractal_type, tol=bbsa_tol,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
            )
            d_bla_table = upload_bla_table_cuda(bla_table, stream_slot, stream=stream)
        else:
            d_bla_table = get_scratch_device_array((1, 1, 1), np.float64, f"bla_dummy_{stream_slot}")

        ref_px_sub = float(target_w - 1) * float(best_u) * float(factor)
        ref_py_sub = float(target_h - 1) * float(best_v) * float(factor)
        ref_cx_f = float(best_cx)
        ref_cy_f = float(best_cy)

        if factor == 1:
            mandelbrot_unified_floatexp_kernel[blocks, threads, stream](
                d_out_rgb, d_glitch_out, d_active_iter, d_dummy_edge, d_lut, target_w, target_h, max_iter,
                d_ref_re, d_ref_im, ref_len,
                ref_px_sub, ref_py_sub, dx_sub_norm_f, dy_sub_norm_f, E_scale,
                ref_cx_f, ref_cy_f, factor, color_scheme_id, fractal_type,
                float(julia_cx), float(julia_cy),
                bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                0, d_glitch_in, int(gen_n), float(gen_kr), float(gen_ki), 0, strict_glitch_flag,
                float(palette_offset), float(color_density), float(color_contrast),
                d_bla_table
            )
        else:
            d_iter_map = d_active_iter
            d_edge_mask = get_scratch_device_array((target_h, target_w), np.uint8, f"edge_mask_{stream_slot}")

            mandelbrot_unified_floatexp_kernel[blocks, threads, stream](
                d_out_rgb, d_glitch_out, d_iter_map, d_dummy_edge, d_lut, target_w, target_h, max_iter,
                d_ref_re, d_ref_im, ref_len,
                ref_px_sub, ref_py_sub, dx_sub_norm_f, dy_sub_norm_f, E_scale,
                ref_cx_f, ref_cy_f, factor, color_scheme_id, fractal_type,
                float(julia_cx), float(julia_cy),
                bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                0, d_glitch_in, int(gen_n), float(gen_kr), float(gen_ki), 0, strict_glitch_flag,
                float(palette_offset), float(color_density), float(color_contrast),
                d_bla_table
            )

            edge_detect_kernel[blocks, threads, stream](
                d_iter_map, d_edge_mask, target_w, target_h, edge_thresh_f
            )

            mandelbrot_unified_floatexp_kernel[blocks, threads, stream](
                d_out_rgb, d_glitch_out, d_iter_map, d_edge_mask, d_lut, target_w, target_h, max_iter,
                d_ref_re, d_ref_im, ref_len,
                ref_px_sub, ref_py_sub, dx_sub_norm_f, dy_sub_norm_f, E_scale,
                ref_cx_f, ref_cy_f, factor, color_scheme_id, fractal_type,
                float(julia_cx), float(julia_cy),
                bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                0, d_glitch_in, int(gen_n), float(gen_kr), float(gen_ki), 2, strict_glitch_flag,
                float(palette_offset), float(color_density), float(color_contrast),
                d_bla_table
            )
    else:
        strict_glitch_flag = 1 if ("Auto Rebasing" in glitch_mode or "Strict" in glitch_mode or "Pauldelbrot" in glitch_mode) else 0

        d_glitch_out = get_scratch_device_array((target_h, target_w), np.uint8, f"glitch_out_{stream_slot}")
        d_glitch_in = get_scratch_device_array((target_h, target_w), np.uint8, f"glitch_in_{stream_slot}")

        best_cx, best_cy, best_u, best_v, ref_re, ref_im, ref_len = find_optimal_reference_orbit(
            center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
            max_iter, target_w, target_h, fractal_type=fractal_type, julia_cx=julia_cx, julia_cy=julia_cy,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_floatexp=False
        )
        max_r = float(plot_width_dec) * 0.75
        bbsa_inv_r = 1.0 / max_r if max_r > 0.0 else 1.0
        d_ref_re_buf = get_scratch_device_array_1d(ref_len + 1, np.float64, f"ref_re_{stream_slot}")
        d_ref_im_buf = get_scratch_device_array_1d(ref_len + 1, np.float64, f"ref_im_{stream_slot}")
        d_ref_re = d_ref_re_buf[:ref_len + 1]
        d_ref_im = d_ref_im_buf[:ref_len + 1]
        d_ref_re.copy_to_device(ref_re[:ref_len + 1], stream=stream)
        d_ref_im.copy_to_device(ref_im[:ref_len + 1], stream=stream)
        bbsa_res = compute_bbsa_coeffs_cuda(
            d_ref_re, d_ref_im, ref_len, max_r, fractal_type,
            bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
            gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, E_scale=0, stream=stream,
            h_ref_re=ref_re, h_ref_im=ref_im
        )
        bbsa_skip = bbsa_res[0]
        bbsa_coeffs = bbsa_res[1:17]

        if bbsa_order == 32 and fractal_type in (0, 2, 4):
            bla_table = build_bla_table(
                ref_re, ref_im, ref_len, fractal_type=fractal_type, tol=bbsa_tol,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
            )
            d_bla_table = upload_bla_table_cuda(bla_table, stream_slot, stream=stream)
        else:
            d_bla_table = get_scratch_device_array((1, 1, 1), np.float64, f"bla_dummy_{stream_slot}")

        ref_px_sub = float(target_w - 1) * float(best_u) * float(factor)
        ref_py_sub = float(target_h - 1) * float(best_v) * float(factor)
        ref_cx_f = float(best_cx)
        ref_cy_f = float(best_cy)

        if factor == 1:
            mandelbrot_unified_perturbation_kernel[blocks, threads, stream](
                d_out_rgb, d_glitch_out, d_active_iter, d_dummy_edge, d_lut, target_w, target_h, max_iter,
                d_ref_re, d_ref_im, ref_len,
                ref_px_sub, ref_py_sub, dx_sub_f, dy_sub_f,
                ref_cx_f, ref_cy_f, factor, color_scheme_id, fractal_type,
                float(julia_cx), float(julia_cy), bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                0, d_glitch_in, int(gen_n), float(gen_kr), float(gen_ki), 0, strict_glitch_flag,
                float(palette_offset), float(color_density), float(color_contrast),
                d_bla_table
            )
        else:
            d_iter_map = d_active_iter
            d_edge_mask = get_scratch_device_array((target_h, target_w), np.uint8, f"edge_mask_{stream_slot}")

            mandelbrot_unified_perturbation_kernel[blocks, threads, stream](
                d_out_rgb, d_glitch_out, d_iter_map, d_dummy_edge, d_lut, target_w, target_h, max_iter,
                d_ref_re, d_ref_im, ref_len,
                ref_px_sub, ref_py_sub, dx_sub_f, dy_sub_f,
                ref_cx_f, ref_cy_f, factor, color_scheme_id, fractal_type,
                float(julia_cx), float(julia_cy), bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                0, d_glitch_in, int(gen_n), float(gen_kr), float(gen_ki), 0, strict_glitch_flag,
                float(palette_offset), float(color_density), float(color_contrast),
                d_bla_table
            )

            edge_detect_kernel[blocks, threads, stream](
                d_iter_map, d_edge_mask, target_w, target_h, edge_thresh_f
            )

            mandelbrot_unified_perturbation_kernel[blocks, threads, stream](
                d_out_rgb, d_glitch_out, d_iter_map, d_edge_mask, d_lut, target_w, target_h, max_iter,
                d_ref_re, d_ref_im, ref_len,
                ref_px_sub, ref_py_sub, dx_sub_f, dy_sub_f,
                ref_cx_f, ref_cy_f, factor, color_scheme_id, fractal_type,
                float(julia_cx), float(julia_cy), bbsa_skip, bbsa_inv_r, *bbsa_coeffs,
                0, d_glitch_in, int(gen_n), float(gen_kr), float(gen_ki), 2, strict_glitch_flag,
                float(palette_offset), float(color_density), float(color_contrast),
                d_bla_table
            )

    allow_rebasing = ("Auto Rebasing" in glitch_mode or "Strict" in glitch_mode or "Pauldelbrot" in glitch_mode)
    if (use_floatexp or use_perturbation) and allow_rebasing and (plot_width_dec < Decimal("1e-12")):
        is_exhaustive = ("Strict" in glitch_mode)
        max_rebase_passes = 4
        curr_pass = 0
        init_r = int(np.clip(ref_py_sub / factor, 0, target_h - 1))
        init_c = int(np.clip(ref_px_sub / factor, 0, target_w - 1))
        visited_rebase_coords = {(init_r, init_c)}
        prev_glitch_count = None
        no_progress_passes = 0

        while curr_pass < max_rebase_passes:
            h_glitch_map = d_glitch_out.copy_to_host(stream=stream)
            _cuda_sync_stream(stream)

            glitched_indices = np.argwhere(h_glitch_map > 0)
            num_glitches = len(glitched_indices)
            if num_glitches == 0:
                break

            if is_exhaustive and num_glitches <= 512:
                break

            if prev_glitch_count is not None and num_glitches >= prev_glitch_count:
                no_progress_passes += 1
                if no_progress_passes >= 2:
                    break
            else:
                no_progress_passes = 0
            prev_glitch_count = num_glitches

            curr_pass += 1

            # Cluster-targeted rebase point selection
            bin_r = np.clip((glitched_indices[:, 0] * 16) // target_h, 0, 15)
            bin_c = np.clip((glitched_indices[:, 1] * 16) // target_w, 0, 15)
            bin_idx = bin_r * 16 + bin_c
            counts = np.bincount(bin_idx, minlength=256)
            sorted_bins = np.argsort(-counts)

            chosen_pixel = None
            h_iter_map_tmp = None

            for b in sorted_bins:
                if counts[b] == 0:
                    break
                cluster_pixels = glitched_indices[bin_idx == b]

                if h_iter_map_tmp is None:
                    h_iter_map_tmp = d_active_iter.copy_to_host(stream=stream)
                    _cuda_sync_stream(stream)

                cluster_iters = h_iter_map_tmp[cluster_pixels[:, 0], cluster_pixels[:, 1]]
                center_r = np.median(cluster_pixels[:, 0])
                center_c = np.median(cluster_pixels[:, 1])

                # 1. First check for interior/un-escaped pixels in the glitched cluster
                interior_mask = (cluster_iters < 0) | (cluster_iters >= max_iter)
                if np.any(interior_mask):
                    int_pixels = cluster_pixels[interior_mask]
                    dists = (int_pixels[:, 0] - center_r)**2 + (int_pixels[:, 1] - center_c)**2
                    dist_order = np.argsort(dists)
                    for d_idx in dist_order:
                        cand_coord = (int(int_pixels[d_idx, 0]), int(int_pixels[d_idx, 1]))
                        if cand_coord not in visited_rebase_coords:
                            chosen_pixel = cand_coord
                            break

                # 2. If no unvisited interior pixels, select escaped candidate with highest iteration count
                if chosen_pixel is None:
                    escaped_mask = (cluster_iters > 0) & (cluster_iters < max_iter)
                    candidates = cluster_pixels[escaped_mask]
                    if len(candidates) > 0:
                        cand_iters = cluster_iters[escaped_mask]
                        best_cand_order = np.argsort(-cand_iters)
                        for c_idx in best_cand_order:
                            cand_coord = (int(candidates[c_idx, 0]), int(candidates[c_idx, 1]))
                            if cand_coord not in visited_rebase_coords:
                                chosen_pixel = cand_coord
                                break

                # 3. Fallback: closest candidate to cluster center
                if chosen_pixel is None:
                    dists = (cluster_pixels[:, 0] - center_r)**2 + (cluster_pixels[:, 1] - center_c)**2
                    dist_order = np.argsort(dists)
                    for d_idx in dist_order:
                        cand_coord = (int(cluster_pixels[d_idx, 0]), int(cluster_pixels[d_idx, 1]))
                        if cand_coord not in visited_rebase_coords:
                            chosen_pixel = cand_coord
                            break

                if chosen_pixel is not None:
                    break

            if chosen_pixel is None:
                for g_idx in range(len(glitched_indices)):
                    cand = (int(glitched_indices[g_idx, 0]), int(glitched_indices[g_idx, 1]))
                    if cand not in visited_rebase_coords:
                        chosen_pixel = cand
                        break

            if chosen_pixel is None:
                break

            g_row, g_col = chosen_pixel
            visited_rebase_coords.add((g_row, g_col))

            x_min_dec = center_x_dec - (plot_width_dec / Decimal(2))
            y_max_dec = center_y_dec + (plot_height_dec / Decimal(2))
            rebase_cx_dec = x_min_dec + (Decimal(int(g_col)) / Decimal(max(1, target_w - 1))) * plot_width_dec
            rebase_cy_dec = y_max_dec - (Decimal(int(g_row)) / Decimal(max(1, target_h - 1))) * plot_height_dec

            bits_needed = compute_required_prec_bits(plot_width_dec)
            r_re, r_im, r_len = compute_single_highprec_orbit(
                rebase_cx_dec, rebase_cy_dec, max_iter, fractal_type, julia_cx, julia_cy, bits_needed,
                gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
            )

            if r_len > 100:
                cache_reference_orbit(
                    rebase_cx_dec, rebase_cy_dec, plot_width_dec, max_iter,
                    r_re, r_im, r_len,
                    fractal_type=fractal_type, julia_cx=julia_cx, julia_cy=julia_cy,
                    gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, is_floatexp=use_floatexp
                )

            d_ref_re_buf = get_scratch_device_array_1d(r_len + 1, np.float64, f"rebase_re_{stream_slot}")
            d_ref_im_buf = get_scratch_device_array_1d(r_len + 1, np.float64, f"rebase_im_{stream_slot}")
            d_ref_re = d_ref_re_buf[:r_len + 1]
            d_ref_im = d_ref_im_buf[:r_len + 1]
            d_ref_re.copy_to_device(r_re[:r_len + 1], stream=stream)
            d_ref_im.copy_to_device(r_im[:r_len + 1], stream=stream)
            ref_px_rebase = float(g_col) * float(factor)
            ref_py_rebase = float(g_row) * float(factor)
            d_glitch_in.copy_to_device(h_glitch_map, stream=stream)
            
            rebase_mode = 0 if factor == 1 else 1

            if use_floatexp:
                r_bbsa_res = compute_bbsa_coeffs_cuda(
                    d_ref_re, d_ref_im, r_len, max_r_norm, fractal_type,
                    bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
                    gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, E_scale=E_scale, stream=stream,
                    h_ref_re=r_re, h_ref_im=r_im
                )
                r_bbsa_skip = r_bbsa_res[0]
                r_bbsa_coeffs = r_bbsa_res[1:17]

                if bbsa_order == 32 and fractal_type in (0, 2, 4):
                    r_bla_table = build_bla_table(
                        r_re, r_im, r_len, fractal_type=fractal_type, tol=bbsa_tol,
                        gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
                    )
                    d_bla_table = upload_bla_table_cuda(r_bla_table, stream_slot, stream=stream)
                else:
                    d_bla_table = get_scratch_device_array((1, 1, 1), np.float64, f"bla_dummy_{stream_slot}")

                if color_scheme_id == 11 and d_cdf_lut is not None:
                    ssaa_resolve_hist_floatexp_kernel[blocks, threads, stream](
                        d_out_rgb, d_glitch_out, d_iter_map, d_dummy_edge, d_cdf_lut, d_lut, target_w, target_h, max_iter,
                        d_ref_re, d_ref_im, r_len,
                        ref_px_rebase, ref_py_rebase,
                        dx_sub_norm_f, dy_sub_norm_f, E_scale,
                        float(rebase_cx_dec), float(rebase_cy_dec), factor, fractal_type,
                        float(julia_cx), float(julia_cy),
                        r_bbsa_skip, bbsa_inv_r, *r_bbsa_coeffs,
                        curr_pass, d_glitch_in, int(gen_n), float(gen_kr), float(gen_ki), strict_glitch_flag,
                        num_bins, float(palette_offset), float(color_density), float(color_contrast)
                    )
                else:
                    mandelbrot_unified_floatexp_kernel[blocks, threads, stream](
                        d_out_rgb, d_glitch_out, d_active_iter, d_dummy_edge, d_lut, target_w, target_h, max_iter,
                        d_ref_re, d_ref_im, r_len,
                        ref_px_rebase, ref_py_rebase,
                        dx_sub_norm_f, dy_sub_norm_f, E_scale,
                        float(rebase_cx_dec), float(rebase_cy_dec), factor, color_scheme_id, fractal_type,
                        float(julia_cx), float(julia_cy),
                        r_bbsa_skip, bbsa_inv_r, *r_bbsa_coeffs,
                        curr_pass, d_glitch_in, int(gen_n), float(gen_kr), float(gen_ki), rebase_mode, strict_glitch_flag,
                        float(palette_offset), float(color_density), float(color_contrast),
                        d_bla_table
                    )
            else:
                r_bbsa_res = compute_bbsa_coeffs_cuda(
                    d_ref_re, d_ref_im, r_len, max_r, fractal_type,
                    bbsa_tol=bbsa_tol, bbsa_order=bbsa_order,
                    gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, E_scale=0, stream=stream,
                    h_ref_re=r_re, h_ref_im=r_im
                )
                r_bbsa_skip = r_bbsa_res[0]
                r_bbsa_coeffs = r_bbsa_res[1:17]

                if bbsa_order == 32:
                    r_bla_table = build_bla_table(
                        r_re, r_im, r_len, fractal_type=fractal_type, tol=bbsa_tol,
                        gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki
                    )
                    d_bla_table = upload_bla_table_cuda(r_bla_table, stream_slot, stream=stream)
                else:
                    d_bla_table = get_scratch_device_array((1, 1, 1), np.float64, f"bla_dummy_{stream_slot}")

                if color_scheme_id == 11 and d_cdf_lut is not None:
                    ssaa_resolve_hist_perturbation_kernel[blocks, threads, stream](
                        d_out_rgb, d_glitch_out, d_iter_map, d_dummy_edge, d_cdf_lut, d_lut, target_w, target_h, max_iter,
                        d_ref_re, d_ref_im, r_len,
                        ref_px_rebase, ref_py_rebase,
                        dx_sub_f, dy_sub_f,
                        float(rebase_cx_dec), float(rebase_cy_dec), factor, fractal_type,
                        float(julia_cx), float(julia_cy), r_bbsa_skip, bbsa_inv_r, *r_bbsa_coeffs,
                        curr_pass, d_glitch_in, int(gen_n), float(gen_kr), float(gen_ki), strict_glitch_flag,
                        num_bins, float(palette_offset), float(color_density), float(color_contrast)
                    )
                else:
                    mandelbrot_unified_perturbation_kernel[blocks, threads, stream](
                        d_out_rgb, d_glitch_out, d_active_iter, d_dummy_edge, d_lut, target_w, target_h, max_iter,
                        d_ref_re, d_ref_im, r_len,
                        ref_px_rebase, ref_py_rebase,
                        dx_sub_f, dy_sub_f,
                        float(rebase_cx_dec), float(rebase_cy_dec), factor, color_scheme_id, fractal_type,
                        float(julia_cx), float(julia_cy), r_bbsa_skip, bbsa_inv_r, *r_bbsa_coeffs,
                        curr_pass, d_glitch_in, int(gen_n), float(gen_kr), float(gen_ki), rebase_mode, strict_glitch_flag,
                        float(palette_offset), float(color_density), float(color_contrast),
                        d_bla_table
                    )

        if is_exhaustive:
            h_glitch_map = d_glitch_out.copy_to_host(stream=stream)
            _cuda_sync_stream(stream)
            glitched_indices = np.argwhere(h_glitch_map > 0)
            if len(glitched_indices) > 0:
                eval_glitched_pixels_fallback_cuda(
                    d_out_rgb, d_glitch_out, d_lut, glitched_indices,
                    target_w, target_h, max_iter,
                    center_x_dec, center_y_dec, plot_width_dec, plot_height_dec,
                    color_scheme_id, fractal_type, julia_cx, julia_cy,
                    gen_n=gen_n, gen_kr=gen_kr, gen_ki=gen_ki, palette_offset=palette_offset,
                    color_density=color_density, color_contrast=color_contrast,
                    d_cdf_lut=d_cdf_lut, num_bins=num_bins if color_scheme_id == 11 else 2048,
                    cmap_name=cmap_name, is_hdr=is_hdr, stream=stream,
                    d_iter_map=d_active_iter, stream_slot=stream_slot
                )

    out_ary = d_out_rgb.copy_to_host(ary=h_rgb_buf, stream=stream)
    if sync_stream:
        _cuda_sync_stream(stream)
    return out_ary


def recolor_frame(params: RenderParameters, d_iter_buf=None, h_iter_buf=None):
    """
    Instantly recolors an already-computed frame from its cached iteration map (~0.08ms GPU / ~1ms CPU).
    Does NOT recompute reference orbits, BLA tables, or pixel iterations.
    """
    target_w = params.target_w
    target_h = params.target_h
    cmap_name = params.cmap_name
    color_scheme_id = params.color_scheme_id
    palette_offset = params.palette_offset
    color_density = params.color_density
    color_contrast = params.color_contrast
    is_hdr = params.is_hdr
    compute_device = params.compute_device
    stream = params.stream if params.stream is not None else 0
    sync_stream = getattr(params, 'sync_stream', True)

    d_iter = d_iter_buf if d_iter_buf is not None else getattr(params, 'd_iter_buf', None)
    h_iter = h_iter_buf if h_iter_buf is not None else getattr(params, 'h_iter_buf', None)

    if compute_device == "GPU" and HAS_CUDA and d_iter is not None:
        stream_slot = get_stream_slot(stream, explicit_slot=getattr(params, 'stream_slot', None))
        dtype_val = np.uint16 if is_hdr else np.uint8
        d_out_rgb = params.d_rgb_buf if params.d_rgb_buf is not None else get_scratch_device_array((target_h, target_w, 3), dtype_val, f"out_rgb_{stream_slot}")
        d_lut = get_gpu_cmap_lut(cmap_name, lut_size=2048, is_hdr=is_hdr)

        threads = (16, 16)
        blocks = ((target_w + 15) // 16, (target_h + 15) // 16)

        if color_scheme_id == 11:
            num_bins = 65536
            d_hist = get_scratch_device_array((num_bins,), np.int32, f"hist_bins_{stream_slot}")
            d_block_sums = get_scratch_device_array((128,), np.float64, f"block_sums_{stream_slot}")
            d_scan_out = get_scratch_device_array((num_bins,), np.float64, f"scan_out_{stream_slot}")
            d_cdf_lut = get_scratch_device_array((num_bins,), np.float32, f"cdf_lut_{stream_slot}")

            zero_histogram_cuda_kernel[128, 512, stream](d_hist, num_bins)
            build_histogram_kernel[blocks, threads, stream](
                d_iter, d_hist, target_w, target_h, params.max_iter, num_bins
            )
            scan_blocks_cuda_kernel[128, 512, stream](d_hist, d_block_sums, d_scan_out)
            scan_single_block_cuda_kernel[1, 128, stream](d_block_sums, 128)
            finalize_normalized_cdf_cuda_kernel[128, 512, stream](d_scan_out, d_block_sums, d_cdf_lut, num_bins)
            apply_histogram_equalization_kernel[blocks, threads, stream](
                d_out_rgb, d_iter, d_cdf_lut, d_lut, target_w, target_h, params.max_iter, num_bins,
                float(palette_offset), float(color_density), float(color_contrast)
            )
        else:
            recolor_iter_map_cuda_kernel[blocks, threads, stream](
                d_out_rgb, d_iter, d_lut, target_w, target_h,
                int(color_scheme_id), float(palette_offset),
                float(color_density), float(color_contrast)
            )

        out_ary = d_out_rgb.copy_to_host(ary=params.h_rgb_buf, stream=stream)
        if sync_stream:
            _cuda_sync_stream(stream)
        return out_ary
    elif h_iter is not None:
        dtype_val = np.uint16 if is_hdr else np.uint8
        if params.h_rgb_buf is not None and params.h_rgb_buf.shape == (target_h, target_w, 3) and params.h_rgb_buf.dtype == dtype_val:
            out_rgb = params.h_rgb_buf
        else:
            out_rgb = np.empty((target_h, target_w, 3), dtype=dtype_val)
        lut = get_cpu_cmap_lut(cmap_name, lut_size=2048, is_hdr=is_hdr)
        set_num_threads(max(1, min(MAX_SYSTEM_CORES, int(params.cpu_threads))))
        if color_scheme_id == 11:
            num_bins = 65536
            hist = np.zeros(num_bins, dtype=np.int64)
            cpu_build_histogram(h_iter, hist, target_w, target_h, params.max_iter, num_bins)
            cdf = np.cumsum(hist, dtype=np.float64)
            total_esc = cdf[-1]
            cdf_lut = (cdf / total_esc).astype(np.float32) if total_esc > 0 else np.linspace(0.0, 1.0, num_bins, dtype=np.float32)
            cpu_recolor_hist_kernel(
                out_rgb, h_iter, cdf_lut, lut, target_w, target_h, params.max_iter, num_bins,
                float(palette_offset), float(color_density), float(color_contrast)
            )
        else:
            cpu_recolor_iter_map_kernel(
                out_rgb, h_iter, lut, target_w, target_h,
                int(color_scheme_id), float(palette_offset),
                float(color_density), float(color_contrast)
            )
        return out_rgb
    else:
        return render_frame(params)