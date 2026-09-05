from config import RenderParameters
from math_core.cuda_kernels import HAS_CUDA, init_primary_cuda_device
from math_core.engine import (
    clear_scratch_buffer_pool,
    compute_bbsa_coeffs_cuda,
    compute_dynamic_max_iter,
    eval_glitched_pixels_fallback_cuda,
    get_scratch_device_array,
    probe_adaptive_max_iter,
    render_frame,
    render_frame_cpu,
    render_frame_ssaa,
)
from math_core.high_precision import (
    cache_reference_orbit,
    compute_bbsa_coeffs,
    compute_highprec_escape_scalar,
    compute_single_highprec_orbit,
    find_optimal_reference_orbit,
    to_dd,
)

__all__ = [
    "HAS_CUDA",
    "init_primary_cuda_device",
    "RenderParameters",
    "render_frame",
    "render_frame_ssaa",
    "render_frame_cpu",
    "compute_dynamic_max_iter",
    "probe_adaptive_max_iter",
    "eval_glitched_pixels_fallback_cuda",
    "get_scratch_device_array",
    "clear_scratch_buffer_pool",
    "to_dd",
    "compute_bbsa_coeffs",
    "compute_bbsa_coeffs_cuda",
    "compute_highprec_escape_scalar",
    "compute_single_highprec_orbit",
    "find_optimal_reference_orbit",
    "cache_reference_orbit",
]