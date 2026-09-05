import os
import sys
import json
import math
from dataclasses import dataclass, asdict
from decimal import Decimal, getcontext
import numpy as np
import psutil

import decimal

# Set high decimal precision context globally and for DefaultContext (inherited by all new threads)
decimal.DefaultContext.prec = 500
getcontext().prec = 500

# --- System Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_SYSTEM_CORES = psutil.cpu_count(logical=True) or 4
GLOBAL_MAX_ITER = 500_000_000
DEFAULT_MAX_ITER = 500
DEFAULT_COLOR_DENSITY = 1.0
DEFAULT_COLOR_CONTRAST = 1.0

# Math constants
INV_LN2_F32 = 1.4426950408889634  # 1.0 / ln(2)
LN_BAILOUT_LOG = 1.7129177284488854  # ln(0.5 * ln(65536.0)) = ln(8 * ln(2))

# GMPY2 Detection
try:
    import gmpy2
    HAS_GMPY2 = True
except ImportError:
    HAS_GMPY2 = False


def compute_required_prec_bits(plot_width_dec):
    """Calculates optimal GMPY2/Decimal precision bits based on zoom depth without excessive limb overhead."""
    try:
        p = Decimal(str(plot_width_dec))
        if p > 0:
            adj = -int(p.adjusted())
            # Guard bits ensuring numeric stability over long iteration chains at ultra-deep zooms
            bits = int(adj * 3.321928094887362) + 128
            return max(64, bits)
        else:
            return 256
    except Exception:
        return 256


def compute_floatexp_scale(plot_width_dec):
    """Computes base-2 exponent scale E_scale such that delta_c * 2^E_scale is in O(1) range."""
    try:
        p = Decimal(str(plot_width_dec))
        if p <= 0:
            return 0
        adj = p.adjusted()
        return max(0, int(-adj * 3.32192809488736235))
    except Exception:
        return 0


def ensure_decimal_precision(plot_width_dec, extra_digits=150):
    """Dynamically scales Decimal getcontext().prec and DefaultContext.prec if zoom depth exceeds default precision."""
    try:
        p = Decimal(str(plot_width_dec))
        if p > 0:
            needed = max(500, -int(p.adjusted()) + extra_digits)
            if decimal.DefaultContext.prec < needed:
                decimal.DefaultContext.prec = needed
            if getcontext().prec < needed:
                getcontext().prec = needed
    except Exception:
        pass


# --- Fractal Definitions & Defaults ---
FRACTAL_NAMES = [
    "Mandelbrot",
    "Burning Ship",
    "Julia Set (Mandelbrot)",
    "Julia Set (Burning Ship)",
    "General Mandelbrot"
]

DEFAULT_JULIA_MANDELBROT_VIEW = (Decimal("0.0"), Decimal("0.0"), Decimal("3.5"))
DEFAULT_JULIA_BURNING_SHIP_VIEW = (Decimal("0.0"), Decimal("0.0"), Decimal("4.0"))
DEFAULT_JULIA_VIEW = DEFAULT_JULIA_MANDELBROT_VIEW

DEFAULT_JULIA_MANDELBROT_PARAMS = (Decimal("-0.67"), Decimal("0.37"))
DEFAULT_JULIA_BURNING_SHIP_PARAMS = (Decimal("0.35"), Decimal("-0.05"))

FRACTAL_DEFAULTS = {
    0: (Decimal("-0.65"), Decimal("0.0"), Decimal("4.0")),
    1: (Decimal("-0.45"), Decimal("0.5"), Decimal("3.5")),
    2: DEFAULT_JULIA_MANDELBROT_VIEW,
    3: DEFAULT_JULIA_BURNING_SHIP_VIEW,
    4: (Decimal("0.0"), Decimal("0.0"), Decimal("4.0"))
}

# --- Color Scheme Identifiers ---
COLOR_SCHEMES = {
    "Sqrt Fold": 0,
    "Logarithmic": 1,
    "Sigmoidal Tanh": 2,
    "Exponential Flare": 3,
    "High-Frequency Ripples": 4,
    "Golden Ratio Dual Sine": 5,
    "Cosine Smooth": 6,
    "Harmonic Cascade": 7,
    "Quintic S-Curve": 8,
    "Chirp Sweep": 9,
    "Soft Damping": 10,
    "Histogram Equalized": 11
}

# --- Precision & Algorithm Options ---
PRECISION_OPTIONS = {
    "1e-12 (FP64)": Decimal("1e-12"),
    "1e-24 (DD)": Decimal("1e-24"),
    "1e-300 (Perturbation)": Decimal("1e-300"),
    "1e-10000+ (Floatexp)": Decimal("1e-646000000")
}

BBSA_OPTIONS = {
    "Off": {"order": 0, "tol": 0.0},
    "4th-order": {"order": 4, "tol": 1e-4},
    "8th-order": {"order": 8, "tol": 1e-4},
    "Chained BLA": {"order": 32, "tol": 1e-6}
}

GLITCH_CORRECTION_MODES = [
    "Off (Single-Ref)",
    "Pauldelbrot",
    "Strict Metric"
]

DYNAMIC_ITER_MODES = [
    "Off",
    "On (Sqrt Scaling)",
    "On (Linear Scaling)",
    "On (Adaptive Preview)"
]

VIDEO_ENCODER_OPTIONS = [
    "NVIDIA NVENC (Hardware H.264 / HEVC / AV1)",
    "Software (CPU libx264 / libx265)",
    "AV1 (CPU libaom-av1)",
    "VP9 WebM (CPU libvpx-vp9)"
]


# --- Presets Loader ---
def load_builtin_presets():
    presets_path = os.path.join(SCRIPT_DIR, "presets.json")
    if os.path.exists(presets_path):
        try:
            with open(presets_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data.get("bookmarks", [])
                elif isinstance(data, list):
                    return data
        except Exception:
            pass
    return [
        {
            "name": "Classic Overview",
            "fractal_type": 0,
            "center_x": "-0.65",
            "center_y": "0.0",
            "plot_width": "4.0",
            "max_iter": 1000,
            "cmap_name": "inferno",
            "color_scheme_id": 0,
            "precision_mode": "1e-300 (Perturbation)",
            "palette_offset": 0.0,
            "bbsa_accuracy": "4th-order",
            "glitch_mode": "Pauldelbrot",
            "dynamic_iter_mode": "Off",
            "ssaa_factor": 1,
            "edge_threshold": 0.35,
            "julia_cx": "-0.67",
            "julia_cy": "0.37",
            "gen_n": 3,
            "gen_kr": 0.25,
            "gen_ki": 1.0
        }
    ]


BUILTIN_PRESETS = load_builtin_presets()


# --- Render Parameters Dataclass ---
@dataclass
class RenderParameters:
    target_w: int = 1280
    target_h: int = 720
    max_iter: int = 1000
    center_x_dec: Decimal = Decimal("-0.75")
    center_y_dec: Decimal = Decimal("0.0")
    plot_width_dec: Decimal = Decimal("4.0")
    cmap_name: str = "inferno"
    ssaa_factor: int = 1
    edge_threshold: float = 0.35
    color_scheme_id: int = 0
    precision_mode: str = "1e-300 (Perturbation)"
    fractal_type: int = 0
    julia_cx: float = 0.0
    julia_cy: float = 0.0
    bbsa_tol: float = 1e-4
    bbsa_order: int = 4
    glitch_mode: str = "Off (Single-Ref)"
    gen_n: int = 3
    gen_kr: float = 0.25
    gen_ki: float = 1.0
    d_rgb_buf: object = None
    h_rgb_buf: object = None
    stream: object = 0
    is_hdr: bool = False
    compute_device: str = "GPU"
    cpu_threads: int = 2
    palette_offset: float = 0.0
    color_density: float = 1.0
    color_contrast: float = 1.0
    sync_stream: bool = True
    cancel_token: object = None
    d_iter_buf: object = None
    h_iter_buf: object = None
    stream_slot: int = 0

    def to_dict(self):
        return asdict(self)

    def as_dict(self):
        return asdict(self)

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, key, default=None):
        return getattr(self, key, default)


# --- Utility Functions ---

def format_eta_time(seconds):
    """Formats raw seconds into human-readable duration strings (e.g. '1m 23s', '4h 12m')."""
    if seconds is None or math.isnan(seconds) or math.isinf(seconds) or seconds < 0 or seconds > 8640000:
        return "--:--"
    sec = int(round(seconds))
    if sec < 60:
        return f"{sec}s"
    elif sec < 3600:
        m = sec // 60
        s = sec % 60
        return f"{m}m {s:02d}s"
    else:
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h}h {m:02d}m {s:02d}s"


def get_real_world_scale(plot_width):
    """
    Calculates the apparent size of the entire fractal as the user zooms in,
    assuming the full fractal overview (w = 4.0) corresponds to 30 cm (0.30 m) on screen.
    As you zoom in (w decreases), the apparent full fractal size increases:
        meters = (4.0 / w) * 0.30 m = 1.20 / w
    Formats using units >= 1 cm (cm, m, km, AU, ly, MW, U).
    Handles arbitrary-precision Decimal zoom depths (e.g. 1e-10000) without float underflow.
    """
    try:
        w_dec = Decimal(str(plot_width))
        if w_dec <= Decimal("0"):
            return "30.00 cm"

        ensure_decimal_precision(w_dec, extra_digits=50)
        meters = Decimal("1.20") / w_dec

        # Astronomical scales in Decimal
        DEC_U = Decimal("8.79847933896e26")
        DEC_MW = Decimal("9.460730472e20")
        DEC_LY = Decimal("9.460730472e15")
        DEC_AU = Decimal("1.495978707e11")
        DEC_KM = Decimal("1.0e3")
        DEC_M = Decimal("1.0")

        if meters >= Decimal("8.79847933896e32"):
            val = meters / DEC_U
            return f"{val:.2e} U"
        elif meters >= DEC_U:
            val = meters / DEC_U
            return f"{val:.2f} U"
        elif meters >= DEC_MW:
            val = meters / DEC_MW
            return f"{val:.2f} MW"
        elif meters >= DEC_LY:
            val = meters / DEC_LY
            return f"{val:.2f} ly"
        elif meters >= DEC_AU:
            val = meters / DEC_AU
            return f"{val:.2f} AU"
        elif meters >= Decimal("1.0e9"):
            val = meters / DEC_KM
            return f"{val:.2e} km"
        elif meters >= DEC_KM:
            val = meters / DEC_KM
            return f"{val:.2f} km"
        elif meters >= DEC_M:
            return f"{meters:.2f} m"
        else:
            cm = meters * Decimal("100.0")
            return f"{cm:.2f} cm"
    except Exception:
        try:
            return f"{Decimal(str(plot_width)):.2e}"
        except Exception:
            return "30.00 cm"


def parse_real_world_scale(scale_str):
    """
    Parses a scale string (e.g. '30 cm', '300 cm', '3 m', '500 km', '1.5 AU', '2.3 ly', '5.2 MW', '1.0 U', '1e500 U')
    back into complex plot width assuming the full overview (w = 4.0) is 30 cm (0.30 m):
        w = (4.0 * 0.30) / meters = 1.20 / meters
    Handles arbitrary-precision Decimal scales.
    """
    scale_str = str(scale_str).strip()
    if not scale_str:
        return Decimal("4.0")

    units_table = {
        'u': Decimal("8.79847933896e26"),
        'mw': Decimal("9.460730472e20"),
        'ly': Decimal("9.460730472e15"),
        'au': Decimal("1.495978707e11"),
        'km': Decimal("1.0e3"),
        'm': Decimal("1.0"),
        'cm': Decimal("0.01")
    }

    cleaned = scale_str.lower()
    for unit, factor in sorted(units_table.items(), key=lambda x: -len(x[0])):
        if cleaned.endswith(unit):
            num_part = cleaned[:-len(unit)].strip()
            try:
                num_dec = Decimal(num_part)
                meters = num_dec * factor
                if meters > Decimal("0"):
                    ensure_decimal_precision(Decimal("1.0") / meters, extra_digits=50)
                    w_dec = Decimal("1.20") / meters
                    return w_dec
            except Exception:
                pass

    try:
        w_val = Decimal(scale_str)
        return w_val if w_val > Decimal("0") else Decimal("4.0")
    except Exception:
        return Decimal("4.0")


def catmull_rom_spline(p0, p1, p2, p3, t):
    """
    Evaluates standard Centripetal/Uniform Catmull-Rom spline at parameter t in [0, 1].
    Supports both scalar Decimal/float values and vector tuples/lists (cx, cy, w) with full type-safety.
    """
    t_dec = Decimal(str(t))
    t2 = t_dec * t_dec
    t3 = t2 * t_dec

    is_scalar = not hasattr(p0, '__len__')
    pts0 = [p0] if is_scalar else p0
    pts1 = [p1] if is_scalar else p1
    pts2 = [p2] if is_scalar else p2
    pts3 = [p3] if is_scalar else p3

    res = []
    for i in range(len(pts0)):
        v0 = Decimal(str(pts0[i]))
        v1 = Decimal(str(pts1[i]))
        v2 = Decimal(str(pts2[i]))
        v3 = Decimal(str(pts3[i]))

        val = Decimal("0.5") * (
            (Decimal("2.0") * v1) +
            (-v0 + v2) * t_dec +
            (Decimal("2.0") * v0 - Decimal("5.0") * v1 + Decimal("4.0") * v2 - v3) * t2 +
            (-v0 + Decimal("3.0") * v1 - Decimal("3.0") * v2 + v3) * t3
        )
        res.append(val)

    return res[0] if is_scalar else tuple(res)


def check_ram_sufficiency(*args, **kwargs):
    """
    Validates whether system RAM is sufficient for high-resolution rendering.
    Supports two calling conventions:
      1. UI Dialog Prompt: check_ram_sufficiency(parent_widget, required_ram_bytes, task_name="Export Task") -> bool
      2. Resolution Check: check_ram_sufficiency(target_w, target_h, is_hdr=False, ssaa_factor=1) -> (is_sufficient, req_gb, avail_gb)
    """
    available_bytes = psutil.virtual_memory().available

    # Case 1: UI Prompt convention: (parent, required_bytes, task_name)
    if len(args) >= 2 and (not isinstance(args[0], (int, float)) or isinstance(args[1], (int, float, str)) and len(args) >= 3 and isinstance(args[2], str)):
        parent = args[0]
        try:
            required_bytes = int(args[1])
        except Exception:
            required_bytes = 0
        task_name = str(args[2]) if len(args) > 2 else kwargs.get("task_name", "Export Task")

        if required_bytes > 0 and available_bytes < required_bytes:
            req_gb = required_bytes / (1024 ** 3)
            avail_gb = available_bytes / (1024 ** 3)
            try:
                from PyQt5.QtWidgets import QMessageBox
                ret = QMessageBox.question(
                    parent if hasattr(parent, 'winId') else None,
                    "High RAM Usage Warning",
                    f"The requested {task_name} requires approximately {req_gb:.2f} GB of RAM, "
                    f"but only {avail_gb:.2f} GB is currently available.\n\n"
                    f"Proceeding may cause significant disk paging or an out-of-memory error.\n\n"
                    f"Do you want to proceed anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                return ret == QMessageBox.Yes
            except Exception:
                return True
        return True

    # Case 2: Resolution convention: (target_w, target_h, is_hdr, ssaa_factor)
    target_w = args[0] if len(args) > 0 else kwargs.get('target_w', 1280)
    target_h = args[1] if len(args) > 1 else kwargs.get('target_h', 720)
    is_hdr = args[2] if len(args) > 2 else kwargs.get('is_hdr', False)
    ssaa_factor = args[3] if len(args) > 3 else kwargs.get('ssaa_factor', 1)

    try:
        bytes_per_sample = 2 if is_hdr else 1
        total_pixels = int(target_w) * int(target_h) * (max(1, int(ssaa_factor)) ** 2)
        raw_frame_bytes = total_pixels * 3 * bytes_per_sample
        estimated_required_bytes = raw_frame_bytes * 4

        required_gb = estimated_required_bytes / (1024 ** 3)
        available_gb = available_bytes / (1024 ** 3)
        return (available_bytes >= estimated_required_bytes), required_gb, available_gb
    except Exception:
        return True, 0.0, available_bytes / (1024 ** 3)


def get_active_engine_name(plot_width_dec, precision_mode="1e-300 (Perturbation)"):
    """Determines the active computation engine name based on zoom depth and precision mode."""
    try:
        w_dec = Decimal(str(plot_width_dec))
        if w_dec < Decimal("1e-300"):
            return "FloatExp"
        elif w_dec >= Decimal("1e-12"):
            if "dd" in str(precision_mode).lower():
                return "Double-Double"
            return "FP64"
        elif w_dec >= Decimal("1e-24") and "dd" in str(precision_mode).lower():
            return "Double-Double"
        else:
            return "Perturbation"
    except Exception:
        return "FP64"


def draw_video_hud_overlay(img_array, plot_width_dec, cx=None, cy=None, is_hdr=False, max_iter=None):
    """
    Draws a compact telemetry glassmorphic HUD badge at the top-left corner on host image arrays.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        h, w = img_array.shape[:2]
        
        # Format strings
        scale_txt = get_real_world_scale(plot_width_dec)
        w_txt = f"{Decimal(str(plot_width_dec)):.3e}"
        iter_txt = f" | Iter: {int(float(max_iter)):,}" if max_iter is not None else ""
        if cx is not None and cy is not None:
            coord_txt = f"Center: ({Decimal(str(cx)):.6e}, {Decimal(str(cy)):.6e}) | Width: {w_txt} | Scale: {scale_txt}{iter_txt}"
        else:
            coord_txt = f"Width: {w_txt} | Scale: {scale_txt}{iter_txt}"

        # Font sizing based on resolution
        font_size = max(14, int(h * 0.022))
        try:
            font = ImageFont.truetype("segoeui.ttf", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

        # Measure text boundaries to fit background tightly and calculate symmetric padding
        bbox = (0, 0, len(coord_txt) * int(font_size * 0.6), font_size)
        try:
            dummy_img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            dummy_draw = ImageDraw.Draw(dummy_img)
            bbox = dummy_draw.textbbox((0, 0), coord_txt, font=font)
        except Exception:
            try:
                tw, th = font.getsize(coord_txt)
                bbox = (0, 0, tw, th)
            except Exception:
                pass

        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        margin_x = max(12, int(w * 0.012))
        margin_y = max(12, int(h * 0.015))
        pad_x = max(12, int(font_size * 0.55))
        pad_y = max(8, int(font_size * 0.35))

        rect_x0 = margin_x
        rect_y0 = margin_y
        rect_x1 = margin_x + text_w + 2 * pad_x
        rect_y1 = margin_y + text_h + 2 * pad_y
        radius = max(4, int(font_size * 0.25))

        badge_w = rect_x1 - rect_x0 + 1
        badge_h = rect_y1 - rect_y0 + 1

        # Render HUD badge strictly onto a compact local ROI image
        badge_img = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(badge_img)

        try:
            draw.rounded_rectangle([(0, 0), (badge_w - 1, badge_h - 1)], radius=radius, fill=(15, 23, 42, 195), outline=(51, 65, 85, 220), width=1)
        except Exception:
            draw.rectangle([(0, 0), (badge_w - 1, badge_h - 1)], fill=(15, 23, 42, 195), outline=(51, 65, 85, 220))

        tx = pad_x - bbox[0]
        ty = pad_y - bbox[1]
        draw.text((tx, ty), coord_txt, fill=(248, 250, 252, 255), font=font)

        badge_arr = np.array(badge_img)
        alpha = badge_arr[:, :, 3:4].astype(np.float32) / 255.0
        badge_rgb = badge_arr[:, :, :3].astype(np.float32)

        dst_y0 = max(0, min(h, rect_y0))
        dst_y1 = max(0, min(h, rect_y0 + badge_h))
        dst_x0 = max(0, min(w, rect_x0))
        dst_x1 = max(0, min(w, rect_x0 + badge_w))

        sl_h = dst_y1 - dst_y0
        sl_w = dst_x1 - dst_x0
        if sl_h <= 0 or sl_w <= 0:
            return

        alpha_crop = alpha[:sl_h, :sl_w]
        badge_rgb_crop = badge_rgb[:sl_h, :sl_w]
        dest_crop = img_array[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)

        if is_hdr or img_array.dtype == np.uint16:
            badge_rgb_crop = badge_rgb_crop * 257.0
            blended = dest_crop * (1.0 - alpha_crop) + badge_rgb_crop * alpha_crop
            img_array[dst_y0:dst_y1, dst_x0:dst_x1] = np.clip(blended, 0.0, 65535.0).astype(np.uint16)
        else:
            blended = dest_crop * (1.0 - alpha_crop) + badge_rgb_crop * alpha_crop
            img_array[dst_y0:dst_y1, dst_x0:dst_x1] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    except Exception:
        pass
