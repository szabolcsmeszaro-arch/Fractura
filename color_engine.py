import os
import types
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
from PyQt5.QtGui import QImage

PALETTABLE_CMAPS = {}
CATEGORY_ORDER = [
    "★ Popular / Recommended", "CartoColors", "cmocean", "ColorBrewer2",
    "Cubehelix", "Light & Bartlein", "matplotlib", "MyCarta", "Scientific", "Tableau"
]
CATEGORIZED_CMAPS = {cat: [] for cat in CATEGORY_ORDER}

POPULAR_CANDIDATES = [
    'inferno', 'magma', 'plasma', 'hot', 'viridis', 'cividis',
    'twilight_shifted', 'ocean', 'coolwarm',
    'cubehelix', 'rainbow', 'colorbrewer.diverging.RdBu_10',
    'colorbrewer.diverging.RdYlBu_10', 'colorbrewer.diverging.PuOr_10', 
    'colorbrewer.diverging.Spectral_10', 'colorbrewer.sequential.Greys_9_r', 
    'cmocean.sequential.Ice_10'
]

PALETTABLE_MAP = {
    'cartocolors': 'CartoColors', 'cmocean': 'cmocean', 'colorbrewer': 'ColorBrewer2',
    'cubehelix': 'Cubehelix', 'lightbartlein': 'Light & Bartlein', 'matplotlib': 'matplotlib',
    'mycarta': 'MyCarta', 'scientific': 'Scientific', 'tableau': 'Tableau'
}

def test_colormap_callable(cmap_obj):
    if not callable(cmap_obj):
        return False
    try:
        sample = cmap_obj(np.array([0.0, 0.5, 1.0]))
        return isinstance(sample, np.ndarray) and sample.shape == (3, 4)
    except Exception:
        return False

try:
    import palettable
    visited_modules = set()

    def harvest_palettable(mod, prefix=""):
        if mod in visited_modules:
            return
        visited_modules.add(mod)
        for attr_name in dir(mod):
            if attr_name.startswith('_'):
                continue
            try:
                obj = getattr(mod, attr_name)
            except Exception:
                continue
            if isinstance(obj, types.ModuleType) and obj.__name__.startswith('palettable'):
                harvest_palettable(obj, f"{prefix}{attr_name}." if prefix else f"{attr_name}.")
            elif hasattr(obj, 'mpl_colormap'):
                cmap_candidate = getattr(obj, 'mpl_colormap', None)
                if test_colormap_callable(cmap_candidate):
                    key = f"{prefix}{attr_name}" if prefix else attr_name
                    PALETTABLE_CMAPS[key] = cmap_candidate
                    target_cat = PALETTABLE_MAP.get(key.split('.')[0].lower())
                    if target_cat in CATEGORIZED_CMAPS:
                        CATEGORIZED_CMAPS[target_cat].append(key)

    harvest_palettable(palettable)
except ImportError:
    pass

valid_mpl_cmaps = []
for name in plt.colormaps():
    try:
        if test_colormap_callable(plt.get_cmap(name)):
            valid_mpl_cmaps.append(name)
    except Exception:
        continue

valid_mpl_cmaps = sorted(valid_mpl_cmaps, key=str.lower)
CATEGORIZED_CMAPS["matplotlib"].extend(valid_mpl_cmaps)

for name in POPULAR_CANDIDATES:
    if name in valid_mpl_cmaps or name in PALETTABLE_CMAPS:
        CATEGORIZED_CMAPS["★ Popular / Recommended"].append(name)

for cat in CATEGORIZED_CMAPS:
    if cat != "★ Popular / Recommended":
        CATEGORIZED_CMAPS[cat] = sorted(set(CATEGORIZED_CMAPS[cat]), key=str.lower)

def resolve_colormap(cmap_name):
    cmap = PALETTABLE_CMAPS.get(cmap_name)
    if cmap is None:
        try:
            cmap = plt.get_cmap(cmap_name)
        except Exception:
            cmap = None
    return cmap if test_colormap_callable(cmap) else plt.get_cmap('inferno')

def save_image_buffer(path, buf, is_hdr=False):
    h, w, _ = buf.shape
    if is_hdr:
        rgba16 = np.empty((h, w, 4), dtype=np.uint16)
        rgba16[:, :, :3] = buf
        rgba16[:, :, 3] = 65535
        qimg = QImage(rgba16.data, w, h, 8 * w, QImage.Format_RGBA64)
    else:
        qimg = QImage(buf.data, w, h, 3 * w, QImage.Format_RGB888)
    ext = os.path.splitext(path)[1].lower()
    fmt = "PNG"
    if ext in ('.jpg', '.jpeg'):
        fmt = "JPG"
    elif ext == '.bmp':
        fmt = "BMP"
    if not qimg.save(path, fmt):
        raise RuntimeError(f"Failed to save image to:\n{path}")

GPU_CMAP_LUT_CACHE = {}
CPU_CMAP_LUT_CACHE = {}

def get_cpu_cmap_lut(cmap_name, lut_size=2048, is_hdr=False):
    key = (cmap_name, lut_size, is_hdr)
    if key not in CPU_CMAP_LUT_CACHE:
        cmap = resolve_colormap(cmap_name)
        scale = 65535.0 if is_hdr else 255.0
        rgb_f = (cmap(np.linspace(0, 1, lut_size))[:, :3] * scale).astype(np.float32)
        CPU_CMAP_LUT_CACHE[key] = rgb_f
    return CPU_CMAP_LUT_CACHE[key]

def get_gpu_cmap_lut(cmap_name, lut_size=2048, is_hdr=False):
    key = (cmap_name, lut_size, is_hdr)
    if key not in GPU_CMAP_LUT_CACHE:
        cpu_lut = get_cpu_cmap_lut(cmap_name, lut_size=lut_size, is_hdr=is_hdr)
        GPU_CMAP_LUT_CACHE[key] = cuda.to_device(cpu_lut)
    return GPU_CMAP_LUT_CACHE[key]