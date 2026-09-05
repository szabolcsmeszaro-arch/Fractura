from decimal import Decimal
import time
from numba import cuda
import numpy as np
from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QCursor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from config import (
    DEFAULT_JULIA_MANDELBROT_PARAMS, DEFAULT_MAX_ITER, FRACTAL_DEFAULTS,
    RenderParameters, catmull_rom_spline
)
from math_core.cuda_kernels import HAS_CUDA
from math_core.engine import compute_dynamic_max_iter, render_frame, recolor_frame


class JuliaPreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 270)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.title_lbl = QLabel("✨ Live Julia Set Preview (HUD)")
        self.title_lbl.setStyleSheet("color: #60a5fa; font-weight: bold; font-size: 12px;")
        self.img_lbl = QLabel()
        self.img_lbl.setFixedSize(268, 200)
        self.coord_lbl = QLabel("c = 0.000000 + 0.000000i")
        self.coord_lbl.setStyleSheet("font-family: monospace")

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.img_lbl)
        layout.addWidget(self.coord_lbl)

        self.thumb_w, self.thumb_h = 268, 200
        self.d_thumb_buf = cuda.device_array((self.thumb_h, self.thumb_w, 3), dtype=np.uint8) if HAS_CUDA else None
        self.h_thumb_buf = cuda.pinned_array((self.thumb_h, self.thumb_w, 3), dtype=np.uint8) if HAS_CUDA else np.empty((self.thumb_h, self.thumb_w, 3), dtype=np.uint8)
        self.stream = cuda.stream() if HAS_CUDA else 0

    def update_preview(self, cx_dec, cy_dec, cmap_name='inferno', color_scheme_id=0, parent_fractal_type=0, compute_device="GPU", cpu_threads=2, color_density=1.0, color_contrast=1.0):
        self.coord_lbl.setText(f"c = {float(cx_dec):+.6f} {float(cy_dec):+.6f}i")
        preview_type = 3 if parent_fractal_type == 1 else 2
        self.title_lbl.setText(f"✨ Live {'BS ' if preview_type == 3 else ''}Julia Set Preview (HUD)")

        use_gpu = (compute_device == "GPU" and HAS_CUDA)
        preview_params = RenderParameters(
            target_w=self.thumb_w, target_h=self.thumb_h, max_iter=300, center_x_dec=Decimal("0.0"), center_y_dec=Decimal("0.0"),
            plot_width_dec=Decimal("3.0"), cmap_name=cmap_name, ssaa_factor=1, edge_threshold=0.35,
            color_scheme_id=color_scheme_id, precision_mode="1e-12 (FP64)", fractal_type=preview_type,
            julia_cx=float(cx_dec), julia_cy=float(cy_dec), bbsa_tol=0.0, bbsa_order=0, glitch_mode="Off (Single-Ref)",
            d_rgb_buf=self.d_thumb_buf if use_gpu else None, h_rgb_buf=self.h_thumb_buf,
            stream=self.stream if use_gpu else 0, is_hdr=False,
            compute_device=compute_device, cpu_threads=cpu_threads,
            color_density=color_density, color_contrast=color_contrast
        )
        rgb = render_frame(preview_params)
        qimg = QImage(rgb.data, self.thumb_w, self.thumb_h, 3 * self.thumb_w, QImage.Format_RGB888)
        self.img_lbl.setPixmap(QPixmap.fromImage(qimg))


class FractalCanvas(QWidget):
    status_changed = pyqtSignal(object, object, object, float, int)
    keyframe_recorded = pyqtSignal(object, object, object)
    hover_coord_changed = pyqtSignal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.fractal_type = 0
        self.center_x, self.center_y, self.plot_width = FRACTAL_DEFAULTS[0]
        self.min_plot_width = Decimal("1e-300")
        self.precision_mode = "1e-300 (Perturbation)"
        self.bbsa_accuracy = "4th-order"
        self.bbsa_tol = 1e-4
        self.bbsa_order = 4
        self.glitch_mode = "Off (Single-Ref)"
        self.dynamic_iter_mode = "Off"
        self.max_iter = DEFAULT_MAX_ITER
        self.cmap_name = 'inferno'
        self.color_scheme_id = 0
        self.palette_offset = 0.0
        self.color_density = 1.0
        self.color_contrast = 1.0
        self.ssaa_factor = 1
        self.edge_threshold = 0.35

        self.compute_device = "GPU" if HAS_CUDA else "CPU"
        self.cpu_threads = 2
        self.click_zoom_factor = Decimal("5.0")
        self.wheel_zoom_factor = Decimal("1.25")

        self.gen_n = 3
        self.gen_kr = 0.25
        self.gen_ki = 1.0

        self.julia_cx, self.julia_cy = DEFAULT_JULIA_MANDELBROT_PARAMS
        self.drag_start_pos = None
        self.drag_offset = QPoint(0, 0)
        self.current_pixmap = None
        self.last_compute_time = 0.0

        self.d_rgb_buf = None
        self.h_rgb_buf = None
        self.d_iter_buf = None
        self.h_iter_buf = None
        self.cached_geo_key = None
        self.rgb_shape = (0, 0)
        self.rgb_device = None
        self.show_waypoints_overlay = False
        self.setFixedSize(1280, 720)

    def set_palette_offset(self, offset):
        """Sets real-time palette phase offset and triggers instant GPU/CPU re-shading (~0.1ms)."""
        self.palette_offset = float(offset) % 1.0
        self.recolor_canvas()

    def set_color_density(self, density):
        """Sets color density and triggers instant GPU/CPU re-shading (~0.1ms)."""
        self.color_density = max(0.001, min(10.0, float(density)))
        self.recolor_canvas()

    def set_color_contrast(self, contrast):
        """Sets color contrast and triggers instant GPU/CPU re-shading (~0.1ms)."""
        self.color_contrast = max(0.001, min(10.0, float(contrast)))
        self.recolor_canvas()

    def screen_to_complex(self, px, py, w, h):
        aspect = Decimal(h) / Decimal(w)
        plot_height = self.plot_width * aspect
        return (self.center_x + (Decimal(px) / Decimal(w - 1) - Decimal("0.5")) * self.plot_width,
                self.center_y + (Decimal("0.5") - Decimal(py) / Decimal(h - 1)) * plot_height)

    def complex_to_screen(self, cx_dec, cy_dec, w, h):
        aspect = Decimal(h) / Decimal(w)
        plot_height = self.plot_width * aspect
        if self.plot_width <= Decimal("0") or plot_height <= Decimal("0"):
            return 0.0, 0.0
        try:
            px = float((Decimal(str(cx_dec)) - self.center_x) / self.plot_width + Decimal("0.5")) * float(w - 1)
            py = float(Decimal("0.5") - (Decimal(str(cy_dec)) - self.center_y) / plot_height) * float(h - 1)
            return px, py
        except Exception:
            return 0.0, 0.0

    def draw_waypoints_overlay(self, painter):
        viewer = self.window()
        waypoints = getattr(viewer, 'waypoints', [])
        if not waypoints:
            return

        w, h = self.width(), self.height()
        if w <= 1 or h <= 1:
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        off_x = float(self.drag_offset.x())
        off_y = float(self.drag_offset.y())
        n_pts = len(waypoints)

        # 1. Draw spline flight trajectory connecting all waypoints if 2 or more
        if n_pts >= 2:
            path = QPainterPath()
            first_point = True

            for k in range(n_pts - 1):
                idx0 = max(0, k - 1)
                idx1 = k
                idx2 = min(n_pts - 1, k + 1)
                idx3 = min(n_pts - 1, k + 2)

                for step in range(21):
                    t = Decimal(str(step / 20.0))
                    cur_cx = catmull_rom_spline(waypoints[idx0][0], waypoints[idx1][0], waypoints[idx2][0], waypoints[idx3][0], t)
                    cur_cy = catmull_rom_spline(waypoints[idx0][1], waypoints[idx1][1], waypoints[idx2][1], waypoints[idx3][1], t)
                    px, py = self.complex_to_screen(cur_cx, cur_cy, w, h)
                    sx, sy = px + off_x, py + off_y

                    if first_point:
                        path.moveTo(sx, sy)
                        first_point = False
                    else:
                        path.lineTo(sx, sy)

            # Outer glow pen
            glow_pen = QPen(QColor(6, 182, 212, 60), 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(glow_pen)
            painter.drawPath(path)

            # Crisp dashed core path
            core_pen = QPen(QColor(34, 211, 238, 230), 2, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin)
            core_pen.setDashPattern([6, 4])
            painter.setPen(core_pen)
            painter.drawPath(path)

        # 2. Draw sleek waypoint beacon pins with numbers
        pin_font = QFont("Segoe UI", 9, QFont.Bold)
        painter.setFont(pin_font)

        for idx, (wp_cx, wp_cy, wp_w) in enumerate(waypoints, 1):
            px, py = self.complex_to_screen(wp_cx, wp_cy, w, h)
            sx, sy = px + off_x, py + off_y

            # Skip if excessively far offscreen
            if sx < -200 or sx > w + 200 or sy < -200 or sy > h + 200:
                continue

            # Field of view target framing box if visible
            try:
                scale_ratio = float(Decimal(str(wp_w)) / self.plot_width)
                frame_w = scale_ratio * float(w)
                frame_h = scale_ratio * float(h)
                if 12.0 < frame_w < 3000.0:
                    painter.setPen(QPen(QColor(34, 211, 238, 70), 1, Qt.DashLine))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(QRectF(sx - frame_w / 2.0, sy - frame_h / 2.0, frame_w, frame_h))
            except Exception:
                pass

            # Outer aura ring
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(6, 182, 212, 60)))
            painter.drawEllipse(QPointF(sx, sy), 17.0, 17.0)

            # Inner beacon disc
            painter.setPen(QPen(QColor(34, 211, 238, 240), 2.0))
            painter.setBrush(QBrush(QColor(15, 23, 42, 230)))
            painter.drawEllipse(QPointF(sx, sy), 12.0, 12.0)

            # Centered waypoint number
            painter.setPen(QColor(255, 255, 255))
            text_rect = QRectF(sx - 12.0, sy - 12.0, 24.0, 24.0)
            painter.drawText(text_rect, Qt.AlignCenter, str(idx))

        painter.restore()

    def render_fractal(self):
        w, h = self.width(), self.height()
        if w <= 1 or h <= 1:
            return

        if self.plot_width < self.min_plot_width:
            self.plot_width = self.min_plot_width

        if self.rgb_shape != (h, w) or self.rgb_device != self.compute_device:
            if HAS_CUDA and self.compute_device == "GPU":
                self.d_rgb_buf = cuda.device_array((h, w, 3), dtype=np.uint8)
                self.h_rgb_buf = cuda.pinned_array((h, w, 3), dtype=np.uint8)
                self.d_iter_buf = cuda.device_array((h, w), dtype=np.float32)
                self.h_iter_buf = None
            else:
                self.d_rgb_buf = None
                self.h_rgb_buf = np.empty((h, w, 3), dtype=np.uint8)
                self.d_iter_buf = None
                self.h_iter_buf = np.empty((h, w), dtype=np.float32)
            self.rgb_shape = (h, w)
            self.rgb_device = self.compute_device
            self.cached_geo_key = None

        t0 = time.perf_counter()
        params = self.get_render_parameters(w, h, self.h_rgb_buf, self.d_rgb_buf, is_hdr=False)
        rgb = render_frame(params)
        self.last_compute_time = (time.perf_counter() - t0) * 1000
        self.current_pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888))
        self.drag_offset = QPoint(0, 0)
        self.cached_geo_key = (
            self.center_x, self.center_y, self.plot_width, params.max_iter, self.fractal_type,
            float(self.julia_cx), float(self.julia_cy), self.gen_n, self.gen_kr, self.gen_ki,
            w, h, self.ssaa_factor, self.compute_device
        )
        self.update()
        self.status_changed.emit(self.center_x, self.center_y, self.plot_width, self.last_compute_time, params.max_iter)

    def recolor_canvas(self):
        """
        Instantly recolors the canvas from the cached iteration map (~0.08ms GPU / ~1ms CPU)
        without recalculating reference orbits, BLA tables, or pixel iterations.
        Falls back to render_fractal() if geometry changed or cache is unavailable.
        """
        w, h = self.width(), self.height()
        params = self.get_render_parameters(w, h, self.h_rgb_buf, self.d_rgb_buf, is_hdr=False)
        current_geo_key = (
            self.center_x, self.center_y, self.plot_width, params.max_iter, self.fractal_type,
            float(self.julia_cx), float(self.julia_cy), self.gen_n, self.gen_kr, self.gen_ki,
            w, h, self.ssaa_factor, self.compute_device
        )
        has_valid_cache = (
            self.cached_geo_key == current_geo_key and
            ((self.compute_device == "GPU" and self.d_iter_buf is not None) or
             (self.compute_device == "CPU" and self.h_iter_buf is not None))
        )
        if not has_valid_cache:
            self.render_fractal()
            return

        t0 = time.perf_counter()
        rgb = recolor_frame(params, d_iter_buf=self.d_iter_buf, h_iter_buf=self.h_iter_buf)
        self.last_compute_time = (time.perf_counter() - t0) * 1000
        self.current_pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888))
        self.update()
        self.status_changed.emit(self.center_x, self.center_y, self.plot_width, self.last_compute_time, params.max_iter)

    def get_render_parameters(self, target_w=None, target_h=None, h_rgb_buf=None, d_rgb_buf=None, is_hdr=False) -> RenderParameters:
        """Returns a RenderParameters dataclass representing current canvas state."""
        w = target_w if target_w is not None else self.width()
        h = target_h if target_h is not None else self.height()
        eff_iter = compute_dynamic_max_iter(
            self.max_iter, self.plot_width, self.dynamic_iter_mode,
            center_x_dec=self.center_x, center_y_dec=self.center_y,
            fractal_type=self.fractal_type, julia_cx=float(self.julia_cx), julia_cy=float(self.julia_cy),
            gen_n=self.gen_n, gen_kr=self.gen_kr, gen_ki=self.gen_ki,
            precision_mode=self.precision_mode, bbsa_tol=self.bbsa_tol, bbsa_order=self.bbsa_order,
            compute_device=self.compute_device, cpu_threads=self.cpu_threads
        )
        return RenderParameters(
            target_w=w, target_h=h, max_iter=eff_iter,
            center_x_dec=self.center_x, center_y_dec=self.center_y, plot_width_dec=self.plot_width,
            cmap_name=self.cmap_name, ssaa_factor=self.ssaa_factor, edge_threshold=self.edge_threshold,
            color_scheme_id=self.color_scheme_id, precision_mode=self.precision_mode,
            fractal_type=self.fractal_type, julia_cx=float(self.julia_cx), julia_cy=float(self.julia_cy),
            bbsa_tol=self.bbsa_tol, bbsa_order=self.bbsa_order, glitch_mode=self.glitch_mode,
            gen_n=self.gen_n, gen_kr=self.gen_kr, gen_ki=self.gen_ki,
            d_rgb_buf=d_rgb_buf if d_rgb_buf is not None else self.d_rgb_buf,
            h_rgb_buf=h_rgb_buf if h_rgb_buf is not None else self.h_rgb_buf,
            stream=0,
            is_hdr=is_hdr, compute_device=self.compute_device, cpu_threads=self.cpu_threads,
            palette_offset=self.palette_offset,
            color_density=self.color_density, color_contrast=self.color_contrast,
            d_iter_buf=self.d_iter_buf,
            h_iter_buf=self.h_iter_buf
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.current_pixmap:
            painter.drawPixmap(self.drag_offset, self.current_pixmap)
        else:
            painter.fillRect(self.rect(), Qt.black)

        if self.show_waypoints_overlay:
            self.draw_waypoints_overlay(painter)

    def zoom_at_point(self, px, py, factor):
        w, h = self.width(), self.height()
        target_cx, target_cy = self.screen_to_complex(px, py, w, h)
        new_w = max(self.min_plot_width, self.plot_width / factor)
        actual_factor = self.plot_width / new_w
        if actual_factor != Decimal("1.0"):
            self.plot_width = new_w
            self.center_x = target_cx + (self.center_x - target_cx) / actual_factor
            self.center_y = target_cy + (self.center_y - target_cy) / actual_factor
            self.render_fractal()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self.wheel_zoom_factor if hasattr(self, 'wheel_zoom_factor') and self.wheel_zoom_factor > Decimal("1.0") else Decimal("1.25")
        zoom_factor = factor if delta > 0 else (Decimal("1.0") / factor)
        self.zoom_at_point(event.x(), event.y(), zoom_factor)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            cx, cy = self.screen_to_complex(event.x(), event.y(), self.width(), self.height())
            viewer = self.window()
            if hasattr(viewer, 'julia_preview') and viewer.julia_preview.isVisible() and self.fractal_type in (0, 1):
                viewer.on_jump_to_julia(cx, cy)
            else:
                self.keyframe_recorded.emit(cx, cy, self.plot_width)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_pos = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        elif event.button() == Qt.RightButton:
            self.zoom_at_point(event.x(), event.y(), self.click_zoom_factor)

    def mouseMoveEvent(self, event):
        cx, cy = self.screen_to_complex(event.x(), event.y(), self.width(), self.height())
        self.hover_coord_changed.emit(cx, cy)
        if self.drag_start_pos is not None:
            self.drag_offset = event.pos() - self.drag_start_pos
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drag_start_pos is not None:
            self.setCursor(QCursor(Qt.ArrowCursor))
            dx, dy = event.pos().x() - self.drag_start_pos.x(), event.pos().y() - self.drag_start_pos.y()
            w, h = self.width(), self.height()
            self.center_x -= (Decimal(dx) / Decimal(w - 1)) * self.plot_width
            self.center_y += (Decimal(dy) / Decimal(h - 1)) * (self.plot_width * (Decimal(h) / Decimal(w)))
            self.drag_start_pos = None
            self.render_fractal()

    def keyPressEvent(self, event):
        viewer = self.window()
        if hasattr(viewer, 'keyPressEvent'):
            viewer.keyPressEvent(event)
        else:
            super().keyPressEvent(event)