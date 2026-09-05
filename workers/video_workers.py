import math
from decimal import Decimal, getcontext
import numpy as np

from config import ensure_decimal_precision
from workers.base import BaseVideoRenderWorker


class UnifiedVideoRenderWorker(BaseVideoRenderWorker):
    """Interpolates waypoint paths using scale-space parameterization for constant optical speed without drift."""
    def __init__(self, output_path, target_w, target_h, fps, bitrate_kbps, duration_sec,
                 waypoints, max_iter, cmap_name, rate_control="crf", crf=16, **kwargs):
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints are required.")
        super().__init__(output_path, target_w, target_h, fps, bitrate_kbps, duration_sec, max_iter, cmap_name,
                         rate_control=rate_control, crf=crf, **kwargs)
        self.waypoints = [(Decimal(str(wp[0])), Decimal(str(wp[1])), Decimal(str(wp[2]))) for wp in waypoints]
        min_w = min(wp[2] for wp in self.waypoints)
        ensure_decimal_precision(min_w, extra_digits=100)
        self.is_simple_zoom = (len(self.waypoints) == 2)
        if self.is_simple_zoom:
            self.start_cx, self.start_cy, self.start_w = self.waypoints[0]
            self.end_cx, self.end_cy, self.end_w = self.waypoints[1]
            self.delta_cx = self.end_cx - self.start_cx
            self.delta_cy = self.end_cy - self.start_cy
            self.delta_w = self.start_w - self.end_w
            self.delta_log_w = self.end_w.ln() - self.start_w.ln()
            self.has_zoom = abs(float(self.delta_log_w)) > 1e-9
            self.is_zoom_in = (self.start_w > self.end_w)
        else:
            self.total_segments = len(self.waypoints) - 1
            self._build_segment_arc_length_lut()

    def _build_segment_arc_length_lut(self):
        """Precomputes scale-space optical arc lengths for each segment to ensure uniform speed."""
        min_w = min(wp[2] for wp in self.waypoints)
        ensure_decimal_precision(min_w, extra_digits=100)
        n_pts = len(self.waypoints)
        self.seg_arc_lengths = []
        cum = 0.0
        self.cum_arc_lengths = [0.0]

        for k in range(n_pts - 1):
            cx0, cy0, w0 = self.waypoints[k]
            cx1, cy1, w1 = self.waypoints[k + 1]
            d_log_w = float(w1.ln() - w0.ln())
            
            # Scale-relative displacement
            max_w = max(w0, w1)
            dx_rel = float((cx1 - cx0) / max_w) if max_w > 0 else 0.0
            dy_rel = float((cy1 - cy0) / max_w) if max_w > 0 else 0.0
            d_pan = math.sqrt(dx_rel * dx_rel + dy_rel * dy_rel)

            ds = math.sqrt(d_log_w * d_log_w + d_pan * d_pan)
            ds = max(1e-6, ds)
            self.seg_arc_lengths.append(ds)
            cum += ds
            self.cum_arc_lengths.append(cum)

        total_len = self.cum_arc_lengths[-1]
        self.norm_seg_arcs = [c / total_len for c in self.cum_arc_lengths]

    def get_frame_state(self, i, total_frames):
        min_w = min(self.start_w, self.end_w) if self.is_simple_zoom else min(wp[2] for wp in self.waypoints)
        ensure_decimal_precision(min_w, extra_digits=100)
        inv_total_frames = Decimal(1) / Decimal(max(1, total_frames - 1))
        if self.is_simple_zoom:
            t_dec = Decimal(i) * inv_total_frames
            if self.has_zoom and self.delta_w != Decimal(0):
                cur_w = self.start_w * (t_dec * self.delta_log_w).exp()
                if self.is_zoom_in:
                    # Numerically stable focal tracking for zoom-in (zero catastrophic cancellation)
                    ratio = (cur_w - self.end_w) / (self.start_w - self.end_w)
                    cur_cx = self.end_cx - ratio * self.delta_cx
                    cur_cy = self.end_cy - ratio * self.delta_cy
                else:
                    # Numerically stable focal tracking for zoom-out
                    ratio = (cur_w - self.start_w) / (self.end_w - self.start_w)
                    cur_cx = self.start_cx + ratio * self.delta_cx
                    cur_cy = self.start_cy + ratio * self.delta_cy
            else:
                cur_w = self.start_w
                cur_cx = self.start_cx + t_dec * self.delta_cx
                cur_cy = self.start_cy + t_dec * self.delta_cy
        else:
            tau = float(i) / max(1.0, float(total_frames - 1))
            n_segs = self.total_segments
            
            # Find which segment tau falls into
            seg_idx = 0
            for s in range(n_segs):
                if self.norm_seg_arcs[s] <= tau <= self.norm_seg_arcs[s + 1]:
                    seg_idx = s
                    break
                if s == n_segs - 1:
                    seg_idx = s

            s_start = self.norm_seg_arcs[seg_idx]
            s_end = self.norm_seg_arcs[seg_idx + 1]
            seg_span = max(1e-9, s_end - s_start)
            t_local = max(0.0, min(1.0, (tau - s_start) / seg_span))
            t_dec = Decimal(str(t_local))

            k0_cx, k0_cy, k0_w = self.waypoints[seg_idx]
            k1_cx, k1_cy, k1_w = self.waypoints[seg_idx + 1]
            delta_cx = k1_cx - k0_cx
            delta_cy = k1_cy - k0_cy
            delta_ln_w = k1_w.ln() - k0_w.ln()

            if abs(float(delta_ln_w)) > 1e-9 and (k0_w != k1_w):
                cur_w = k0_w * (t_dec * delta_ln_w).exp()
                if k0_w > k1_w:
                    ratio = (cur_w - k1_w) / (k0_w - k1_w)
                    cur_cx = k1_cx - ratio * delta_cx
                    cur_cy = k1_cy - ratio * delta_cy
                else:
                    ratio = (cur_w - k0_w) / (k1_w - k0_w)
                    cur_cx = k0_cx + ratio * delta_cx
                    cur_cy = k0_cy + ratio * delta_cy
            else:
                cur_w = k0_w
                cur_cx = k0_cx + t_dec * delta_cx
                cur_cy = k0_cy + t_dec * delta_cy

        return cur_cx, cur_cy, cur_w, {}


class GeneralMandelbrotMorphVideoWorker(BaseVideoRenderWorker):
    """Morphs General Mandelbrot k-parameter across the timeline."""
    def __init__(self, output_path, target_w, target_h, fps, bitrate_kbps, duration_sec,
                 cx, cy, w, start_kr, end_kr, start_ki, end_ki, gen_n, max_iter, cmap_name,
                 rate_control="crf", crf=16, **kwargs):
        super().__init__(output_path, target_w, target_h, fps, bitrate_kbps, duration_sec, max_iter, cmap_name,
                         rate_control=rate_control, crf=crf, **kwargs)
        self.cx = Decimal(str(cx))
        self.cy = Decimal(str(cy))
        self.w = Decimal(str(w))
        self.start_kr = float(start_kr)
        self.end_kr = float(end_kr)
        self.start_ki = float(start_ki)
        self.end_ki = float(end_ki)
        self.gen_n = int(gen_n)

    def get_frame_state(self, i, total_frames):
        t = float(i) / max(1.0, float(total_frames - 1))
        cur_kr = self.start_kr + t * (self.end_kr - self.start_kr)
        cur_ki = self.start_ki + t * (self.end_ki - self.start_ki)
        return self.cx, self.cy, self.w, {
            'gen_kr': cur_kr, 'gen_ki': cur_ki,
            'gen_n': self.gen_n, 'fractal_type': 4
        }


class IterationRevealVideoWorker(BaseVideoRenderWorker):
    """Sweeps max_iter across the timeline while keeping coordinates fixed."""
    def __init__(self, output_path, target_w, target_h, fps, bitrate_kbps, duration_sec,
                 cx, cy, w, start_iter, end_iter, sweep_curve, max_iter, cmap_name,
                 rate_control="crf", crf=16, **kwargs):
        super().__init__(output_path, target_w, target_h, fps, bitrate_kbps, duration_sec, max_iter, cmap_name,
                         rate_control=rate_control, crf=crf, **kwargs)
        self.cx = Decimal(str(cx))
        self.cy = Decimal(str(cy))
        self.w = Decimal(str(w))
        self.start_iter = max(1, int(start_iter))
        self.end_iter = max(self.start_iter, int(end_iter))
        self.sweep_curve = str(sweep_curve).lower()

    def get_frame_state(self, i, total_frames):
        t = float(i) / max(1.0, float(total_frames - 1))
        if "log" in self.sweep_curve or "exp" in self.sweep_curve:
            ratio = self.end_iter / max(1.0, float(self.start_iter))
            cur_iter = int(round(float(self.start_iter) * (ratio ** t)))
        elif "quad" in self.sweep_curve:
            cur_iter = int(round(self.start_iter + (t ** 2) * (self.end_iter - self.start_iter)))
        elif "cube" in self.sweep_curve or "cubic" in self.sweep_curve:
            cur_iter = int(round(self.start_iter + (t ** 3) * (self.end_iter - self.start_iter)))
        else:
            cur_iter = int(round(self.start_iter + t * (self.end_iter - self.start_iter)))

        cur_iter = max(1, cur_iter)
        return self.cx, self.cy, self.w, {'max_iter': cur_iter}