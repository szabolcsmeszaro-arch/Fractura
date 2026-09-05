from collections import deque
from decimal import Decimal, getcontext
import json
import math
import os
import queue
import threading
import time
from numba import cuda
import numpy as np

from color_engine import save_image_buffer
from config import RenderParameters, catmull_rom_spline, draw_video_hud_overlay, ensure_decimal_precision
from math_core.cuda_kernels import HAS_CUDA
from math_core.engine import compute_dynamic_max_iter, render_frame
from workers.base import (
    BaseFractalWorker, downsample_to_preview_256x144, safe_queue_drain_and_terminate
)


class ImageRenderWorker(BaseFractalWorker):
    def __init__(self, path, target_w, target_h, max_iter, cx, cy, w, cmap_name, ssaa_factor, overlay_hud=False, **kwargs):
        super().__init__(target_w, target_h, max_iter, cmap_name, ssaa_factor=ssaa_factor, **kwargs)
        self.path = path
        self.cx = Decimal(str(cx))
        self.cy = Decimal(str(cy))
        self.w = Decimal(str(w))
        self.overlay_hud = bool(overlay_hud)

    def run(self):
        try:
            ensure_decimal_precision(self.w)
            if self.compute_device == "GPU" and HAS_CUDA:
                cuda.select_device(0)
            t0 = time.perf_counter()
            dtype_val = np.uint16 if self.is_hdr else np.uint8
            d_rgb_buf = cuda.device_array((self.target_h, self.target_w, 3), dtype=dtype_val) if (self.compute_device == "GPU" and HAS_CUDA) else None
            h_rgb_buf = cuda.pinned_array((self.target_h, self.target_w, 3), dtype=dtype_val) if (self.compute_device == "GPU" and HAS_CUDA) else np.empty((self.target_h, self.target_w, 3), dtype=dtype_val)

            eff_iter = compute_dynamic_max_iter(
                self.max_iter, self.w, self.dynamic_iter_mode,
                center_x_dec=self.cx, center_y_dec=self.cy,
                fractal_type=self.fractal_type, julia_cx=float(self.julia_cx), julia_cy=float(self.julia_cy),
                gen_n=self.gen_n, gen_kr=self.gen_kr, gen_ki=self.gen_ki,
                precision_mode=self.precision_mode, bbsa_tol=self.bbsa_tol, bbsa_order=self.bbsa_order,
                compute_device=self.compute_device, cpu_threads=self.cpu_threads
            )

            frame_params = RenderParameters(
                target_w=self.target_w, target_h=self.target_h, max_iter=eff_iter,
                center_x_dec=self.cx, center_y_dec=self.cy, plot_width_dec=self.w,
                cmap_name=self.cmap_name, ssaa_factor=self.ssaa_factor, edge_threshold=self.edge_threshold,
                color_scheme_id=self.color_scheme_id, precision_mode=self.precision_mode,
                fractal_type=self.fractal_type, julia_cx=float(self.julia_cx), julia_cy=float(self.julia_cy),
                bbsa_tol=self.bbsa_tol, bbsa_order=self.bbsa_order, glitch_mode=self.glitch_mode,
                gen_n=self.gen_n, gen_kr=self.gen_kr, gen_ki=self.gen_ki,
                d_rgb_buf=d_rgb_buf, h_rgb_buf=h_rgb_buf, is_hdr=self.is_hdr,
                compute_device=self.compute_device, cpu_threads=self.cpu_threads, palette_offset=self.palette_offset,
                color_density=self.color_density, color_contrast=self.color_contrast
            )
            render_frame(frame_params)
            if self.overlay_hud:
                draw_video_hud_overlay(h_rgb_buf, self.w, cx=self.cx, cy=self.cy, is_hdr=self.is_hdr, max_iter=eff_iter)
            save_image_buffer(self.path, h_rgb_buf, is_hdr=self.is_hdr)
            self.finished.emit(self.path, time.perf_counter() - t0)
        except Exception as e:
            self.error.emit(str(e))


class ImageSequenceRenderWorker(BaseFractalWorker):
    def __init__(self, output_dir, file_prefix, num_images, target_w, target_h, max_iter,
                 cmap_name, waypoints=None, cx=None, cy=None, start_w=None, end_w=None, overlay_hud=False,
                 resume_existing=True, **kwargs):
        super().__init__(target_w, target_h, max_iter, cmap_name, **kwargs)
        target_w_val = min(Decimal(str(start_w or 1)), Decimal(str(end_w or 1)))
        ensure_decimal_precision(target_w_val)
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.num_images = max(1, int(num_images))
        self.overlay_hud = bool(overlay_hud)
        self.resume_existing = bool(resume_existing)

        if waypoints and len(waypoints) > 2:
            self.waypoints = [(Decimal(str(w[0])), Decimal(str(w[1])), Decimal(str(w[2]))) for w in waypoints]
            self.is_multi_waypoint = True
            self.total_segments = len(self.waypoints) - 1
            self._build_segment_arc_length_lut()
        elif waypoints and len(waypoints) == 2:
            self.is_multi_waypoint = False
            self.start_cx = Decimal(str(waypoints[0][0]))
            self.start_cy = Decimal(str(waypoints[0][1]))
            self.start_w = Decimal(str(waypoints[0][2]))
            self.end_cx = Decimal(str(waypoints[1][0]))
            self.end_cy = Decimal(str(waypoints[1][1]))
            self.end_w = Decimal(str(waypoints[1][2]))
            self.delta_cx = self.end_cx - self.start_cx
            self.delta_cy = self.end_cy - self.start_cy
            self.delta_w = self.start_w - self.end_w
            self.delta_log_w = self.end_w.ln() - self.start_w.ln()
            self.has_zoom = abs(float(self.delta_log_w)) > 1e-9
            self.is_zoom_in = (self.start_w > self.end_w)
        elif waypoints and len(waypoints) == 1:
            self.is_multi_waypoint = False
            self.start_cx = Decimal(str(waypoints[0][0]))
            self.start_cy = Decimal(str(waypoints[0][1]))
            self.start_w = Decimal(str(start_w if start_w is not None else "4.0"))
            self.end_cx = Decimal(str(waypoints[0][0]))
            self.end_cy = Decimal(str(waypoints[0][1]))
            self.end_w = Decimal(str(waypoints[0][2]))
            self.delta_cx = self.end_cx - self.start_cx
            self.delta_cy = self.end_cy - self.start_cy
            self.delta_w = self.start_w - self.end_w
            self.delta_log_w = self.end_w.ln() - self.start_w.ln()
            self.has_zoom = abs(float(self.delta_log_w)) > 1e-9
            self.is_zoom_in = (self.start_w > self.end_w)
        else:
            self.is_multi_waypoint = False
            self.start_cx = Decimal(str(cx if cx is not None else "0.0"))
            self.start_cy = Decimal(str(cy if cy is not None else "0.0"))
            self.start_w = Decimal(str(start_w if start_w is not None else "4.0"))
            self.end_cx = self.start_cx
            self.end_cy = self.start_cy
            self.end_w = Decimal(str(end_w if end_w is not None else self.start_w))
            self.delta_cx = self.end_cx - self.start_cx
            self.delta_cy = self.end_cy - self.start_cy
            self.delta_w = self.start_w - self.end_w
            self.delta_log_w = self.end_w.ln() - self.start_w.ln()
            self.has_zoom = abs(float(self.delta_log_w)) > 1e-9
        min_zoom_w = min(self.start_w, self.end_w) if (hasattr(self, 'start_w') and hasattr(self, 'end_w')) else (self.end_w if hasattr(self, 'end_w') else self.start_w)
        ensure_decimal_precision(min_zoom_w, extra_digits=100)

    def save_session_manifest(self):
        """Saves session metadata to JSON so the exact sequence can be resumed or inspected later."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            manifest_path = os.path.join(self.output_dir, f"{self.file_prefix}_session.json")
            wps_data = []
            if hasattr(self, 'waypoints') and self.waypoints:
                wps_data = [{"center_x": str(cx), "center_y": str(cy), "plot_width": str(w)} for cx, cy, w in self.waypoints]
            elif hasattr(self, 'start_w'):
                wps_data = [
                    {"center_x": str(self.start_cx), "center_y": str(self.start_cy), "plot_width": str(self.start_w)},
                    {"center_x": str(self.end_cx), "center_y": str(self.end_cy), "plot_width": str(self.end_w)}
                ]
            
            data = {
                "file_prefix": self.file_prefix,
                "num_images": self.num_images,
                "target_w": self.target_w,
                "target_h": self.target_h,
                "ssaa_factor": self.ssaa_factor,
                "max_iter": self.max_iter,
                "cmap_name": self.cmap_name,
                "color_scheme_id": self.color_scheme_id,
                "palette_offset": self.palette_offset,
                "color_density": self.color_density,
                "color_contrast": self.color_contrast,
                "precision_mode": self.precision_mode,
                "fractal_type": self.fractal_type,
                "julia_cx": str(self.julia_cx),
                "julia_cy": str(self.julia_cy),
                "gen_n": self.gen_n,
                "gen_kr": self.gen_kr,
                "gen_ki": self.gen_ki,
                "bbsa_tol": self.bbsa_tol,
                "bbsa_order": self.bbsa_order,
                "glitch_mode": self.glitch_mode,
                "dynamic_iter_mode": self.dynamic_iter_mode,
                "edge_threshold": self.edge_threshold,
                "is_hdr": self.is_hdr,
                "overlay_hud": self.overlay_hud,
                "waypoints": wps_data
            }
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except Exception:
            pass

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

    def run(self):
        min_w = getattr(self, 'end_w', getattr(self, 'start_w', Decimal("1.0")))
        ensure_decimal_precision(min_w, extra_digits=100)
        self.save_session_manifest()

        writer_queue = queue.Queue(maxsize=3)
        free_buffers = queue.Queue()
        writer_err = []

        def disk_saver():
            try:
                while True:
                    item = writer_queue.get()
                    if item is None:
                        writer_queue.task_done()
                        break
                    full_path, buf, is_hdr = item
                    save_image_buffer(full_path, buf, is_hdr)
                    free_buffers.put(buf)
                    writer_queue.task_done()
            except Exception as ex:
                writer_err.append(str(ex))

        try:
            if hasattr(self, 'start_w') and hasattr(self, 'end_w') and self.start_w is not None and self.end_w is not None:
                ensure_decimal_precision(min(self.start_w, self.end_w))
            elif hasattr(self, 'waypoints') and self.waypoints:
                ensure_decimal_precision(min(wp[2] for wp in self.waypoints))
            elif hasattr(self, 'end_w') and self.end_w is not None:
                ensure_decimal_precision(self.end_w)
            elif hasattr(self, 'start_w') and self.start_w is not None:
                ensure_decimal_precision(self.start_w)
            if self.compute_device == "GPU" and HAS_CUDA:
                cuda.select_device(0)
            t0 = time.perf_counter()
            dtype_val = np.uint16 if self.is_hdr else np.uint8
            streams = [cuda.stream(), cuda.stream()] if (self.compute_device == "GPU" and HAS_CUDA) else [0, 0]
            d_rgb_bufs = [cuda.device_array((self.target_h, self.target_w, 3), dtype=dtype_val) for _ in range(2)] if (self.compute_device == "GPU" and HAS_CUDA) else [None, None]
            
            for _ in range(4):
                buf = cuda.pinned_array((self.target_h, self.target_w, 3), dtype=dtype_val) if (self.compute_device == "GPU" and HAS_CUDA) else np.empty((self.target_h, self.target_w, 3), dtype=dtype_val)
                free_buffers.put(buf)

            saver_thread = threading.Thread(target=disk_saver, daemon=True)
            saver_thread.start()

            inv_denom = Decimal(1) / Decimal(max(1, self.num_images - 1))
            recent_times = deque(maxlen=100)
            pending_frames = deque()
            pad_digits = max(4, len(str(self.num_images)))
            use_async = (self.compute_device == "GPU" and HAS_CUDA)
            last_completion_time = [time.perf_counter()]

            try:
                existing_files = set(os.listdir(self.output_dir))
            except Exception:
                existing_files = set()

            def process_completed_image(job):
                f_idx, f_t0, f_w, f_cx, f_cy, f_iter, f_buf, f_fname, f_stream = job
                if f_stream != 0 and HAS_CUDA:
                    f_stream.synchronize()
                if self.overlay_hud:
                    draw_video_hud_overlay(f_buf, f_w, cx=f_cx, cy=f_cy, is_hdr=self.is_hdr, max_iter=f_iter)
                if self.preview_hud:
                    self.preview_updated.emit(downsample_to_preview_256x144(f_buf, is_hdr=self.is_hdr))
                writer_queue.put((os.path.join(self.output_dir, f_fname), f_buf, self.is_hdr))
                now = time.perf_counter()
                elapsed = max(1e-6, now - last_completion_time[0])
                last_completion_time[0] = now
                recent_times.append(elapsed)
                self.progress_updated.emit(f_idx + 1, self.num_images, 1.0 / elapsed, len(recent_times) / max(1e-6, sum(recent_times)))

            for idx in range(self.num_images):
                if self.is_cancelled or writer_err:
                    break
                frame_t0 = time.perf_counter()
                slot = idx % 2
                stream = streams[slot]

                if self.is_multi_waypoint:
                    tau = float(idx) / max(1.0, float(self.num_images - 1))
                    n_segs = self.total_segments
                    
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
                        current_w = k0_w * (t_dec * delta_ln_w).exp()
                        if k0_w > k1_w:
                            ratio = (current_w - k1_w) / (k0_w - k1_w)
                            current_cx = k1_cx - ratio * delta_cx
                            current_cy = k1_cy - ratio * delta_cy
                        else:
                            ratio = (current_w - k0_w) / (k1_w - k0_w)
                            current_cx = k0_cx + ratio * delta_cx
                            current_cy = k0_cy + ratio * delta_cy
                    else:
                        current_w = k0_w
                        current_cx = k0_cx + t_dec * delta_cx
                        current_cy = k0_cy + t_dec * delta_cy
                else:
                    t_dec = Decimal(idx) * inv_denom
                    if self.has_zoom and self.delta_w != Decimal(0):
                        current_w = self.start_w * (t_dec * self.delta_log_w).exp()
                        if self.is_zoom_in:
                            ratio = (current_w - self.end_w) / (self.start_w - self.end_w)
                            current_cx = self.end_cx - ratio * self.delta_cx
                            current_cy = self.end_cy - ratio * self.delta_cy
                        else:
                            ratio = (current_w - self.start_w) / (self.end_w - self.start_w)
                            current_cx = self.start_cx + ratio * self.delta_cx
                            current_cy = self.start_cy + ratio * self.delta_cy
                    else:
                        current_w = self.start_w
                        current_cx = self.start_cx + t_dec * self.delta_cx
                        current_cy = self.start_cy + t_dec * self.delta_cy

                hdr_tag = "_16bitHDR" if self.is_hdr else ""
                file_name = (f"{self.file_prefix}_{idx+1:0{pad_digits}d}_{self.target_w}x{self.target_h}_"
                    f"{self.ssaa_factor}xSSAA{hdr_tag}_"
                    f"{Decimal(str(current_cx)):.6e}_{Decimal(str(current_cy)):.6e}_{Decimal(str(current_w)):.6e}.png"
                )

                # Skip frame if resume is enabled and image already exists on disk
                if self.resume_existing:
                    target_prefix = f"{self.file_prefix}_{idx+1:0{pad_digits}d}_"
                    matched_file = None
                    if file_name in existing_files:
                        matched_file = file_name
                    else:
                        for ef in existing_files:
                            if ef.startswith(target_prefix) and ef.endswith(".png"):
                                matched_file = ef
                                break
                    if matched_file:
                        full_match_path = os.path.join(self.output_dir, matched_file)
                        if os.path.exists(full_match_path) and os.path.getsize(full_match_path) > 1024:
                            self.progress_updated.emit(idx + 1, self.num_images, 0.0, 0.0)
                            continue

                eff_iter = compute_dynamic_max_iter(
                    self.max_iter, current_w, self.dynamic_iter_mode,
                    center_x_dec=current_cx, center_y_dec=current_cy,
                    fractal_type=self.fractal_type, julia_cx=float(self.julia_cx), julia_cy=float(self.julia_cy),
                    gen_n=self.gen_n, gen_kr=self.gen_kr, gen_ki=self.gen_ki,
                    precision_mode=self.precision_mode, bbsa_tol=self.bbsa_tol, bbsa_order=self.bbsa_order,
                    compute_device=self.compute_device, cpu_threads=self.cpu_threads
                )
                
                h_buf = None
                while not self.is_cancelled and not writer_err:
                    try:
                        h_buf = free_buffers.get(timeout=0.1)
                        break
                    except queue.Empty:
                        continue

                if self.is_cancelled or writer_err or h_buf is None:
                    break

                frame_params = RenderParameters(
                    target_w=self.target_w, target_h=self.target_h, max_iter=eff_iter,
                    center_x_dec=current_cx, center_y_dec=current_cy, plot_width_dec=current_w,
                    cmap_name=self.cmap_name, ssaa_factor=self.ssaa_factor, edge_threshold=self.edge_threshold,
                    color_scheme_id=self.color_scheme_id, precision_mode=self.precision_mode,
                    fractal_type=self.fractal_type, julia_cx=float(self.julia_cx), julia_cy=float(self.julia_cy),
                    bbsa_tol=self.bbsa_tol, bbsa_order=self.bbsa_order, glitch_mode=self.glitch_mode,
                    gen_n=self.gen_n, gen_kr=self.gen_kr, gen_ki=self.gen_ki,
                    d_rgb_buf=d_rgb_bufs[slot], h_rgb_buf=h_buf, stream=stream,
                    stream_slot=(slot + 1),
                    is_hdr=self.is_hdr, compute_device=self.compute_device, cpu_threads=self.cpu_threads,
                    palette_offset=self.palette_offset,
                    color_density=self.color_density, color_contrast=self.color_contrast,
                    sync_stream=(not use_async)
                )
                render_frame(frame_params)

                pending_frames.append((idx, frame_t0, current_w, current_cx, current_cy, eff_iter, h_buf, file_name, stream))

                if len(pending_frames) >= 2:
                    process_completed_image(pending_frames.popleft())

            # Drain remaining in-flight images
            while pending_frames and not self.is_cancelled and not writer_err:
                process_completed_image(pending_frames.popleft())

            if not self.is_cancelled and not writer_err:
                writer_queue.put(None)
                saver_thread.join()
            else:
                safe_queue_drain_and_terminate(writer_queue)
                saver_thread.join(timeout=1.0)

            if self.is_cancelled:
                return
            if writer_err:
                raise RuntimeError(writer_err[0])
            self.finished.emit(self.output_dir, time.perf_counter() - t0)
        except Exception as e:
            safe_queue_drain_and_terminate(writer_queue)
            self.error.emit(str(e))


class MosaicRenderWorker(BaseFractalWorker):
    def __init__(self, output_dir, file_prefix, grid_n, target_w, target_h, max_iter,
                 canvas_cx, canvas_cy, plot_width,
                 cmap_name, auto_stitch=True, delete_tiles_after_stitch=True, overlay_hud=False, **kwargs):
        super().__init__(target_w, target_h, max_iter, cmap_name, **kwargs)
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.grid_n = grid_n
        self.canvas_cx = Decimal(str(canvas_cx))
        self.canvas_cy = Decimal(str(canvas_cy))
        self.plot_width = Decimal(str(plot_width))
        self.auto_stitch = auto_stitch
        self.delete_tiles_after_stitch = delete_tiles_after_stitch
        self.overlay_hud = bool(overlay_hud)

    def run(self):
        writer_queue = queue.Queue(maxsize=4)
        free_buffers = queue.Queue()
        writer_err = []

        def tile_disk_saver():
            try:
                while True:
                    item = writer_queue.get()
                    if item is None:
                        writer_queue.task_done()
                        break
                    full_path, buf, is_hdr, should_save = item
                    if should_save:
                        save_image_buffer(full_path, buf, is_hdr)
                    free_buffers.put(buf)
                    writer_queue.task_done()
            except Exception as ex:
                writer_err.append(str(ex))

        try:
            ensure_decimal_precision(self.plot_width)
            if self.compute_device == "GPU" and HAS_CUDA:
                cuda.select_device(0)
            t0 = time.perf_counter()
            total_tiles = self.grid_n * self.grid_n
            n_dec = Decimal(self.grid_n)
            tile_aspect = Decimal(self.target_h) / Decimal(self.target_w)
            total_plot_h = self.plot_width * tile_aspect
            tile_plot_w = self.plot_width / n_dec
            tile_plot_h = total_plot_h / n_dec
            x_min_total = self.canvas_cx - (self.plot_width / Decimal(2))
            y_max_total = self.canvas_cy + (total_plot_h / Decimal(2))

            total_stitched_w, total_stitched_h = self.target_w * self.grid_n, self.target_h * self.grid_n
            dtype_val = np.uint16 if self.is_hdr else np.uint8
            master_rgb = None
            if self.auto_stitch:
                try:
                    master_rgb = np.zeros((total_stitched_h, total_stitched_w, 3), dtype=dtype_val)
                except MemoryError:
                    master_rgb = None

            streams = [cuda.stream(), cuda.stream()] if (self.compute_device == "GPU" and HAS_CUDA) else [0, 0]
            d_rgb_bufs = [cuda.device_array((self.target_h, self.target_w, 3), dtype=dtype_val) for _ in range(2)] if (self.compute_device == "GPU" and HAS_CUDA) else [None, None]
            for _ in range(4):
                buf = cuda.pinned_array((self.target_h, self.target_w, 3), dtype=dtype_val) if (self.compute_device == "GPU" and HAS_CUDA) else np.empty((self.target_h, self.target_w, 3), dtype=dtype_val)
                free_buffers.put(buf)

            saver_thread = threading.Thread(target=tile_disk_saver, daemon=True)
            saver_thread.start()

            recent_times, tile_index = deque(maxlen=100), 0
            hdr_tag = "_16bitHDR" if self.is_hdr else ""
            base_name = (f"{self.file_prefix}_{self.target_w}x{self.target_h}_"
                f"{self.ssaa_factor}xSSAA{hdr_tag}_"
                f"{Decimal(str(self.canvas_cx)):.6e}_{Decimal(str(self.canvas_cy)):.6e}_{Decimal(str(self.plot_width)):.6e}"
            )
            should_save_individual_tile = not (self.auto_stitch and self.delete_tiles_after_stitch and (master_rgb is not None))
            half_dec = Decimal("0.5")

            for row in range(self.grid_n):
                tile_cy = y_max_total - (Decimal(row) + half_dec) * tile_plot_h
                for col in range(self.grid_n):
                    if self.is_cancelled or writer_err:
                        break
                    tile_t0 = time.perf_counter()
                    tile_cx = x_min_total + (Decimal(col) + half_dec) * tile_plot_w
                    full_path = os.path.join(self.output_dir, f"{base_name}_r{row+1:02d}_c{col+1:02d}.png")

                    eff_iter = compute_dynamic_max_iter(
                        self.max_iter, tile_plot_w, self.dynamic_iter_mode,
                        center_x_dec=tile_cx, center_y_dec=tile_cy,
                        fractal_type=self.fractal_type, julia_cx=float(self.julia_cx), julia_cy=float(self.julia_cy),
                        gen_n=self.gen_n, gen_kr=self.gen_kr, gen_ki=self.gen_ki,
                        precision_mode=self.precision_mode, bbsa_tol=self.bbsa_tol, bbsa_order=self.bbsa_order,
                        compute_device=self.compute_device, cpu_threads=self.cpu_threads
                    )

                    slot = tile_index % 2
                    stream = streams[slot]
                    d_buf = d_rgb_bufs[slot]
                    h_buf = None
                    while not self.is_cancelled and not writer_err:
                        try:
                            h_buf = free_buffers.get(timeout=0.1)
                            break
                        except queue.Empty:
                            continue

                    if self.is_cancelled or writer_err or h_buf is None:
                        break

                    tile_params = RenderParameters(
                        target_w=self.target_w, target_h=self.target_h, max_iter=eff_iter,
                        center_x_dec=tile_cx, center_y_dec=tile_cy, plot_width_dec=tile_plot_w,
                        cmap_name=self.cmap_name, ssaa_factor=self.ssaa_factor, edge_threshold=self.edge_threshold,
                        color_scheme_id=self.color_scheme_id, precision_mode=self.precision_mode,
                        fractal_type=self.fractal_type, julia_cx=float(self.julia_cx), julia_cy=float(self.julia_cy),
                        bbsa_tol=self.bbsa_tol, bbsa_order=self.bbsa_order, glitch_mode=self.glitch_mode,
                        gen_n=self.gen_n, gen_kr=self.gen_kr, gen_ki=self.gen_ki,
                        d_rgb_buf=d_buf, h_rgb_buf=h_buf, stream=stream, is_hdr=self.is_hdr,
                        compute_device=self.compute_device, cpu_threads=self.cpu_threads, palette_offset=self.palette_offset,
                        color_density=self.color_density, color_contrast=self.color_contrast
                    )
                    render_frame(tile_params)

                    if master_rgb is not None:
                        r0, c0 = row * self.target_h, col * self.target_w
                        master_rgb[r0:r0 + self.target_h, c0:c0 + self.target_w] = h_buf

                    if self.overlay_hud:
                        draw_video_hud_overlay(h_buf, tile_plot_w, cx=tile_cx, cy=tile_cy, is_hdr=self.is_hdr, max_iter=eff_iter)

                    if self.preview_hud:
                        self.preview_updated.emit(downsample_to_preview_256x144(h_buf, is_hdr=self.is_hdr))

                    writer_queue.put((full_path, h_buf, self.is_hdr, should_save_individual_tile))

                    tile_index += 1
                    tile_elapsed = time.perf_counter() - tile_t0
                    recent_times.append(tile_elapsed)
                    self.progress_updated.emit(tile_index, total_tiles, 1.0 / max(1e-6, tile_elapsed), len(recent_times) / max(1e-6, sum(recent_times)))

            if not self.is_cancelled and not writer_err:
                writer_queue.put(None)
                saver_thread.join()
            else:
                safe_queue_drain_and_terminate(writer_queue)
                saver_thread.join(timeout=1.0)

            if self.is_cancelled:
                return
            if writer_err:
                raise RuntimeError(writer_err[0])

            if master_rgb is not None:
                if self.overlay_hud:
                    master_eff_iter = compute_dynamic_max_iter(
                        self.max_iter, self.plot_width, self.dynamic_iter_mode,
                        center_x_dec=self.canvas_cx, center_y_dec=self.canvas_cy,
                        fractal_type=self.fractal_type, julia_cx=float(self.julia_cx), julia_cy=float(self.julia_cy),
                        gen_n=self.gen_n, gen_kr=self.gen_kr, gen_ki=self.gen_ki,
                        precision_mode=self.precision_mode, bbsa_tol=self.bbsa_tol, bbsa_order=self.bbsa_order,
                        compute_device=self.compute_device, cpu_threads=self.cpu_threads
                    )
                    draw_video_hud_overlay(master_rgb, self.plot_width, cx=self.canvas_cx, cy=self.canvas_cy, is_hdr=self.is_hdr, max_iter=master_eff_iter)
                master_file_name = (
                    f"{self.file_prefix}_STITCHED_FULL_{total_stitched_w}x{total_stitched_h}_"
                    f"{self.ssaa_factor}xSSAA{hdr_tag}_"
                    f"{Decimal(str(self.canvas_cx)):.6e}_{Decimal(str(self.canvas_cy)):.6e}_{Decimal(str(self.plot_width)):.6e}.png"
                )
                master_path = os.path.join(self.output_dir, master_file_name)
                save_image_buffer(master_path, master_rgb, self.is_hdr)

            self.finished.emit(self.output_dir, time.perf_counter() - t0)
        except Exception as e:
            safe_queue_drain_and_terminate(writer_queue)
            self.error.emit(str(e))


class GeneralMandelbrotMorphImageWorker(BaseFractalWorker):
    """Morphs General Mandelbrot k-parameter across a sequence of high-resolution images."""
    def __init__(self, output_dir, file_prefix, num_images, target_w, target_h, max_iter,
                 cmap_name, cx, cy, w, start_kr, end_kr, start_ki, end_ki, gen_n,
                 overlay_hud=False, **kwargs):
        super().__init__(target_w, target_h, max_iter, cmap_name, **kwargs)
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.num_images = max(1, int(num_images))
        self.cx = Decimal(str(cx))
        self.cy = Decimal(str(cy))
        self.w = Decimal(str(w))
        self.start_kr = float(start_kr)
        self.end_kr = float(end_kr)
        self.start_ki = float(start_ki)
        self.end_ki = float(end_ki)
        self.gen_n = int(gen_n)
        self.overlay_hud = bool(overlay_hud)

    def run(self):
        writer_queue = queue.Queue(maxsize=3)
        free_buffers = queue.Queue()
        writer_err = []

        def disk_saver():
            try:
                while True:
                    item = writer_queue.get()
                    if item is None:
                        writer_queue.task_done()
                        break
                    full_path, buf, is_hdr = item
                    save_image_buffer(full_path, buf, is_hdr)
                    free_buffers.put(buf)
                    writer_queue.task_done()
            except Exception as ex:
                writer_err.append(str(ex))

        try:
            ensure_decimal_precision(self.w)
            if self.compute_device == "GPU" and HAS_CUDA:
                cuda.select_device(0)
            t0 = time.perf_counter()
            dtype_val = np.uint16 if self.is_hdr else np.uint8
            d_rgb_buf = cuda.device_array((self.target_h, self.target_w, 3), dtype=dtype_val) if (self.compute_device == "GPU" and HAS_CUDA) else None

            for _ in range(3):
                buf = cuda.pinned_array((self.target_h, self.target_w, 3), dtype=dtype_val) if (self.compute_device == "GPU" and HAS_CUDA) else np.empty((self.target_h, self.target_w, 3), dtype=dtype_val)
                free_buffers.put(buf)

            saver_thread = threading.Thread(target=disk_saver, daemon=True)
            saver_thread.start()

            inv_denom = float(1.0) / max(1.0, float(self.num_images - 1))
            recent_times = deque(maxlen=100)
            pad_digits = max(4, len(str(self.num_images)))

            for idx in range(self.num_images):
                if self.is_cancelled or writer_err:
                    break
                frame_t0 = time.perf_counter()
                t = float(idx) * inv_denom

                cur_kr = self.start_kr + t * (self.end_kr - self.start_kr)
                cur_ki = self.start_ki + t * (self.end_ki - self.start_ki)

                eff_iter = compute_dynamic_max_iter(
                    self.max_iter, self.w, self.dynamic_iter_mode,
                    center_x_dec=self.cx, center_y_dec=self.cy,
                    fractal_type=4, julia_cx=float(self.julia_cx), julia_cy=float(self.julia_cy),
                    gen_n=self.gen_n, gen_kr=cur_kr, gen_ki=cur_ki,
                    precision_mode=self.precision_mode, bbsa_tol=self.bbsa_tol, bbsa_order=self.bbsa_order,
                    compute_device=self.compute_device, cpu_threads=self.cpu_threads
                )
                hdr_tag = "_16bitHDR" if self.is_hdr else ""
                file_name = (
                    f"{self.file_prefix}_{idx+1:0{pad_digits}d}_{self.target_w}x{self.target_h}_"
                    f"{self.ssaa_factor}xSSAA{hdr_tag}_"
                    f"kr{cur_kr:+.4f}_ki{cur_ki:+.4f}_n{self.gen_n}.png"
                )

                h_buf = None
                while not self.is_cancelled and not writer_err:
                    try:
                        h_buf = free_buffers.get(timeout=0.1)
                        break
                    except queue.Empty:
                        continue

                if self.is_cancelled or writer_err or h_buf is None:
                    break

                frame_params = RenderParameters(
                    target_w=self.target_w, target_h=self.target_h, max_iter=eff_iter,
                    center_x_dec=self.cx, center_y_dec=self.cy, plot_width_dec=self.w,
                    cmap_name=self.cmap_name, ssaa_factor=self.ssaa_factor, edge_threshold=self.edge_threshold,
                    color_scheme_id=self.color_scheme_id, precision_mode=self.precision_mode,
                    fractal_type=4, julia_cx=float(self.julia_cx), julia_cy=float(self.julia_cy),
                    bbsa_tol=self.bbsa_tol, bbsa_order=self.bbsa_order, glitch_mode=self.glitch_mode,
                    gen_n=self.gen_n, gen_kr=cur_kr, gen_ki=cur_ki,
                    d_rgb_buf=d_rgb_buf, h_rgb_buf=h_buf, is_hdr=self.is_hdr,
                    compute_device=self.compute_device, cpu_threads=self.cpu_threads, palette_offset=self.palette_offset,
                    color_density=self.color_density, color_contrast=self.color_contrast
                )
                render_frame(frame_params)
                if self.overlay_hud:
                    draw_video_hud_overlay(h_buf, self.w, cx=self.cx, cy=self.cy, is_hdr=self.is_hdr, max_iter=eff_iter)
                if self.preview_hud:
                    self.preview_updated.emit(downsample_to_preview_256x144(h_buf, is_hdr=self.is_hdr))
                writer_queue.put((os.path.join(self.output_dir, file_name), h_buf, self.is_hdr))
                elapsed = time.perf_counter() - frame_t0
                recent_times.append(elapsed)
                self.progress_updated.emit(idx + 1, self.num_images, 1.0 / max(1e-6, elapsed), len(recent_times) / max(1e-6, sum(recent_times)))

            if not self.is_cancelled and not writer_err:
                writer_queue.put(None)
                saver_thread.join()
            else:
                safe_queue_drain_and_terminate(writer_queue)
                saver_thread.join(timeout=1.0)

            if self.is_cancelled:
                return
            if writer_err:
                raise RuntimeError(writer_err[0])
            self.finished.emit(self.output_dir, time.perf_counter() - t0)
        except Exception as e:
            safe_queue_drain_and_terminate(writer_queue)
            self.error.emit(str(e))