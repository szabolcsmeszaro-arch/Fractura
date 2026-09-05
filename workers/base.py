from collections import deque
from decimal import Decimal, getcontext
import queue
import threading
import time
import imageio_ffmpeg
from numba import cuda
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from math_core.cuda_kernels import HAS_CUDA
from math_core.engine import compute_dynamic_max_iter, render_frame
from config import RenderParameters, draw_video_hud_overlay, ensure_decimal_precision


def safe_queue_drain_and_terminate(q):
    """Safely drains remaining items from a queue and sends termination sentinel."""
    if q is None:
        return
    while not q.empty():
        try:
            q.get_nowait()
            q.task_done()
        except (queue.Empty, ValueError):
            break
    try:
        q.put_nowait(None)
    except queue.Full:
        try:
            q.get_nowait()
            q.task_done()
            q.put_nowait(None)
        except Exception:
            pass


def downsample_to_preview_256x144(buf, is_hdr=False):
    """Ultra-fast 256x144 preview downsampler (<0.06ms) for calculation HUD."""
    h, w = buf.shape[0], buf.shape[1]
    sy = np.linspace(0, h - 1, 144, dtype=np.int32)
    sx = np.linspace(0, w - 1, 256, dtype=np.int32)
    sub = buf[sy[:, None], sx[None, :]]
    if is_hdr or sub.dtype == np.uint16:
        sub = (sub >> 8).astype(np.uint8)
    elif sub.dtype != np.uint8:
        sub = np.clip(sub, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(sub)


downsample_to_preview_192x108 = downsample_to_preview_256x144
downsample_to_preview_128x72 = downsample_to_preview_256x144


class BaseFractalWorker(QThread):
    finished = pyqtSignal(str, float)
    error = pyqtSignal(str)
    progress_updated = pyqtSignal(int, int, float, float)
    preview_updated = pyqtSignal(np.ndarray)

    def __init__(self, target_w=1280, target_h=720, max_iter=1000, cmap_name="inferno", ssaa_factor=1, edge_threshold=0.35,
                 color_scheme_id=0, precision_mode="1e-300 (Perturbation)",
                 fractal_type=0, julia_cx=0.0, julia_cy=0.0, bbsa_tol=1e-4, bbsa_order=4,
                 glitch_mode="Off (Legacy Single-Ref)", dynamic_iter_mode="Off",
                 gen_n=3, gen_kr=0.25, gen_ki=1.0, is_hdr=False,
                 compute_device="GPU", cpu_threads=2, palette_offset=0.0,
                 color_density=1.0, color_contrast=1.0,
                 center_x_dec=Decimal("-0.75"), center_y_dec=Decimal("0.0"), plot_width_dec=Decimal("4.0"),
                 cx=None, cy=None, w=None,
                 render_params=None, preview_hud=False, **kwargs):
        super().__init__()
        if cx is not None: center_x_dec = cx
        if cy is not None: center_y_dec = cy
        if w is not None: plot_width_dec = w

        c_x = center_x_dec if isinstance(center_x_dec, Decimal) else Decimal(str(center_x_dec))
        c_y = center_y_dec if isinstance(center_y_dec, Decimal) else Decimal(str(center_y_dec))
        p_w = plot_width_dec if isinstance(plot_width_dec, Decimal) else Decimal(str(plot_width_dec))

        if render_params is not None and isinstance(render_params, RenderParameters):
            self.p = render_params
        else:
            self.p = RenderParameters(
                target_w=target_w, target_h=target_h, max_iter=max_iter,
                center_x_dec=c_x, center_y_dec=c_y, plot_width_dec=p_w,
                cmap_name=cmap_name, ssaa_factor=ssaa_factor, edge_threshold=float(edge_threshold),
                color_scheme_id=color_scheme_id, precision_mode=precision_mode,
                fractal_type=fractal_type, julia_cx=float(julia_cx), julia_cy=float(julia_cy),
                bbsa_tol=float(bbsa_tol), bbsa_order=int(bbsa_order), glitch_mode=glitch_mode,
                gen_n=int(gen_n), gen_kr=float(gen_kr), gen_ki=float(gen_ki), is_hdr=bool(is_hdr),
                compute_device=compute_device, cpu_threads=int(cpu_threads), palette_offset=float(palette_offset),
                color_density=float(color_density), color_contrast=float(color_contrast)
            )
        self.dynamic_iter_mode = dynamic_iter_mode
        self.preview_hud = bool(preview_hud)
        self.is_cancelled = False

    @property
    def target_w(self): return self.p.target_w
    @target_w.setter
    def target_w(self, val): self.p.target_w = val

    @property
    def target_h(self): return self.p.target_h
    @target_h.setter
    def target_h(self, val): self.p.target_h = val

    @property
    def max_iter(self): return self.p.max_iter
    @max_iter.setter
    def max_iter(self, val): self.p.max_iter = val

    @property
    def cmap_name(self): return self.p.cmap_name
    @cmap_name.setter
    def cmap_name(self, val): self.p.cmap_name = val

    @property
    def ssaa_factor(self): return self.p.ssaa_factor
    @ssaa_factor.setter
    def ssaa_factor(self, val): self.p.ssaa_factor = val

    @property
    def edge_threshold(self): return self.p.edge_threshold
    @edge_threshold.setter
    def edge_threshold(self, val): self.p.edge_threshold = val

    @property
    def color_scheme_id(self): return self.p.color_scheme_id
    @color_scheme_id.setter
    def color_scheme_id(self, val): self.p.color_scheme_id = val

    @property
    def precision_mode(self): return self.p.precision_mode
    @precision_mode.setter
    def precision_mode(self, val): self.p.precision_mode = val

    @property
    def fractal_type(self): return self.p.fractal_type
    @fractal_type.setter
    def fractal_type(self, val): self.p.fractal_type = val

    @property
    def julia_cx(self): return self.p.julia_cx
    @julia_cx.setter
    def julia_cx(self, val): self.p.julia_cx = val

    @property
    def julia_cy(self): return self.p.julia_cy
    @julia_cy.setter
    def julia_cy(self, val): self.p.julia_cy = val

    @property
    def bbsa_tol(self): return self.p.bbsa_tol
    @bbsa_tol.setter
    def bbsa_tol(self, val): self.p.bbsa_tol = val

    @property
    def bbsa_order(self): return self.p.bbsa_order
    @bbsa_order.setter
    def bbsa_order(self, val): self.p.bbsa_order = val

    @property
    def glitch_mode(self): return self.p.glitch_mode
    @glitch_mode.setter
    def glitch_mode(self, val): self.p.glitch_mode = val

    @property
    def gen_n(self): return self.p.gen_n
    @gen_n.setter
    def gen_n(self, val): self.p.gen_n = val

    @property
    def gen_kr(self): return self.p.gen_kr
    @gen_kr.setter
    def gen_kr(self, val): self.p.gen_kr = val

    @property
    def gen_ki(self): return self.p.gen_ki
    @gen_ki.setter
    def gen_ki(self, val): self.p.gen_ki = val

    @property
    def is_hdr(self): return self.p.is_hdr
    @is_hdr.setter
    def is_hdr(self, val): self.p.is_hdr = val

    @property
    def compute_device(self): return self.p.compute_device
    @compute_device.setter
    def compute_device(self, val): self.p.compute_device = val

    @property
    def cpu_threads(self): return self.p.cpu_threads
    @cpu_threads.setter
    def cpu_threads(self, val): self.p.cpu_threads = val

    @property
    def palette_offset(self): return self.p.palette_offset
    @palette_offset.setter
    def palette_offset(self, val): self.p.palette_offset = val

    @property
    def color_density(self): return self.p.color_density
    @color_density.setter
    def color_density(self, val): self.p.color_density = float(val)

    @property
    def color_contrast(self): return self.p.color_contrast
    @color_contrast.setter
    def color_contrast(self, val): self.p.color_contrast = float(val)

    @property
    def center_x_dec(self): return self.p.center_x_dec
    @center_x_dec.setter
    def center_x_dec(self, val): self.p.center_x_dec = Decimal(str(val))

    @property
    def center_y_dec(self): return self.p.center_y_dec
    @center_y_dec.setter
    def center_y_dec(self, val): self.p.center_y_dec = Decimal(str(val))

    @property
    def plot_width_dec(self): return self.p.plot_width_dec
    @plot_width_dec.setter
    def plot_width_dec(self, val): self.p.plot_width_dec = Decimal(str(val))

    @property
    def cx(self): return self.p.center_x_dec
    @cx.setter
    def cx(self, val): self.p.center_x_dec = Decimal(str(val))

    @property
    def cy(self): return self.p.center_y_dec
    @cy.setter
    def cy(self, val): self.p.center_y_dec = Decimal(str(val))

    @property
    def w(self): return self.p.plot_width_dec
    @w.setter
    def w(self, val): self.p.plot_width_dec = Decimal(str(val))

    def cancel(self):
        self.is_cancelled = True

    def render_params(self):
        return {
            'ssaa_factor': self.ssaa_factor,
            'edge_threshold': self.edge_threshold,
            'color_scheme_id': self.color_scheme_id,
            'precision_mode': self.precision_mode,
            'fractal_type': self.fractal_type,
            'julia_cx': self.julia_cx,
            'julia_cy': self.julia_cy,
            'bbsa_tol': self.bbsa_tol,
            'bbsa_order': self.bbsa_order,
            'glitch_mode': self.glitch_mode,
            'gen_n': self.gen_n,
            'gen_kr': self.gen_kr,
            'gen_ki': self.gen_ki,
            'is_hdr': self.is_hdr,
            'compute_device': self.compute_device,
            'cpu_threads': self.cpu_threads,
            'palette_offset': self.palette_offset,
            'color_density': self.color_density,
            'color_contrast': self.color_contrast
        }


class BaseVideoRenderWorker(BaseFractalWorker):
    """Core video rendering engine handling FFmpeg pipe streaming, double-buffering, and HDR metadata."""
    def __init__(self, output_path, target_w, target_h, fps, bitrate_kbps, duration_sec,
                 max_iter, cmap_name, hdr_mode="HDR10 (PQ)", encoder_name="Software (CPU libx264 / libx265)",
                 rate_control="crf", crf=16, palette_cycle_speed=0.0, overlay_hud=False, **kwargs):
        super().__init__(target_w, target_h, max_iter, cmap_name, **kwargs)
        self.output_path = output_path
        self.fps = int(fps)
        self.bitrate_kbps = int(bitrate_kbps)
        self.duration_sec = duration_sec
        self.hdr_mode = hdr_mode
        self.encoder_name = encoder_name
        self.rate_control = str(rate_control).lower()
        self.crf = int(crf)
        self.palette_cycle_speed = float(palette_cycle_speed)
        self.overlay_hud = bool(overlay_hud)

    def get_frame_state(self, frame_idx, total_frames):
        """Override in subclasses. Returns: (cx_dec, cy_dec, w_dec, extra_render_params_dict)"""
        raise NotImplementedError

    def run(self):
        writer_gen, encoder_thread = None, None
        encode_queue = queue.Queue(maxsize=4)
        free_buffers = queue.Queue()
        writer_error = []

        def video_writer_target():
            try:
                while True:
                    item = encode_queue.get()
                    if item is None:
                        encode_queue.task_done()
                        break
                    writer_gen.send(item)
                    free_buffers.put(item)
                    encode_queue.task_done()
                writer_gen.close()
            except Exception as ex:
                writer_error.append(str(ex))
                if writer_gen:
                    try:
                        writer_gen.close()
                    except Exception:
                        pass

        try:
            if hasattr(self, 'start_w') and hasattr(self, 'end_w') and self.start_w is not None and self.end_w is not None:
                ensure_decimal_precision(min(self.start_w, self.end_w))
            elif hasattr(self, 'end_w') and self.end_w is not None:
                ensure_decimal_precision(self.end_w)
            elif hasattr(self, 'waypoints') and self.waypoints:
                ensure_decimal_precision(min(wp[2] for wp in self.waypoints))
            elif hasattr(self, 'w'):
                ensure_decimal_precision(self.w)
            if self.compute_device == "GPU" and HAS_CUDA:
                cuda.select_device(0)
            t0 = time.perf_counter()
            total_frames = max(1, int(self.fps * self.duration_sec))
            dtype_val = np.uint16 if self.is_hdr else np.uint8
            enc_name = str(self.encoder_name).lower()
            is_crf_mode = (self.rate_control == "crf")
            crf_str = str(self.crf)
            bitrate_arg = None if is_crf_mode else f"{self.bitrate_kbps}k"

            if self.is_hdr:
                trc = 'smpte2084' if 'PQ' in self.hdr_mode else 'arib-std-b67'
                trc_val = 16 if 'PQ' in self.hdr_mode else 18
                hdr_color_args = ['-color_primaries', 'bt2020', '-color_trc', trc, '-colorspace', 'bt2020nc', '-color_range', 'tv']

                if 'av1' in enc_name and 'nvenc' in enc_name:
                    output_params = [
                        '-bsf:v', f'av1_metadata=color_primaries=9:transfer_characteristics={trc_val}:matrix_coefficients=9:color_range=tv'
                    ] + hdr_color_args + ['-preset', 'p4', '-tune', 'hq']
                    if is_crf_mode:
                        output_params += ['-rc:v', 'vbr', '-cq:v', crf_str, '-b:v', '0']
                    writer_gen = imageio_ffmpeg.write_frames(
                        self.output_path, size=(self.target_w, self.target_h), fps=self.fps, codec='av1_nvenc',
                        bitrate=bitrate_arg, pix_fmt_in='rgb48le', pix_fmt_out='p010le',
                        macro_block_size=1, output_params=output_params
                    )
                elif 'av1' in enc_name:
                    output_params = [
                        '-bsf:v', f'av1_metadata=color_primaries=9:transfer_characteristics={trc_val}:matrix_coefficients=9:color_range=tv'
                    ] + hdr_color_args + ['-cpu-used', '6']
                    if is_crf_mode:
                        output_params += ['-crf', crf_str, '-b:v', '0']
                    writer_gen = imageio_ffmpeg.write_frames(
                        self.output_path, size=(self.target_w, self.target_h), fps=self.fps, codec='libaom-av1',
                        bitrate=bitrate_arg, pix_fmt_in='rgb48le', pix_fmt_out='yuv420p10le',
                        macro_block_size=1, output_params=output_params
                    )
                elif 'vp9' in enc_name:
                    output_params = [
                        '-bsf:v', 'vp9_metadata=color_space=5:color_range=0'
                    ] + hdr_color_args + ['-profile:v', '2', '-deadline', 'realtime', '-cpu-used', '6']
                    if is_crf_mode:
                        output_params += ['-crf', crf_str, '-b:v', '0']
                    writer_gen = imageio_ffmpeg.write_frames(
                        self.output_path, size=(self.target_w, self.target_h), fps=self.fps, codec='libvpx-vp9',
                        bitrate=bitrate_arg, pix_fmt_in='rgb48le', pix_fmt_out='yuv420p10le',
                        macro_block_size=1, output_params=output_params
                    )
                elif 'nvenc' in enc_name:
                    output_params = [
                        '-bsf:v', f'hevc_metadata=colour_primaries=9:transfer_characteristics={trc_val}:matrix_coefficients=9'
                    ] + hdr_color_args + ['-profile:v', 'main10', '-preset', 'p4', '-tune', 'hq']
                    if is_crf_mode:
                        output_params += ['-rc:v', 'vbr', '-cq:v', crf_str, '-b:v', '0']
                    writer_gen = imageio_ffmpeg.write_frames(
                        self.output_path, size=(self.target_w, self.target_h), fps=self.fps, codec='hevc_nvenc',
                        bitrate=bitrate_arg, pix_fmt_in='rgb48le', pix_fmt_out='p010le',
                        macro_block_size=1, output_params=output_params
                    )
                else:
                    output_params = hdr_color_args + [
                        '-x265-params', f'colorprim=bt2020:transfer={trc}:colormatrix=bt2020nc:range=limited:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400',
                        '-preset', 'fast'
                    ]
                    if is_crf_mode:
                        output_params += ['-crf', crf_str]
                    writer_gen = imageio_ffmpeg.write_frames(
                        self.output_path, size=(self.target_w, self.target_h), fps=self.fps, codec='libx265',
                        bitrate=bitrate_arg, pix_fmt_in='rgb48le', pix_fmt_out='yuv420p10le',
                        macro_block_size=1, output_params=output_params
                    )
            else:
                if 'av1' in enc_name and 'nvenc' in enc_name:
                    output_params = ['-preset', 'p4', '-tune', 'hq']
                    if is_crf_mode:
                        output_params += ['-rc:v', 'vbr', '-cq:v', crf_str, '-b:v', '0']
                    writer_gen = imageio_ffmpeg.write_frames(
                        self.output_path, size=(self.target_w, self.target_h), fps=self.fps, codec='av1_nvenc',
                        bitrate=bitrate_arg, pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
                        macro_block_size=1, output_params=output_params
                    )
                elif 'av1' in enc_name:
                    output_params = ['-cpu-used', '6']
                    if is_crf_mode:
                        output_params += ['-crf', crf_str, '-b:v', '0']
                    writer_gen = imageio_ffmpeg.write_frames(
                        self.output_path, size=(self.target_w, self.target_h), fps=self.fps, codec='libaom-av1',
                        bitrate=bitrate_arg, pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
                        macro_block_size=1, output_params=output_params
                    )
                elif 'vp9' in enc_name:
                    output_params = ['-deadline', 'realtime', '-cpu-used', '6']
                    if is_crf_mode:
                        output_params += ['-crf', crf_str, '-b:v', '0']
                    writer_gen = imageio_ffmpeg.write_frames(
                        self.output_path, size=(self.target_w, self.target_h), fps=self.fps, codec='libvpx-vp9',
                        bitrate=bitrate_arg, pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
                        macro_block_size=1, output_params=output_params
                    )
                elif 'nvenc' in enc_name:
                    output_params = ['-preset', 'p4', '-tune', 'hq']
                    if is_crf_mode:
                        output_params += ['-rc:v', 'vbr', '-cq:v', crf_str, '-b:v', '0']
                    writer_gen = imageio_ffmpeg.write_frames(
                        self.output_path, size=(self.target_w, self.target_h), fps=self.fps, codec='h264_nvenc',
                        bitrate=bitrate_arg, pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
                        macro_block_size=1, output_params=output_params
                    )
                else:
                    output_params = ['-preset', 'fast']
                    if is_crf_mode:
                        output_params += ['-crf', crf_str]
                    writer_gen = imageio_ffmpeg.write_frames(
                        self.output_path, size=(self.target_w, self.target_h), fps=self.fps, codec='libx264',
                        bitrate=bitrate_arg, pix_fmt_in='rgb24', pix_fmt_out='yuv420p',
                        macro_block_size=1, output_params=output_params
                    )

            writer_gen.send(None)
            for _ in range(4):
                buf = cuda.pinned_array((self.target_h, self.target_w, 3), dtype=dtype_val) if (self.compute_device == "GPU" and HAS_CUDA) else np.empty((self.target_h, self.target_w, 3), dtype=dtype_val)
                free_buffers.put(buf)

            encoder_thread = threading.Thread(target=video_writer_target, daemon=True)
            encoder_thread.start()

            streams = [cuda.stream(), cuda.stream()] if (self.compute_device == "GPU" and HAS_CUDA) else [0, 0]
            d_rgb_bufs = [cuda.device_array((self.target_h, self.target_w, 3), dtype=dtype_val) for _ in range(2)] if (self.compute_device == "GPU" and HAS_CUDA) else [None, None]
            recent_times = deque(maxlen=100)
            pending_frames = deque()
            use_async = (self.compute_device == "GPU" and HAS_CUDA)
            last_completion_time = [time.perf_counter()]

            def process_completed_frame(job):
                f_idx, f_t0, f_w, f_cx, f_cy, f_iter, f_buf, f_stream = job
                if f_stream != 0 and HAS_CUDA:
                    f_stream.synchronize()
                if self.overlay_hud:
                    draw_video_hud_overlay(f_buf, f_w, cx=f_cx, cy=f_cy, is_hdr=self.is_hdr, max_iter=f_iter)
                if self.preview_hud:
                    self.preview_updated.emit(downsample_to_preview_256x144(f_buf, is_hdr=self.is_hdr))
                encode_queue.put(f_buf)
                now = time.perf_counter()
                elapsed = max(1e-6, now - last_completion_time[0])
                last_completion_time[0] = now
                recent_times.append(elapsed)
                self.progress_updated.emit(f_idx + 1, total_frames, 1.0 / elapsed, len(recent_times) / max(1e-6, sum(recent_times)))

            for i in range(total_frames):
                frame_t0 = time.perf_counter()
                if self.is_cancelled or writer_error:
                    break

                slot = i % 2
                stream = streams[slot]

                current_cx, current_cy, current_w, frame_overrides = self.get_frame_state(i, total_frames)

                params = self.render_params()
                if self.palette_cycle_speed != 0.0:
                    params['palette_offset'] = (self.palette_offset + float(i) * (self.palette_cycle_speed / max(1, float(self.fps)))) % 1.0
                params.update(frame_overrides)
                target_max_iter = params.pop('max_iter', self.max_iter)

                eff_iter = compute_dynamic_max_iter(
                    target_max_iter, current_w, self.dynamic_iter_mode,
                    center_x_dec=current_cx, center_y_dec=current_cy,
                    fractal_type=params.get('fractal_type', self.fractal_type),
                    julia_cx=float(params.get('julia_cx', self.julia_cx)),
                    julia_cy=float(params.get('julia_cy', self.julia_cy)),
                    gen_n=params.get('gen_n', self.gen_n),
                    gen_kr=params.get('gen_kr', self.gen_kr),
                    gen_ki=params.get('gen_ki', self.gen_ki),
                    precision_mode=self.precision_mode, bbsa_tol=self.bbsa_tol, bbsa_order=self.bbsa_order,
                    compute_device=self.compute_device, cpu_threads=self.cpu_threads
                )

                h_buf = None
                while not self.is_cancelled and not writer_error:
                    try:
                        h_buf = free_buffers.get(timeout=0.1)
                        break
                    except queue.Empty:
                        continue

                if self.is_cancelled or writer_error or h_buf is None:
                    break

                frame_params = RenderParameters(
                    target_w=self.target_w, target_h=self.target_h, max_iter=eff_iter,
                    center_x_dec=current_cx, center_y_dec=current_cy, plot_width_dec=current_w,
                    cmap_name=self.cmap_name, ssaa_factor=self.ssaa_factor, edge_threshold=self.edge_threshold,
                    color_scheme_id=self.color_scheme_id, precision_mode=self.precision_mode,
                    fractal_type=params.get('fractal_type', self.fractal_type),
                    julia_cx=float(params.get('julia_cx', self.julia_cx)),
                    julia_cy=float(params.get('julia_cy', self.julia_cy)),
                    bbsa_tol=self.bbsa_tol, bbsa_order=self.bbsa_order,
                    glitch_mode=self.glitch_mode,
                    gen_n=params.get('gen_n', self.gen_n),
                    gen_kr=params.get('gen_kr', self.gen_kr),
                    gen_ki=params.get('gen_ki', self.gen_ki),
                    d_rgb_buf=d_rgb_bufs[slot], h_rgb_buf=h_buf, stream=stream,
                    stream_slot=(slot + 1),
                    is_hdr=self.is_hdr, compute_device=self.compute_device, cpu_threads=self.cpu_threads,
                    palette_offset=params.get('palette_offset', self.palette_offset),
                    color_density=params.get('color_density', self.color_density),
                    color_contrast=params.get('color_contrast', self.color_contrast),
                    sync_stream=(not use_async)
                )
                render_frame(frame_params)

                pending_frames.append((i, frame_t0, current_w, current_cx, current_cy, eff_iter, h_buf, stream))

                if len(pending_frames) >= 2:
                    process_completed_frame(pending_frames.popleft())

            # Drain remaining in-flight frames
            while pending_frames and not self.is_cancelled and not writer_error:
                process_completed_frame(pending_frames.popleft())

            if not self.is_cancelled and not writer_error:
                encode_queue.put(None)
                if encoder_thread:
                    encoder_thread.join()
            else:
                safe_queue_drain_and_terminate(encode_queue)
                if encoder_thread:
                    encoder_thread.join(timeout=1.0)

            if self.is_cancelled:
                return
            if writer_error:
                raise RuntimeError(writer_error[0])
            self.finished.emit(self.output_path, time.perf_counter() - t0)
        except Exception as e:
            safe_queue_drain_and_terminate(encode_queue)
            if writer_gen:
                try:
                    writer_gen.close()
                except Exception:
                    pass
            self.error.emit(str(e))