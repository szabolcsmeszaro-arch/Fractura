from decimal import Decimal
import json
import os
from PyQt5.QtCore import QLocale, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMessageBox, QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget
)

from config import (
    FRACTAL_DEFAULTS, SCRIPT_DIR, VIDEO_ENCODER_OPTIONS, check_ram_sufficiency, format_eta_time
)
from math_core.cuda_kernels import HAS_CUDA
from math_core.engine import clear_scratch_buffer_pool
from workers.image_workers import (
    GeneralMandelbrotMorphImageWorker, ImageRenderWorker, ImageSequenceRenderWorker, MosaicRenderWorker
)
from workers.video_workers import (
    GeneralMandelbrotMorphVideoWorker, IterationRevealVideoWorker, UnifiedVideoRenderWorker
)


_ACTIVE_EXPORT_WORKERS = set()


def _register_lingering_worker(worker):
    """Retains a reference to an active worker QThread until it terminates, preventing C++ qFatal aborts."""
    if worker is None:
        return
    _ACTIVE_EXPORT_WORKERS.add(worker)

    def _cleanup():
        _ACTIVE_EXPORT_WORKERS.discard(worker)
        if not _ACTIVE_EXPORT_WORKERS:
            try:
                clear_scratch_buffer_pool()
            except Exception:
                pass

    if hasattr(worker, 'finished'):
        try:
            worker.finished.connect(_cleanup)
        except Exception:
            pass


class BaseExportDialog(QDialog):
    STYLE_DEFAULT = """
        QPushButton { background-color: #2E8B57; color: white; font-weight: bold; padding: 8px 16px; border-radius: 4px; }
        QPushButton:hover { background-color: #3cb371; }
        QPushButton:pressed { background-color: #1e5e37; }
        QPushButton:disabled { background-color: #2d3748; color: #718096; }
    """
    STYLE_BUSY = """
        QPushButton { background-color: #d97706; color: white; font-weight: bold; padding: 8px 16px; border-radius: 4px; }
        QPushButton:hover { background-color: #f59e0b; }
        QPushButton:pressed { background-color: #b45309; }
        QPushButton:disabled { background-color: #d97706; color: #ffffff; }
    """
    STYLE_CANCEL = """
        QPushButton { background-color: #4b5563; color: white; padding: 8px 16px; border-radius: 4px; }
        QPushButton:hover { background-color: #6b7280; }
        QPushButton:pressed { background-color: #374151; }
        QPushButton:disabled { background-color: #2d3748; color: #718096; }
    """
    STYLE_BROWSE = """
        QPushButton { background-color: #4b5563; color: white; padding: 4px 10px; border-radius: 4px; }
        QPushButton:hover { background-color: #6b7280; }
        QPushButton:pressed { background-color: #374151; }
        QPushButton:disabled { background-color: #2d3748; color: #718096; }
    """
    STYLE_ADD = """
        QPushButton { background-color: #2E8B57; color: white; padding: 5px 12px; font-weight: bold; border-radius: 4px; }
        QPushButton:hover { background-color: #3cb371; }
        QPushButton:pressed { background-color: #1e5e37; }
        QPushButton:disabled { background-color: #2d3748; color: #718096; }
    """
    STYLE_DELETE = """
        QPushButton { background-color: #dc3545; color: white; padding: 5px 12px; font-weight: bold; border-radius: 4px; }
        QPushButton:hover { background-color: #e04a58; }
        QPushButton:pressed { background-color: #a71d2a; }
        QPushButton:disabled { background-color: #2d3748; color: #718096; }
    """
    STYLE_CLEAR = """
        QPushButton { background-color: #4b5563; color: white; padding: 5px 12px; border-radius: 4px; }
        QPushButton:hover { background-color: #6b7280; }
        QPushButton:pressed { background-color: #374151; }
        QPushButton:disabled { background-color: #2d3748; color: #718096; }
    """

    def __init__(self, viewer, title, width=600, height=400):
        super().__init__(None)
        self.viewer = viewer
        self.worker = None
        self.setWindowTitle(title)
        self.resize(width, height)
        self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)

        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)

        self.dir_edit = None
        self.aa_combo = None

    def done(self, r):
        w = self.worker
        self.worker = None
        if w is not None:
            if hasattr(w, 'cancel'):
                w.cancel()
            if hasattr(w, 'isRunning') and w.isRunning():
                w.wait(1000)
                if w.isRunning():
                    _register_lingering_worker(w)
                    super().done(r)
                    return
        if not _ACTIVE_EXPORT_WORKERS:
            clear_scratch_buffer_pool()
        super().done(r)

    def closeEvent(self, event):
        self.done(QDialog.Rejected)
        event.accept()

    def create_res_combo(self, default_res="1920x1080"):
        combo = QComboBox()
        resolutions = [
            "480x270", "640x360", "960x540", "1280x720", "1920x1080",
            "2560x1440", "3840x2160", "5120x2880", "7680x4320", "10240x5760", "15360x8640"
        ]
        combo.addItems(resolutions)
        if default_res in resolutions:
            combo.setCurrentText(default_res)
        else:
            combo.setCurrentIndex(4)
        return combo

    def create_aa_combo(self, initial_ssaa=1):
        combo = QComboBox()
        combo.addItems(["1x (Off) / Fastest", "2x (SSAA)", "4x (SSAA)", "6x (SSAA)", "8x (SSAA / Slowest)"])
        idx_map = {1: 0, 2: 1, 4: 2, 6: 3, 8: 4}
        combo.setCurrentIndex(idx_map.get(initial_ssaa, 0))
        self.aa_combo = combo
        return combo

    def get_ssaa_factor(self):
        if self.aa_combo is None:
            return 1
        return [1, 2, 4, 6, 8][self.aa_combo.currentIndex()]

    def create_dir_selector(self):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        default_dir = getattr(self.viewer, 'last_export_dir', SCRIPT_DIR)
        self.dir_edit = QLineEdit(default_dir)
        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet(self.STYLE_BROWSE)
        browse_btn.clicked.connect(self._browse_directory)

        layout.addWidget(self.dir_edit, 1)
        layout.addWidget(browse_btn)
        return container

    def _browse_directory(self):
        current_path = self.dir_edit.text().strip() or SCRIPT_DIR
        selected = QFileDialog.getExistingDirectory(self, "Select Destination Folder", current_path)
        if selected:
            self.dir_edit.setText(selected)
            if self.viewer:
                self.viewer.last_export_dir = selected

    def validate_destination_dir(self):
        if self.dir_edit is None:
            return SCRIPT_DIR
        out_dir = self.dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Invalid Folder", "Please select a destination folder.")
            return None
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to access or create output folder:\n{e}")
            return None
        if self.viewer:
            self.viewer.last_export_dir = out_dir
        return out_dir

    def create_preview_hud_ui(self):
        self.preview_hud_check = QCheckBox("Show Live Calculation Preview (256x144 HUD)")
        self.preview_hud_check.setChecked(False)
        self.preview_hud_check.setToolTip("Displays a real-time 256x144 preview HUD of each frame/tile being calculated.")

        self.preview_container = QWidget()
        p_layout = QHBoxLayout(self.preview_container)
        p_layout.setContentsMargins(0, 4, 0, 4)
        p_layout.setAlignment(Qt.AlignCenter)

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(256, 144)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 4px; color: #64748b; font-size: 13px;")
        self.preview_label.setText("256 × 144 HUD")
        p_layout.addWidget(self.preview_label)

        self.preview_container.setVisible(False)
        self.preview_hud_check.toggled.connect(self.on_preview_hud_toggled)
        return self.preview_hud_check, self.preview_container

    def on_preview_hud_toggled(self, checked):
        if hasattr(self, 'preview_container') and self.preview_container is not None:
            self.preview_container.setVisible(checked)
        if hasattr(self, 'worker') and self.worker is not None:
            self.worker.preview_hud = bool(checked)

    def on_preview_updated(self, arr):
        if hasattr(self, 'preview_label') and hasattr(self, 'preview_container') and self.preview_container.isVisible():
            h, w, c = arr.shape
            qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
            self.preview_label.setPixmap(QPixmap.fromImage(qimg))

    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
        super().reject()


class ImageExportDialog(BaseExportDialog):
    def __init__(self, viewer, target_cx, target_cy, target_w, max_iter, cmap_name, initial_ssaa=2, **kwargs):
        super().__init__(viewer, "Save Fractal Image (PNG)", 600, 380)
        self.params = dict(kwargs, target_cx=target_cx, target_cy=target_cy, target_w=target_w, max_iter=max_iter, cmap_name=cmap_name)

        grid = QGridLayout()
        self.res_combo = self.create_res_combo("7680x4320")
        self.aa_combo = self.create_aa_combo(initial_ssaa)
        self.prefix_edit = QLineEdit("fractal_image")
        self.hdr_check = QCheckBox("Export as 16-bit Deep Color PNG (HDR / Wide-Gamut)")
        self.hud_check = QCheckBox("Overlay Telemetry HUD (Display Width & Physical Scale)")
        self.hud_check.setChecked(False)
        self.hud_check.setToolTip("Renders a sleek semi-transparent HUD banner displaying the complex plane width and real-world scale in the status bar format.")

        grid.addWidget(QLabel("Resolution:"), 0, 0)
        grid.addWidget(self.res_combo, 0, 1)
        grid.addWidget(QLabel("Antialiasing (SSAA):"), 1, 0)
        grid.addWidget(self.aa_combo, 1, 1)
        grid.addWidget(QLabel("Save Folder:"), 2, 0)
        grid.addWidget(self.create_dir_selector(), 2, 1)
        grid.addWidget(QLabel("Filename Prefix:"), 3, 0)
        grid.addWidget(self.prefix_edit, 3, 1)
        self.layout.addLayout(grid)
        self.layout.addWidget(self.hdr_check)
        self.layout.addWidget(self.hud_check)

        self.status_lbl = QLabel(f"Target Width: {Decimal(str(target_w)):.3e}")
        self.status_lbl.setStyleSheet("color: #aaaaaa;")
        self.layout.addWidget(self.status_lbl)

        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save PNG")
        self.save_btn.setStyleSheet(self.STYLE_DEFAULT)
        self.save_btn.clicked.connect(self.start_save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(self.STYLE_CANCEL)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(cancel_btn)
        self.layout.addLayout(btn_layout)

    def start_save(self):
        output_dir = self.validate_destination_dir()
        if not output_dir:
            return

        target_w, target_h = map(int, self.res_combo.currentText().split('x'))
        ssaa_factor = self.get_ssaa_factor()
        is_hdr = self.hdr_check.isChecked()

        bytes_per_sample = 2 if is_hdr else 1
        rgb_buf_bytes = target_w * target_h * 3 * bytes_per_sample
        save_overhead_bytes = target_w * target_h * (8 if is_hdr else 3)
        safety_headroom = 128 * 1024 * 1024
        required_ram = rgb_buf_bytes + save_overhead_bytes + safety_headroom

        if not check_ram_sufficiency(self, required_ram, "PNG Image Export"):
            return

        w_tag = f"_{Decimal(str(self.params['target_w'])):.6e}"
        path = os.path.join(output_dir,
            f"{self.prefix_edit.text().strip()}_{self.res_combo.currentText()}_{ssaa_factor}xSSAA{'_16bitHDR' if is_hdr else ''}_{Decimal(str(self.params['target_cx'])):.6e}_{Decimal(str(self.params['target_cy'])):.6e}{w_tag}.png"
        )

        self.save_btn.setEnabled(False)
        self.save_btn.setText("Rendering & Saving...")
        self.save_btn.setStyleSheet(self.STYLE_BUSY)

        self.worker = ImageRenderWorker(
            path, target_w, target_h, self.params['max_iter'], self.params['target_cx'], self.params['target_cy'],
            self.params['target_w'], self.params['cmap_name'], ssaa_factor=ssaa_factor, is_hdr=is_hdr,
            overlay_hud=self.hud_check.isChecked(),
            **{k: v for k, v in self.params.items() if k not in ('target_cx', 'target_cy', 'target_w', 'max_iter', 'cmap_name')}
        )
        self.worker.finished.connect(lambda p, el: (QMessageBox.information(self, "Success", f"Saved in {format_eta_time(el)}:\n{p}"), self.accept()))
        self.worker.error.connect(lambda err: (
            self.save_btn.setEnabled(True),
            self.save_btn.setText("Save PNG"),
            self.save_btn.setStyleSheet(self.STYLE_DEFAULT),
            QMessageBox.critical(self, "Error", err)
        ))
        self.worker.start()


class ImageSequenceExportDialog(BaseExportDialog):
    def __init__(self, viewer, target_cx, target_cy, target_w, max_iter, cmap_name, initial_ssaa=2, **kwargs):
        super().__init__(viewer, "Export Zoom / Pan Image Sequence (PNG Series)", 720, 620)
        self.params = dict(kwargs, target_cx=target_cx, target_cy=target_cy, target_w=target_w, max_iter=max_iter, cmap_name=cmap_name)
        self.waypoints = []

        self.layout.addWidget(QLabel("📍 Recorded Waypoints (Double-Click Canvas to Add):"))
        self.wp_list = QListWidget()
        self.wp_list.setFixedHeight(120)
        self.layout.addWidget(self.wp_list)

        wp_btn_layout = QHBoxLayout()
        add_btn = QPushButton("+ Add Current View")
        add_btn.setStyleSheet(self.STYLE_ADD)
        add_btn.clicked.connect(self.on_add_current_view)
        del_btn = QPushButton("Delete Selected")
        del_btn.setStyleSheet(self.STYLE_DELETE)
        del_btn.clicked.connect(self.on_delete_selected)
        clr_btn = QPushButton("Clear All")
        clr_btn.setStyleSheet(self.STYLE_CLEAR)
        clr_btn.clicked.connect(self.on_clear_waypoints)
        load_session_btn = QPushButton("📂 Load Session...")
        load_session_btn.setToolTip("Load sequence parameters, waypoints, and image count from a previously saved session JSON in the output folder.")
        load_session_btn.clicked.connect(self.on_load_session)

        wp_btn_layout.addWidget(add_btn)
        wp_btn_layout.addWidget(del_btn)
        wp_btn_layout.addWidget(clr_btn)
        wp_btn_layout.addWidget(load_session_btn)
        wp_btn_layout.addStretch()
        self.layout.addLayout(wp_btn_layout)

        grid = QGridLayout()
        count_container = QWidget()
        count_layout = QHBoxLayout(count_container)
        count_layout.setContentsMargins(0, 0, 0, 0)
        self.count_spin = QSpinBox()
        self.count_spin.setRange(2, 1000000)
        self.count_spin.setValue(10)
        self.count_spin.setMinimumWidth(160)

        calc_frames_btn = QPushButton("Auto (60 FPS Video Length)")
        calc_frames_btn.setToolTip("Calculates the number of images to match the optical zoom speed equation of a 60 fps video.")
        calc_frames_btn.clicked.connect(self.on_calc_video_frames)

        count_layout.addWidget(self.count_spin)
        count_layout.addWidget(calc_frames_btn)
        count_layout.addStretch()

        self.res_combo = self.create_res_combo("1920x1080")
        self.aa_combo = self.create_aa_combo(initial_ssaa)
        self.prefix_edit = QLineEdit("fractal_seq")
        self.resume_check = QCheckBox("Resume / Skip already rendered frames in destination folder")
        self.resume_check.setChecked(True)
        self.resume_check.setToolTip("If enabled, previously completed images in the folder will be skipped instantly, resuming the sequence right where you left off.")
        self.hdr_check = QCheckBox("Export as 16-bit Deep Color PNGs (HDR / Wide-Gamut)")
        self.hud_check = QCheckBox("Overlay Telemetry HUD (Display Width & Physical Scale)")
        self.hud_check.setChecked(False)
        self.hud_check.setToolTip("Renders a sleek semi-transparent HUD banner displaying the complex plane width and real-world scale in the status bar format.")

        grid.addWidget(QLabel("Number of Images:"), 0, 0)
        grid.addWidget(count_container, 0, 1)
        grid.addWidget(QLabel("Per-Frame Resolution:"), 1, 0)
        grid.addWidget(self.res_combo, 1, 1)
        grid.addWidget(QLabel("Antialiasing (SSAA):"), 2, 0)
        grid.addWidget(self.aa_combo, 2, 1)
        grid.addWidget(QLabel("Save Folder:"), 3, 0)
        grid.addWidget(self.create_dir_selector(), 3, 1)
        grid.addWidget(QLabel("Filename Prefix:"), 4, 0)
        grid.addWidget(self.prefix_edit, 4, 1)
        self.layout.addLayout(grid)
        self.layout.addWidget(self.resume_check)
        self.layout.addWidget(self.hdr_check)
        self.layout.addWidget(self.hud_check)
        p_chk, p_box = self.create_preview_hud_ui()
        self.layout.addWidget(p_chk)
        self.layout.addWidget(p_box)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Ready to export sequence.")
        self.status_lbl.setStyleSheet("color: #458b00;")
        self.layout.addWidget(self.status_lbl)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Sequence Render")
        self.start_btn.setStyleSheet(self.STYLE_DEFAULT)
        self.start_btn.clicked.connect(self.start_export)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(self.STYLE_CANCEL)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(cancel_btn)
        self.layout.addLayout(btn_layout)

        if self.viewer and hasattr(self.viewer, 'canvas'):
            self.viewer.canvas.keyframe_recorded.connect(self.sync_waypoints_ui)
        self.sync_waypoints_ui()

    def calc_video_frames(self):
        fps = 60
        user_wps = getattr(self.viewer, 'waypoints', []) if self.viewer else []
        if len(user_wps) >= 2:
            duration = max(15, len(user_wps) * 8)
            return duration * fps
        target_w = Decimal(str(self.params.get('target_w', '4.0')))
        default_start_w = FRACTAL_DEFAULTS.get(self.params.get('fractal_type', 0), (0, 0, Decimal("4.0")))[2]
        if target_w > Decimal("0") and default_start_w > target_w:
            duration = max(5, int(0.6 * round(float((default_start_w.ln() - target_w.ln()) / Decimal("10").ln()) * 4.0)))
            return duration * fps
        return 10 * fps

    def on_calc_video_frames(self):
        frames = self.calc_video_frames()
        self.count_spin.setValue(frames)
        duration_sec = frames // 60
        self.status_lbl.setText(f"Calculated {frames:,} images ({duration_sec}s equivalent at 60 fps).")
        self.status_lbl.setStyleSheet("color: #2563eb;")

    def sync_waypoints_ui(self):
        self.wp_list.clear()
        waypoints = getattr(self.viewer, 'waypoints', [])
        for idx, (cx, cy, w) in enumerate(waypoints, 1):
            self.wp_list.addItem(f"Point #{idx:02d} | Center: ({float(cx):.6e}, {float(cy):.6e}) | Width: {Decimal(str(w)):.3e}")

    def add_waypoint(self, cx, cy, w):
        if not hasattr(self.viewer, 'waypoints'):
            self.viewer.waypoints = []
        pt = (Decimal(str(cx)), Decimal(str(cy)), Decimal(str(w)))
        self.viewer.waypoints.append(pt)
        self.sync_waypoints_ui()
        if hasattr(self.viewer, 'canvas'):
            self.viewer.canvas.update()

    def on_add_current_view(self):
        self.add_waypoint(self.viewer.canvas.center_x, self.viewer.canvas.center_y, self.viewer.canvas.plot_width)

    def on_delete_selected(self):
        waypoints = getattr(self.viewer, 'waypoints', [])
        row = self.wp_list.currentRow()
        if 0 <= row < len(waypoints):
            del waypoints[row]
            self.sync_waypoints_ui()
            if hasattr(self.viewer, 'canvas'):
                self.viewer.canvas.update()

    def on_clear_waypoints(self):
        if hasattr(self.viewer, 'waypoints'):
            self.viewer.waypoints.clear()
        self.sync_waypoints_ui()
        if hasattr(self.viewer, 'canvas'):
            self.viewer.canvas.update()

    def on_load_session(self):
        folder = self.dir_edit.text().strip() if hasattr(self, 'dir_edit') and self.dir_edit.text().strip() else os.path.expanduser("~")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Saved Sequence Session Manifest", folder, "JSON Session Files (*_session.json *.json);;All Files (*)"
        )
        if not file_path or not os.path.exists(file_path):
            return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "file_prefix" in data:
                self.prefix_edit.setText(data["file_prefix"])
            if "num_images" in data:
                self.count_spin.setValue(int(data["num_images"]))
            if "target_w" in data and "target_h" in data:
                res_str = f"{data['target_w']}x{data['target_h']}"
                idx = self.res_combo.findText(res_str)
                if idx >= 0:
                    self.res_combo.setCurrentIndex(idx)
                else:
                    self.res_combo.addItem(res_str)
                    self.res_combo.setCurrentText(res_str)
            if "ssaa_factor" in data:
                ssaa_str = f"{data['ssaa_factor']}x"
                idx = self.aa_combo.findText(ssaa_str)
                if idx >= 0:
                    self.aa_combo.setCurrentIndex(idx)
            if "is_hdr" in data:
                self.hdr_check.setChecked(bool(data["is_hdr"]))
            if "overlay_hud" in data:
                self.hud_check.setChecked(bool(data["overlay_hud"]))

            # Set directory to session file's parent folder
            session_dir = os.path.dirname(file_path)
            if hasattr(self, 'dir_edit'):
                self.dir_edit.setText(session_dir)

            # Restore waypoints
            if "waypoints" in data and data["waypoints"]:
                if hasattr(self.viewer, 'waypoints'):
                    self.viewer.waypoints = [
                        (Decimal(str(wp["center_x"])), Decimal(str(wp["center_y"])), Decimal(str(wp["plot_width"])))
                        for wp in data["waypoints"]
                    ]
                self.sync_waypoints_ui()
                if hasattr(self.viewer, 'canvas'):
                    self.viewer.canvas.update()

            # Scan folder for existing frames
            prefix = data.get("file_prefix", self.prefix_edit.text().strip())
            num_imgs = data.get("num_images", self.count_spin.value())
            pad_digits = max(4, len(str(num_imgs)))
            
            existing_count = 0
            try:
                files = set(os.listdir(session_dir))
                for i in range(1, num_imgs + 1):
                    pfx = f"{prefix}_{i:0{pad_digits}d}_"
                    for fname in files:
                        if fname.startswith(pfx) and fname.endswith(".png"):
                            fp = os.path.join(session_dir, fname)
                            if os.path.exists(fp) and os.path.getsize(fp) > 1024:
                                existing_count += 1
                            break
            except Exception:
                pass

            pct = (existing_count / max(1, num_imgs)) * 100.0
            if existing_count > 0:
                self.status_lbl.setText(f"Session loaded: {existing_count:,} / {num_imgs:,} frames ({pct:.1f}%) found on disk. Ready to resume.")
                self.status_lbl.setStyleSheet("color: #2563eb; font-weight: bold;")
            else:
                self.status_lbl.setText(f"Session loaded ({num_imgs:,} total frames). Ready to render.")
                self.status_lbl.setStyleSheet("color: #458b00; font-weight: bold;")

        except Exception as e:
            QMessageBox.warning(self, "Load Session Failed", f"Could not load sequence session:\n{e}")

    def done(self, r):
        if self.viewer and hasattr(self.viewer, 'canvas'):
            try:
                self.viewer.canvas.keyframe_recorded.disconnect(self.sync_waypoints_ui)
            except Exception:
                pass
        super().done(r)

    def start_export(self):
        output_dir = self.validate_destination_dir()
        if not output_dir:
            return

        target_w, target_h = map(int, self.res_combo.currentText().split('x'))
        is_hdr = self.hdr_check.isChecked()

        bytes_per_sample = 2 if is_hdr else 1
        single_frame_bytes = target_w * target_h * 3 * bytes_per_sample
        queue_buffer_pool = 3 * single_frame_bytes
        save_overhead_bytes = target_w * target_h * (8 if is_hdr else 3)
        safety_headroom = 256 * 1024 * 1024
        required_ram = queue_buffer_pool + save_overhead_bytes + safety_headroom

        if not check_ram_sufficiency(self, required_ram, "Zoom Image Sequence"):
            return

        self.start_btn.setEnabled(False)
        self.start_btn.setText("Rendering Sequence...")
        self.start_btn.setStyleSheet(self.STYLE_BUSY)
        self.progress_bar.setVisible(True)

        user_wps = getattr(self.viewer, 'waypoints', [])
        if len(user_wps) >= 2:
            waypoints = list(user_wps)
        elif len(user_wps) == 1:
            default_start_w = FRACTAL_DEFAULTS.get(self.params.get('fractal_type', 0), (0, 0, Decimal("4.0")))[2]
            waypoints = [(user_wps[0][0], user_wps[0][1], default_start_w), user_wps[0]]
        else:
            default_start_w = FRACTAL_DEFAULTS.get(self.params.get('fractal_type', 0), (0, 0, Decimal("4.0")))[2]
            waypoints = [
                (Decimal(str(self.params['target_cx'])), Decimal(str(self.params['target_cy'])), default_start_w),
                (Decimal(str(self.params['target_cx'])), Decimal(str(self.params['target_cy'])), Decimal(str(self.params['target_w'])))
            ]

        is_preview_hud = self.preview_hud_check.isChecked()
        self.worker = ImageSequenceRenderWorker(
            output_dir=output_dir, file_prefix=self.prefix_edit.text().strip() or "fractal_seq",
            num_images=self.count_spin.value(), target_w=target_w, target_h=target_h, max_iter=self.params['max_iter'],
            waypoints=waypoints, cmap_name=self.params['cmap_name'], ssaa_factor=self.get_ssaa_factor(), is_hdr=is_hdr,
            overlay_hud=self.hud_check.isChecked(), preview_hud=is_preview_hud,
            resume_existing=self.resume_check.isChecked(),
            **{k: v for k, v in self.params.items() if k not in ('target_cx', 'target_cy', 'target_w', 'max_iter', 'cmap_name')}
        )
        self.worker.preview_updated.connect(self.on_preview_updated)
        def on_seq_progress(cur, tot, rate, avg):
            self.progress_bar.setValue(int(cur / tot * 100))
            if rate <= 0.0 or avg <= 1e-5:
                self.status_lbl.setText(f"Resuming: skipped {cur}/{tot} frames on disk | ETA: --:--")
            else:
                self.status_lbl.setText(f"Rendered {cur}/{tot} | Rate: {rate:.3f} img/s | ETA: {format_eta_time((tot - cur) / avg)}")

        self.worker.progress_updated.connect(on_seq_progress)
        self.worker.finished.connect(lambda p, el: (QMessageBox.information(self, "Success", f"Rendered in {format_eta_time(el)}:\n{p}"), self.accept()))
        self.worker.error.connect(lambda err: (
            self.start_btn.setEnabled(True),
            self.start_btn.setText("Start Sequence Render"),
            self.start_btn.setStyleSheet(self.STYLE_DEFAULT),
            QMessageBox.critical(self, "Error", err)
        ))
        self.worker.start()


class MosaicExportDialog(BaseExportDialog):
    def __init__(self, viewer, target_cx, target_cy, target_w, max_iter, cmap_name, initial_ssaa=2, **kwargs):
        super().__init__(viewer, "Export Mosaic Grid (PNG Series)", 650, 580)
        self.params = dict(kwargs, target_cx=target_cx, target_cy=target_cy, target_w=target_w, max_iter=max_iter, cmap_name=cmap_name)

        grid = QGridLayout()
        self.grid_spin = QSpinBox()
        self.grid_spin.setRange(2, 1000)
        self.grid_spin.setValue(2)
        self.grid_spin.valueChanged.connect(self.update_stats)

        self.res_combo = self.create_res_combo("7680x4320")
        self.res_combo.currentIndexChanged.connect(self.update_stats)
        self.aa_combo = self.create_aa_combo(initial_ssaa)
        self.prefix_edit = QLineEdit("fractal_mosaic")
        self.hdr_check = QCheckBox("Export as 16-bit Deep Color PNGs (HDR / Wide-Gamut)")
        self.hud_check = QCheckBox("Overlay Telemetry HUD (Coordinates, Width & Real-World Scale)")
        self.hud_check.setChecked(False)
        self.hud_check.setToolTip("Renders a sleek semi-transparent HUD banner displaying the coordinates, complex plane width, and real-world scale on individual tiles and stitched images.")

        grid.addWidget(QLabel("Grid Tiles (X and Y):"), 0, 0)
        grid.addWidget(self.grid_spin, 0, 1)
        grid.addWidget(QLabel("Per-Tile Resolution:"), 1, 0)
        grid.addWidget(self.res_combo, 1, 1)
        grid.addWidget(QLabel("Antialiasing (SSAA):"), 2, 0)
        grid.addWidget(self.aa_combo, 2, 1)
        grid.addWidget(QLabel("Save Folder:"), 3, 0)
        grid.addWidget(self.create_dir_selector(), 3, 1)
        grid.addWidget(QLabel("Filename Prefix:"), 4, 0)
        grid.addWidget(self.prefix_edit, 4, 1)
        self.layout.addLayout(grid)
        self.layout.addWidget(self.hdr_check)
        self.layout.addWidget(self.hud_check)
        p_chk, p_box = self.create_preview_hud_ui()
        self.layout.addWidget(p_chk)
        self.layout.addWidget(p_box)

        self.stitch_check = QCheckBox("Automatically stitch tiles into a single combined PNG")
        self.stitch_check.setChecked(True)
        self.delete_tiles_check = QCheckBox("Delete individual tile images after successful stitching")
        self.delete_tiles_check.setChecked(True)
        self.stitch_check.toggled.connect(self.delete_tiles_check.setEnabled)
        self.layout.addWidget(self.stitch_check)
        self.layout.addWidget(self.delete_tiles_check)

        self.info_lbl = QLabel("")
        self.info_lbl.setStyleSheet("color: #60a5fa; font-weight: bold;")
        self.layout.addWidget(self.info_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Ready to export grid partition.")
        self.status_lbl.setStyleSheet("color: #458b00;")
        self.layout.addWidget(self.status_lbl)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Export Mosaic Grid")
        self.start_btn.setStyleSheet(self.STYLE_DEFAULT)
        self.start_btn.clicked.connect(self.start_export)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(self.STYLE_CANCEL)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(cancel_btn)
        self.layout.addLayout(btn_layout)
        self.update_stats()

    def update_stats(self):
        n = self.grid_spin.value()
        w, h = map(int, self.res_combo.currentText().split('x'))
        self.info_lbl.setText(f"Mosaic: {n}x{n} ({n*n} tiles) | Combined: {w*n}x{h*n} ({(w*n*h*n)/1e6:.1f} MP)")

    def start_export(self):
        output_dir = self.validate_destination_dir()
        if not output_dir:
            return

        target_w, target_h = map(int, self.res_combo.currentText().split('x'))
        grid_n = self.grid_spin.value()
        is_hdr = self.hdr_check.isChecked()
        auto_stitch = self.stitch_check.isChecked()
        bytes_per_sample = 2 if is_hdr else 1

        tile_pool_bytes = 4 * (target_w * target_h * 3 * bytes_per_sample)

        if auto_stitch:
            total_w = target_w * grid_n
            total_h = target_h * grid_n
            master_canvas_bytes = total_w * total_h * 3 * bytes_per_sample
            master_save_overhead = total_w * total_h * (8 if is_hdr else 3)
            stitch_total = master_canvas_bytes + master_save_overhead
        else:
            stitch_total = 0

        safety_headroom = 256 * 1024 * 1024
        required_ram = tile_pool_bytes + stitch_total + safety_headroom

        if not check_ram_sufficiency(self, required_ram, f"Mosaic Grid ({grid_n}x{grid_n}) Export"):
            return

        self.start_btn.setEnabled(False)
        self.start_btn.setText("Rendering Mosaic...")
        self.start_btn.setStyleSheet(self.STYLE_BUSY)
        self.progress_bar.setVisible(True)

        is_preview_hud = self.preview_hud_check.isChecked()
        self.worker = MosaicRenderWorker(
            output_dir=output_dir, file_prefix=self.prefix_edit.text().strip() or "fractal_mosaic",
            grid_n=grid_n, target_w=target_w, target_h=target_h, max_iter=self.params['max_iter'],
            canvas_cx=self.params['target_cx'], canvas_cy=self.params['target_cy'], plot_width=self.params['target_w'], cmap_name=self.params['cmap_name'],
            auto_stitch=auto_stitch, delete_tiles_after_stitch=self.delete_tiles_check.isChecked(),
            ssaa_factor=self.get_ssaa_factor(), is_hdr=is_hdr,
            overlay_hud=self.hud_check.isChecked(), preview_hud=is_preview_hud,
            **{k: v for k, v in self.params.items() if k not in ('target_cx', 'target_cy', 'target_w', 'max_iter', 'cmap_name')}
        )
        self.worker.preview_updated.connect(self.on_preview_updated)
        self.worker.progress_updated.connect(lambda cur, tot, rate, avg: (
            self.progress_bar.setValue(int(cur / tot * 100)),
            self.status_lbl.setText(f"Rendered {cur}/{tot} | Rate: {rate:.3f} tiles/s | ETA: {format_eta_time((tot - cur) / max(1e-6, avg))}")
        ))
        self.worker.finished.connect(lambda p, el: (QMessageBox.information(self, "Success", f"Mosaic completed in {format_eta_time(el)}:\n{p}"), self.accept()))
        self.worker.error.connect(lambda err: (
            self.start_btn.setEnabled(True),
            self.start_btn.setText("Export Mosaic Grid"),
            self.start_btn.setStyleSheet(self.STYLE_DEFAULT),
            QMessageBox.critical(self, "Error", err)
        ))
        self.worker.start()


class GeneralMandelbrotMorphImageExportDialog(BaseExportDialog):
    """Exports General Mandelbrot k-parameter morph as a sequence of PNG images."""
    def __init__(self, viewer, target_cx, target_cy, target_w, max_iter, cmap_name, initial_ssaa=2, **kwargs):
        super().__init__(viewer, "General Mandelbrot: k-Parameter Morph Image Sequence", 720, 640)
        self.params = dict(kwargs, target_cx=target_cx, target_cy=target_cy, target_w=target_w, max_iter=max_iter, cmap_name=cmap_name)
        c_locale = QLocale(QLocale.C)

        self.start_kr_spin = QDoubleSpinBox()
        self.start_kr_spin.setLocale(c_locale)
        self.start_kr_spin.setRange(-2.0, 2.0)
        self.start_kr_spin.setDecimals(4)
        self.start_kr_spin.setValue(self.params.get('gen_kr', 0.25))

        self.end_kr_spin = QDoubleSpinBox()
        self.end_kr_spin.setLocale(c_locale)
        self.end_kr_spin.setRange(-2.0, 2.0)
        self.end_kr_spin.setDecimals(4)
        self.end_kr_spin.setValue(0.75)

        self.start_ki_spin = QDoubleSpinBox()
        self.start_ki_spin.setLocale(c_locale)
        self.start_ki_spin.setRange(-2.0, 2.0)
        self.start_ki_spin.setDecimals(4)
        self.start_ki_spin.setValue(self.params.get('gen_ki', 1.0))

        self.end_ki_spin = QDoubleSpinBox()
        self.end_ki_spin.setLocale(c_locale)
        self.end_ki_spin.setRange(-2.0, 2.0)
        self.end_ki_spin.setDecimals(4)
        self.end_ki_spin.setValue(1.0)

        self.power_spin = QSpinBox()
        self.power_spin.setRange(2, 60)
        self.power_spin.setValue(int(self.params.get('gen_n', 3)))

        grid = QGridLayout()
        k_box = QHBoxLayout()
        k_box.addWidget(QLabel("Start:"))
        k_box.addWidget(self.start_kr_spin)
        k_box.addWidget(QLabel("End:"))
        k_box.addWidget(self.end_kr_spin)

        ki_box = QHBoxLayout()
        ki_box.addWidget(QLabel("Start:"))
        ki_box.addWidget(self.start_ki_spin)
        ki_box.addWidget(QLabel("End:"))
        ki_box.addWidget(self.end_ki_spin)

        grid.addWidget(QLabel("k.re Range:"), 0, 0)
        grid.addLayout(k_box, 0, 1)
        grid.addWidget(QLabel("k.im Range:"), 1, 0)
        grid.addLayout(ki_box, 1, 1)
        grid.addWidget(QLabel("Exponent Power (n):"), 2, 0)
        grid.addWidget(self.power_spin, 2, 1)

        self.count_spin = QSpinBox()
        self.count_spin.setRange(2, 10000)
        self.count_spin.setValue(20)

        self.res_combo = self.create_res_combo("1920x1080")
        self.aa_combo = self.create_aa_combo(initial_ssaa)
        self.prefix_edit = QLineEdit("general_mandel_kmorph")
        self.hdr_check = QCheckBox("Export as 16-bit Deep Color PNGs (HDR / Wide-Gamut)")
        self.hud_check = QCheckBox("Overlay Telemetry HUD (Coordinates, Width & Real-World Scale)")
        self.hud_check.setChecked(False)
        self.hud_check.setToolTip("Renders a sleek semi-transparent HUD banner displaying the coordinates, complex plane width, and real-world scale.")

        grid.addWidget(QLabel("Number of Images:"), 3, 0)
        grid.addWidget(self.count_spin, 3, 1)
        grid.addWidget(QLabel("Per-Frame Resolution:"), 4, 0)
        grid.addWidget(self.res_combo, 4, 1)
        grid.addWidget(QLabel("Antialiasing (SSAA):"), 5, 0)
        grid.addWidget(self.aa_combo, 5, 1)
        grid.addWidget(QLabel("Save Folder:"), 6, 0)
        grid.addWidget(self.create_dir_selector(), 6, 1)
        grid.addWidget(QLabel("Filename Prefix:"), 7, 0)
        grid.addWidget(self.prefix_edit, 7, 1)

        self.layout.addLayout(grid)
        self.layout.addWidget(self.hdr_check)
        self.layout.addWidget(self.hud_check)
        p_chk, p_box = self.create_preview_hud_ui()
        self.layout.addWidget(p_chk)
        self.layout.addWidget(p_box)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("Ready to export k-morph image series.")
        self.status_lbl.setStyleSheet("color: #458b00;")
        self.layout.addWidget(self.status_lbl)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Morph Render")
        self.start_btn.setStyleSheet(self.STYLE_DEFAULT)
        self.start_btn.clicked.connect(self.start_export)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(self.STYLE_CANCEL)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(cancel_btn)
        self.layout.addLayout(btn_layout)

    def start_export(self):
        output_dir = self.validate_destination_dir()
        if not output_dir:
            return

        target_w, target_h = map(int, self.res_combo.currentText().split('x'))
        is_hdr = self.hdr_check.isChecked()

        bytes_per_sample = 2 if is_hdr else 1
        single_frame_bytes = target_w * target_h * 3 * bytes_per_sample
        queue_buffer_pool = 3 * single_frame_bytes
        save_overhead_bytes = target_w * target_h * (8 if is_hdr else 3)
        safety_headroom = 256 * 1024 * 1024
        required_ram = queue_buffer_pool + save_overhead_bytes + safety_headroom

        if not check_ram_sufficiency(self, required_ram, "k-Morph Image Sequence"):
            return

        self.start_btn.setEnabled(False)
        self.start_btn.setText("Rendering Morph...")
        self.start_btn.setStyleSheet(self.STYLE_BUSY)
        self.progress_bar.setVisible(True)

        is_preview_hud = self.preview_hud_check.isChecked()
        self.worker = GeneralMandelbrotMorphImageWorker(
            output_dir=output_dir, file_prefix=self.prefix_edit.text().strip() or "general_mandel_kmorph",
            num_images=self.count_spin.value(), target_w=target_w, target_h=target_h, max_iter=self.params['max_iter'],
            cx=self.params['target_cx'], cy=self.params['target_cy'], w=self.params['target_w'],
            start_kr=self.start_kr_spin.value(), end_kr=self.end_kr_spin.value(),
            start_ki=self.start_ki_spin.value(), end_ki=self.end_ki_spin.value(),
            gen_n=self.power_spin.value(),
            cmap_name=self.params['cmap_name'], ssaa_factor=self.get_ssaa_factor(), is_hdr=is_hdr,
            overlay_hud=self.hud_check.isChecked(), preview_hud=is_preview_hud,
            **{k: v for k, v in self.params.items() if k not in ('target_cx', 'target_cy', 'target_w', 'max_iter', 'cmap_name', 'gen_n', 'gen_kr', 'gen_ki')}
        )
        self.worker.preview_updated.connect(self.on_preview_updated)
        self.worker.progress_updated.connect(lambda cur, tot, rate, avg: (
            self.progress_bar.setValue(int(cur / tot * 100)),
            self.status_lbl.setText(f"Rendered {cur}/{tot} | Rate: {rate:.3f} img/s | ETA: {format_eta_time((tot - cur) / max(1e-6, avg))}")
        ))
        self.worker.finished.connect(lambda p, el: (QMessageBox.information(self, "Success", f"Rendered in {format_eta_time(el)}:\n{p}"), self.accept()))
        self.worker.error.connect(lambda err: (
            self.start_btn.setEnabled(True),
            self.start_btn.setText("Start Morph Render"),
            self.start_btn.setStyleSheet(self.STYLE_DEFAULT),
            QMessageBox.critical(self, "Error", err)
        ))
        self.worker.start()


class BaseVideoExportDialog(BaseExportDialog):
    """Shared base UI and execution harness for all video generation modes."""
    def __init__(self, viewer, title, width=760, height=640, default_prefix="fractal_video", initial_ssaa=4):
        super().__init__(viewer, title, width, height)
        self.res_combo = self.create_res_combo("1920x1080")
        self.aa_combo = self.create_aa_combo(initial_ssaa)

        self.encoder_combo = QComboBox()
        self.encoder_combo.addItems(VIDEO_ENCODER_OPTIONS)
        self.encoder_combo.setCurrentText(VIDEO_ENCODER_OPTIONS[0])

        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["24", "25", "30", "50", "60", "100", "120"])
        self.fps_combo.setCurrentText("60")

        self.rate_control_combo = QComboBox()
        self.rate_control_combo.addItems(["Constant Rate Factor (CRF)", "Target Bitrate (kbps)"])
        self.rate_control_combo.setCurrentText("Constant Rate Factor (CRF)")
        self.rate_control_combo.currentTextChanged.connect(self.on_rate_control_changed)

        self.crf_spin = QSpinBox()
        self.crf_spin.setRange(0, 51)
        self.crf_spin.setValue(16)
        self.crf_spin.setToolTip("CRF (0–51): 0 = Lossless, 14–18 = Visually Lossless, 23 = Default, 28+ = High Compression")

        self.bitrate_combo = QComboBox()
        self.bitrate_combo.setEditable(True)
        self.bitrate_combo.addItems(["5000", "10000", "20000", "50000", "100000"])
        self.bitrate_combo.setCurrentText("20000")
        self.bitrate_combo.setVisible(False)

        self.len_spin = QSpinBox()
        self.len_spin.setRange(1, 86400)
        self.len_spin.setValue(10)

        self.cycle_speed_spin = QDoubleSpinBox()
        self.cycle_speed_spin.setRange(-10.0, 10.0)
        self.cycle_speed_spin.setSingleStep(0.05)
        self.cycle_speed_spin.setDecimals(2)
        self.cycle_speed_spin.setValue(0.0)
        self.cycle_speed_spin.setToolTip("Dynamic color cycle rotation speed (revolutions per second). 0.0 = static palette.")

        self.prefix_edit = QLineEdit(default_prefix)

        self.hdr_check = QCheckBox("Enable HDR Video (10-bit BT.2020 / HEVC)")
        self.hdr_combo = QComboBox()
        self.hdr_combo.addItems(["HDR10 (PQ / SMPTE ST 2084)", "HLG (ARIB STD-B67)"])
        self.hdr_combo.setEnabled(False)
        self.hdr_check.toggled.connect(self.hdr_combo.setEnabled)

        self.hud_check = QCheckBox("Overlay Telemetry HUD (Display Width & Physical Scale)")
        self.hud_check.setChecked(False)
        self.hud_check.setToolTip("Renders a sleek semi-transparent HUD banner displaying the complex plane width and real-world scale in the status bar format.")

        self.create_preview_hud_ui()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self.status_lbl = QLabel("Ready to export video.")
        self.status_lbl.setStyleSheet("color: #458b00;")

        self.start_btn = QPushButton("Start Rendering MP4")
        self.start_btn.setStyleSheet(self.STYLE_DEFAULT)
        self.start_btn.clicked.connect(self.start_render)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(self.STYLE_CANCEL)
        self.cancel_btn.clicked.connect(self.reject)

    def on_rate_control_changed(self, text):
        is_crf = ("CRF" in text)
        if hasattr(self, 'lbl_rate_val'):
            self.lbl_rate_val.setText("CRF Value (0–51):" if is_crf else "Bitrate (kbps):")
        self.crf_spin.setVisible(is_crf)
        self.bitrate_combo.setVisible(not is_crf)

    def append_standard_video_controls(self, grid, start_row=0):
        grid.addWidget(QLabel("Resolution:"), start_row, 0)
        grid.addWidget(self.res_combo, start_row, 1)
        grid.addWidget(QLabel("Antialiasing (SSAA):"), start_row + 1, 0)
        grid.addWidget(self.aa_combo, start_row + 1, 1)
        grid.addWidget(QLabel("Encoder:"), start_row + 2, 0)
        grid.addWidget(self.encoder_combo, start_row + 2, 1)
        grid.addWidget(QLabel("Framerate (FPS):"), start_row + 3, 0)
        grid.addWidget(self.fps_combo, start_row + 3, 1)

        grid.addWidget(QLabel("Rate Control:"), start_row + 4, 0)
        grid.addWidget(self.rate_control_combo, start_row + 4, 1)

        self.lbl_rate_val = QLabel("CRF Value (0–51):" if "CRF" in self.rate_control_combo.currentText() else "Bitrate (kbps):")
        grid.addWidget(self.lbl_rate_val, start_row + 5, 0)

        rate_val_container = QWidget()
        rate_val_layout = QHBoxLayout(rate_val_container)
        rate_val_layout.setContentsMargins(0, 0, 0, 0)
        rate_val_layout.addWidget(self.crf_spin)
        rate_val_layout.addWidget(self.bitrate_combo)
        grid.addWidget(rate_val_container, start_row + 5, 1)

        grid.addWidget(QLabel("Length (seconds):"), start_row + 6, 0)
        grid.addWidget(self.len_spin, start_row + 6, 1)
        grid.addWidget(QLabel("Palette Cycling (rev/s):"), start_row + 7, 0)
        grid.addWidget(self.cycle_speed_spin, start_row + 7, 1)
        grid.addWidget(QLabel("Save Folder:"), start_row + 8, 0)
        grid.addWidget(self.create_dir_selector(), start_row + 8, 1)
        grid.addWidget(QLabel("Filename Prefix:"), start_row + 9, 0)
        grid.addWidget(self.prefix_edit, start_row + 9, 1)

    def append_standard_bottom_layout(self):
        hdr_layout = QHBoxLayout()
        hdr_layout.addWidget(self.hdr_check)
        hdr_layout.addWidget(self.hdr_combo)
        self.layout.addLayout(hdr_layout)
        self.layout.addWidget(self.hud_check)
        self.layout.addWidget(self.preview_hud_check)
        self.layout.addWidget(self.preview_container)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.status_lbl)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.cancel_btn)
        self.layout.addLayout(btn_layout)

    def launch_video_worker(self, worker_cls, worker_kwargs):
        output_dir = self.validate_destination_dir()
        if not output_dir:
            return

        target_w, target_h = map(int, self.res_combo.currentText().split('x'))
        ssaa_factor = self.get_ssaa_factor()
        is_hdr = self.hdr_check.isChecked()
        ext = ".webm" if "VP9" in self.encoder_combo.currentText() else ".mp4"
        out_path = os.path.join(output_dir, f"{self.prefix_edit.text().strip()}_{self.res_combo.currentText()}_{ssaa_factor}xSSAA{'_HDR' if is_hdr else ''}{ext}")

        self.start_btn.setEnabled(False)
        self.start_btn.setText("Rendering Video...")
        self.start_btn.setStyleSheet(self.STYLE_BUSY)
        self.progress_bar.setVisible(True)

        rate_ctrl = "crf" if "CRF" in self.rate_control_combo.currentText() else "bitrate"
        try:
            bitrate_val = int(self.bitrate_combo.currentText().replace('k', '').replace('K', '').strip())
        except ValueError:
            bitrate_val = 20000

        is_preview_hud = self.preview_hud_check.isChecked()
        self.worker = worker_cls(
            output_path=out_path, target_w=target_w, target_h=target_h,
            fps=int(self.fps_combo.currentText()),
            rate_control=rate_ctrl,
            crf=self.crf_spin.value(),
            bitrate_kbps=bitrate_val,
            duration_sec=self.len_spin.value(),
            palette_cycle_speed=self.cycle_speed_spin.value(),
            ssaa_factor=ssaa_factor, is_hdr=is_hdr,
            hdr_mode=self.hdr_combo.currentText(),
            encoder_name=self.encoder_combo.currentText(),
            overlay_hud=self.hud_check.isChecked(),
            preview_hud=is_preview_hud,
            **worker_kwargs
        )
        self.worker.preview_updated.connect(self.on_preview_updated)
        self.worker.progress_updated.connect(lambda cur, tot, rate, avg: (
            self.progress_bar.setValue(int(cur / tot * 100)),
            self.status_lbl.setText(f"Frame {cur}/{tot} | FPS: {rate:.3f} (R. Avg: {avg:.3f}) | ETA: {format_eta_time((tot - cur) / max(1e-6, avg))}")
        ))
        self.worker.finished.connect(lambda p, el: (QMessageBox.information(self, "Success", f"Video exported in {format_eta_time(el)}:\n{p}"), self.accept()))
        self.worker.error.connect(lambda err: (
            self.start_btn.setEnabled(True),
            self.start_btn.setText("Start Rendering MP4"),
            self.start_btn.setStyleSheet(self.STYLE_DEFAULT),
            QMessageBox.critical(self, "Error", err)
        ))
        self.worker.start()

    def start_render(self):
        raise NotImplementedError


class UnifiedVideoExportDialog(BaseVideoExportDialog):
    """Exports smooth MP4 video zoom-in or multi-waypoint camera path with scale-space arc-length speed."""
    def __init__(self, viewer, waypoints=None, initial_ssaa=4, **kwargs):
        super().__init__(viewer, "Export Zoom & Pan MP4 Video", 760, 720, default_prefix="fractal_video", initial_ssaa=initial_ssaa)
        self.params = kwargs

        self.layout.addWidget(QLabel("📍 Waypoints & Camera Path (Optional; Double-Click on Canvas to Add):"))
        self.wp_list = QListWidget()
        self.wp_list.setFixedHeight(120)
        self.layout.addWidget(self.wp_list)

        wp_btn_layout = QHBoxLayout()
        add_btn = QPushButton("+ Add Current View")
        add_btn.setStyleSheet(self.STYLE_ADD)
        add_btn.clicked.connect(self.on_add_current_view)
        del_btn = QPushButton("Delete Selected")
        del_btn.setStyleSheet(self.STYLE_DELETE)
        del_btn.clicked.connect(self.on_delete_selected)
        clr_btn = QPushButton("Clear All")
        clr_btn.setStyleSheet(self.STYLE_CLEAR)
        clr_btn.clicked.connect(self.on_clear_waypoints)
        wp_btn_layout.addWidget(add_btn)
        wp_btn_layout.addWidget(del_btn)
        wp_btn_layout.addWidget(clr_btn)
        wp_btn_layout.addStretch()
        self.layout.addLayout(wp_btn_layout)

        grid = QGridLayout()
        self.append_standard_video_controls(grid, start_row=0)
        self.layout.addLayout(grid)
        self.append_standard_bottom_layout()

        self.len_spin.setValue(self.calc_default_duration())

        if self.viewer and hasattr(self.viewer, 'canvas'):
            self.viewer.canvas.keyframe_recorded.connect(self.sync_waypoints_ui)
        self.sync_waypoints_ui()

    def calc_default_duration(self):
        if self.viewer and hasattr(self.viewer, 'waypoints') and len(self.viewer.waypoints) >= 2:
            return max(15, len(self.viewer.waypoints) * 8)
        target_w = Decimal(str(self.params.get('target_w', '4.0')))
        default_start_w = FRACTAL_DEFAULTS.get(self.params.get('fractal_type', 0), (0, 0, Decimal("4.0")))[2]
        if target_w > Decimal("0") and default_start_w > target_w:
            return max(5, int(0.6*round(float((default_start_w.ln() - target_w.ln()) / Decimal("10").ln()) * 4.0)))
        return 10

    def sync_waypoints_ui(self):
        self.wp_list.clear()
        if not self.viewer or not hasattr(self.viewer, 'waypoints'):
            return
        for idx, (cx, cy, w) in enumerate(self.viewer.waypoints, 1):
            self.wp_list.addItem(f"Point #{idx:02d} | Center: ({float(cx):.6e}, {float(cy):.6e}) | Width: {Decimal(str(w)):.3e}")

    def add_waypoint(self, cx, cy, w):
        if not hasattr(self.viewer, 'waypoints'):
            self.viewer.waypoints = []
        self.viewer.waypoints.append((Decimal(str(cx)), Decimal(str(cy)), Decimal(str(w))))
        self.sync_waypoints_ui()
        if hasattr(self.viewer, 'canvas'):
            self.viewer.canvas.update()

    def on_add_current_view(self):
        self.add_waypoint(self.viewer.canvas.center_x, self.viewer.canvas.center_y, self.viewer.canvas.plot_width)

    def on_delete_selected(self):
        if not hasattr(self.viewer, 'waypoints'):
            return
        row = self.wp_list.currentRow()
        if 0 <= row < len(self.viewer.waypoints):
            del self.viewer.waypoints[row]
            self.sync_waypoints_ui()
            if hasattr(self.viewer, 'canvas'):
                self.viewer.canvas.update()

    def on_clear_waypoints(self):
        if hasattr(self.viewer, 'waypoints'):
            self.viewer.waypoints.clear()
        self.sync_waypoints_ui()
        if hasattr(self.viewer, 'canvas'):
            self.viewer.canvas.update()

    def start_render(self):
        user_wps = getattr(self.viewer, 'waypoints', [])
        if len(user_wps) >= 2:
            waypoints = list(user_wps)
        elif len(user_wps) == 1:
            default_start_w = FRACTAL_DEFAULTS.get(self.params.get('fractal_type', 0), (0, 0, Decimal("4.0")))[2]
            waypoints = [(user_wps[0][0], user_wps[0][1], default_start_w), user_wps[0]]
        else:
            default_start_w = FRACTAL_DEFAULTS.get(self.params.get('fractal_type', 0), (0, 0, Decimal("4.0")))[2]
            waypoints = [
                (Decimal(str(self.params['target_cx'])), Decimal(str(self.params['target_cy'])), default_start_w),
                (Decimal(str(self.params['target_cx'])), Decimal(str(self.params['target_cy'])), Decimal(str(self.params['target_w'])))
            ]

        worker_kwargs = dict(
            waypoints=waypoints,
            max_iter=self.params['max_iter'],
            cmap_name=self.params['cmap_name'],
            **{k: v for k, v in self.params.items() if k not in ('target_cx', 'target_cy', 'target_w', 'max_iter', 'cmap_name')}
        )
        self.launch_video_worker(UnifiedVideoRenderWorker, worker_kwargs)

    def done(self, r):
        if self.viewer and hasattr(self.viewer, 'canvas'):
            try:
                self.viewer.canvas.keyframe_recorded.disconnect(self.sync_waypoints_ui)
            except Exception:
                pass
        super().done(r)


class GeneralMandelbrotMorphExportDialog(BaseVideoExportDialog):
    def __init__(self, viewer, target_cx, target_cy, target_w, max_iter, cmap_name, initial_ssaa=2, **kwargs):
        super().__init__(viewer, "General Mandelbrot: k-Parameter Morph Video Generator", 760, 680, default_prefix="general_mandel_kmorph", initial_ssaa=initial_ssaa)
        self.params = dict(kwargs, target_cx=target_cx, target_cy=target_cy, target_w=target_w, max_iter=max_iter, cmap_name=cmap_name)
        c_locale = QLocale(QLocale.C)

        self.start_kr_spin = QDoubleSpinBox()
        self.start_kr_spin.setLocale(c_locale)
        self.start_kr_spin.setRange(-2.0, 2.0)
        self.start_kr_spin.setDecimals(4)
        self.start_kr_spin.setValue(self.params.get('gen_kr', 0.25))

        self.end_kr_spin = QDoubleSpinBox()
        self.end_kr_spin.setLocale(c_locale)
        self.end_kr_spin.setRange(-2.0, 2.0)
        self.end_kr_spin.setDecimals(4)
        self.end_kr_spin.setValue(0.75)

        self.start_ki_spin = QDoubleSpinBox()
        self.start_ki_spin.setLocale(c_locale)
        self.start_ki_spin.setRange(-2.0, 2.0)
        self.start_ki_spin.setDecimals(4)
        self.start_ki_spin.setValue(self.params.get('gen_ki', 1.0))

        self.end_ki_spin = QDoubleSpinBox()
        self.end_ki_spin.setLocale(c_locale)
        self.end_ki_spin.setRange(-2.0, 2.0)
        self.end_ki_spin.setDecimals(4)
        self.end_ki_spin.setValue(1.0)

        self.power_spin = QSpinBox()
        self.power_spin.setRange(2, 60)
        self.power_spin.setValue(int(self.params.get('gen_n', 3)))

        grid = QGridLayout()
        k_box = QHBoxLayout()
        k_box.addWidget(QLabel("Start:"))
        k_box.addWidget(self.start_kr_spin)
        k_box.addWidget(QLabel("End:"))
        k_box.addWidget(self.end_kr_spin)

        ki_box = QHBoxLayout()
        ki_box.addWidget(QLabel("Start:"))
        ki_box.addWidget(self.start_ki_spin)
        ki_box.addWidget(QLabel("End:"))
        ki_box.addWidget(self.end_ki_spin)

        grid.addWidget(QLabel("k.re Range:"), 0, 0)
        grid.addLayout(k_box, 0, 1)
        grid.addWidget(QLabel("k.im Range:"), 1, 0)
        grid.addLayout(ki_box, 1, 1)
        grid.addWidget(QLabel("Exponent Power (n):"), 2, 0)
        grid.addWidget(self.power_spin, 2, 1)

        self.append_standard_video_controls(grid, start_row=3)
        self.layout.addLayout(grid)
        self.append_standard_bottom_layout()

    def start_render(self):
        worker_kwargs = dict(
            cx=self.params['target_cx'],
            cy=self.params['target_cy'],
            w=self.params['target_w'],
            start_kr=self.start_kr_spin.value(),
            end_kr=self.end_kr_spin.value(),
            start_ki=self.start_ki_spin.value(),
            end_ki=self.end_ki_spin.value(),
            gen_n=self.power_spin.value(),
            max_iter=self.params['max_iter'],
            cmap_name=self.params['cmap_name'],
            **{k: v for k, v in self.params.items() if k not in ('target_cx', 'target_cy', 'target_w', 'max_iter', 'cmap_name', 'gen_n', 'gen_kr', 'gen_ki')}
        )
        self.launch_video_worker(GeneralMandelbrotMorphVideoWorker, worker_kwargs)


class IterationRevealExportDialog(BaseVideoExportDialog):
    """Generates timelapse video sweeping Max Iterations from start_iter to end_iter."""
    def __init__(self, viewer, target_cx, target_cy, target_w, max_iter, cmap_name, initial_ssaa=2, **kwargs):
        super().__init__(viewer, "Iteration Reveal Timelapse Generator", 760, 680, default_prefix="iter_reveal_timelapse", initial_ssaa=initial_ssaa)
        self.params = dict(kwargs, target_cx=target_cx, target_cy=target_cy, target_w=target_w, max_iter=max_iter, cmap_name=cmap_name)

        self.start_iter_spin = QSpinBox()
        self.start_iter_spin.setRange(1, 1000000)
        self.start_iter_spin.setValue(10)

        self.end_iter_spin = QSpinBox()
        self.end_iter_spin.setRange(1, 1000000)
        self.end_iter_spin.setValue(1000)

        self.sweep_curve_combo = QComboBox()
        self.sweep_curve_combo.addItems([
            "Logarithmic / Exponential (Smooth Revelation across Decades)",
            "Quadratic (t^2)",
            "Cubic (t^3)",
            "Linear (Constant Iteration Increment)"
        ])
        self.sweep_curve_combo.setCurrentIndex(0)

        grid = QGridLayout()
        iter_box = QHBoxLayout()
        iter_box.addWidget(QLabel("Start:"))
        iter_box.addWidget(self.start_iter_spin)
        iter_box.addWidget(QLabel("End:"))
        iter_box.addWidget(self.end_iter_spin)

        grid.addWidget(QLabel("Iteration Range:"), 0, 0)
        grid.addLayout(iter_box, 0, 1)
        grid.addWidget(QLabel("Progression Curve:"), 1, 0)
        grid.addWidget(self.sweep_curve_combo, 1, 1)

        self.append_standard_video_controls(grid, start_row=2)
        self.layout.addLayout(grid)
        self.append_standard_bottom_layout()

    def start_render(self):
        worker_kwargs = dict(
            cx=self.params['target_cx'],
            cy=self.params['target_cy'],
            w=self.params['target_w'],
            start_iter=self.start_iter_spin.value(),
            end_iter=self.end_iter_spin.value(),
            sweep_curve=self.sweep_curve_combo.currentText(),
            max_iter=self.end_iter_spin.value(),
            cmap_name=self.params['cmap_name'],
            **{k: v for k, v in self.params.items() if k not in ('target_cx', 'target_cy', 'target_w', 'max_iter', 'cmap_name')}
        )
        self.launch_video_worker(IterationRevealVideoWorker, worker_kwargs)