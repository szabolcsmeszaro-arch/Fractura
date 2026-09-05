from decimal import Decimal
from PyQt5.QtCore import QEvent, QLocale, Qt, QTimer
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QAbstractItemView, QAbstractSpinBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QSpinBox, QStatusBar, QTextEdit, QVBoxLayout, QWidget,
    qApp
)

from color_engine import CATEGORIZED_CMAPS
from config import (
    BBSA_OPTIONS, BUILTIN_PRESETS, COLOR_SCHEMES,
    DEFAULT_JULIA_MANDELBROT_PARAMS, DEFAULT_JULIA_BURNING_SHIP_PARAMS,
    DEFAULT_JULIA_VIEW, DEFAULT_MAX_ITER, DYNAMIC_ITER_MODES, FRACTAL_DEFAULTS,
    FRACTAL_NAMES, GLITCH_CORRECTION_MODES, GLOBAL_MAX_ITER,
    MAX_SYSTEM_CORES, PRECISION_OPTIONS, SCRIPT_DIR, get_active_engine_name,
    get_real_world_scale, parse_real_world_scale
)
from math_core.cuda_kernels import HAS_CUDA
from math_core.engine import clear_scratch_buffer_pool
from math_core.high_precision import clear_reference_orbit_cache
from ui.canvas import FractalCanvas, JuliaPreviewWidget
from ui.dialogs_export import (
    GeneralMandelbrotMorphExportDialog, GeneralMandelbrotMorphImageExportDialog,
    ImageExportDialog, ImageSequenceExportDialog, IterationRevealExportDialog,
    MosaicExportDialog, UnifiedVideoExportDialog
)
from ui.dialogs_presets import BookmarkPresetsDialog


class MandelbrotViewer(QMainWindow):
    STYLE_HUD_OFF = """
        QPushButton { background-color: #374151; color: white; font-weight: bold; padding: 8px 14px; border-radius: 4px; border: 1px solid #4b5563; }
        QPushButton:hover { background-color: #4b5563; border-color: #6b7280; }
        QPushButton:pressed { background-color: #1f2937; }
        QPushButton:disabled { background-color: #2d3748; color: #718096; border-color: #2d3748; }
    """
    STYLE_HUD_ON = """
        QPushButton { background-color: #7c3aed; color: white; font-weight: bold; padding: 8px 14px; border-radius: 4px; border: 1px solid #a78bfa; }
        QPushButton:hover { background-color: #8b5cf6; border-color: #c4b5fd; }
        QPushButton:pressed { background-color: #6d28d9; }
        QPushButton:disabled { background-color: #2d3748; color: #718096; border-color: #2d3748; }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Apeiron GPU/CPU Fractal Explorer")
        self.last_export_dir = SCRIPT_DIR
        self.bookmarks = [dict(p, is_builtin=True) for p in BUILTIN_PRESETS]
        self.waypoints = []
        self.dialogs = {}

        self.canvas = FractalCanvas(self)
        self.julia_preview = JuliaPreviewWidget(self)
        self.julia_preview.setVisible(False)

        self.palette_timer = QTimer(self)
        self.palette_timer.setInterval(16)
        self.palette_timer.timeout.connect(self._on_palette_timer_tick)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        control_bar = QWidget()
        control_layout = QVBoxLayout(control_bar)
        control_layout.setContentsMargins(10, 8, 10, 8)
        control_layout.setSpacing(8)

        # Row 1: Operations, Device & Cores
        r1 = QHBoxLayout()
        r1.setSpacing(10)
        bm_btn = QPushButton("Presets")
        bm_btn.setStyleSheet("""
            QPushButton { background-color: #675B58; color: white; font-weight: bold; padding: 8px 14px; border-radius: 4px; }
            QPushButton:hover { background-color: #7d706d; }
            QPushButton:pressed { background-color: #524744; }
            QPushButton:disabled { background-color: #2d3748; color: #718096; }
        """)
        bm_btn.clicked.connect(lambda: self.open_dialog('bookmark', BookmarkPresetsDialog, self))
        r1.addWidget(bm_btn)

        buttons_def = [
            ("Save PNG", lambda: self.open_export_dialog('image', ImageExportDialog, initial_ssaa=4)),
            ("Image Zoom", lambda: self.open_export_dialog('image_seq', ImageSequenceExportDialog, initial_ssaa=4)),
            ("Image Mosaic", lambda: self.open_export_dialog('mosaic', MosaicExportDialog, initial_ssaa=4)),
            ("Video Zoom", lambda: self.open_export_dialog('video', UnifiedVideoExportDialog, initial_ssaa=4)),
            ("k-Morph Image", lambda: self.open_export_dialog('k_morph_img', GeneralMandelbrotMorphImageExportDialog, initial_ssaa=4)),
            ("k-Morph Video", lambda: self.open_export_dialog('k_morph', GeneralMandelbrotMorphExportDialog, initial_ssaa=4)),
            ("Iter Reveal", lambda: self.open_export_dialog('iter_reveal', IterationRevealExportDialog, initial_ssaa=4))
        ]
        for name, callback in buttons_def:
            btn = QPushButton(name)
            btn.setStyleSheet("""
                QPushButton { background-color: #227442; color: white; font-weight: bold; padding: 8px 14px; border-radius: 4px; }
                QPushButton:hover { background-color: #2e9e5b; }
                QPushButton:pressed { background-color: #1b5e35; }
                QPushButton:disabled { background-color: #2d3748; color: #718096; }
            """)
            btn.clicked.connect(callback)
            r1.addWidget(btn)

        self.preview_toggle_btn = QPushButton("Julia HUD")
        self.preview_toggle_btn.setStyleSheet(self.STYLE_HUD_OFF)
        self.preview_toggle_btn.clicked.connect(self.toggle_julia_preview)
        r1.addWidget(self.preview_toggle_btn)

        self.waypoints_toggle_btn = QPushButton("Waypoints: OFF")
        self.waypoints_toggle_btn.setStyleSheet(self.STYLE_HUD_OFF)
        self.waypoints_toggle_btn.setToolTip("Toggle visibility of waypoint beacon pins and spline flight trajectory curves (Shortcut: H)")
        self.waypoints_toggle_btn.clicked.connect(self.toggle_waypoints_overlay)
        r1.addWidget(self.waypoints_toggle_btn)

        r1.addStretch()
        control_layout.addLayout(r1)

        # Row 2: Fractal Type, Iterations, SSAA, Palettes, Precision
        r2 = QHBoxLayout()
        r2.setSpacing(12)

        lbl_fractal = QLabel("Fractal:")
        lbl_fractal.setStyleSheet("color: #000000; font-weight: 900")
        r2.addWidget(lbl_fractal)

        self.fractal_combo = QComboBox()
        self.fractal_combo.setFixedWidth(240)
        self.fractal_combo.addItems(FRACTAL_NAMES)
        self.fractal_combo.currentIndexChanged.connect(self.on_fractal_type_change)

        self.fractal_combo.setStyleSheet("""
            QComboBox {background-color: #DA8A67; color: #ffffff; font-weight: bold; border: 1px solid #4a5568; border-radius: 4px; padding: 4px 8px; }
            QComboBox:hover { border: 1px solid #60a5fa; }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView {background-color: #DA8A67; color: #ffffff; selection-background-color: #3b82f6; selection-color: #ffffff; } """)

        r2.addWidget(self.fractal_combo)

        c_locale = QLocale(QLocale.C)

        r2.addWidget(QLabel("Max Iter:"))
        self.iter_spin = QSpinBox()
        self.iter_spin.setMinimumWidth(120)
        self.iter_spin.setRange(10, GLOBAL_MAX_ITER)
        self.iter_spin.setValue(DEFAULT_MAX_ITER)
        self.iter_spin.setSingleStep(250)
        self.iter_spin.valueChanged.connect(lambda v: (setattr(self.canvas, 'max_iter', v), self.canvas.render_fractal()))
        r2.addWidget(self.iter_spin)

        r2.addWidget(QLabel("AA:"))
        self.aa_combo = QComboBox()
        self.aa_combo.addItems(["1x (Off / Fastest)", "2x (SSAA)", "4x (SSAA)", "6x (SSAA)", "8x (SSAA / Slowest)"])
        self.aa_combo.currentIndexChanged.connect(lambda i: (setattr(self.canvas, 'ssaa_factor', [1, 2, 4, 6, 8][i]), self.canvas.render_fractal()))
        r2.addWidget(self.aa_combo)

        r2.addWidget(QLabel("AA Edge Tol:"))
        self.edge_spin = QDoubleSpinBox()
        self.edge_spin.setLocale(c_locale)
        self.edge_spin.setRange(0.001, 2.0)
        self.edge_spin.setSingleStep(0.02)
        self.edge_spin.setDecimals(3)
        self.edge_spin.setValue(self.canvas.edge_threshold)
        self.edge_spin.valueChanged.connect(lambda v: (setattr(self.canvas, 'edge_threshold', float(v)), self.canvas.render_fractal()))
        r2.addWidget(self.edge_spin)

        r2.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["GPU", "CPU"] if HAS_CUDA else ["CPU"])
        self.device_combo.setCurrentText("GPU" if HAS_CUDA else "CPU")
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        r2.addWidget(self.device_combo)

        self.lbl_cores = QLabel("Cores:")
        self.cores_spin = QSpinBox()
        self.cores_spin.setRange(1, MAX_SYSTEM_CORES)
        self.cores_spin.setValue(min(2, MAX_SYSTEM_CORES))
        self.cores_spin.valueChanged.connect(self.on_cpu_cores_changed)
        self.lbl_cores.setEnabled(not HAS_CUDA)
        self.cores_spin.setEnabled(not HAS_CUDA)
        r2.addWidget(self.lbl_cores)
        r2.addWidget(self.cores_spin)

        r2.addWidget(QLabel("Canvas:"))
        self.canvas_size_combo = QComboBox()
        self.canvas_size_combo.addItems(["480x270", "640x360", "960x540", "1280x720"])
        self.canvas_size_combo.setCurrentText("1280x720")
        self.canvas_size_combo.currentTextChanged.connect(self.on_canvas_size_changed)
        r2.addWidget(self.canvas_size_combo)
        r2.addStretch()
        control_layout.addLayout(r2)

        # Row 3: Min Width, BBSA, Glitch Correction, Dynamic Iter, Zoom Factor
        r3 = QHBoxLayout()
        r3.setSpacing(12)

        r3.addWidget(QLabel("Min Width:"))
        self.prec_combo = QComboBox()
        self.prec_combo.addItems(["1e-12 (FP64)", "1e-24 (DD)", "1e-300 (Perturbation)", "1e-10000+ (Floatexp)"])
        self.prec_combo.setCurrentText("1e-300 (Perturbation)")
        self.prec_combo.currentTextChanged.connect(self.on_prec_changed_combo)
        r3.addWidget(self.prec_combo)

        r3.addWidget(QLabel("BBSA:"))
        self.bbsa_combo = QComboBox()
        self.bbsa_combo.addItems(list(BBSA_OPTIONS.keys()))
        self.bbsa_combo.setCurrentText("4th-order")
        self.bbsa_combo.currentTextChanged.connect(self.on_bbsa_mode_changed)
        r3.addWidget(self.bbsa_combo)

        r3.addWidget(QLabel("Glitch Corr:"))
        self.glitch_combo = QComboBox()
        self.glitch_combo.addItems(GLITCH_CORRECTION_MODES)
        self.glitch_combo.setCurrentText(self.canvas.glitch_mode)
        self.glitch_combo.currentTextChanged.connect(self.on_glitch_mode_changed)
        r3.addWidget(self.glitch_combo)

        r3.addWidget(QLabel("Dynamic Iter:"))
        self.dynamic_iter_combo = QComboBox()
        self.dynamic_iter_combo.addItems(DYNAMIC_ITER_MODES)
        self.dynamic_iter_combo.setCurrentText(self.canvas.dynamic_iter_mode)
        self.dynamic_iter_combo.currentTextChanged.connect(self.on_dynamic_iter_changed)
        r3.addWidget(self.dynamic_iter_combo)

        r3.addWidget(QLabel("Zoom Factor:"))
        self.zoom_factor_edit = QLineEdit("5")
        self.zoom_factor_edit.setFixedWidth(55)
        self.zoom_factor_edit.setAlignment(Qt.AlignCenter)
        zf_validator = QDoubleValidator(1.0, 1000.0, 4, self)
        zf_validator.setNotation(QDoubleValidator.StandardNotation)
        zf_validator.setLocale(c_locale)
        self.zoom_factor_edit.setValidator(zf_validator)
        self.zoom_factor_edit.setToolTip("Zoom Factor: Multiplier for Zoom In/Out buttons and Right-Click zoom (1 to 1000). Default is 5.")
        self.zoom_factor_edit.textChanged.connect(self.on_zoom_factor_changed)
        self.zoom_factor_edit.editingFinished.connect(self.on_zoom_factor_editing_finished)
        self.zoom_factor_edit.returnPressed.connect(self.on_zoom_factor_return_pressed)
        r3.addWidget(self.zoom_factor_edit)

        r3.addWidget(QLabel("Wheel Zoom:"))
        self.wheel_zoom_edit = QLineEdit("1.25")
        self.wheel_zoom_edit.setFixedWidth(55)
        self.wheel_zoom_edit.setAlignment(Qt.AlignCenter)
        wz_validator = QDoubleValidator(1.001, 1000.0, 4, self)
        wz_validator.setNotation(QDoubleValidator.StandardNotation)
        wz_validator.setLocale(c_locale)
        self.wheel_zoom_edit.setValidator(wz_validator)
        self.wheel_zoom_edit.setToolTip("Wheel Zoom: Multiplier for mouse wheel zoom in / out (1.001 to 1000). Default is 1.25.")
        self.wheel_zoom_edit.textChanged.connect(self.on_wheel_zoom_changed)
        self.wheel_zoom_edit.editingFinished.connect(self.on_wheel_zoom_editing_finished)
        self.wheel_zoom_edit.returnPressed.connect(self.on_wheel_zoom_return_pressed)
        r3.addWidget(self.wheel_zoom_edit)

        r3.addStretch()
        control_layout.addLayout(r3)

        # Row 4: Viewport Coordinates & Controls
        r4 = QHBoxLayout()
        r4.setSpacing(12)
        self.cx_edit = QLineEdit(str(self.canvas.center_x))
        self.cx_edit.setFixedWidth(140)
        self.cy_edit = QLineEdit(str(self.canvas.center_y))
        self.cy_edit.setFixedWidth(140)
        self.w_edit = QLineEdit(str(self.canvas.plot_width))
        self.w_edit.setFixedWidth(95)
        self.scale_edit = QLineEdit(get_real_world_scale(self.canvas.plot_width))
        self.scale_edit.setFixedWidth(95)
        self.scale_edit.setToolTip("Real-world scale (e.g. '30 cm', '500 km', '1.5 AU', '2.3 ly', '5.2 MW', '1.0 U'). Press Enter or click Set.")
        self.last_width_source = 'width'
        self.w_edit.textEdited.connect(lambda: setattr(self, 'last_width_source', 'width'))
        self.scale_edit.textEdited.connect(lambda: setattr(self, 'last_width_source', 'scale'))

        self.lbl_jx, self.lbl_jy = QLabel("Julia cX:"), QLabel("Julia cY:")
        self.julia_cx_spin = QDoubleSpinBox()
        self.julia_cx_spin.setFixedWidth(90)
        self.julia_cx_spin.setLocale(c_locale)
        self.julia_cx_spin.setRange(-4.0, 4.0)
        self.julia_cx_spin.setSingleStep(0.001)
        self.julia_cx_spin.setDecimals(4)
        self.julia_cx_spin.setValue(float(self.canvas.julia_cx))
        self.julia_cx_spin.valueChanged.connect(self.on_julia_params_changed)

        self.julia_cy_spin = QDoubleSpinBox()
        self.julia_cy_spin.setFixedWidth(90)
        self.julia_cy_spin.setLocale(c_locale)
        self.julia_cy_spin.setRange(-4.0, 4.0)
        self.julia_cy_spin.setSingleStep(0.001)
        self.julia_cy_spin.setDecimals(4)
        self.julia_cy_spin.setValue(float(self.canvas.julia_cy))
        self.julia_cy_spin.valueChanged.connect(self.on_julia_params_changed)

        self.lbl_gn = QLabel("n:")
        self.gen_n_spin = QSpinBox()
        self.gen_n_spin.setFixedWidth(80)
        self.gen_n_spin.setRange(2, 60)
        self.gen_n_spin.setValue(self.canvas.gen_n)
        self.gen_n_spin.valueChanged.connect(self.on_gen_params_changed)

        self.lbl_gkr = QLabel("k Re:")
        self.gen_kr_spin = QDoubleSpinBox()
        self.gen_kr_spin.setFixedWidth(90)
        self.gen_kr_spin.setLocale(c_locale)
        self.gen_kr_spin.setRange(-2.0, 2.0)
        self.gen_kr_spin.setSingleStep(0.001)
        self.gen_kr_spin.setDecimals(3)
        self.gen_kr_spin.setValue(self.canvas.gen_kr)
        self.gen_kr_spin.valueChanged.connect(self.on_gen_params_changed)

        self.lbl_gki = QLabel("k Im:")
        self.gen_ki_spin = QDoubleSpinBox()
        self.gen_ki_spin.setFixedWidth(90)
        self.gen_ki_spin.setLocale(c_locale)
        self.gen_ki_spin.setRange(-2.0, 2.0)
        self.gen_ki_spin.setSingleStep(0.001)
        self.gen_ki_spin.setDecimals(3)
        self.gen_ki_spin.setValue(self.canvas.gen_ki)
        self.gen_ki_spin.valueChanged.connect(self.on_gen_params_changed)

        for w in [self.cx_edit, self.cy_edit, self.w_edit, self.scale_edit]:
            w.returnPressed.connect(self.on_apply_coords)

        r4.addWidget(QLabel("X:"))
        r4.addWidget(self.cx_edit)
        r4.addWidget(QLabel("Y:"))
        r4.addWidget(self.cy_edit)
        r4.addWidget(QLabel("Width:"))
        r4.addWidget(self.w_edit)
        r4.addWidget(QLabel("Scale:"))
        r4.addWidget(self.scale_edit)

        r4.addWidget(self.lbl_jx)
        r4.addWidget(self.julia_cx_spin)
        r4.addWidget(self.lbl_jy)
        r4.addWidget(self.julia_cy_spin)

        r4.addWidget(self.lbl_gn)
        r4.addWidget(self.gen_n_spin)
        r4.addWidget(self.lbl_gkr)
        r4.addWidget(self.gen_kr_spin)
        r4.addWidget(self.lbl_gki)
        r4.addWidget(self.gen_ki_spin)

        self.update_dynamic_control_visibility(self.canvas.fractal_type)

        nav_btn_style = """
            QPushButton { background-color: #3b82f6; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px; }
            QPushButton:hover { background-color: #60a5fa; }
            QPushButton:pressed { background-color: #2563eb; }
            QPushButton:disabled { background-color: #2d3748; color: #718096; }
        """
        apply_btn = QPushButton("Set")
        apply_btn.setStyleSheet(nav_btn_style)
        apply_btn.clicked.connect(self.on_apply_coords)
        reset_btn = QPushButton("Reset")
        reset_btn.setStyleSheet(nav_btn_style)
        reset_btn.clicked.connect(self.on_reset_view)
        zin_btn = QPushButton("Zoom In")
        zin_btn.setStyleSheet(nav_btn_style)
        zin_btn.clicked.connect(self.on_zoom_in_clicked)
        zout_btn = QPushButton("Zoom Out")
        zout_btn.setStyleSheet(nav_btn_style)
        zout_btn.clicked.connect(self.on_zoom_out_clicked)

        r4.addWidget(apply_btn)
        r4.addWidget(reset_btn)
        r4.addWidget(zin_btn)
        r4.addWidget(zout_btn)
        r4.addStretch()
        control_layout.addLayout(r4)

        main_layout.addWidget(control_bar)

        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(12)

        sidebar = QWidget()
        sidebar.setFixedWidth(340)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(8)

        scheme_box = QHBoxLayout()
        scheme_box.addWidget(QLabel("Color Scheme:"))
        self.scheme_combo = QComboBox()
        self.scheme_combo.setMaxVisibleItems(len(COLOR_SCHEMES))
        self.scheme_combo.view().setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scheme_combo.addItems(list(COLOR_SCHEMES.keys()))
        self.scheme_combo.currentIndexChanged.connect(self.on_color_scheme_changed)
        scheme_box.addWidget(self.scheme_combo)
        sidebar_layout.addLayout(scheme_box)

        sidebar_layout.addWidget(QLabel("🎨 Colormap Library"))

        self.cmap_search = QLineEdit()
        self.cmap_search.setPlaceholderText("🔍 Search in category...")
        self.cmap_search.textChanged.connect(self.filter_colormaps)
        sidebar_layout.addWidget(self.cmap_search)

        self.category_combo = QComboBox()
        self.category_combo.addItems(list(CATEGORIZED_CMAPS.keys()))
        self.category_combo.currentTextChanged.connect(lambda cat: self.populate_cmap_list(CATEGORIZED_CMAPS.get(cat, [])))
        sidebar_layout.addWidget(self.category_combo)

        self.cmap_list = QListWidget()
        self.cmap_list.currentTextChanged.connect(self.on_cmap_selected)
        sidebar_layout.addWidget(self.cmap_list)

        phase_box = QHBoxLayout()
        phase_box.addWidget(QLabel("Phase Shift:"))
        self.phase_spin = QDoubleSpinBox()
        self.phase_spin.setRange(0.0, 1.0)
        self.phase_spin.setSingleStep(0.02)
        self.phase_spin.setDecimals(3)
        self.phase_spin.setValue(0.0)
        self.phase_spin.setWrapping(True)
        self.phase_spin.setToolTip("Shift palette phase contours in real time. Hotkeys: [ and ]")
        self.phase_spin.valueChanged.connect(self.on_phase_offset_changed)
        phase_box.addWidget(self.phase_spin)
        sidebar_layout.addLayout(phase_box)

        tuning_box = QVBoxLayout()
        density_row = QHBoxLayout()
        density_row.addWidget(QLabel("Density:"))
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setLocale(c_locale)
        self.density_spin.setRange(0.001, 10.000)
        self.density_spin.setSingleStep(0.001)
        self.density_spin.setDecimals(3)
        self.density_spin.setValue(1.000)
        self.density_spin.setToolTip("Control color cycle frequency across all schemes (Range: 0.001 to 10.0, Default: 1.000)")
        self.density_spin.valueChanged.connect(lambda v: self.canvas.set_color_density(float(v)))
        density_row.addWidget(self.density_spin)
        tuning_box.addLayout(density_row)

        contrast_row = QHBoxLayout()
        contrast_row.addWidget(QLabel("Contrast / Curve:"))
        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setLocale(c_locale)
        self.contrast_spin.setRange(0.001, 10.000)
        self.contrast_spin.setSingleStep(0.001)
        self.contrast_spin.setDecimals(3)
        self.contrast_spin.setValue(1.000)
        self.contrast_spin.setToolTip("Control non-linear contrast and curvature response (Range: 0.001 to 10.0, Default: 1.000)")
        self.contrast_spin.valueChanged.connect(lambda v: self.canvas.set_color_contrast(float(v)))
        contrast_row.addWidget(self.contrast_spin)
        tuning_box.addLayout(contrast_row)

        sidebar_layout.addLayout(tuning_box)

        content_layout.addWidget(sidebar)

        self.canvas.status_changed.connect(self.update_status_display)
        self.canvas.hover_coord_changed.connect(self.on_canvas_hover)
        self.canvas.keyframe_recorded.connect(self.on_waypoint_recorded)
        content_layout.addWidget(self.canvas, 1, Qt.AlignCenter)
        content_layout.addWidget(self.julia_preview)
        main_layout.addWidget(content_widget, 1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self.adjustSize()
        self.base_window_height = max(800, self.height())
        self.update_window_width()
        self.populate_cmap_list(CATEGORIZED_CMAPS["★ Popular / Recommended"], select_name='inferno')

        for btn in self.findChildren(QPushButton):
            btn.setFocusPolicy(Qt.NoFocus)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        if qApp is not None:
            qApp.installEventFilter(self)

        self.canvas.render_fractal()

    def on_waypoint_recorded(self, cx, cy, w):
        pt = (Decimal(str(cx)), Decimal(str(cy)), Decimal(str(w)))
        self.waypoints.append(pt)
        self.canvas.update()
        for dlg in self.dialogs.values():
            if hasattr(dlg, 'sync_waypoints_ui') and dlg.isVisible():
                dlg.sync_waypoints_ui()

    def on_device_changed(self, device):
        self.canvas.compute_device = device
        is_cpu = (device == "CPU")
        self.lbl_cores.setEnabled(is_cpu)
        self.cores_spin.setEnabled(is_cpu)

        self.prec_combo.setEnabled(True)
        self.canvas.min_plot_width = PRECISION_OPTIONS.get(self.prec_combo.currentText(), Decimal("1e-300"))
        is_perturbation = ("Perturbation" in self.prec_combo.currentText() or "Floatexp" in self.prec_combo.currentText())
        self.bbsa_combo.setEnabled(is_perturbation)
        self.glitch_combo.setEnabled(is_perturbation)
        self.canvas.render_fractal()

    def on_cpu_cores_changed(self, val):
        self.canvas.cpu_threads = val
        if self.canvas.compute_device == "CPU":
            self.canvas.render_fractal()

    def update_dynamic_control_visibility(self, ft):
        is_julia = ft in (2, 3)
        is_general = (ft == 4)
        for w in [self.lbl_jx, self.lbl_jy, self.julia_cx_spin, self.julia_cy_spin]:
            w.setVisible(is_julia)
        for w in [self.lbl_gn, self.gen_n_spin, self.lbl_gkr, self.gen_kr_spin, self.lbl_gki, self.gen_ki_spin]:
            w.setVisible(is_general)

        # For Mandelbrot and Burning Ship (ft 0 and 1), enlarge X, Y, Width, Scale inputs.
        # For the other 3 types (Julia Mandelbrot, Julia Burning Ship, General Mandelbrot), keep them compact.
        if ft in (0, 1):
            self.cx_edit.setFixedWidth(250)
            self.cy_edit.setFixedWidth(250)
            self.w_edit.setFixedWidth(160)
            self.scale_edit.setFixedWidth(160)
        else:
            self.cx_edit.setFixedWidth(140)
            self.cy_edit.setFixedWidth(140)
            self.w_edit.setFixedWidth(95)
            self.scale_edit.setFixedWidth(95)

    def on_julia_params_changed(self):
        try:
            self.canvas.julia_cx = Decimal(str(self.julia_cx_spin.value()))
            self.canvas.julia_cy = Decimal(str(self.julia_cy_spin.value()))
            self.canvas.render_fractal()
        except Exception as e:
            QMessageBox.warning(self, "Invalid Julia Parameters", str(e))

    def on_gen_params_changed(self):
        try:
            self.canvas.gen_n = self.gen_n_spin.value()
            self.canvas.gen_kr = float(self.gen_kr_spin.value())
            self.canvas.gen_ki = float(self.gen_ki_spin.value())
            self.canvas.render_fractal()
        except Exception as e:
            QMessageBox.warning(self, "Invalid Parameters", str(e))

    def open_dialog(self, key, dialog_cls, *args, **kwargs):
        if key in self.dialogs and self.dialogs[key].isVisible():
            self.dialogs[key].raise_()
            self.dialogs[key].activateWindow()
            if key in ('image_seq', 'video'):
                self.set_waypoints_overlay(True)
            return
        dlg = dialog_cls(*args, **kwargs)
        self.dialogs[key] = dlg
        if key in ('image_seq', 'video'):
            self.set_waypoints_overlay(True)
            dlg.finished.connect(lambda _=None: self.set_waypoints_overlay(False))
        dlg.show()

    def open_export_dialog(self, key, dialog_cls, **kwargs):
        self.open_dialog(
            key, dialog_cls, viewer=self, target_cx=self.canvas.center_x, target_cy=self.canvas.center_y,
            target_w=self.canvas.plot_width, max_iter=self.canvas.max_iter, cmap_name=self.canvas.cmap_name,
            edge_threshold=self.canvas.edge_threshold,
            color_scheme_id=self.canvas.color_scheme_id, precision_mode=self.canvas.precision_mode,
            fractal_type=self.canvas.fractal_type, julia_cx=float(self.canvas.julia_cx),
            julia_cy=float(self.canvas.julia_cy), bbsa_tol=self.canvas.bbsa_tol,
            bbsa_order=self.canvas.bbsa_order,
            glitch_mode=self.canvas.glitch_mode, dynamic_iter_mode=self.canvas.dynamic_iter_mode,
            gen_n=self.canvas.gen_n, gen_kr=self.canvas.gen_kr, gen_ki=self.canvas.gen_ki,
            compute_device=self.canvas.compute_device, cpu_threads=self.canvas.cpu_threads,
            palette_offset=self.canvas.palette_offset,
            color_density=self.canvas.color_density, color_contrast=self.canvas.color_contrast,
            **kwargs
        )

    def on_color_scheme_changed(self):
        scheme_id = COLOR_SCHEMES.get(self.scheme_combo.currentText(), 0)
        self.canvas.color_scheme_id = scheme_id
        self.canvas.color_density = 1.0
        self.canvas.color_contrast = 1.0
        self.density_spin.blockSignals(True)
        self.contrast_spin.blockSignals(True)
        self.density_spin.setValue(1.000)
        self.contrast_spin.setValue(1.000)
        self.density_spin.blockSignals(False)
        self.contrast_spin.blockSignals(False)
        self.canvas.recolor_canvas()

    def on_bbsa_mode_changed(self, text):
        self.canvas.bbsa_accuracy = text
        opts = BBSA_OPTIONS.get(text, BBSA_OPTIONS["4th-order"])
        self.canvas.bbsa_order = opts["order"]
        self.canvas.bbsa_tol = opts["tol"]
        self.canvas.render_fractal()

    def update_bbsa_combo_items(self, width=None, trigger_render=False):
        """Restricts BBSA options to ['Off', 'Chained BLA'] in FloatExp (< 1e-300),
        automatically selecting 'Chained BLA' since polynomial Taylor BBSA is unavailable."""
        w = width if width is not None else self.canvas.plot_width
        try:
            is_floatexp = Decimal(str(w)) < Decimal("1e-300")
        except Exception:
            is_floatexp = False

        desired_items = ["Off", "Chained BLA"] if is_floatexp else list(BBSA_OPTIONS.keys())
        curr_items = [self.bbsa_combo.itemText(i) for i in range(self.bbsa_combo.count())]

        if curr_items != desired_items:
            current_text = self.bbsa_combo.currentText()
            self.bbsa_combo.blockSignals(True)
            self.bbsa_combo.clear()
            self.bbsa_combo.addItems(desired_items)

            if is_floatexp:
                new_text = "Off" if current_text == "Off" else "Chained BLA"
            else:
                new_text = current_text if current_text in desired_items else "4th-order"

            self.bbsa_combo.setCurrentText(new_text)
            self.bbsa_combo.blockSignals(False)

            self.canvas.bbsa_accuracy = new_text
            opts = BBSA_OPTIONS.get(new_text, BBSA_OPTIONS["Chained BLA" if is_floatexp else "4th-order"])
            self.canvas.bbsa_order = opts["order"]
            self.canvas.bbsa_tol = opts["tol"]
            if trigger_render:
                self.canvas.render_fractal()

    def on_glitch_mode_changed(self, text):
        self.canvas.glitch_mode = text
        self.canvas.render_fractal()

    def on_dynamic_iter_changed(self, text):
        self.canvas.dynamic_iter_mode = text
        self.canvas.render_fractal()

    def update_window_width(self):
        """Locks the main window width and height to match the standard canvas + sidebar, expanding width only when Julia HUD is active."""
        base_w = 1280 + 372
        if self.julia_preview.isVisible():
            base_w += self.julia_preview.width() + 12
        self.setFixedWidth(base_w)
        if hasattr(self, 'base_window_height'):
            self.setFixedHeight(self.base_window_height)

    def on_canvas_size_changed(self, text):
        w, h = map(int, text.split('x'))
        self.canvas.setFixedSize(w, h)
        self.update_window_width()
        self.canvas.render_fractal()

    def toggle_julia_preview(self):
        new_state = not self.julia_preview.isVisible()
        self.julia_preview.setVisible(new_state)
        self.preview_toggle_btn.setStyleSheet(self.STYLE_HUD_ON if new_state else self.STYLE_HUD_OFF)
        self.update_window_width()

    def set_waypoints_overlay(self, state: bool):
        self.canvas.show_waypoints_overlay = bool(state)
        self.waypoints_toggle_btn.setText("Waypoints: ON" if self.canvas.show_waypoints_overlay else "Waypoints: OFF")
        self.waypoints_toggle_btn.setStyleSheet(self.STYLE_HUD_ON if self.canvas.show_waypoints_overlay else self.STYLE_HUD_OFF)
        self.canvas.update()

    def toggle_waypoints_overlay(self):
        self.set_waypoints_overlay(not self.canvas.show_waypoints_overlay)

    def on_canvas_hover(self, cx_dec, cy_dec):
        if self.julia_preview.isVisible() and self.canvas.fractal_type in (0, 1):
            self.julia_preview.update_preview(
                cx_dec, cy_dec, self.canvas.cmap_name, self.canvas.color_scheme_id, self.canvas.fractal_type,
                compute_device=self.canvas.compute_device, cpu_threads=self.canvas.cpu_threads,
                color_density=self.canvas.color_density, color_contrast=self.canvas.color_contrast
            )

    def on_jump_to_julia(self, jx_dec, jy_dec):
        self.toggle_julia_preview()
        self.canvas.fractal_type = 3 if self.canvas.fractal_type == 1 else 2
        self.canvas.julia_cx, self.canvas.julia_cy = jx_dec, jy_dec
        self.julia_cx_spin.blockSignals(True)
        self.julia_cy_spin.blockSignals(True)
        self.julia_cx_spin.setValue(float(jx_dec))
        self.julia_cy_spin.setValue(float(jy_dec))
        self.julia_cx_spin.blockSignals(False)
        self.julia_cy_spin.blockSignals(False)
        self.canvas.center_x, self.canvas.center_y, self.canvas.plot_width = FRACTAL_DEFAULTS[self.canvas.fractal_type]
        self.fractal_combo.setCurrentIndex(self.canvas.fractal_type)
        self.update_dynamic_control_visibility(self.canvas.fractal_type)
        self.canvas.render_fractal()

    def set_viewport_defaults(self, ft):
        self.canvas.fractal_type = ft
        def_cx, def_cy, def_w = FRACTAL_DEFAULTS.get(ft, DEFAULT_JULIA_VIEW)
        self.canvas.center_x = Decimal(str(def_cx))
        self.canvas.center_y = Decimal(str(def_cy))
        self.canvas.plot_width = Decimal(str(def_w))

        if ft in (2, 3):
            jx, jy = DEFAULT_JULIA_MANDELBROT_PARAMS if ft == 2 else DEFAULT_JULIA_BURNING_SHIP_PARAMS
            self.canvas.julia_cx = Decimal(str(jx))
            self.canvas.julia_cy = Decimal(str(jy))
            self.julia_cx_spin.blockSignals(True)
            self.julia_cy_spin.blockSignals(True)
            self.julia_cx_spin.setValue(float(self.canvas.julia_cx))
            self.julia_cy_spin.setValue(float(self.canvas.julia_cy))
            self.julia_cx_spin.blockSignals(False)
            self.julia_cy_spin.blockSignals(False)
        elif ft == 4:
            self.canvas.gen_n = 3
            self.canvas.gen_kr = 0.25
            self.canvas.gen_ki = 1.0
            self.gen_n_spin.setValue(3)
            self.gen_kr_spin.setValue(0.25)
            self.gen_ki_spin.setValue(1.0)

        self.cx_edit.setText(str(self.canvas.center_x))
        self.cx_edit.setCursorPosition(0)
        self.cy_edit.setText(str(self.canvas.center_y))
        self.cy_edit.setCursorPosition(0)
        self.w_edit.setText(str(self.canvas.plot_width))
        self.scale_edit.setText(get_real_world_scale(self.canvas.plot_width))
        self.scale_edit.setCursorPosition(0)

        self.fractal_combo.blockSignals(True)
        self.fractal_combo.setCurrentIndex(ft)
        self.fractal_combo.blockSignals(False)

        self.update_dynamic_control_visibility(ft)
        self.canvas.render_fractal()

    def on_fractal_type_change(self, index):
        if self.canvas.fractal_type != index:
            clear_reference_orbit_cache()
            self.set_viewport_defaults(index)

    def get_zoom_factor(self):
        try:
            val = Decimal(str(self.zoom_factor_edit.text()).strip())
            if val < Decimal("1.0"):
                return Decimal("1.0")
            elif val > Decimal("1000.0"):
                return Decimal("1000.0")
            return val
        except Exception:
            pass
        return Decimal("5.0")

    def on_zoom_factor_changed(self, text):
        factor = self.get_zoom_factor()
        self.canvas.click_zoom_factor = factor

    def on_zoom_factor_editing_finished(self):
        factor = self.get_zoom_factor()
        s = f"{factor:f}".rstrip('0').rstrip('.') if '.' in f"{factor:f}" else str(factor)
        if self.zoom_factor_edit.text().strip() != s:
            self.zoom_factor_edit.setText(s)
        self.canvas.click_zoom_factor = factor

    def on_zoom_factor_return_pressed(self):
        self.on_zoom_factor_editing_finished()

    def get_wheel_zoom_factor(self):
        try:
            val = Decimal(str(self.wheel_zoom_edit.text()).strip())
            if val < Decimal("1.001"):
                return Decimal("1.001")
            elif val > Decimal("1000.0"):
                return Decimal("1000.0")
            return val
        except Exception:
            pass
        return Decimal("1.25")

    def on_wheel_zoom_changed(self, text):
        factor = self.get_wheel_zoom_factor()
        self.canvas.wheel_zoom_factor = factor

    def on_wheel_zoom_editing_finished(self):
        factor = self.get_wheel_zoom_factor()
        s = f"{factor:f}".rstrip('0').rstrip('.') if '.' in f"{factor:f}" else str(factor)
        if self.wheel_zoom_edit.text().strip() != s:
            self.wheel_zoom_edit.setText(s)
        self.canvas.wheel_zoom_factor = factor

    def on_wheel_zoom_return_pressed(self):
        self.on_wheel_zoom_editing_finished()

    def on_zoom_in_clicked(self):
        factor = self.get_zoom_factor()
        new_w = max(self.canvas.min_plot_width, self.canvas.plot_width / factor)
        if new_w != self.canvas.plot_width:
            self.canvas.plot_width = new_w
            self.canvas.render_fractal()

    def on_zoom_out_clicked(self):
        factor = self.get_zoom_factor()
        self.canvas.plot_width = self.canvas.plot_width * factor
        self.canvas.render_fractal()

    def _set_combo_safe(self, combo, text):
        if text is None:
            return
        # 1. Exact match
        idx = combo.findText(str(text))
        # 2. Case-insensitive match
        if idx == -1:
            text_lower = str(text).strip().lower()
            for i in range(combo.count()):
                if combo.itemText(i).strip().lower() == text_lower:
                    idx = i
                    break
        # 3. Substring / partial match
        if idx == -1:
            text_lower = str(text).strip().lower()
            for i in range(combo.count()):
                item_lower = combo.itemText(i).strip().lower()
                if text_lower in item_lower or item_lower in text_lower:
                    idx = i
                    break
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def on_reset_view(self):
        """Full factory reset to default overview and baseline rendering engine settings."""
        default_state = {
            "name": "Default Overview",
            "fractal_type": 0,
            "center_x": "-0.65",
            "center_y": "0.0",
            "plot_width": "4.0",
            "max_iter": DEFAULT_MAX_ITER,
            "cmap_name": "inferno",
            "color_scheme_id": 0,
            "precision_mode": "1e-300 (Perturbation)",
            "palette_offset": 0.0,
            "color_density": 1.0,
            "color_contrast": 1.0,
            "bbsa_accuracy": "4th-order",
            "glitch_mode": "Off (Single-Ref)",
            "dynamic_iter_mode": "Off",
            "ssaa_factor": 1,
            "edge_threshold": 0.35,
            "julia_cx": "-0.67",
            "julia_cy": "0.37",
            "gen_n": 3,
            "gen_kr": 0.25,
            "gen_ki": 1.0,
            "keyframes": []
        }
        self.apply_state_dict(default_state)

    def on_prec_changed_combo(self, text):
        self.canvas.precision_mode = text
        self.canvas.min_plot_width = PRECISION_OPTIONS.get(text, Decimal("1e-300"))
        is_perturbation = ("Perturbation" in text or "Floatexp" in text)
        self.bbsa_combo.setEnabled(is_perturbation)
        self.glitch_combo.setEnabled(is_perturbation)

        self.canvas.plot_width = max(self.canvas.min_plot_width, self.canvas.plot_width)
        self.update_bbsa_combo_items(self.canvas.plot_width, trigger_render=False)
        self.canvas.render_fractal()

    def select_colormap_in_ui(self, cmap_name):
        """Finds the category containing cmap_name and updates the sidebar selection."""
        current_cat = self.category_combo.currentText()
        target_cat = None
        matched_name = cmap_name

        # 1. Check if the current category already contains the colormap
        current_cmaps = CATEGORIZED_CMAPS.get(current_cat, [])
        if cmap_name in current_cmaps:
            target_cat = current_cat
        # 2. Check Popular / Recommended next
        elif cmap_name in CATEGORIZED_CMAPS.get("★ Popular / Recommended", []):
            target_cat = "★ Popular / Recommended"
        # 3. Search across all other categories (exact match)
        else:
            for cat, cmaps in CATEGORIZED_CMAPS.items():
                if cmap_name in cmaps:
                    target_cat = cat
                    break

        # 4. If not found, search case-insensitively
        if not target_cat:
            cmap_lower = cmap_name.lower()
            for cat, cmaps in CATEGORIZED_CMAPS.items():
                for m in cmaps:
                    if m.lower() == cmap_lower:
                        target_cat = cat
                        matched_name = m
                        break
                if target_cat:
                    break

        # 5. If still not found, add to matplotlib or Popular so it is always present
        if not target_cat:
            target_cat = "matplotlib" if "matplotlib" in CATEGORIZED_CMAPS else "★ Popular / Recommended"
            if matched_name not in CATEGORIZED_CMAPS[target_cat]:
                CATEGORIZED_CMAPS[target_cat].append(matched_name)

        # Clear search filter so the target colormap is always visible
        self.cmap_search.blockSignals(True)
        self.cmap_search.clear()
        self.cmap_search.blockSignals(False)

        # Ensure target_cat exists in category combo
        if self.category_combo.findText(target_cat) == -1:
            self.category_combo.addItem(target_cat)

        self.category_combo.blockSignals(True)
        self._set_combo_safe(self.category_combo, target_cat)
        self.category_combo.blockSignals(False)
        self.populate_cmap_list(CATEGORIZED_CMAPS.get(target_cat, []), select_name=matched_name)

    def populate_cmap_list(self, cmaps, select_name=None):
        self.cmap_list.blockSignals(True)
        self.cmap_list.clear()
        f = self.cmap_search.text().strip().lower()
        target = (select_name or self.canvas.cmap_name).strip().lower()
        sel_row = -1

        for name in cmaps:
            if f and f not in name.lower():
                continue
            self.cmap_list.addItem(name)
            if name.strip().lower() == target:
                sel_row = self.cmap_list.count() - 1

        if sel_row >= 0:
            self.cmap_list.setCurrentRow(sel_row)
            item = self.cmap_list.item(sel_row)
            if item is not None:
                self.cmap_list.scrollToItem(item, QAbstractItemView.PositionAtCenter)
        self.cmap_list.blockSignals(False)

    def filter_colormaps(self, _):
        self.populate_cmap_list(CATEGORIZED_CMAPS.get(self.category_combo.currentText(), []))

    def on_cmap_selected(self, name):
        if name:
            self.canvas.cmap_name = name
            self.canvas.recolor_canvas()

    def on_phase_offset_changed(self, val):
        self.canvas.set_palette_offset(float(val))

    def step_palette_offset(self, delta):
        new_val = round((self.canvas.palette_offset + delta) % 1.0, 4)
        self.phase_spin.blockSignals(True)
        self.phase_spin.setValue(new_val)
        self.phase_spin.blockSignals(False)
        self.canvas.set_palette_offset(new_val)

    def _on_palette_timer_tick(self):
        new_val = round((self.canvas.palette_offset + 0.0035) % 1.0, 4)
        self.phase_spin.blockSignals(True)
        self.phase_spin.setValue(new_val)
        self.phase_spin.blockSignals(False)
        self.canvas.set_palette_offset(new_val)

    def toggle_palette_playback(self):
        if self.palette_timer.isActive():
            self.palette_timer.stop()
        else:
            self.palette_timer.start(16)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            active_window = qApp.activeWindow()
            if active_window is not None and active_window != self:
                return super().eventFilter(obj, event)
            if qApp.activeModalWidget() is not None:
                return super().eventFilter(obj, event)

            focus_widget = qApp.focusWidget() or self.focusWidget()
            is_text_input = isinstance(obj, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)) or \
                            isinstance(focus_widget, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox))
            
            if is_text_input:
                return super().eventFilter(obj, event)

            if event.key() == Qt.Key_Space:
                self.toggle_palette_playback()
                return True
            elif event.key() == Qt.Key_R:
                self.on_reset_view()
                return True
            elif event.key() == Qt.Key_H:
                self.toggle_waypoints_overlay()
                return True
            elif event.key() == Qt.Key_BracketLeft:
                self.step_palette_offset(-0.02)
                return True
            elif event.key() == Qt.Key_BracketRight:
                self.step_palette_offset(0.02)
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        active_window = qApp.activeWindow()
        if active_window is not None and active_window != self:
            super().keyPressEvent(event)
            return
        if qApp.activeModalWidget() is not None:
            super().keyPressEvent(event)
            return

        focus_widget = qApp.focusWidget() or self.focusWidget()
        if isinstance(focus_widget, (QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox)):
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key_BracketLeft:
            self.step_palette_offset(-0.02)
        elif event.key() == Qt.Key_BracketRight:
            self.step_palette_offset(0.02)
        elif event.key() == Qt.Key_Space:
            self.toggle_palette_playback()
        elif event.key() == Qt.Key_R:
            self.on_reset_view()
        elif event.key() == Qt.Key_H:
            self.toggle_waypoints_overlay()
        else:
            super().keyPressEvent(event)

    def on_apply_coords(self):
        try:
            self.canvas.center_x = Decimal(self.cx_edit.text().strip())
            self.canvas.center_y = Decimal(self.cy_edit.text().strip())

            if getattr(self, 'last_width_source', 'width') == 'scale' or self.scale_edit.hasFocus():
                w = parse_real_world_scale(self.scale_edit.text().strip())
            else:
                w = Decimal(self.w_edit.text().strip())

            if w <= Decimal(0):
                raise ValueError("Width must be positive.")
            self.canvas.plot_width = max(self.canvas.min_plot_width, w)
            if self.canvas.fractal_type in (2, 3):
                self.canvas.julia_cx = Decimal(str(self.julia_cx_spin.value()))
                self.canvas.julia_cy = Decimal(str(self.julia_cy_spin.value()))
            elif self.canvas.fractal_type == 4:
                self.canvas.gen_n = self.gen_n_spin.value()
                self.canvas.gen_kr = float(self.gen_kr_spin.value())
                self.canvas.gen_ki = float(self.gen_ki_spin.value())
            self.canvas.render_fractal()
        except Exception as e:
            QMessageBox.warning(self, "Invalid Coordinates", str(e))

    def get_current_state_dict(self, name="Custom View"):
        return {
            "name": name, "fractal_type": int(self.canvas.fractal_type),
            "center_x": str(self.canvas.center_x), "center_y": str(self.canvas.center_y), "plot_width": str(self.canvas.plot_width),
            "max_iter": int(self.canvas.max_iter), "cmap_name": self.canvas.cmap_name,
            "color_scheme_id": int(self.canvas.color_scheme_id), "precision_mode": self.canvas.precision_mode,
            "palette_offset": float(self.canvas.palette_offset),
            "color_density": float(self.canvas.color_density),
            "color_contrast": float(self.canvas.color_contrast),
            "bbsa_accuracy": self.canvas.bbsa_accuracy, "glitch_mode": self.canvas.glitch_mode,
            "dynamic_iter_mode": self.canvas.dynamic_iter_mode,
            "ssaa_factor": int(self.canvas.ssaa_factor),
            "edge_threshold": float(self.canvas.edge_threshold),
            "julia_cx": str(self.canvas.julia_cx), "julia_cy": str(self.canvas.julia_cy),
            "gen_n": int(self.canvas.gen_n), "gen_kr": float(self.canvas.gen_kr), "gen_ki": float(self.canvas.gen_ki),
            "click_zoom_factor": str(self.canvas.click_zoom_factor), "wheel_zoom_factor": str(self.canvas.wheel_zoom_factor),
            "keyframes": [{"center_x": str(cx), "center_y": str(cy), "plot_width": str(w)} for cx, cy, w in self.waypoints]
        }

    def apply_state_dict(self, s):
        try:
            # 1. Fractal Type
            raw_ft = s.get("fractal_type", s.get("fractal", 0))
            if isinstance(raw_ft, int):
                ft = raw_ft
            elif str(raw_ft).isdigit():
                ft = int(raw_ft)
            elif raw_ft in FRACTAL_NAMES:
                ft = FRACTAL_NAMES.index(raw_ft)
            else:
                ft = 0
                for idx, fname in enumerate(FRACTAL_NAMES):
                    if fname.lower() == str(raw_ft).strip().lower():
                        ft = idx
                        break
            self.canvas.fractal_type = max(0, min(ft, len(FRACTAL_NAMES) - 1))

            # 2. Viewport Coordinates & Max Iterations
            self.canvas.center_x = Decimal(str(s.get("center_x", s.get("cx", "-0.65"))))
            self.canvas.center_y = Decimal(str(s.get("center_y", s.get("cy", "0.0"))))
            self.canvas.plot_width = Decimal(str(s.get("plot_width", s.get("width", "4.0"))))
            self.canvas.max_iter = int(s.get("max_iter", s.get("iterations", DEFAULT_MAX_ITER)))

            # 3. Color Scheme
            raw_scheme = s.get("color_scheme_id", s.get("color_scheme", s.get("scheme", 0)))
            if isinstance(raw_scheme, int):
                scheme_id = raw_scheme
            elif str(raw_scheme).isdigit():
                scheme_id = int(raw_scheme)
            elif raw_scheme in COLOR_SCHEMES:
                scheme_id = COLOR_SCHEMES[raw_scheme]
            else:
                scheme_id = 0
                for name, cid in COLOR_SCHEMES.items():
                    if name.lower() == str(raw_scheme).strip().lower():
                        scheme_id = cid
                        break
            self.canvas.color_scheme_id = scheme_id

            # 4. Color tuning params
            self.canvas.palette_offset = float(s.get("palette_offset", s.get("phase", 0.0)))
            self.canvas.color_density = float(s.get("color_density", s.get("density", 1.0)))
            self.canvas.color_contrast = float(s.get("color_contrast", s.get("contrast", 1.0)))

            # 5. Precision Mode
            raw_prec = str(s.get("precision_mode", s.get("precision", ""))).strip()
            raw_prec_lower = raw_prec.lower()
            if self.canvas.plot_width < Decimal("1e-300") or "floatexp" in raw_prec_lower or "10000" in raw_prec:
                self.canvas.precision_mode = "1e-10000+ (Floatexp)"
            elif "dd" in raw_prec_lower or "double" in raw_prec_lower or "24" in raw_prec:
                self.canvas.precision_mode = "1e-24 (DD)"
            elif "fp64" in raw_prec_lower or ("12" in raw_prec and "1e-12" in raw_prec):
                self.canvas.precision_mode = "1e-12 (FP64)"
            else:
                self.canvas.precision_mode = "1e-300 (Perturbation)"
            self.canvas.min_plot_width = PRECISION_OPTIONS.get(self.canvas.precision_mode, Decimal("1e-300"))

            # 6. BBSA Accuracy
            raw_bbsa = str(s.get("bbsa_accuracy", s.get("bbsa", "4th-order"))).strip()
            raw_bbsa_lower = raw_bbsa.lower()
            if "piecewise" in raw_bbsa_lower or "multi-stage" in raw_bbsa_lower or "8th" in raw_bbsa_lower:
                bbsa_text = "8th-order"
            elif "bla" in raw_bbsa_lower or "32" in raw_bbsa_lower:
                bbsa_text = "Chained BLA"
            elif raw_bbsa_lower in ("off", "0", "none", "false"):
                bbsa_text = "Off"
            elif "4th" in raw_bbsa_lower or "4" in raw_bbsa_lower:
                bbsa_text = "4th-order"
            elif raw_bbsa in BBSA_OPTIONS:
                bbsa_text = raw_bbsa
            else:
                bbsa_text = "4th-order"

            try:
                target_pw = Decimal(str(s.get("plot_width", self.canvas.plot_width)))
                if target_pw < Decimal("1e-300") and bbsa_text in ("4th-order", "8th-order"):
                    bbsa_text = "Chained BLA"
            except Exception:
                pass

            self.canvas.bbsa_accuracy = bbsa_text
            opts = BBSA_OPTIONS.get(bbsa_text, BBSA_OPTIONS["4th-order"])
            self.canvas.bbsa_order = opts["order"]
            self.canvas.bbsa_tol = opts["tol"]

            # 7. Glitch Correction Mode
            raw_glitch = str(s.get("glitch_mode", s.get("glitch_correction", s.get("glitch", "")))).strip()
            raw_glitch_lower = raw_glitch.lower()
            if "strict" in raw_glitch_lower:
                glitch_text = "Strict Metric"
            elif "pauldelbrot" in raw_glitch_lower or "rebas" in raw_glitch_lower:
                glitch_text = "Pauldelbrot"
            elif "off" in raw_glitch_lower or "single" in raw_glitch_lower or "legacy" in raw_glitch_lower:
                glitch_text = "Off (Single-Ref)"
            elif raw_glitch in GLITCH_CORRECTION_MODES:
                glitch_text = raw_glitch
            else:
                glitch_text = "Off (Single-Ref)"
            self.canvas.glitch_mode = glitch_text

            # 8. Dynamic Iteration Mode
            raw_dyn = str(s.get("dynamic_iter_mode", s.get("dynamic_iterations", s.get("dynamic_iter", "Off")))).strip()
            raw_dyn_lower = raw_dyn.lower()
            if "linear" in raw_dyn_lower:
                dyn_text = "On (Linear Scaling)"
            elif "adaptive" in raw_dyn_lower or "preview" in raw_dyn_lower:
                dyn_text = "On (Adaptive Preview)"
            elif "sqrt" in raw_dyn_lower or raw_dyn in ("On", "True", "1"):
                dyn_text = "On (Sqrt Scaling)"
            elif "off" in raw_dyn_lower or raw_dyn in ("False", "0"):
                dyn_text = "Off"
            elif raw_dyn in DYNAMIC_ITER_MODES:
                dyn_text = raw_dyn
            else:
                dyn_text = "Off"
            self.canvas.dynamic_iter_mode = dyn_text

            # 9. Antialiasing (SSAA)
            raw_ssaa = str(s.get("ssaa_factor", s.get("ssaa", 1)))
            ssaa_val = 1
            for factor in [8, 6, 4, 2, 1]:
                if str(factor) in raw_ssaa:
                    ssaa_val = factor
                    break
            self.canvas.ssaa_factor = ssaa_val

            self.canvas.edge_threshold = float(s.get("edge_threshold", 0.35))
            self.canvas.julia_cx = Decimal(str(s.get("julia_cx", "-0.67")))
            self.canvas.julia_cy = Decimal(str(s.get("julia_cy", "0.37")))
            self.canvas.gen_n = int(s.get("gen_n", 3))
            self.canvas.gen_kr = float(s.get("gen_kr", s.get("gen_k", 0.25)))
            self.canvas.gen_ki = float(s.get("gen_ki", 1.0))
            self.canvas.cmap_name = str(s.get("cmap_name", s.get("colormap", "inferno")))

            # 10. Update Colormap Library selection
            self.select_colormap_in_ui(self.canvas.cmap_name)

            # 11. Update Controls with Signal Blocking
            controls = [
                self.gen_n_spin, self.gen_kr_spin, self.gen_ki_spin,
                self.julia_cx_spin, self.julia_cy_spin,
                self.edge_spin, self.fractal_combo, self.iter_spin,
                self.prec_combo, self.bbsa_combo, self.glitch_combo,
                self.dynamic_iter_combo, self.aa_combo, self.scheme_combo,
                self.phase_spin, self.density_spin, self.contrast_spin
            ]

            for w in controls:
                w.blockSignals(True)

            try:
                self.gen_n_spin.setValue(self.canvas.gen_n)
                self.gen_kr_spin.setValue(self.canvas.gen_kr)
                self.gen_ki_spin.setValue(self.canvas.gen_ki)
                self.julia_cx_spin.setValue(float(self.canvas.julia_cx))
                self.julia_cy_spin.setValue(float(self.canvas.julia_cy))
                self.edge_spin.setValue(self.canvas.edge_threshold)
                self.phase_spin.setValue(self.canvas.palette_offset)
                self.density_spin.setValue(self.canvas.color_density)
                self.contrast_spin.setValue(self.canvas.color_contrast)

                self.fractal_combo.setCurrentIndex(self.canvas.fractal_type)
                self.iter_spin.setValue(self.canvas.max_iter)

                self._set_combo_safe(self.prec_combo, self.canvas.precision_mode)
                self.update_bbsa_combo_items(self.canvas.plot_width, trigger_render=False)
                self._set_combo_safe(self.bbsa_combo, self.canvas.bbsa_accuracy)
                self._set_combo_safe(self.glitch_combo, self.canvas.glitch_mode)
                self._set_combo_safe(self.dynamic_iter_combo, self.canvas.dynamic_iter_mode)
                self.aa_combo.setCurrentIndex({1: 0, 2: 1, 4: 2, 6: 3, 8: 4}.get(self.canvas.ssaa_factor, 0))

                scheme_names = [k for k, v in COLOR_SCHEMES.items() if v == self.canvas.color_scheme_id]
                if scheme_names:
                    self._set_combo_safe(self.scheme_combo, scheme_names[0])
                else:
                    self.scheme_combo.setCurrentIndex(0)

                if "compute_device" in s or "device" in s:
                    dev_str = str(s.get("compute_device", s.get("device", ""))).upper()
                    if dev_str in ("GPU", "CPU"):
                        if dev_str == "GPU" and not HAS_CUDA:
                            dev_str = "CPU"
                        self._set_combo_safe(self.device_combo, dev_str)
                        self.canvas.compute_device = dev_str

                if "canvas_size" in s:
                    self._set_combo_safe(self.canvas_size_combo, str(s["canvas_size"]))

                if "click_zoom_factor" in s or "zoom_factor" in s:
                    zf = str(s.get("click_zoom_factor", s.get("zoom_factor", "5")))
                    self.zoom_factor_edit.setText(zf)
                    self.on_zoom_factor_editing_finished()

                if "wheel_zoom_factor" in s or "wheel_zoom" in s:
                    wz = str(s.get("wheel_zoom_factor", s.get("wheel_zoom", "1.25")))
                    self.wheel_zoom_edit.setText(wz)
                    self.on_wheel_zoom_editing_finished()

                self.cx_edit.setText(str(self.canvas.center_x))
                self.cx_edit.setCursorPosition(0)
                self.cy_edit.setText(str(self.canvas.center_y))
                self.cy_edit.setCursorPosition(0)
                self.w_edit.setText(str(self.canvas.plot_width))
                self.scale_edit.setText(get_real_world_scale(self.canvas.plot_width))
                self.scale_edit.setCursorPosition(0)
            finally:
                for w in controls:
                    w.blockSignals(False)

            is_pert = ("Perturbation" in self.canvas.precision_mode or "Floatexp" in self.canvas.precision_mode)
            self.bbsa_combo.setEnabled(is_pert)
            self.glitch_combo.setEnabled(is_pert)

            self.preview_toggle_btn.setEnabled(self.canvas.fractal_type in (0, 1))
            if self.canvas.fractal_type not in (0, 1) and self.julia_preview.isVisible():
                self.julia_preview.setVisible(False)
                self.preview_toggle_btn.setStyleSheet(self.STYLE_HUD_OFF)

            self.update_dynamic_control_visibility(self.canvas.fractal_type)
            self.update_window_width()

            self.waypoints = [(Decimal(str(k["center_x"])), Decimal(str(k["center_y"])), Decimal(str(k["plot_width"]))) for k in s.get("keyframes", s.get("waypoints", []))]
            self.set_waypoints_overlay(len(self.waypoints) > 0)

            for dlg in self.dialogs.values():
                if dlg and dlg.isVisible():
                    if hasattr(dlg, 'sync_waypoints_ui'):
                        dlg.sync_waypoints_ui()
                    if hasattr(dlg, 'params') and isinstance(dlg.params, dict):
                        dlg.params.update({
                            'target_cx': self.canvas.center_x,
                            'target_cy': self.canvas.center_y,
                            'target_w': self.canvas.plot_width,
                            'max_iter': self.canvas.max_iter,
                            'cmap_name': self.canvas.cmap_name,
                            'fractal_type': self.canvas.fractal_type,
                            'precision_mode': self.canvas.precision_mode,
                            'color_scheme_id': self.canvas.color_scheme_id,
                            'glitch_mode': self.canvas.glitch_mode,
                            'dynamic_iter_mode': self.canvas.dynamic_iter_mode,
                            'bbsa_accuracy': self.canvas.bbsa_accuracy
                        })
                        if hasattr(dlg, 'status_lbl') and dlg.status_lbl:
                            dlg.status_lbl.setText(f"Target Width: {Decimal(str(self.canvas.plot_width)):.3e}")

            preset_name = s.get("name")
            if preset_name:
                self.setWindowTitle(f"Apeiron - {preset_name}")

            clear_reference_orbit_cache()
            self.canvas.render_fractal()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to restore state:\n{e}")

    def update_status_display(self, cx, cy, width, rtime, eff_iter):
        self.update_bbsa_combo_items(width, trigger_render=False)
        self.cx_edit.setText(str(cx))
        self.cx_edit.setCursorPosition(0)
        self.cy_edit.setText(str(cy))
        self.cy_edit.setCursorPosition(0)
        self.w_edit.setText(str(width))
        self.scale_edit.setText(get_real_world_scale(width))
        self.scale_edit.setCursorPosition(0)
        self.last_width_source = 'width'
        try:
            w_str = f"{Decimal(str(width)):.3e}"
            cx_s = f"{Decimal(str(cx)):.6f}"
            cy_s = f"{Decimal(str(cy)):.6f}"
        except Exception:
            w_str = str(width)
            cx_s = str(cx)
            cy_s = str(cy)

        title_str = FRACTAL_NAMES[self.canvas.fractal_type]
        dev_tag = f"Device: {self.canvas.compute_device}" + (f" ({self.canvas.cpu_threads}T)" if self.canvas.compute_device == "CPU" else "")
        engine_tag = f"Engine: {get_active_engine_name(width, self.canvas.precision_mode)}"
        
        fps = 1000.0 / rtime if rtime > 0 else 0.0
        
        iter_info = f"Iter: {eff_iter}" if self.canvas.dynamic_iter_mode == "Off" else f"Iter: {self.canvas.max_iter} -> {eff_iter}"
        self.status.showMessage(
            f"[{title_str}] | {dev_tag} | {engine_tag} | Center: ({cx_s}, {cy_s}) | Width: {w_str} | "
            f"Scale: {get_real_world_scale(width)} | {iter_info} | Render: {rtime:.2f} ms ({fps:.2f} FPS)"
        )

    def closeEvent(self, event):
        """Cleanly cancels active export workers and releases GPU scratch buffers before exit."""
        if hasattr(self, 'dialogs'):
            for dlg in list(self.dialogs.values()):
                try:
                    if hasattr(dlg, 'worker') and dlg.worker is not None:
                        if hasattr(dlg.worker, 'cancel'):
                            dlg.worker.cancel()
                        if hasattr(dlg.worker, 'isRunning') and dlg.worker.isRunning():
                            dlg.worker.wait(1000)
                    dlg.close()
                except Exception:
                    pass
        clear_reference_orbit_cache()
        clear_scratch_buffer_pool()
        event.accept()