import json
import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QTextEdit, QVBoxLayout
)

from config import FRACTAL_NAMES, SCRIPT_DIR, load_builtin_presets


class BookmarkPresetsDialog(QDialog):
    def __init__(self, viewer):
        super().__init__(None)
        self.viewer = viewer
        self.setWindowTitle("Bookmarks & Curated Presets")
        self.resize(750, 620)
        self.setWindowFlags(Qt.Window)
        self.filtered_indices = []

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        header_lbl = QLabel("Saved Bookmarks & Curated Locations")
        header_lbl.setStyleSheet("font-weight: bold; font-size: 15px;")
        layout.addWidget(header_lbl)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("🔍 Search bookmarks & presets (title, type, description)...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setStyleSheet("""
            QLineEdit {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #9ca3af;
                border-radius: 4px;
                padding: 6px 10px;
            }
            QLineEdit:focus {
                border: 1.5px solid #2563eb;
            }
        """)
        self.search_edit.textChanged.connect(self.on_filter_text_changed)
        layout.addWidget(self.search_edit)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.on_selection_changed)
        self.list_widget.itemDoubleClicked.connect(self.on_load_selected)
        layout.addWidget(self.list_widget, 1)

        self.details_box = QTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setFixedHeight(130)
        self.details_box.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.details_box)

        btn_l1 = QHBoxLayout()
        btn_l1.setSpacing(8)

        self.load_btn = QPushButton("Load & Jump to Location")
        self.load_btn.setStyleSheet("""
            QPushButton { background-color: #4863A0; color: white; padding: 6px 12px; font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background-color: #5a77b8; }
            QPushButton:disabled { background-color: #2d3748; color: #718096; }
        """)
        self.load_btn.clicked.connect(self.on_load_selected)

        self.add_btn = QPushButton("+ Bookmark Current View")
        self.add_btn.setStyleSheet("""
            QPushButton { background-color: #2E8B57; color: white; padding: 6px 12px; font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background-color: #3cb371; }
            QPushButton:disabled { background-color: #2d3748; color: #718096; }
        """)
        self.add_btn.clicked.connect(self.on_add_current_view)

        self.rename_btn = QPushButton("✏️ Rename")
        self.rename_btn.setStyleSheet("""
            QPushButton { background-color: #4b5563; color: white; padding: 6px 12px; font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background-color: #6b7280; }
            QPushButton:disabled { background-color: #2d3748; color: #718096; }
        """)
        self.rename_btn.clicked.connect(self.on_rename_bookmark)

        self.delete_btn = QPushButton("🗑️ Delete")
        self.delete_btn.setStyleSheet("""
            QPushButton { background-color: #dc3545; color: white; padding: 6px 12px; font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background-color: #e04a58; }
            QPushButton:disabled { background-color: #2d3748; color: #718096; }
        """)
        self.delete_btn.clicked.connect(self.on_delete_bookmark)

        btn_l1.addWidget(self.load_btn)
        btn_l1.addWidget(self.add_btn)
        btn_l1.addWidget(self.rename_btn)
        btn_l1.addWidget(self.delete_btn)
        layout.addLayout(btn_l1)

        btn_l2 = QHBoxLayout()
        btn_l2.setSpacing(8)

        export_btn = QPushButton("💾 Export to JSON...")
        export_btn.setStyleSheet("""
            QPushButton { background-color: #374151; color: white; padding: 6px 12px; border-radius: 4px; }
            QPushButton:hover { background-color: #4b5563; }
            QPushButton:pressed { background-color: #1f2937; }
            QPushButton:disabled { background-color: #2d3748; color: #718096; }
        """)
        export_btn.clicked.connect(self.on_export_json)

        import_btn = QPushButton("📂 Import from JSON...")
        import_btn.setStyleSheet("""
            QPushButton { background-color: #374151; color: white; padding: 6px 12px; border-radius: 4px; }
            QPushButton:hover { background-color: #4b5563; }
            QPushButton:pressed { background-color: #1f2937; }
            QPushButton:disabled { background-color: #2d3748; color: #718096; }
        """)
        import_btn.clicked.connect(self.on_import_json)

        reset_btn = QPushButton("🔄 Reset to Defaults")
        reset_btn.setStyleSheet("""
            QPushButton { background-color: #374151; color: white; padding: 6px 12px; border-radius: 4px; }
            QPushButton:hover { background-color: #4b5563; }
            QPushButton:pressed { background-color: #1f2937; }
            QPushButton:disabled { background-color: #2d3748; color: #718096; }
        """)
        reset_btn.clicked.connect(self.on_reset_defaults)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton { background-color: #4b5563; color: white; padding: 6px 14px; border-radius: 4px; }
            QPushButton:hover { background-color: #6b7280; }
            QPushButton:pressed { background-color: #374151; }
            QPushButton:disabled { background-color: #2d3748; color: #718096; }
        """)
        close_btn.clicked.connect(self.accept)

        btn_l2.addWidget(export_btn)
        btn_l2.addWidget(import_btn)
        btn_l2.addWidget(reset_btn)
        btn_l2.addStretch()
        btn_l2.addWidget(close_btn)
        layout.addLayout(btn_l2)
        self.refresh_list()

    def on_filter_text_changed(self):
        self.refresh_list(0)

    def get_real_index(self, row):
        if 0 <= row < len(self.filtered_indices):
            return self.filtered_indices[row]
        return -1

    def refresh_list(self, select_idx=0):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        self.filtered_indices.clear()
        f_tags = ["M", "BS", "J", "BS-J", "GM"]
        query = self.search_edit.text().strip().lower() if hasattr(self, 'search_edit') else ""

        for orig_idx, bm in enumerate(self.viewer.bookmarks):
            f_type_idx = bm.get("fractal_type", 0)
            tag_label = f_tags[f_type_idx] if 0 <= f_type_idx < len(f_tags) else 'M'
            tag = f"[{'PRESET' if bm.get('is_builtin', False) else 'CUSTOM'}:{tag_label}]"
            name = bm.get("name", "")
            desc = bm.get("description", "")
            fname = FRACTAL_NAMES[f_type_idx] if f_type_idx < len(FRACTAL_NAMES) else ""

            if query:
                searchable = f"{tag_label} {name} {desc} {fname}".lower()
                if query not in searchable:
                    continue

            self.filtered_indices.append(orig_idx)
            kf_count = len(bm.get("keyframes", []))
            self.list_widget.addItem(QListWidgetItem(f"{tag} {name}{f' | {kf_count} Waypoints' if kf_count > 0 else ''}"))

        if self.filtered_indices:
            select_idx = max(0, min(select_idx, len(self.filtered_indices) - 1))
            self.list_widget.setCurrentRow(select_idx)
        else:
            self.on_selection_changed(-1)
        self.list_widget.blockSignals(False)
        self.on_selection_changed(self.list_widget.currentRow())

    def on_selection_changed(self, row):
        real_idx = self.get_real_index(row)
        if real_idx < 0 or real_idx >= len(self.viewer.bookmarks):
            self.details_box.clear()
            self.load_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self.rename_btn.setEnabled(False)
            return

        bm = self.viewer.bookmarks[real_idx]
        self.load_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self.rename_btn.setEnabled(True)

        f_type_idx = bm.get("fractal_type", 0)
        extra_info = ""
        if f_type_idx in (2, 3):
            extra_info = f" | Julia: ({bm.get('julia_cx', '0.0')}, {bm.get('julia_cy', '0.0')})"
        elif f_type_idx == 4:
            try:
                kr = float(bm.get('gen_kr', bm.get('gen_k', 0.25)))
            except (ValueError, TypeError):
                kr = 0.25
            try:
                ki = float(bm.get('gen_ki', 1.0))
            except (ValueError, TypeError):
                ki = 1.0
            k_str = f"{kr:+.4f}{ki:+.4f}i" if ki != 0.0 else f"{kr:+.4f}"
            extra_info = f" | Formula: z^{bm.get('gen_n', 3)} + ({k_str})*z"

        try:
            dens = float(bm.get('color_density', 1.0))
        except (ValueError, TypeError):
            dens = 1.0
        try:
            cont = float(bm.get('color_contrast', 1.0))
        except (ValueError, TypeError):
            cont = 1.0

        self.details_box.setPlainText(
            f"Name: {bm.get('name', 'Unnamed')} [{FRACTAL_NAMES[f_type_idx] if f_type_idx < len(FRACTAL_NAMES) else 'Fractal'}]{extra_info}\n"
            f"Center: ({bm.get('center_x', '0.0')}, {bm.get('center_y', '0.0')}) | Plot Width: {bm.get('plot_width', '4.0')}\n"
            f"Colormap: {bm.get('cmap_name', 'inferno')} | Scheme: {bm.get('color_scheme_id', 0)} | Density: {dens:.2f} | Contrast: {cont:.2f} | Max Iter: {bm.get('max_iter', 1000)}\n"
            f"Precision: {bm.get('precision_mode', '1e-300 (Perturbation)')} | BBSA: {bm.get('bbsa_accuracy', '4th-order')} | Glitch Mode: {bm.get('glitch_mode', 'Off (Single-Ref)')} | Dynamic Iter: {bm.get('dynamic_iter_mode', 'Off')} | SSAA: {bm.get('ssaa_factor', 1)}x | AA Edge Tol: {bm.get('edge_threshold', 0.35)} | Waypoints: {len(bm.get('keyframes', []))}"
        )

    def on_load_selected(self):
        real_idx = self.get_real_index(self.list_widget.currentRow())
        if 0 <= real_idx < len(self.viewer.bookmarks):
            self.viewer.apply_state_dict(self.viewer.bookmarks[real_idx])
            self.accept()

    def on_add_current_view(self):
        name, ok = QInputDialog.getText(self, "New Bookmark", "Enter a title for this bookmark:")
        if ok and name.strip():
            state = self.viewer.get_current_state_dict(name=name.strip())
            state["is_builtin"] = False
            self.viewer.bookmarks.append(state)
            if self.search_edit.text().strip():
                self.search_edit.clear()
            self.refresh_list(len(self.viewer.bookmarks) - 1)

    def on_rename_bookmark(self):
        real_idx = self.get_real_index(self.list_widget.currentRow())
        if 0 <= real_idx < len(self.viewer.bookmarks):
            bm = self.viewer.bookmarks[real_idx]
            current_name = bm.get("name", "")
            new_name, ok = QInputDialog.getText(self, "Rename Bookmark", "Enter new name:", text=current_name)
            if ok and new_name.strip():
                bm["name"] = new_name.strip()
                bm["is_builtin"] = False
                self.refresh_list(self.list_widget.currentRow())

    def on_delete_bookmark(self):
        real_idx = self.get_real_index(self.list_widget.currentRow())
        if 0 <= real_idx < len(self.viewer.bookmarks):
            bm = self.viewer.bookmarks[real_idx]
            name = bm.get("name", "this bookmark")
            if QMessageBox.question(self, "Delete Bookmark", f"Delete '{name}'?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                del self.viewer.bookmarks[real_idx]
                self.refresh_list(max(0, self.list_widget.currentRow() - 1))

    def on_reset_defaults(self):
        if QMessageBox.question(
            self,
            "Reset Defaults",
            "Reset bookmarks list to the built-in presets?\nCustom bookmarks in the current list will be replaced.",
            QMessageBox.Yes | QMessageBox.No
        ) == QMessageBox.Yes:
            self.viewer.bookmarks = [dict(p, is_builtin=True) for p in load_builtin_presets()]
            self.refresh_list(0)

    def on_export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Bookmarks", os.path.join(getattr(self.viewer, 'last_export_dir', SCRIPT_DIR), "fractal_bookmarks.json"), "JSON (*.json)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"version": "1.0", "bookmarks": self.viewer.bookmarks}, f, indent=4)
                self.viewer.last_export_dir = os.path.dirname(path)
                QMessageBox.information(self, "Export Successful", f"Saved {len(self.viewer.bookmarks)} bookmarks.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def on_import_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Bookmarks", getattr(self.viewer, 'last_export_dir', SCRIPT_DIR), "JSON (*.json)")
        if path:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    imported = data.get("bookmarks", [])
                elif isinstance(data, list):
                    imported = data
                else:
                    imported = []
                self.viewer.bookmarks.clear()
                for bm in imported:
                    if isinstance(bm, dict) and "center_x" in bm:
                        bm["is_builtin"] = False
                        self.viewer.bookmarks.append(bm)
                self.viewer.last_export_dir = os.path.dirname(path)
                self.refresh_list(0)
                QMessageBox.information(self, "Import Successful", f"Loaded {len(self.viewer.bookmarks)} bookmarks.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))