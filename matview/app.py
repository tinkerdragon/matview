"""Application window: toolbar, live viewport, control panel, shortcuts."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QDesktopServices,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTabBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .controls import ControlPanel
from .export import ManimExporter, build_payload
from .viewport import Viewport

DISPLAY_FLAGS = [
    ("grid", "Grid"),
    ("reference", "Ghost grid (original space)"),
    ("basis", "Basis vectors"),
    ("unit", "Unit cell"),
    ("eigen", "Eigenvectors"),
    ("trails", "Motion trails"),
    ("labels", "Labels"),
]

STYLE = """
QMainWindow { background: #0A0E15; }
QWidget#panel { background: #0C1119; }
QToolBar { background: #0C1119; border-bottom: 1px solid #16222F; padding: 5px 8px; spacing: 6px; }
QToolBar QToolButton { background: transparent; border: 1px solid transparent; border-radius: 4px;
                       padding: 4px 9px; color: #A9B8C9; }
QToolBar QToolButton:hover { background: #16202C; }
QToolBar QToolButton:checked { background: #17303D; border-color: #58C4DD; color: #BFEAF5; }
QToolBar QToolButton:disabled { color: #3E4C5C; background: transparent; border-color: transparent; }
QPushButton { background: #16202C; border: 1px solid #22344A; border-radius: 4px;
              padding: 5px 9px; color: #C9D4E3; }
QPushButton:hover { background: #1D2A3A; border-color: #33506E; }
QPushButton:pressed { background: #101922; }
QPushButton:disabled { color: #4E5D70; border-color: #1A2532; }
QPushButton:checked { background: #17303D; border-color: #58C4DD; color: #BFEAF5; }
QPushButton#apply { background: #14424F; border: 1px solid #58C4DD; color: #D6F2FA; font-weight: 600; }
QPushButton#apply:hover { background: #1B5769; }
QGroupBox QPushButton { padding-top: 4px; padding-bottom: 4px; }
QGroupBox { border: 1px solid #1A2735; border-radius: 7px; margin-top: 16px;
            padding-top: 8px; background: #0C1119; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px;
                   padding: 0 4px; color: #6E8299; font-size: 10px; font-weight: 600; }
QDoubleSpinBox, QLineEdit, QComboBox { background: #080C12; border: 1px solid #1E2C3D;
                                       border-radius: 4px; padding: 3px 5px; color: #D6E2F0;
                                       selection-background-color: #2B6C7E; }
QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus { border-color: #58C4DD; }
QDoubleSpinBox QLineEdit { color: #D6E2F0; background: transparent; border: none; padding: 0; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { width: 0px; border: none; background: transparent; }
QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow { width: 0px; height: 0px; image: none; }
QComboBox QAbstractItemView { background: #0C1119; border: 1px solid #22344A;
                              selection-background-color: #17303D; }
QListWidget { background: #080C12; border: 1px solid #1E2C3D; border-radius: 4px;
              color: #B9C7D6; }
QListWidget::item { padding: 3px 6px; border-radius: 3px; }
QListWidget::item:selected { background: #17303D; color: #E6F3F8; }
QCheckBox { color: #B9C7D6; spacing: 6px; }
QSlider::groove:horizontal { height: 4px; background: #1A2735; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #2B6C7E; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; height: 12px; margin: -5px 0;
                             border-radius: 6px; background: #58C4DD; }
QScrollArea { border: none; background: #0C1119; }
QScrollBar:vertical { background: #0C1119; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #243546; min-height: 30px; border-radius: 5px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QStatusBar { background: #080C12; color: #6E8299; border-top: 1px solid #141F2B; }
QLabel { color: #B9C7D6; }
QLabel#muted { color: #6E8299; font-size: 11px; }
QLabel#dimLabel { color: #8AA0B6; font-family: Consolas; }
QLabel#title { color: #7D8FA6; font-size: 11px; font-weight: 700; }
QToolTip { background: #16202C; color: #C9D4E3; border: 1px solid #2A3E55; padding: 3px; }
QMenu { background: #0C1119; border: 1px solid #22344A; padding: 4px; color: #C9D4E3; }
QMenu::item { padding: 6px 28px 6px 10px; border-radius: 4px; }
QMenu::item:selected { background: #17303D; }
QMenu::item:disabled { color: #4E5D70; }
QMenu::separator { height: 1px; background: #1E2C3D; margin: 4px 8px; }
QWidget#bottombar { background: #0C1119; border-top: 1px solid #16222F; }
QTabBar::tab { background: #0C1119; color: #8FA3B8; padding: 4px 12px; margin-right: 2px;
               border: 1px solid #1E2C3D; border-radius: 4px; }
QTabBar::tab:selected { background: #17303D; color: #E6F3F8; border-color: #58C4DD; }
QTabBar::tab:hover { background: #16202C; }
QWidget#bottombar QToolButton { background: transparent; border: 1px solid #22344A;
                                border-radius: 4px; padding: 1px 7px; color: #A9B8C9; }
QWidget#bottombar QToolButton:hover { background: #16202C; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Matview")
        self.resize(1380, 880)
        self.setMinimumSize(1080, 700)

        self._quality = "m"
        self._last_video: str | None = None
        self._sessions: list[tuple[QWidget, Viewport, ControlPanel]] = []
        self._session_counter = 0
        self._ready = False
        self._sync_view = False
        self.exporter = ManimExporter(self)

        self.stack = QStackedWidget()
        self.tabs = QTabBar()
        self.tabs.setExpanding(False)
        self.tabs.setDrawBase(False)
        self.tabs.setTabsClosable(True)
        self.tabs.currentChanged.connect(self._switch_session)
        self.tabs.tabCloseRequested.connect(self._close_session)

        bottom = QWidget()
        bottom.setObjectName("bottombar")
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(8, 4, 8, 4)
        bottom_layout.setSpacing(4)
        bottom_layout.addWidget(self.tabs)
        new_button = QToolButton()
        new_button.setText("+")
        new_button.setToolTip("New render instance")
        new_button.clicked.connect(lambda: self._new_session())
        bottom_layout.addWidget(new_button)
        close_button = QToolButton()
        close_button.setText("×")
        close_button.setToolTip("Close this render instance")
        close_button.clicked.connect(lambda: self._close_session(self.tabs.currentIndex()))
        bottom_layout.addWidget(close_button)
        bottom_layout.addStretch(1)

        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.stack, 1)
        outer.addWidget(bottom)
        self.setCentralWidget(central)

        self._new_session()
        self.setStyleSheet(STYLE)
        self._build_toolbar()
        self._build_shortcuts()
        self.exporter.logLine.connect(self._export_log)
        self.exporter.finished.connect(self._on_export_finished)

        self.hint = QLabel()
        self.hint.setObjectName("muted")
        self.statusBar().addPermanentWidget(self.hint)
        self.fps_label = QLabel("— fps")
        self.fps_label.setObjectName("muted")
        self.fps_label.setMinimumWidth(96)
        self.statusBar().addPermanentWidget(self.fps_label)
        self.statusBar().showMessage("Ready — edit a matrix, then Apply.", 4000)

        self._ready = True
        self._sync_toolbar()
        self._update_hint()

    # --------------------------------------------------------------- sessions
    def _new_session(self) -> None:
        viewport = Viewport()
        panel = ControlPanel(viewport)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setFixedWidth(396)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(viewport, 1)
        layout.addWidget(scroll)

        self._sessions.append((container, viewport, panel))
        self.stack.addWidget(container)
        self._session_counter += 1
        self.tabs.addTab(f"Space {self._session_counter}")

        viewport.statusMessage.connect(self._session_status)
        viewport.fpsChanged.connect(self._session_fps)
        viewport.cameraChanged.connect(self._session_camera_changed)
        viewport.stateChanged.connect(self._session_changed)

        if self._sync_view and getattr(self, "viewport", None) is not None:
            self._sync_camera_to(self.viewport, viewport)

        index = self.tabs.count() - 1
        self.tabs.setCurrentIndex(index)
        self._switch_session(index)
        if self._ready:
            self.statusBar().showMessage(f"New render instance · Space {self._session_counter}", 2500)

    def _switch_session(self, index: int) -> None:
        if not (0 <= index < len(self._sessions)):
            return
        self.stack.setCurrentIndex(index)
        _, viewport, panel = self._sessions[index]
        previous = getattr(self, "viewport", None)
        if self._sync_view and previous is not None and previous is not viewport:
            self._sync_camera_to(previous, viewport)
        self.viewport = viewport
        self.panel = panel
        if not self._ready:
            return
        self.mode_2d.setChecked(viewport.dim == 2)
        self.mode_3d.setChecked(viewport.dim == 3)
        self._sync_toolbar()
        self._update_hint()
        if viewport.timeline is not None:
            self.fps_label.setText("— fps")
            self.fps_label.setStyleSheet("color:#6E8299")

    def _close_session(self, index: int) -> None:
        if len(self._sessions) <= 1:
            if self._ready:
                self.statusBar().showMessage("At least one render instance must stay.", 3000)
            return
        if not (0 <= index < len(self._sessions)):
            return
        active = self.tabs.currentIndex()
        container, viewport, panel = self._sessions.pop(index)
        self.stack.removeWidget(container)
        container.deleteLater()
        self.tabs.blockSignals(True)
        self.tabs.removeTab(index)
        self.tabs.blockSignals(False)
        if index < active:
            active -= 1
        active = max(0, min(active, len(self._sessions) - 1))
        self.tabs.setCurrentIndex(active)
        self._switch_session(active)

    def _session_status(self, message: str) -> None:
        if self.sender() is self.viewport:
            self._show_status(message)

    def _session_fps(self, value: float) -> None:
        if self.sender() is self.viewport:
            self._on_fps(value)

    def _session_changed(self) -> None:
        if self.sender() is self.viewport:
            self._sync_toolbar()

    def _session_camera_changed(self) -> None:
        if self.sender() is not self.viewport:
            return
        if self._sync_view:
            self._push_camera(self.viewport)
        self._sync_toolbar()

    def _toggle_sync_view(self, enabled: bool) -> None:
        self._sync_view = bool(enabled)
        if self._sync_view:
            self._push_camera(self.viewport)
            self.statusBar().showMessage("Perspective sync on.", 2500)
        else:
            self.statusBar().showMessage("Perspective sync off.", 2500)

    def _push_camera(self, source: Viewport) -> None:
        for container, viewport, panel in self._sessions:
            if viewport is not source:
                self._sync_camera_to(source, viewport)

    @staticmethod
    def _sync_camera_to(source: Viewport, target: Viewport) -> None:
        src, dst = source.camera, target.camera
        dst.target[:] = src.target
        if target.dim == 3:
            if source.dim == 3:
                dst.yaw = src.yaw
                dst.pitch = src.pitch
                dst.ortho = src.ortho
                dst.ortho_height = src.ortho_height
                dst.distance = src.distance
                dst.fov = src.fov
            else:
                if dst.ortho:
                    dst.ortho_height = src.ortho_height
                else:
                    dst.distance = src.ortho_height / (2.0 * math.tan(dst.fov * 0.5))
        else:
            if source.dim == 2 or src.ortho:
                dst.ortho_height = src.ortho_height
            else:
                dst.ortho_height = 2.0 * src.distance * math.tan(src.fov * 0.5)
        target._dirty = True
        target.update()

    # ------------------------------------------------------------------ chrome
    def _build_toolbar(self) -> None:
        toolbar = QToolBar("View")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(toolbar)

        title = QLabel("MATVIEW")
        title.setObjectName("title")
        toolbar.addWidget(title)
        toolbar.addSeparator()

        self.mode_2d = QPushButton("2D")
        self.mode_3d = QPushButton("3D")
        for button in (self.mode_2d, self.mode_3d):
            button.setCheckable(True)
            button.setFixedWidth(46)
            toolbar.addWidget(button)
        self.mode_2d.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.mode_2d, 2)
        self.mode_group.addButton(self.mode_3d, 3)
        self.mode_group.idClicked.connect(self.set_mode)

        toolbar.addSeparator()
        self.camera_buttons: dict[str, QToolButton] = {}
        for name, label in (
            ("iso", "ISO"),
            ("top", "TOP"),
            ("front", "FRONT"),
            ("right", "RIGHT"),
        ):
            button = QToolButton()
            button.setText(label)
            button.setToolTip(f"{label.title()} view")
            button.clicked.connect(lambda _=False, n=name: self.viewport.set_camera_view(n))
            toolbar.addWidget(button)
            self.camera_buttons[name] = button

        self.reset_button = QToolButton()
        self.reset_button.setText("RESET VIEW")
        self.reset_button.setToolTip("Reset camera (R)")
        self.reset_button.clicked.connect(self.viewport.reset_camera)
        toolbar.addWidget(self.reset_button)

        toolbar.addSeparator()
        self.ortho_button = QToolButton()
        self.ortho_button.setText("ORTHO")
        self.ortho_button.setCheckable(True)
        self.ortho_button.setToolTip("Toggle orthographic / perspective projection (F)")
        self.ortho_button.clicked.connect(self.viewport.toggle_projection)
        toolbar.addWidget(self.ortho_button)

        toolbar.addSeparator()
        self.sync_button = QToolButton()
        self.sync_button.setText("SYNC VIEW")
        self.sync_button.setCheckable(True)
        self.sync_button.setToolTip(
            "Sync the camera across render instances: 3D viewpoints fully, "
            "2D tabs share pan and zoom"
        )
        self.sync_button.toggled.connect(self._toggle_sync_view)
        toolbar.addWidget(self.sync_button)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)

        self.settings_button = QToolButton()
        self.settings_button.setText("SETTINGS")
        self.settings_button.setToolTip("Display settings")
        self.settings_button.setPopupMode(QToolButton.InstantPopup)
        settings_menu = QMenu(self.settings_button)
        self.flag_actions: dict[str, QAction] = {}
        for key, label in DISPLAY_FLAGS:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(bool(self.viewport.flags.get(key, False)))
            action.toggled.connect(lambda checked, k=key: self._set_flag(k, checked))
            settings_menu.addAction(action)
            self.flag_actions[key] = action
        self.settings_button.setMenu(settings_menu)
        toolbar.addWidget(self.settings_button)

        self.export_button = QToolButton()
        self.export_button.setText("EXPORT")
        self.export_button.setToolTip("Render with Manim or save a snapshot")
        self.export_button.setPopupMode(QToolButton.InstantPopup)
        export_menu = QMenu(self.export_button)
        self.render_action = export_menu.addAction("Render MP4")
        self.render_action.triggered.connect(self._start_export)
        self.cancel_action = export_menu.addAction("Cancel render")
        self.cancel_action.setEnabled(False)
        self.cancel_action.triggered.connect(self._cancel_export)
        quality_menu = export_menu.addMenu("Quality")
        self.quality_group = QActionGroup(self)
        self.quality_group.setExclusive(True)
        for label, key in (
            ("Draft · 480p", "l"),
            ("Standard · 720p", "m"),
            ("High · 1080p", "h"),
        ):
            action = quality_menu.addAction(label)
            action.setCheckable(True)
            action.setData(key)
            action.setChecked(key == self._quality)
            action.triggered.connect(lambda _=False, k=key: self._set_quality(k))
            self.quality_group.addAction(action)
        export_menu.addSeparator()
        self.png_action = export_menu.addAction("Save PNG snapshot…")
        self.png_action.triggered.connect(self._save_png)
        self.open_action = export_menu.addAction("Open rendered video")
        self.open_action.setEnabled(False)
        self.open_action.triggered.connect(self._open_video)
        self.export_button.setMenu(export_menu)
        toolbar.addWidget(self.export_button)

    def _build_shortcuts(self) -> None:
        bindings = [
            ("1", lambda: self.set_mode(2)),
            ("2", lambda: self.set_mode(3)),
            ("Return", self.panel.apply_pending),
            ("Enter", self.panel.apply_pending),
            ("Space", self.viewport.replay),
            ("Ctrl+Z", lambda: self.viewport.remove_step(len(self.viewport.stack))),
            ("R", self.viewport.reset_camera),
            ("F", self.viewport.toggle_projection),
            ("G", lambda: self._toggle_flag("grid")),
            ("B", lambda: self._toggle_flag("basis")),
            ("U", lambda: self._toggle_flag("unit")),
            ("E", lambda: self._toggle_flag("eigen")),
            ("T", lambda: self._toggle_flag("trails")),
        ]
        self._shortcuts = []
        for keys, slot in bindings:
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)

    # ------------------------------------------------------------------ state
    def set_mode(self, dim: int) -> None:
        self.mode_2d.setChecked(dim == 2)
        self.mode_3d.setChecked(dim == 3)
        self.viewport.set_dim(dim)
        self._sync_toolbar()
        self._update_hint()

    def _sync_toolbar(self) -> None:
        if not self._ready:
            return
        three = self.viewport.dim == 3
        for button in self.camera_buttons.values():
            button.setEnabled(three)
        self.ortho_button.setEnabled(three)
        self.ortho_button.setChecked(self.viewport.camera.ortho)
        for key, action in getattr(self, "flag_actions", {}).items():
            checked = bool(self.viewport.flags.get(key, False))
            if action.isChecked() != checked:
                action.blockSignals(True)
                action.setChecked(checked)
                action.blockSignals(False)
        if self.viewport.timeline is None:
            idle_text = f"idle · {self.viewport.target_fps:.0f} Hz"
            if self.fps_label.text() != idle_text:
                self.fps_label.setText(idle_text)
                self.fps_label.setStyleSheet("color:#6E8299")

    def _update_hint(self) -> None:
        if self.viewport.dim == 3:
            text = "Drag orbit  ·  Shift+drag pan  ·  Wheel zoom  ·  Double-click reset"
        else:
            text = "Drag pan  ·  Wheel zoom  ·  Double-click reset"
        self.hint.setText(text)

    def _set_flag(self, name: str, value: bool) -> None:
        self.viewport.set_flag(name, value)
        action = self.flag_actions.get(name)
        if action is not None and action.isChecked() != value:
            action.blockSignals(True)
            action.setChecked(value)
            action.blockSignals(False)

    def _toggle_flag(self, name: str) -> None:
        self._set_flag(name, not self.viewport.flags.get(name, False))

    def _set_quality(self, key: str) -> None:
        self._quality = key
        for action in self.quality_group.actions():
            action.blockSignals(True)
            action.setChecked(action.data() == key)
            action.blockSignals(False)

    def _start_export(self) -> None:
        if self.exporter.busy():
            return
        payload = build_payload(self.viewport)
        if payload is None:
            self._show_status("Apply at least one matrix before rendering.")
            return
        self.render_action.setEnabled(False)
        self.cancel_action.setEnabled(True)
        self.open_action.setEnabled(False)
        self._show_status("Rendering with Manim…")
        self.exporter.render(payload, self._quality)

    def _cancel_export(self) -> None:
        self.exporter.cancel()

    def _export_log(self, line: str) -> None:
        line = line.strip()
        if line:
            self._show_status(line[-160:])

    def _on_export_finished(self, ok: bool, path: str) -> None:
        self.render_action.setEnabled(True)
        self.cancel_action.setEnabled(False)
        if ok:
            self._last_video = path
            self.open_action.setEnabled(True)
            self._show_status(f"Rendered → {path}")
        else:
            self._show_status(f"Render failed: {path}")

    def _save_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save snapshot", "matview-snapshot.png", "PNG image (*.png)"
        )
        if not path:
            return
        image = self.viewport.snapshot()
        if image.save(path):
            self._show_status(f"Saved {path}")
        else:
            self._show_status("Could not save the snapshot.")

    def _open_video(self) -> None:
        if self._last_video:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._last_video))

    def _show_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 6000)

    def _on_fps(self, value: float) -> None:
        if self.viewport.timeline is None:
            return
        target = self.viewport.target_fps
        self.fps_label.setText(f"{value:3.0f} / {target:.0f} fps")
        color = "#83C167" if value >= target * 0.85 else "#F0AC5F"
        self.fps_label.setStyleSheet(f"color:{color}")
