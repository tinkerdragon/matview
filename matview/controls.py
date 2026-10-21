"""Right-hand control panel: matrix entry, rotation, space states, vectors."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from . import core
from .expr import evaluate
from .viewport import VectorSpec, Viewport



class MatrixDisplay(QWidget):
    """Small bracket-rendered matrix with sign coloured entries."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.matrix = np.eye(2)
        self.setMinimumHeight(44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_matrix(self, matrix: np.ndarray) -> None:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape == self.matrix.shape and np.allclose(matrix, self.matrix, atol=4e-3):
            return
        self.matrix = matrix.copy()
        self.update()

    def sizeHint(self) -> QSize:
        rows = self.matrix.shape[0]
        return QSize(230, 20 * rows + 10)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        font = QFont("Cascadia Mono", 10)
        font.setStyleHint(QFont.Monospace)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        rows, cols = self.matrix.shape
        cell_w = max(metrics.horizontalAdvance("-0.00") + 12, 52)
        cell_h = metrics.height() + 5
        block_w = cols * cell_w
        block_h = rows * cell_h
        x0 = (self.width() - block_w) * 0.5
        y0 = (self.height() - block_h) * 0.5

        bracket = QPen(QColor("#49637D"), 1.6)
        bracket.setCapStyle(Qt.RoundCap)
        painter.setPen(bracket)
        arm = 5.0
        painter.drawLine(x0 - 9, y0 - 5, x0 - 9, y0 + block_h + 5)
        painter.drawLine(x0 - 9, y0 - 5, x0 - 9 + arm, y0 - 5)
        painter.drawLine(x0 - 9, y0 + block_h + 5, x0 - 9 + arm, y0 + block_h + 5)
        right = x0 + block_w + 9
        painter.drawLine(right, y0 - 5, right, y0 + block_h + 5)
        painter.drawLine(right - arm, y0 - 5, right, y0 - 5)
        painter.drawLine(right - arm, y0 + block_h + 5, right, y0 + block_h + 5)

        for r in range(rows):
            for c in range(cols):
                value = float(self.matrix[r, c])
                if abs(value) < 5e-3:
                    color = QColor("#5C6B7A")
                    text = "0.00"
                elif value > 0:
                    color = QColor("#9CDCFE")
                    text = f"{value:.2f}"
                else:
                    color = QColor("#F48771")
                    text = f"{value:.2f}"
                painter.setPen(color)
                cell = (
                    x0 + c * cell_w,
                    y0 + r * cell_h,
                    cell_w,
                    cell_h,
                )
                painter.drawText(
                    int(cell[0]),
                    int(cell[1]),
                    int(cell[2]),
                    int(cell[3]),
                    Qt.AlignCenter,
                    text,
                )


class MatrixField(QLineEdit):
    """Matrix entry accepting a number or an arithmetic expression."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(28)
        self.setFont(QFont("Consolas", 10))
        self.setToolTip(
            "Number or expression: sqrt(2)/2, pi/4, 2**0.5, cos(pi/3)\n"
            "Enter applies the matrix."
        )


class ControlPanel(QWidget):
    def __init__(self, viewport: Viewport, parent=None):
        super().__init__(parent)
        self.viewport = viewport
        self._syncing = False
        self._vector_rows = []
        self._vector_signature = None
        self._stack_signature = None
        self._dim = viewport.dim
        self._invalid_fields: list[tuple[int, int]] = []
        self._invalid_key = None

        self.setObjectName("panel")
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(10, 10, 10, 10)
        self._root.setSpacing(8)

        self._build_matrix()
        self._build_rotation()
        self._build_stack()
        self._build_vectors()
        self._root.addStretch(1)

        viewport.stateChanged.connect(self.refresh)
        viewport.frameChanged.connect(self.refresh_live)
        self.refresh()

    # ------------------------------------------------------------------ build
    def _group(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(9, 12, 9, 8)
        layout.setSpacing(5)
        self._root.addWidget(box)
        return box, layout

    def _build_matrix(self) -> None:
        box, layout = self._group("TRANSFORM MATRIX")
        self.det_pending = QLabel("det = +1.00")
        self.det_pending.setObjectName("dimLabel")
        self.spin_host = QWidget()
        self.spin_grid = QGridLayout(self.spin_host)
        self.spin_grid.setContentsMargins(0, 0, 0, 0)
        self.spin_grid.setSpacing(5)
        layout.addWidget(self.spin_host)

        self.spins: list[list[QDoubleSpinBox]] = []
        self._rebuild_spins()

        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(self.det_pending, 1)
        for text, tip, slot in (
            ("Aᵀ", "Transpose the pending matrix", self._transpose),
            ("A⁻¹", "Invert the pending matrix", self._invert),
            ("I", "Reset the pending matrix to the identity", self._identity),
        ):
            button = QPushButton(text)
            button.setFixedWidth(36)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            row.addWidget(button)
        layout.addLayout(row)

        self.apply_button = QPushButton("Apply matrix      ⏎")
        self.apply_button.setObjectName("apply")
        self.apply_button.setMinimumHeight(32)
        self.apply_button.clicked.connect(self._apply)
        layout.addWidget(self.apply_button)

    def _build_rotation(self) -> None:
        box, layout = self._group("ROTATION")
        rotate_row = QHBoxLayout()
        rotate_row.setSpacing(6)
        rotate_row.addWidget(QLabel("Rotate"))
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setRange(-180, 180)
        self.angle_slider.setValue(0)
        self.angle_slider.valueChanged.connect(self._rotation_changed)
        rotate_row.addWidget(self.angle_slider, 1)
        self.angle_label = QLabel("0°")
        self.angle_label.setFixedWidth(38)
        self.angle_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        rotate_row.addWidget(self.angle_label)
        layout.addLayout(rotate_row)

        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["x axis", "y axis", "z axis"])
        self.axis_combo.setCurrentIndex(2)
        self.axis_combo.currentIndexChanged.connect(lambda _: self._rotation_changed(self.angle_slider.value()))
        layout.addWidget(self.axis_combo)
        self.axis_combo.setVisible(self.viewport.dim == 3)

    def _build_stack(self) -> None:
        box, layout = self._group("STATES  ·  CURRENT SPACE")
        self.matrix_display = MatrixDisplay()
        layout.addWidget(self.matrix_display)
        self.det_label = QLabel("det = +1.00")
        layout.addWidget(self.det_label)
        self.eigen_label = QLabel("λ = 1.00, 1.00")
        self.eigen_label.setWordWrap(True)
        self.eigen_label.setObjectName("muted")
        layout.addWidget(self.eigen_label)

        self.stack_list = QListWidget()
        self.stack_list.setFont(QFont("Consolas", 9))
        self.stack_list.setMinimumHeight(74)
        self.stack_list.setMaximumHeight(82)
        self.stack_list.itemSelectionChanged.connect(self._stack_selection_changed)
        self.stack_list.itemDoubleClicked.connect(lambda _: self._remove_selected())
        layout.addWidget(self.stack_list)

        row = QHBoxLayout()
        row.setSpacing(5)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self._remove_selected)
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(lambda: self.viewport.remove_step(len(self.viewport.stack)))
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.viewport.clear_stack)
        for button in (self.remove_button, self.undo_button, self.clear_button):
            row.addWidget(button)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Duration"))
        self.duration_slider = QSlider(Qt.Horizontal)
        self.duration_slider.setRange(200, 3000)
        self.duration_slider.setValue(1000)
        self.duration_slider.valueChanged.connect(self._duration_changed)
        row.addWidget(self.duration_slider, 1)
        self.duration_label = QLabel("1.00 s")
        self.duration_label.setFixedWidth(44)
        self.duration_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.duration_label)
        layout.addLayout(row)

        replay = QPushButton("Replay from identity   (Space)")
        replay.clicked.connect(self.viewport.replay)
        layout.addWidget(replay)

    def _build_vectors(self) -> None:
        box, layout = self._group("VECTORS")
        self.vector_host = QWidget()
        self.vector_layout = QVBoxLayout(self.vector_host)
        self.vector_layout.setContentsMargins(0, 0, 0, 0)
        self.vector_layout.setSpacing(4)
        layout.addWidget(self.vector_host)
        add = QPushButton("+ Add vector")
        add.clicked.connect(self._add_vector)
        layout.addWidget(add)

    # ------------------------------------------------------------------ state
    def refresh(self) -> None:
        if self.viewport.dim != self._dim:
            self._dim = self.viewport.dim
            self._rebuild_spins()
            self.axis_combo.setVisible(self.viewport.dim == 3)
        self._sync_editor()
        editor_matrix, editor_ok = self._read_editor()
        if editor_ok and self.angle_slider.value() != 0 and np.allclose(
            editor_matrix, np.eye(self.viewport.dim), atol=1e-9
        ):
            self.angle_slider.blockSignals(True)
            self.angle_slider.setValue(0)
            self.angle_slider.blockSignals(False)
            self.angle_label.setText("0°")
        self.refresh_live()
        self._refresh_stack()
        self._refresh_vectors()
        self._refresh_stack_buttons()

    def refresh_live(self) -> None:
        displayed = self.viewport.displayed
        self.matrix_display.set_matrix(displayed)
        determinant = core.determinant(displayed)
        if abs(determinant) < 5e-3:
            color, note = "#F48771", "space collapsed"
        elif determinant > 0:
            color, note = "#83C167", "orientation preserved"
        else:
            color, note = "#F48771", "orientation flipped"
        unit = "volume" if self.viewport.dim == 3 else "area"
        det_text = (
            f"<span style='color:{color}'>det = {determinant:+.2f}</span>"
            f" <span style='color:#7D8FA6'>· {unit} ×{abs(determinant):.2f} · {note}</span>"
        )
        if det_text != getattr(self, "_det_text", None):
            self._det_text = det_text
            self.det_label.setText(det_text)
            self.det_label.setTextFormat(Qt.RichText)

        values = core.eigen_data(displayed)
        real = [datum for datum in values if datum.is_real]
        complex_values = [datum for datum in values if not datum.is_real]
        parts = [f"{datum.value.real:+.2f}" for datum in real]
        if complex_values:
            pair = complex_values[0]
            parts.append(f"{pair.value.real:+.2f} ± {abs(pair.value.imag):.2f}i")
        eigen_text = "λ = " + ",  ".join(parts) if parts else "λ = —"
        if eigen_text != getattr(self, "_eigen_text", None):
            self._eigen_text = eigen_text
            self.eigen_label.setText(eigen_text)

    def _refresh_stack(self) -> None:
        signature = tuple(
            (round(float(value), 6) for value in matrix.flatten()) for matrix in self.viewport.stack
        )
        if signature == self._stack_signature:
            return
        self._stack_signature = signature
        selected = self.viewport.active
        states = core.cumulative_states(self.viewport.stack, self.viewport.dim)
        self.stack_list.blockSignals(True)
        self.stack_list.clear()
        for index, state in enumerate(states):
            label = "original" if index == 0 else core.geometric_series(
                self.viewport.stack[index - 1]
            )
            determinant = core.determinant(state)
            item = QListWidgetItem(f"{index:02d}  {label:<11} det {determinant:+.2f}")
            item.setToolTip(core.pretty(state, 2))
            self.stack_list.addItem(item)
        if 0 <= selected < self.stack_list.count():
            self.stack_list.setCurrentRow(selected)
        self.stack_list.blockSignals(False)
        self._refresh_stack_buttons()

    def _refresh_stack_buttons(self) -> None:
        count = len(self.viewport.stack)
        self.undo_button.setEnabled(count > 0)
        self.clear_button.setEnabled(count > 0)
        self.remove_button.setEnabled(count > 0 and self.stack_list.currentRow() >= 1)

    def _refresh_vectors(self) -> None:
        signature = (self.viewport.dim, len(self.viewport.vectors))
        if signature == self._vector_signature:
            return
        self._vector_signature = signature
        while self.vector_layout.count():
            item = self.vector_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._vector_rows = []
        for index, spec in enumerate(self.viewport.vectors):
            self._vector_rows.append(self._vector_row(index, spec))

    def _vector_row(self, index: int, spec: VectorSpec) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        swatch = QPushButton()
        swatch.setFixedSize(17, 17)
        swatch.setToolTip("Change colour")
        swatch.setStyleSheet(
            f"background:{spec.color}; border:1px solid #2A3A50; border-radius:3px;"
        )
        swatch.clicked.connect(lambda _, i=index: self._pick_color(i))
        layout.addWidget(swatch)

        name = QLineEdit(spec.label)
        name.setFixedWidth(36)
        name.setMaxLength(4)
        name.setAlignment(Qt.AlignCenter)
        name.textChanged.connect(lambda text, i=index: self._vector_text(i, text))
        layout.addWidget(name)

        for axis in range(self.viewport.dim):
            spin = QDoubleSpinBox()
            spin.setRange(-20, 20)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setAlignment(Qt.AlignCenter)
            spin.setValue(float(spec.coords[axis]))
            spin.setMinimumWidth(54)
            spin.setToolTip("xyz"[axis])
            spin.valueChanged.connect(
                lambda value, i=index, a=axis: self._vector_value(i, a, value)
            )
            self._style_spin(spin)
            layout.addWidget(spin, 1)

        remove = QPushButton("×")
        remove.setFixedWidth(20)
        remove.setToolTip("Delete this vector")
        remove.clicked.connect(lambda _, i=index: self._remove_vector(i))
        layout.addWidget(remove)
        row.setParent(self.vector_host)
        self.vector_layout.addWidget(row)
        return row

    def _style_spin(self, spin: QDoubleSpinBox) -> None:
        """Colour the internal line edit explicitly: Qt does not always inherit
        the stylesheet colour into QAbstractSpinBox's child editor."""
        editor = spin.lineEdit()
        palette = editor.palette()
        palette.setColor(QPalette.Text, QColor("#D6E2F0"))
        palette.setColor(QPalette.Base, QColor("#080C12"))
        editor.setPalette(palette)
        editor.setStyleSheet("color:#D6E2F0; background:transparent; border:none; padding:0;")

    # ----------------------------------------------------------------- matrix
    def _rebuild_spins(self) -> None:
        while self.spin_grid.count():
            item = self.spin_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.spins = []
        dim = self.viewport.dim
        for r in range(dim):
            row = []
            for c in range(dim):
                field = MatrixField()
                field.textChanged.connect(self._editor_changed)
                field.installEventFilter(self)
                self.spin_grid.addWidget(field, r, c)
                row.append(field)
            self.spins.append(row)
        self._sync_editor()

    @staticmethod
    def _format_value(value: float) -> str:
        if abs(value) < 5e-7:
            return "0"
        return f"{value:.5g}"

    def _sync_editor(self) -> None:
        self._syncing = True
        for r, row in enumerate(self.spins):
            for c, field in enumerate(row):
                field.blockSignals(True)
                field.setText(self._format_value(float(self.viewport.editor[r, c])))
                field.setStyleSheet("")
                field.blockSignals(False)
        self._syncing = False
        self._invalid_fields = []
        self._invalid_key = None
        self._update_pending_det(self.viewport.editor)

    def _write_editor(self, matrix: np.ndarray) -> None:
        matrix = core.embed(np.asarray(matrix, dtype=float), self.viewport.dim)
        self._write_fields(matrix)
        self._editor_changed()

    def _write_fields(self, matrix: np.ndarray) -> None:
        self._syncing = True
        for r, row in enumerate(self.spins):
            for c, field in enumerate(row):
                field.blockSignals(True)
                field.setText(self._format_value(float(matrix[r, c])))
                field.setStyleSheet("")
                field.blockSignals(False)
        self._syncing = False
        self._invalid_fields = []
        self._invalid_key = None

    def _read_editor(self, mark: bool = True) -> tuple[np.ndarray, bool]:
        dim = self.viewport.dim
        matrix = np.zeros((dim, dim))
        invalid: list[tuple[int, int]] = []
        for r, row in enumerate(self.spins):
            for c, field in enumerate(row):
                try:
                    matrix[r, c] = evaluate(field.text())
                except ValueError:
                    invalid.append((r, c))
        if mark:
            self._mark_invalid(invalid)
        return matrix, not invalid

    def _mark_invalid(self, invalid: list[tuple[int, int]]) -> None:
        key = frozenset(invalid)
        self._invalid_fields = list(invalid)
        if key == self._invalid_key:
            return
        self._invalid_key = key
        error_style = (
            "QLineEdit { color:#F48771; border:1px solid #F48771; "
            "background:#080C12; border-radius:4px; padding:3px 5px; }"
        )
        for r, row in enumerate(self.spins):
            for c, field in enumerate(row):
                field.setStyleSheet(error_style if (r, c) in key else "")

    def _editor_changed(self) -> None:
        if self._syncing:
            return
        matrix, ok = self._read_editor()
        if not ok:
            self._update_pending_det(None)
            return
        self.viewport.set_editor(matrix)
        self._update_pending_det(matrix)

    def _update_pending_det(self, matrix: np.ndarray | None) -> None:
        if matrix is None:
            self.det_pending.setText("det = —")
        else:
            self.det_pending.setText(f"det = {core.determinant(matrix):+.2f}")

    def eventFilter(self, obj, event) -> bool:
        # Enter inside a matrix cell applies the pending matrix and returns to
        # the first cell, so a second matrix can be typed immediately.
        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.apply_pending()
            return True
        return super().eventFilter(obj, event)

    def apply_pending(self) -> None:
        self._apply()

    def _apply(self) -> None:
        matrix, ok = self._read_editor()
        if not ok:
            self.viewport.statusMessage.emit("Fix the highlighted matrix entry first.")
            self._focus_first_invalid()
            return
        self.viewport.set_editor(matrix)
        self.viewport.apply_editor()
        QTimer.singleShot(0, self._focus_first_entry)

    def _focus_first_entry(self) -> None:
        if self.spins:
            field = self.spins[0][0]
            field.setFocus(Qt.OtherFocusReason)
            field.selectAll()

    def _focus_first_invalid(self) -> None:
        if self._invalid_fields:
            r, c = self._invalid_fields[0]
            field = self.spins[r][c]
            field.setFocus(Qt.OtherFocusReason)
            field.selectAll()

    def _identity(self) -> None:
        self._write_editor(np.eye(self.viewport.dim))
        self.angle_slider.blockSignals(True)
        self.angle_slider.setValue(0)
        self.angle_slider.blockSignals(False)
        self.angle_label.setText("0°")

    def _transpose(self) -> None:
        matrix, ok = self._read_editor()
        if not ok:
            self.viewport.statusMessage.emit("Fix the highlighted matrix entry first.")
            return
        self._write_editor(matrix.T)

    def _invert(self) -> None:
        matrix, ok = self._read_editor()
        if not ok:
            self.viewport.statusMessage.emit("Fix the highlighted matrix entry first.")
            return
        if abs(core.determinant(matrix)) < 1e-6:
            self.viewport.statusMessage.emit("A singular matrix has no inverse.")
            return
        self._write_editor(np.linalg.inv(matrix))

    def _rotation_changed(self, value: int) -> None:
        self.angle_label.setText(f"{value}°")
        if self._syncing:
            return
        axis = 2 if self.viewport.dim == 2 else self.axis_combo.currentIndex()
        self._write_fields(core.rotation_matrix(self.viewport.dim, value, axis))
        self._editor_changed()

    # ---------------------------------------------------------------- vectors
    def _vector_spec(self, index: int) -> VectorSpec | None:
        if 0 <= index < len(self.viewport.vectors):
            return self.viewport.vectors[index]
        return None

    def _vector_value(self, index: int, axis: int, value: float) -> None:
        spec = self._vector_spec(index)
        if spec is None:
            return
        spec.coords[axis] = value
        self.viewport.set_vectors(self.viewport.vectors)

    def _vector_text(self, index: int, text: str) -> None:
        spec = self._vector_spec(index)
        if spec is None:
            return
        spec.label = text
        self.viewport.set_vectors(self.viewport.vectors)

    def _pick_color(self, index: int) -> None:
        spec = self._vector_spec(index)
        if spec is None:
            return
        color = QColorDialog.getColor(QColor(spec.color), self, "Vector colour")
        if color.isValid():
            spec.color = color.name()
            self.viewport.set_vectors(self.viewport.vectors)
            self._vector_signature = None
            self._refresh_vectors()

    def _add_vector(self) -> None:
        dim = self.viewport.dim
        index = len(self.viewport.vectors)
        coords = np.array([2.0, 1.0, 0.0]) if dim == 2 else np.array([1.0, 1.0, 1.0])
        color = core.MANIM_COLORS[(index + 3) % len(core.MANIM_COLORS)]
        self.viewport.vectors.append(VectorSpec(f"v{index + 1}", coords, color))
        self.viewport.set_vectors(self.viewport.vectors)

    def _remove_vector(self, index: int) -> None:
        if 0 <= index < len(self.viewport.vectors):
            self.viewport.vectors.pop(index)
            self.viewport.set_vectors(self.viewport.vectors)

    # ----------------------------------------------------------------- events
    def _stack_selection_changed(self) -> None:
        self._refresh_stack_buttons()
        row = self.stack_list.currentRow()
        if row >= 0:
            self.viewport.select_step(row)

    def _remove_selected(self) -> None:
        index = self.stack_list.currentRow()
        if index >= 1:
            self.viewport.remove_step(index)

    def _duration_changed(self, value: int) -> None:
        self.viewport.set_duration(value / 1000.0)
        self.duration_label.setText(f"{value / 1000:.2f} s")
