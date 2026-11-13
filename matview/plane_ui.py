"""Control panel for the Ax=b plane-intersection mode."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import planes
from .controls import MatrixField
from .expr import evaluate
from .plane_viewport import PlaneViewport

PRESETS: tuple[tuple[str, str], ...] = (
    ("Unique point", "unique"),
    ("Intersection line", "line"),
    ("Coincident planes", "coincident"),
    ("Parallel (no solution)", "parallel"),
    ("Triangular prism (no solution)", "prism"),
    ("Near-singular", "near"),
)

PRESET_SYSTEMS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "unique": (planes.DEFAULT_A.copy(), planes.DEFAULT_B.copy()),
    "line": (
        np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 2.0, 1.0]]),
        np.array([3.0, 5.0, 8.0]),
    ),
    "coincident": (
        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]]),
        np.array([3.0, 6.0, -3.0]),
    ),
    "parallel": (
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([1.0, 2.0, 3.0]),
    ),
    "prism": (
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        np.array([0.0, 0.0, 1.0]),
    ),
    "near": (
        np.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.000001], [1.0, 1.000001, 1.0]]
        ),
        np.array([3.0, 3.000001, 3.000001]),
    ),
}

BADGE_COLORS = {
    "unique": "#83C167",
    "line": "#58C4DD",
    "plane": "#58C4DD",
    "none": "#FC6255",
    "all": "#8AA0B6",
}

BADGE_TEXT = {
    "unique": "UNIQUE SOLUTION",
    "line": "INTERSECTION LINE",
    "plane": "COINCIDENT PLANES",
    "none": "NO SOLUTION",
    "all": "ALL OF R³",
}


class PlanePanel(QWidget):
    def __init__(self, viewport: PlaneViewport, parent=None):
        super().__init__(parent)
        self.viewport = viewport
        self._syncing = False
        self._invalid = False
        self.setObjectName("panel")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self._build_equations(root)
        self._build_solution(root)
        self._build_presets(root)
        self._build_display(root)
        root.addStretch(1)

        viewport.stateChanged.connect(self.refresh)
        self._write_system(self.viewport.system_a, self.viewport.system_b, fit=False)
        self.refresh()

    # ------------------------------------------------------------------ build
    @staticmethod
    def _group(root: QVBoxLayout, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(9, 12, 9, 8)
        layout.setSpacing(5)
        root.addWidget(box)
        return box, layout

    def _build_equations(self, root: QVBoxLayout) -> None:
        box, layout = self._group(root, "AX = B  ·  THREE PLANES")
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(5)
        grid.setVerticalSpacing(4)
        for column, title in enumerate(("x", "y", "z", "", "b")):
            label = QLabel(title)
            label.setAlignment(Qt.AlignCenter)
            label.setObjectName("muted")
            grid.addWidget(label, 0, column)
        self.a_fields: list[list[MatrixField]] = []
        self.b_fields: list[MatrixField] = []
        for row in range(3):
            fields = []
            for column in range(3):
                field = self._field(grid, row + 1, column)
                fields.append(field)
            equals = QLabel("=")
            equals.setAlignment(Qt.AlignCenter)
            equals.setObjectName("muted")
            grid.addWidget(equals, row + 1, 3)
            b_field = self._field(grid, row + 1, 4)
            self.a_fields.append(fields)
            self.b_fields.append(b_field)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        grid.addWidget(spacer, 0, 5)
        grid.setColumnStretch(5, 1)
        layout.addLayout(grid)
        note = QLabel("Rows are planes · entries accept expressions")
        note.setObjectName("muted")
        layout.addWidget(note)

    def _field(self, grid: QGridLayout, row: int, column: int) -> MatrixField:
        field = MatrixField()
        field.setFixedWidth(52)
        field.setMinimumHeight(26)
        field.textChanged.connect(self._changed)
        grid.addWidget(field, row, column)
        return field

    def _build_solution(self, root: QVBoxLayout) -> None:
        box, layout = self._group(root, "SOLUTION")
        self.badge = QLabel("—")
        self.badge.setStyleSheet("font-weight:700; color:#8AA0B6;")
        layout.addWidget(self.badge)
        self.detail = QLabel("—")
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.RichText)
        layout.addWidget(self.detail)
        self.metrics = QLabel("—")
        self.metrics.setWordWrap(True)
        self.metrics.setObjectName("muted")
        layout.addWidget(self.metrics)
        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color:#F0AC5F;")
        self.warning.setVisible(False)
        layout.addWidget(self.warning)

        self.equation_labels: list[QLabel] = []
        for row in range(3):
            label = QLabel("—")
            label.setTextFormat(Qt.RichText)
            label.setWordWrap(True)
            self.equation_labels.append(label)
            layout.addWidget(label)

    def _build_presets(self, root: QVBoxLayout) -> None:
        box, layout = self._group(root, "PRESETS")
        self.preset_combo = QComboBox()
        for label, key in PRESETS:
            self.preset_combo.addItem(label, key)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        layout.addWidget(self.preset_combo)
        random_button = QPushButton("Random full-rank system")
        random_button.clicked.connect(lambda: self._random())
        layout.addWidget(random_button)

    def _build_display(self, root: QVBoxLayout) -> None:
        box, layout = self._group(root, "DISPLAY")
        self.flag_boxes: dict[str, QCheckBox] = {}
        for key, label in (
            ("planes", "Planes"),
            ("plane_grid", "Plane grids"),
            ("normals", "Normals"),
            ("intersection", "Intersection"),
            ("labels", "Labels"),
            ("grid", "Spatial grid"),
        ):
            box_check = QCheckBox(label)
            box_check.setChecked(bool(self.viewport.flags.get(key, False)))
            box_check.toggled.connect(
                lambda checked, k=key: self.viewport.set_flag(k, checked)
            )
            layout.addWidget(box_check)
            self.flag_boxes[key] = box_check
        fit = QPushButton("Fit view to solution")
        fit.clicked.connect(lambda: self.viewport.fit_view())
        layout.addWidget(fit)

    # ------------------------------------------------------------------ state
    def _write_system(self, A: np.ndarray, b: np.ndarray, fit: bool = True) -> None:
        self._syncing = True
        try:
            for row in range(3):
                for column in range(3):
                    field = self.a_fields[row][column]
                    field.blockSignals(True)
                    field.setText(self._format(float(A[row, column])))
                    field.setStyleSheet("")
                    field.blockSignals(False)
                self.b_fields[row].blockSignals(True)
                self.b_fields[row].setText(self._format(float(b[row])))
                self.b_fields[row].setStyleSheet("")
                self.b_fields[row].blockSignals(False)
        finally:
            self._syncing = False
        self._invalid = False
        self.viewport.set_system(A, b)
        if fit:
            self.viewport.fit_view()

    @staticmethod
    def _format(value: float) -> str:
        if abs(value) < 5e-7:
            return "0"
        return f"{value:.6g}"

    def _changed(self) -> None:
        if self._syncing:
            return
        A = np.zeros((3, 3))
        b = np.zeros(3)
        invalid: list[tuple[str, int, int]] = []
        for row in range(3):
            for column in range(3):
                try:
                    A[row, column] = evaluate(self.a_fields[row][column].text())
                except ValueError:
                    invalid.append(("a", row, column))
            try:
                b[row] = evaluate(self.b_fields[row].text())
            except ValueError:
                invalid.append(("b", row, 0))
        self._mark_invalid(invalid)
        if invalid:
            return
        try:
            self.viewport.set_system(A, b)
        except (ValueError, np.linalg.LinAlgError):
            self._invalid = True
            self.badge.setText("INVALID INPUT")
            self.badge.setStyleSheet("font-weight:700; color:#F48771;")
            self.detail.setText("The system contains a non-finite value.")
            self.metrics.setText("—")
            self.warning.setVisible(False)

    def _mark_invalid(self, invalid: list[tuple[str, int, int]]) -> None:
        self._invalid = bool(invalid)
        key = set(invalid)
        style = (
            "QLineEdit { color:#F48771; border:1px solid #F48771; "
            "background:#080C12; border-radius:4px; padding:3px 5px; }"
        )
        for row in range(3):
            for column in range(3):
                field = self.a_fields[row][column]
                field.setStyleSheet(style if ("a", row, column) in key else "")
            field = self.b_fields[row]
            field.setStyleSheet(style if ("b", row, 0) in key else "")
        if self._invalid:
            self.badge.setText("INVALID INPUT")
            self.badge.setStyleSheet("font-weight:700; color:#F48771;")
            self.detail.setText("Fix the highlighted entries; the scene keeps the last valid system.")
            self.metrics.setText("—")
            self.warning.setVisible(False)

    def refresh(self) -> None:
        if self._invalid:
            return
        solution = self.viewport.solution
        color = BADGE_COLORS.get(solution.kind, "#8AA0B6")
        text = BADGE_TEXT.get(solution.kind, solution.kind.upper())
        if solution.ill_conditioned:
            text += "  ·  ILL-CONDITIONED"
        self.badge.setText(text)
        self.badge.setStyleSheet(f"font-weight:700; color:{color};")

        if solution.kind == "unique":
            self.detail.setText(
                f"<span style='color:#F5D547; font-family:Consolas'>"
                f"x = {planes.format_point(solution.point)}</span>"
            )
        elif solution.kind == "line":
            self.detail.setText(
                "<span style='color:#F5D547; font-family:Consolas'>"
                f"p = {planes.format_point(solution.point)}<br>"
                f"d = {planes.format_direction(solution.direction)}<br>"
                "x = p + t d</span>"
            )
        elif solution.kind == "plane":
            self.detail.setText(
                "<span style='color:#F5D547; font-family:Consolas'>"
                f"p = {planes.format_point(solution.point)}<br>"
                "every point of the plane is a solution</span>"
            )
        else:
            self.detail.setText(f"<span style='color:{color}'>{solution.reason}</span>")

        self.metrics.setText(
            f"det A = {solution.det:+.3g}   ·   rank A = {solution.rank_a}   ·   "
            f"rank [A|b] = {solution.rank_aug}   ·   min ‖Ax−b‖ = {solution.residual:.2e}"
        )
        if solution.ill_conditioned:
            ratio = float(solution.singular_values[-1] / solution.singular_values[0])
            self.warning.setText(
                f"Ill-conditioned: σmin/σmax = {ratio:.2e}; the intersection point is "
                "sensitive to small entry changes."
            )
            self.warning.setVisible(True)
        else:
            self.warning.setVisible(False)

        for row, label in enumerate(self.equation_labels):
            coefficient = self.viewport.system_a[row]
            rhs = float(self.viewport.system_b[row])
            color = (
                planes.PLANE_COLORS[row]
                if np.linalg.norm(coefficient) > 1e-9
                else "#5C6B7A"
            )
            equation = planes.format_linear_combination(coefficient, rhs)
            label.setText(
                f"<span style='color:{color}; font-family:Consolas'>P{row + 1}  "
                f"{equation}</span>"
            )

    # ---------------------------------------------------------------- presets
    def _preset_changed(self, index: int) -> None:
        if self._syncing or index < 0:
            return
        key = self.preset_combo.itemData(index)
        system = PRESET_SYSTEMS.get(str(key))
        if system is None:
            return
        self._write_system(system[0], system[1], fit=True)

    def _random(self) -> None:
        rng = np.random.default_rng()
        A = np.eye(3)
        b = np.zeros(3)
        for _ in range(50):
            candidate = rng.integers(-3, 4, size=(3, 3)).astype(float)
            if abs(float(np.linalg.det(candidate))) >= 1.0:
                A = candidate
                b = rng.integers(-4, 5, size=3).astype(float)
                break
        self._write_system(A, b, fit=True)

