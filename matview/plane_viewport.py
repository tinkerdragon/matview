"""Live 3D renderer for the Ax=b plane-intersection mode."""

from __future__ import annotations

import numpy as np
from OpenGL import GL
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter

from . import gl, planes
from .viewport import Viewport

PLANE_HALF_EXTENT = 6.0
PLANE_FILL_ALPHA = 0.16
PLANE_BORDER_ALPHA = 0.82
PLANE_GRID_ALPHA = 0.16
NORMAL_LENGTH = 1.6


class PlaneViewport(Viewport):
    """The same camera and label system, rendering three planes instead of a map."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.kind = "planes"
        # Plane mode is 3D-only.  Use the base implementation directly so the
        # subclass guard below cannot swallow this initial transition.
        Viewport.set_dim(self, 3)
        self.flags.update(
            {
                "reference": False,
                "basis": False,
                "unit": False,
                "eigen": False,
                "trails": False,
                "grid": False,
                "labels": True,
                "planes": True,
                "plane_grid": True,
                "normals": False,
                "intersection": True,
            }
        )
        self.R = PLANE_HALF_EXTENT
        # The base viewport starts with one user vector; plane mode has none.
        self.vectors = []
        self.system_a = planes.DEFAULT_A.copy()
        self.system_b = planes.DEFAULT_B.copy()
        self.solution = planes.classify(self.system_a, self.system_b)

        self._plane_dirty = True
        self._fill_arrays: list[np.ndarray] = []
        self._fill_centers: list[np.ndarray] = []
        self._plane_border_data = np.zeros((0, gl.LINE_STRIDE), dtype=np.float32)
        self._plane_grid_data = np.zeros((0, gl.LINE_STRIDE), dtype=np.float32)
        self._normal_line_data = np.zeros((0, gl.LINE_STRIDE), dtype=np.float32)
        self._normal_head_data = np.zeros((0, gl.TRI_STRIDE), dtype=np.float32)
        self._intersection_line_data = np.zeros((0, gl.LINE_STRIDE), dtype=np.float32)

        self._plane_fill_batch: gl.TriBatch | None = None
        self._plane_border_batch: gl.LineBatch | None = None
        self._plane_grid_batch: gl.LineBatch | None = None
        self._normal_line_batch: gl.LineBatch | None = None
        self._normal_head_batch: gl.TriBatch | None = None
        self._intersection_line_batch: gl.LineBatch | None = None

        self._rebuild_plane_geometry()
        self.camera.target[:] = self.focus_point()
        self.fit_view(emit=False)

    # ------------------------------------------------------------------ state
    def set_dim(self, dim: int) -> None:
        # Plane tabs are always R3; ignore toolbar attempts to switch them.
        if dim == 3 and self.dim != 3:
            super().set_dim(3)

    def set_system(self, A, b) -> None:
        matrix = np.asarray(A, dtype=float)
        vector = np.asarray(b, dtype=float).reshape(3)
        solution = planes.classify(matrix, vector)
        self.system_a = matrix.copy()
        self.system_b = vector.copy()
        self.solution = solution
        self._rebuild_plane_geometry()
        self.stateChanged.emit()
        self._dirty = True
        self.update()

    def focus_point(self) -> np.ndarray:
        solution = self.solution
        if solution.point is not None:
            return np.asarray(solution.point, dtype=float).copy()
        if solution.planes:
            return np.mean([plane.foot for plane in solution.planes], axis=0)
        return np.zeros(3)

    def fit_view(self, emit: bool = True) -> None:
        extent = self.R
        self.camera.target[:] = self.focus_point()
        if self.camera.ortho:
            self.camera.ortho_height = 3.0 * extent
        else:
            self.camera.distance = 4.2 * extent
        self._dirty = True
        self.update()
        if emit:
            self.cameraChanged.emit()

    def reset_camera(self) -> None:
        self.fit_view()

    def toggle_projection(self) -> None:
        if self.dim == 3:
            self.camera.ortho = not self.camera.ortho
            self.fit_view()

    def set_flag(self, name: str, value: bool) -> None:
        super().set_flag(name, value)
        if name == "intersection":
            self._rebuild_plane_geometry()

    def _center_for(self, plane: planes.Plane) -> np.ndarray:
        if self.solution.kind in ("unique", "line", "plane"):
            return np.asarray(self.solution.point, dtype=float)
        return np.asarray(plane.foot, dtype=float)

    # -------------------------------------------------------------- geometry
    def _rebuild_plane_geometry(self) -> None:
        fills: list[np.ndarray] = []
        centers: list[np.ndarray] = []
        border_starts: list[np.ndarray] = []
        border_ends: list[np.ndarray] = []
        border_colors: list[np.ndarray] = []
        grid_starts: list[np.ndarray] = []
        grid_ends: list[np.ndarray] = []
        grid_colors: list[np.ndarray] = []
        normal_starts: list[np.ndarray] = []
        normal_ends: list[np.ndarray] = []
        normal_colors: list[np.ndarray] = []
        heads: list[np.ndarray] = []
        radius = self.R

        for plane in self.solution.planes:
            center = self._center_for(plane)
            u = plane.u * radius
            v = plane.v * radius
            corners = np.array(
                [
                    center - u - v,
                    center + u - v,
                    center + u + v,
                    center - u + v,
                ]
            )
            triangles = np.array(
                [[corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]]
            )
            fills.append(
                gl.flat_triangles_vertices(triangles, plane.color, PLANE_FILL_ALPHA)
            )
            centers.append(center.copy())

            color = gl.hex_rgba(plane.color, PLANE_BORDER_ALPHA)
            for index in range(4):
                border_starts.append(corners[index])
                border_ends.append(corners[(index + 1) % 4])
                border_colors.append(color)

            grid_color = gl.hex_rgba(plane.color, PLANE_GRID_ALPHA)
            step = 2.0
            offsets = np.arange(-radius + step, radius, step)
            for offset in offsets:
                grid_starts.append(center + plane.u * offset - v)
                grid_ends.append(center + plane.u * offset + v)
                grid_colors.append(grid_color)
                grid_starts.append(center + plane.v * offset - u)
                grid_ends.append(center + plane.v * offset + u)
                grid_colors.append(grid_color)

            normal_start = center
            normal_end = center + plane.unit_normal * NORMAL_LENGTH
            normal_starts.append(normal_start)
            normal_ends.append(normal_end)
            normal_colors.append(gl.hex_rgba(plane.color, 0.9))
            heads.append(
                gl.cone_vertices(
                    normal_end,
                    plane.unit_normal,
                    0.26,
                    0.085,
                    plane.color,
                    alpha=0.92,
                    segments=16,
                )
            )


        self._fill_arrays = fills
        self._fill_centers = centers
        self._plane_border_data = self._concat_lines(
            border_starts, border_ends, border_colors
        )
        self._plane_grid_data = self._concat_lines(grid_starts, grid_ends, grid_colors)
        self._normal_line_data = self._concat_lines(
            normal_starts, normal_ends, normal_colors
        )
        self._normal_head_data = (
            np.vstack(heads) if heads else np.zeros((0, gl.TRI_STRIDE), dtype=np.float32)
        )

        solution = self.solution
        if (
            self.flags.get("intersection", True)
            and solution.kind == "line"
            and solution.direction is not None
        ):
            start = solution.point - solution.direction * radius * 1.45
            end = solution.point + solution.direction * radius * 1.45
            self._intersection_line_data = gl.segment_vertices(
                [start], [end], planes.SOLUTION_COLOR, 0.95
            )
        else:
            self._intersection_line_data = np.zeros(
                (0, gl.LINE_STRIDE), dtype=np.float32
            )
        self._plane_dirty = True

    @staticmethod
    def _concat_lines(starts, ends, colors) -> np.ndarray:
        if not starts:
            return np.zeros((0, gl.LINE_STRIDE), dtype=np.float32)
        return gl.segment_vertices(np.asarray(starts), np.asarray(ends), np.asarray(colors))

    # ------------------------------------------------------------------- GL
    def initializeGL(self) -> None:
        super().initializeGL()
        self._plane_fill_batch = gl.TriBatch()
        self._plane_border_batch = gl.LineBatch()
        self._plane_grid_batch = gl.LineBatch()
        self._normal_line_batch = gl.LineBatch()
        self._normal_head_batch = gl.TriBatch()
        self._intersection_line_batch = gl.LineBatch()
        self._plane_dirty = True
        self._upload_plane_geometry()

    def _upload_plane_geometry(self) -> None:
        if self._plane_fill_batch is None:
            return
        self._plane_border_batch.upload(self._plane_border_data)
        self._plane_grid_batch.upload(self._plane_grid_data)
        self._normal_line_batch.upload(self._normal_line_data)
        self._normal_head_batch.upload(self._normal_head_data)
        self._intersection_line_batch.upload(self._intersection_line_data)
        self._plane_dirty = False

    def _sorted_fill_data(self) -> np.ndarray:
        if not self._fill_arrays:
            return np.zeros((0, gl.TRI_STRIDE), dtype=np.float32)
        eye = self.camera.eye()
        order = sorted(
            range(len(self._fill_arrays)),
            key=lambda index: -float(np.linalg.norm(self._fill_centers[index] - eye)),
        )
        return np.vstack([self._fill_arrays[index] for index in order])

    def _render_extra(self, pixel_w: int, pixel_h: int) -> None:
        if self._plane_fill_batch is None:
            return
        if self._plane_dirty:
            self._upload_plane_geometry()

        identity = np.eye(4)
        line = self._line_prog
        line.use()
        line.mat4("u_viewproj", self._vp)
        line.vec2("u_res", pixel_w, pixel_h)

        if self.flags.get("planes", True):
            GL.glDepthMask(GL.GL_FALSE)
            self._plane_fill_batch.upload(self._sorted_fill_data())
            self._tris(self._plane_fill_batch, identity, 1.0, lit=False)
            GL.glDepthMask(GL.GL_TRUE)
            if self.flags.get("plane_grid", True):
                self._lines(self._plane_grid_batch, identity, 1.0, 1.0)
            self._lines(self._plane_border_batch, identity, 1.35, 1.0)

        if self.flags.get("normals", False):
            self._lines(self._normal_line_batch, identity, 1.6, 1.0)
            self._tris(self._normal_head_batch, identity, 1.0, lit=True)

        if self.flags.get("intersection", True) and self.solution.kind == "line":
            self._lines(self._intersection_line_batch, identity, 3.2, 1.0)

    # --------------------------------------------------------------- overlays
    def _draw_overlays(self, painter: QPainter) -> None:
        solution = self.solution
        if not self.flags.get("intersection", True) or solution.kind != "unique":
            return
        if solution.point is None:
            return
        screen = self._project(solution.point)
        if screen is None or not (
            -30 <= screen.x() <= self.width() + 30
            and -30 <= screen.y() <= self.height() + 30
        ):
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        halo = QColor(planes.SOLUTION_COLOR)
        halo.setAlpha(42)
        painter.setPen(Qt.NoPen)
        painter.setBrush(halo)
        painter.drawEllipse(screen, 12.0, 12.0)
        core = QColor(planes.SOLUTION_COLOR)
        painter.setBrush(core)
        painter.drawEllipse(screen, 5.0, 5.0)
        painter.setBrush(QColor(255, 255, 255, 230))
        painter.drawEllipse(screen, 1.8, 1.8)
        painter.setBrush(Qt.NoBrush)

    def _collect_labels(self, eigen_labels):
        labels = super()._collect_labels(eigen_labels)
        if not self.flags.get("labels", True):
            return labels

        # Pick a visible corner for each plane label; the fixed geometric corner
        # can be off-screen depending on the camera and the plane orientation.
        # Prefer corners inside the view and away from labels already placed.
        used: list[tuple[float, float]] = []
        visible_planes = (
            self.solution.planes if self.flags.get("planes", True) else ()
        )
        for plane in visible_planes:
            center = self._center_for(plane)
            radius = self.R * 0.62
            corners = [
                center + plane.u * radius + plane.v * radius,
                center - plane.u * radius + plane.v * radius,
                center - plane.u * radius - plane.v * radius,
                center + plane.u * radius - plane.v * radius,
            ]
            best = None
            best_screen = None
            best_score = -1e9
            for shift in range(4):
                candidate = corners[(plane.index + shift) % 4]
                screen = self._project(candidate)
                if screen is None:
                    continue
                inside = (
                    36.0 <= screen.x() <= self.width() - 36.0
                    and 30.0 <= screen.y() <= self.height() - 30.0
                )
                center_distance = (
                    (screen.x() - self.width() * 0.5) ** 2
                    + (screen.y() - self.height() * 0.5) ** 2
                ) ** 0.5
                separation = min(
                    (
                        (screen.x() - other[0]) ** 2 + (screen.y() - other[1]) ** 2
                    )
                    ** 0.5
                    for other in used
                ) if used else 1000.0
                score = (
                    (3.0 if inside else 0.0)
                    + min(separation, 220.0) / 55.0
                    - center_distance * 0.002
                )
                if score > best_score:
                    best_score = score
                    best = candidate
                    best_screen = screen
            if best is not None and best_screen is not None:
                used.append((best_screen.x(), best_screen.y()))
                labels.append(
                    (f"P{plane.index + 1}", best, plane.color, 11, 6.0, -6.0)
                )
        solution = self.solution
        if solution.point is None or not self.flags.get("intersection", True):
            return labels
        if solution.kind == "unique":
            text = f"x = {planes.format_point(solution.point)}"
            labels.append((text, solution.point, planes.SOLUTION_COLOR, 11, 12.0, -12.0))
        elif solution.kind == "line":
            labels.append(
                ("x = p + t d", solution.point, planes.SOLUTION_COLOR, 11, 12.0, -12.0)
            )
        return labels
