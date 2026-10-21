"""The live 2D/3D viewport: space rendering, camera interaction, playback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OpenGL import GL
from PySide6.QtCore import QElapsedTimer, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from . import core, gl
from .camera import Camera

BACKGROUND = (0.035, 0.047, 0.067, 1.0)
GRID_MINOR = "#16222F"
GRID_MAJOR = "#203346"
GRID_LATTICE = "#28405A"
REFERENCE = "#26384A"
AXIS_COLOR = "#5A7089"
UNIT_COLOR = "#58C4DD"
EIGEN_COLOR = "#F5D547"
GHOST_TINT = (0.35, 0.77, 0.87)


@dataclass
class VectorSpec:
    label: str
    coords: np.ndarray
    color: str


def _basis(dim: int, index: int) -> np.ndarray:
    import numpy as np

    vector = np.zeros(3)
    vector[index] = 1.0
    return vector[:dim]


class Viewport(QOpenGLWidget):
    stateChanged = Signal()
    frameChanged = Signal()
    cameraChanged = Signal()
    fpsChanged = Signal(float)
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 420)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(False)
        self.setCursor(Qt.CrossCursor)

        self.dim = 2
        self.stack: list[np.ndarray] = []
        # State index: 0 is the original (identity) space, k is the space after
        # the first k transformations.
        self.active = 0
        self.editor = np.eye(2)
        self.vectors: list[VectorSpec] = [
            VectorSpec("v", np.array([1.6, 1.1, 0.0]), core.MANIM_COLORS[3])
        ]
        self.flags = {
            "grid": True,
            "reference": False,
            "basis": True,
            "unit": True,
            "eigen": False,
            "trails": True,
            "labels": True,
        }
        self.duration = 1.0

        self.camera = Camera()
        self.camera.set_mode(2)
        self.displayed = np.eye(2)
        self.timeline: core.Timeline | None = None

        self._clock = QElapsedTimer()
        self._clock.start()
        self._timeline_start = 0
        self._progress = 0.0
        self._dirty = True

        self._drag_mode: str | None = None
        self._last_pos = None

        self._line_prog: gl.Shader | None = None
        self._tri_prog: gl.Shader | None = None
        self._grid_batch: gl.LineBatch | None = None
        self._unit_line_batch: gl.LineBatch | None = None
        self._unit_fill_batch: gl.TriBatch | None = None
        self._basis_line_batch: gl.LineBatch | None = None
        self._vector_line_batch: gl.LineBatch | None = None
        self._head_batch: gl.TriBatch | None = None
        self._eigen_batch: gl.LineBatch | None = None
        self._scratch_batch: gl.LineBatch | None = None
        self._trail_batches: list[gl.LineBatch] = []
        self._static_dirty = False
        self._static_grid = np.zeros((0, gl.LINE_STRIDE), dtype=np.float32)
        self._static_unit_line = np.zeros((0, gl.LINE_STRIDE), dtype=np.float32)
        self._static_unit_fill = np.zeros((0, gl.TRI_STRIDE), dtype=np.float32)
        self._axis_starts = np.zeros((0, 3))
        self._axis_ends = np.zeros((0, 3))
        self._axis_colors = np.zeros((0, 4))
        self._trail_data = [
            np.zeros((0, gl.LINE_STRIDE), dtype=np.float32) for _ in range(3)
        ]
        self._trail_dirty = False

        self._vp = np.eye(4)
        self._sample_labels: list[tuple[str, np.ndarray, str, int, float, float]] = []
        self._origin_screen = None

        self.target_fps = 165
        screen = self.screen()
        if screen is not None and screen.refreshRate() > 1.0:
            self.target_fps = max(30.0, float(screen.refreshRate()))
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.CoarseTimer)
        self._timer.setInterval(10)  # watchdog only; frames are vsync-paced
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self.frameSwapped.connect(self._on_frame_swapped)
        self._frames = 0
        self._fps_since = 0
        self._last_live_emit = 0

    # ------------------------------------------------------------------ state
    @property
    def composite(self) -> np.ndarray:
        return core.compose(self.stack[: self.active], self.dim)

    def preview_matrix(self) -> np.ndarray | None:
        if self.active < len(self.stack):
            return None
        editor = core.embed(self.editor, self.dim)
        if np.allclose(editor, np.eye(self.dim), atol=1e-12):
            return None
        return editor @ self.displayed

    def set_dim(self, dim: int) -> None:
        if dim == self.dim:
            return
        if self.timeline is not None:
            # land the running transition before changing space dimensions
            self.displayed = self.timeline.states[-1].copy()
            self.timeline = None
        self.dim = dim
        self.stack = [core.embed(matrix, dim) for matrix in self.stack]
        self.editor = core.embed(self.editor, dim)
        self.displayed = core.embed(self.displayed, dim)
        self.timeline = None
        self._progress = 0.0
        self._clear_trails()
        self.camera.set_mode(dim)
        self._rebuild_static()
        self.stateChanged.emit()
        self.cameraChanged.emit()
        self._dirty = True

    def set_editor(self, matrix: np.ndarray) -> None:
        self.editor = core.embed(matrix, self.dim)
        self._dirty = True

    def set_duration(self, seconds: float) -> None:
        self.duration = max(0.05, float(seconds))

    def apply_editor(self) -> None:
        matrix = self.editor.copy()
        if np.allclose(matrix, np.eye(self.dim), atol=1e-12):
            self.statusMessage.emit("Matrix is the identity — edit an entry or choose a preset.")
            return
        start = self.displayed.copy()
        self.stack.append(matrix)
        self.active = len(self.stack)
        target = core.compose(self.stack, self.dim)
        self.editor = np.eye(self.dim)
        self._start_timeline([start, target], self.duration)
        self.statusMessage.emit(
            f"Applied {core.geometric_series(matrix)} · det {core.determinant(matrix):+.2f}"
        )
        self.stateChanged.emit()

    def remove_step(self, index: int) -> None:
        """Delete the state at ``index`` (0 is the original and cannot go)."""
        if not 1 <= index <= len(self.stack):
            return
        start = self.displayed.copy()
        self.stack.pop(index - 1)
        self.active = min(self.active, len(self.stack))
        target = core.compose(self.stack[: self.active], self.dim)
        self._start_timeline([start, target], min(self.duration, 0.9))
        self.statusMessage.emit(f"Removed state {index}.")
        self.stateChanged.emit()

    def select_step(self, index: int) -> None:
        """Show the state of the space at ``index`` (0 = original, identity)."""
        self.active = max(0, min(int(index), len(self.stack)))
        target = self.composite
        if np.allclose(target, self.displayed, atol=1e-9):
            self.stateChanged.emit()
            return
        self._start_timeline([self.displayed.copy(), target], min(self.duration, 0.55))
        if self.active == 0:
            self.statusMessage.emit("Viewing the original state — nothing applied.")
        elif self.active < len(self.stack):
            self.statusMessage.emit(
                f"Viewing state {self.active} of {len(self.stack) + 1} — nothing applied."
            )
        self.stateChanged.emit()

    def clear_stack(self) -> None:
        if not self.stack:
            return
        start = self.displayed.copy()
        self.stack.clear()
        self.active = 0
        self._start_timeline([start, np.eye(self.dim)], min(self.duration, 0.9))
        self.statusMessage.emit("Cleared to the original state.")
        self.stateChanged.emit()

    def replay(self) -> None:
        if not self.stack:
            self.statusMessage.emit("Apply at least one matrix first.")
            return
        self.active = len(self.stack)
        states = core.cumulative_states(self.stack, self.dim)
        self._start_timeline(states, min(self.duration, 1.25))
        self.stateChanged.emit()

    def set_vectors(self, vectors: list[VectorSpec]) -> None:
        self.vectors = vectors
        self.stateChanged.emit()
        self._dirty = True

    def set_flag(self, name: str, value: bool) -> None:
        self.flags[name] = bool(value)
        self._dirty = True

    def reset_camera(self) -> None:
        self.camera.reset(self.dim, keep_projection=True)
        self._dirty = True
        self.cameraChanged.emit()

    def set_camera_view(self, name: str) -> None:
        if self.dim == 3:
            self.camera.set_view(name)
            self._dirty = True
            self.cameraChanged.emit()

    def toggle_projection(self) -> None:
        if self.dim == 3:
            self.camera.ortho = not self.camera.ortho
            self._dirty = True
            self.cameraChanged.emit()

    # -------------------------------------------------------------- animation
    def _start_timeline(self, states, step: float) -> None:
        self.timeline = core.Timeline(states, step)
        self._timeline_start = self._clock.elapsed()
        self._progress = 0.0
        self._build_trails()
        self._dirty = True

    def _on_frame_swapped(self) -> None:
        # Render-loop pacing: advance the animation at the swap, then request the
        # next frame. Overlapping updates are coalesced by Qt, so this holds the
        # animation at the display refresh instead of an independent 6 ms timer.
        if self.timeline is None:
            return
        self._advance()
        self.update()

    def _tick(self) -> None:
        # Watchdog for cases where frameSwapped stops (occluded or hidden window).
        if self.timeline is not None:
            self._advance()
        if self._dirty and self.isVisible():
            self._dirty = False
            self.update()

    def _advance(self) -> None:
        if self.timeline is None:
            return
        elapsed = (self._clock.elapsed() - self._timeline_start) / 1000.0
        total = self.timeline.total
        if total <= 1e-6 or elapsed >= total:
            self.displayed = self.timeline.states[-1].copy()
            self.timeline = None
            self._progress = 0.0
            self._clear_trails()
            self.stateChanged.emit()
        else:
            self._progress = elapsed / total
            self.displayed = self.timeline.at(elapsed)[0]
            now = self._clock.elapsed()
            if now - self._last_live_emit >= 100:
                self._last_live_emit = now
                self.frameChanged.emit()
        self._dirty = True

    def _clear_trails(self) -> None:
        # GL buffers can only be touched with the widget context current, so the
        # CPU-side geometry is written now and uploaded on the next paint.
        self._trail_data = [
            np.zeros((0, gl.LINE_STRIDE), dtype=np.float32) for _ in range(3)
        ]
        self._trail_dirty = True

    def _build_trails(self) -> None:
        self._clear_trails()
        if self.timeline is None or not self.flags.get("trails"):
            return
        basis = [_basis(self.dim, i) for i in range(self.dim)]
        trails = [[(self.timeline.states[0] @ b)[: self.dim]] for b in basis]
        for transition in self.timeline.transitions:
            for step in range(1, 25):
                state = transition.at(core.smoothstep(step / 24.0))
                for index, b in enumerate(basis):
                    trails[index].append((state @ b)[: self.dim])
        for index, points in enumerate(trails):
            points = np.asarray(points, dtype=float)
            count = len(points)
            if count < 2:
                continue
            points3 = np.zeros((count, 3))
            points3[:, : self.dim] = points
            color = gl.hex_rgba(core.BASIS_COLORS[index])
            colors = np.tile(color, (count - 1, 1))
            colors[:, 3] *= np.linspace(0.04, 0.5, count - 1)
            self._trail_data[index] = gl.segment_vertices(points3[:-1], points3[1:], colors)

    def _upload_trails(self) -> None:
        if not self._trail_dirty or len(self._trail_batches) < 3:
            return
        for batch, data in zip(self._trail_batches, self._trail_data):
            batch.upload(data)
        self._trail_dirty = False

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._dirty = True
        self.update()

    # ------------------------------------------------------------------- GL
    def initializeGL(self) -> None:
        GL.glClearColor(*BACKGROUND)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFuncSeparate(
            GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
            GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA,
        )
        GL.glDisable(GL.GL_CULL_FACE)
        self._line_prog = gl.make_line_shader()
        self._tri_prog = gl.make_tri_shader()
        self._grid_batch = gl.LineBatch()
        self._unit_line_batch = gl.LineBatch()
        self._unit_fill_batch = gl.TriBatch()
        self._basis_line_batch = gl.LineBatch()
        self._vector_line_batch = gl.LineBatch()
        self._head_batch = gl.TriBatch()
        self._eigen_batch = gl.LineBatch()
        self._scratch_batch = gl.LineBatch()
        self._trail_batches = [gl.LineBatch() for _ in range(3)]
        self._rebuild_static()
        self._upload_static()

    def _rebuild_static(self) -> None:
        """Build the space skeleton on the CPU.

        * 2D: a 1-unit plane grid.
        * 3D: a full cubic lattice -- segments along all three axes at every
          integer coordinate -- so a matrix visibly warps a volume of space,
          not just a single plane.
        """
        starts, ends, colors = [], [], []
        if self.dim == 2:
            extent = 10.0
            axis_extent = 10.0
            minor = gl.hex_rgba(GRID_MINOR, 1.0)
            major = gl.hex_rgba(GRID_MAJOR, 1.0)
            for i in range(-10, 11):
                color = major if i % 5 == 0 else minor
                starts.append([i, -extent, 0.0])
                ends.append([i, extent, 0.0])
                colors.append(color)
                starts.append([-extent, i, 0.0])
                ends.append([extent, i, 0.0])
                colors.append(color)
        else:
            lattice = 6
            spacing = 3
            axis_extent = 7.0
            for axis in range(3):
                others = [index for index in range(3) if index != axis]
                for a in range(-lattice, lattice + 1, spacing):
                    for b in range(-lattice, lattice + 1, spacing):
                        p0 = np.zeros(3)
                        p1 = np.zeros(3)
                        p0[axis] = -lattice
                        p1[axis] = lattice
                        p0[others[0]] = p1[others[0]] = a
                        p0[others[1]] = p1[others[1]] = b
                        midpoint = (p0 + p1) * 0.5
                        falloff = 1.0 - 0.35 * (
                            np.linalg.norm(midpoint) / (lattice * np.sqrt(3.0))
                        )
                        on_plane = axis != 2 and b == 0
                        alpha = min(1.0, (0.95 if on_plane else 0.62) * falloff)
                        color = gl.hex_rgba(
                            GRID_MAJOR if on_plane else GRID_LATTICE, alpha
                        )
                        starts.append(p0)
                        ends.append(p1)
                        colors.append(color)
        self._static_grid = gl.segment_vertices(starts, ends, np.asarray(colors))

        axis_starts, axis_ends, axis_colors = [], [], []
        axis_rgba = gl.hex_rgba(AXIS_COLOR, 0.45 if self.dim == 3 else 0.6)
        for axis in range(self.dim):
            direction = np.zeros(3)
            direction[axis] = 1.0
            axis_starts.append(-axis_extent * direction)
            axis_ends.append(axis_extent * direction)
            axis_colors.append(axis_rgba)
        self._axis_starts = np.asarray(axis_starts)
        self._axis_ends = np.asarray(axis_ends)
        self._axis_colors = np.asarray(axis_colors)

        line_points = []
        fill = np.zeros((0, gl.TRI_STRIDE), dtype=np.float32)
        if self.dim == 2:
            loop = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
            line_points.append(gl.polyline_vertices(loop, UNIT_COLOR, 0.85))
            triangles = np.array(
                [[[0, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 0, 0], [1, 1, 0], [0, 1, 0]]],
                dtype=float,
            )
            fill = gl.flat_triangles_vertices(triangles, UNIT_COLOR, 0.11)
        else:
            corners = np.array(
                [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
            )
            edges = []
            for i, a in enumerate(corners):
                for b in corners[i + 1 :]:
                    if np.sum(np.abs(a - b) > 1e-9) == 1:
                        edges.append((a, b))
            for a, b in edges:
                line_points.append(gl.segment_vertices(a, b, gl.hex_rgba(UNIT_COLOR, 0.8)))
        self._static_unit_line = (
            np.vstack(line_points) if line_points else np.zeros((0, gl.LINE_STRIDE))
        )
        self._static_unit_fill = fill
        self._static_dirty = True
        self._dirty = True

    def _upload_static(self) -> None:
        if not self._static_dirty or self._grid_batch is None:
            return
        self._grid_batch.upload(self._static_grid)
        self._unit_line_batch.upload(self._static_unit_line)
        self._unit_fill_batch.upload(self._static_unit_fill)
        self._static_dirty = False

    def paintGL(self) -> None:
        width = max(1, self.width())
        height = max(1, self.height())
        dpr = self.devicePixelRatioF()
        pixel_w = max(1, int(width * dpr))
        pixel_h = max(1, int(height * dpr))

        view = self.camera.view_matrix()
        projection = self.camera.projection_matrix(width / height)
        self._vp = projection @ view

        painter = QPainter(self)
        painter.beginNativePainting()
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glViewport(0, 0, pixel_w, pixel_h)
        GL.glClearColor(*BACKGROUND)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self._render(pixel_w, pixel_h)
        painter.endNativePainting()
        self._draw_labels(painter)
        painter.end()

        self._frames += 1
        now = self._clock.elapsed()
        if now - self._fps_since >= 500:
            fps = self._frames * 1000.0 / max(1, now - self._fps_since)
            self._frames = 0
            self._fps_since = now
            self.fpsChanged.emit(fps)

    def _render(self, pixel_w: int, pixel_h: int) -> None:
        if self._line_prog is None or self._grid_batch is None:
            return
        self._upload_static()
        self._upload_trails()
        self._pixel_w = pixel_w
        self._pixel_h = pixel_h
        GL.glColorMask(True, True, True, True)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFuncSeparate(
            GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
            GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA,
        )
        depth = self.dim == 3
        if depth:
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LEQUAL)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)

        line = self._line_prog
        line.use()
        line.mat4("u_viewproj", self._vp)
        line.vec2("u_res", pixel_w, pixel_h)

        displayed_model = gl.model_matrix(self.displayed)
        identity_model = np.eye(4)

        if self.flags["reference"]:
            ghost_alpha = 0.70 if self.dim == 2 else 0.46
            self._lines(self._grid_batch, identity_model, 1.0, ghost_alpha, (0.62, 0.80, 0.95))
        if self.flags["grid"]:
            self._lines(self._grid_batch, displayed_model, 1.0, 1.0)

        ghost = self.preview_matrix()
        if ghost is not None:
            model = gl.model_matrix(ghost)
            self._lines(self._grid_batch, model, 1.1, 0.75, (0.5, 0.9, 1.0))
            if self.flags["unit"]:
                self._lines(self._unit_line_batch, model, 1.8, 0.85, (0.5, 0.9, 1.0))

        if self.flags["unit"]:
            if depth:
                GL.glDepthMask(GL.GL_FALSE)
            self._tris(self._unit_fill_batch, displayed_model, 1.0, lit=False)
            if depth:
                GL.glDepthMask(GL.GL_TRUE)
            self._lines(self._unit_line_batch, displayed_model, 1.8, 0.9)

        self._lines_segments(
            self._axis_starts,
            self._axis_ends,
            self._axis_colors,
            displayed_model,
            1.2,
            1.0,
            batch=self._scratch_batch,
        )

        # trails are already in world space
        if self.flags["trails"] and self.timeline is not None:
            for batch in self._trail_batches[: self.dim]:
                count = int(batch.count * self._progress) // 6 * 6
                self._lines(batch, identity_model, 2.0, 1.0, count=count)

        self._build_arrow_geometry()
        self._lines(self._basis_line_batch, identity_model, 3.0, 1.0)
        self._lines(self._vector_line_batch, identity_model, 2.4, 1.0)

        eigen_segments, eigen_labels = self._eigen_geometry()
        if self.flags["eigen"] and eigen_segments:
            starts = np.array([s for s, _ in eigen_segments])
            ends = np.array([e for _, e in eigen_segments])
            self._lines_segments(starts, ends, gl.hex_rgba(EIGEN_COLOR, 0.75), identity_model, 1.6, 1.0)

        tri = self._tri_prog
        tri.use()
        tri.mat4("u_viewproj", self._vp)
        tri.mat4("u_model", identity_model)
        tri.vec3("u_light", 0.4, -0.65, 0.65)
        tri.float("u_lit", 1.0 if depth else 0.0)
        self._head_batch.draw()

        self._sample_labels = self._collect_labels(eigen_labels)

        if depth:
            GL.glDisable(GL.GL_DEPTH_TEST)

    def _lines(self, batch, model, width, alpha, tint=(1.0, 1.0, 1.0), count=None) -> None:
        if batch is None or batch.count == 0:
            return
        program = self._line_prog
        program.mat4("u_model", model)
        # Widths are authored in logical pixels; convert to device pixels so
        # strokes stay hairline-crisp on HiDPI displays instead of sub-pixel.
        program.float("u_width", width * self.devicePixelRatioF())
        program.float("u_alpha", alpha)
        program.vec3("u_tint", *tint)
        batch.draw(count)

    def _lines_segments(self, starts, ends, colors, model, width, alpha, batch=None) -> None:
        target = batch or self._eigen_batch
        if target is None:
            return
        target.upload(gl.segment_vertices(starts, ends, colors, alpha))
        self._lines(target, model, width, 1.0)

    def _tris(self, batch, model, alpha, lit=False) -> None:
        if batch is None or batch.count == 0:
            return
        program = self._tri_prog
        program.use()
        program.mat4("u_viewproj", self._vp)
        program.mat4("u_model", model)
        program.vec3("u_light", 0.4, -0.65, 0.65)
        program.float("u_lit", 1.0 if lit else 0.0)
        batch.draw()
        self._line_prog.use()
        self._line_prog.mat4("u_viewproj", self._vp)
        self._line_prog.vec2("u_res", self._pixel_w, self._pixel_h)

    # -------------------------------------------------------------- geometry
    def _build_arrow_geometry(self) -> None:
        if self._basis_line_batch is None:
            return
        wpp = self.camera.world_per_pixel(self.height())
        basis_vertices = []
        user_vertices = []
        heads = []
        basis = core.BASIS_COLORS[: self.dim]

        entries: list[tuple[np.ndarray, str, bool]] = []
        if self.flags["basis"]:
            for index in range(self.dim):
                entries.append((_basis(self.dim, index), basis[index], True))
        for spec in self.vectors:
            entries.append((np.asarray(spec.coords, dtype=float)[: self.dim], spec.color, False))

        for coords, color, is_basis in entries:
            tip = self.displayed @ coords
            tip3 = np.zeros(3)
            tip3[: self.dim] = tip
            length = float(np.linalg.norm(tip3))
            if length < 1e-7:
                continue
            direction = tip3 / length
            head_len = (14.0 if is_basis else 13.0) * wpp
            head_len = min(max(head_len, 0.055), length * 0.42)
            head_radius = head_len * 0.37
            shaft_end = tip3 - direction * head_len * 0.86
            if np.linalg.norm(shaft_end) < length * 0.2:
                shaft_end = tip3 * 0.4
            target = basis_vertices if is_basis else user_vertices
            target.append(gl.segment_vertices(np.zeros(3), shaft_end, gl.hex_rgba(color)))
            if self.dim == 2:
                perp = np.array([-direction[1], direction[0], 0.0])
                base = tip3 - direction * head_len
                triangles = np.array(
                    [[tip3, base + perp * head_radius, base - perp * head_radius]]
                )
                heads.append(gl.flat_triangles_vertices(triangles, color))
            else:
                heads.append(
                    gl.cone_vertices(tip3, direction, head_len, head_radius, color, segments=20)
                )

        self._basis_line_batch.upload(
            np.vstack(basis_vertices) if basis_vertices else np.zeros((0, gl.LINE_STRIDE))
        )
        self._vector_line_batch.upload(
            np.vstack(user_vertices) if user_vertices else np.zeros((0, gl.LINE_STRIDE))
        )
        self._head_batch.upload(
            np.vstack(heads) if heads else np.zeros((0, gl.TRI_STRIDE))
        )

    def _eigen_geometry(self):
        segments = []
        labels = []
        if not self.flags["eigen"]:
            return segments, labels
        matrix = self.displayed
        if np.allclose(matrix, np.eye(self.dim), atol=1e-9):
            return segments, labels
        extent = 6.0 if self.dim == 3 else 8.0
        seen = set()
        for datum in core.eigen_data(matrix):
            if not datum.is_real or datum.vector is None:
                continue
            vector = np.zeros(3)
            vector[: self.dim] = datum.vector[: self.dim]
            norm = np.linalg.norm(vector)
            if norm < 1e-9:
                continue
            vector = vector / norm
            if vector[np.argmax(np.abs(vector))] < 0:
                vector = -vector
            key = tuple(np.round(vector, 3))
            if key in seen:
                continue
            seen.add(key)
            length = extent
            pieces = 22
            for index in range(pieces):
                start = -length + 2 * length * index / pieces
                stop = start + (2 * length / pieces) * 0.55
                segments.append((vector * start, vector * stop))
            labels.append(
                (f"λ={datum.value.real:+.2f}", vector * (length * 0.94), EIGEN_COLOR)
            )
        return segments, labels

    # ----------------------------------------------------------------- labels
    def _project(self, point: np.ndarray) -> QPointF | None:
        clip = self._vp @ np.array([point[0], point[1], point[2], 1.0])
        if clip[3] <= 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        if ndc[2] < -1.2 or ndc[2] > 1.2:
            return None
        x = (ndc[0] * 0.5 + 0.5) * self.width()
        y = (1.0 - (ndc[1] * 0.5 + 0.5)) * self.height()
        return QPointF(x, y)

    def _collect_labels(self, eigen_labels):
        labels: list[tuple[str, np.ndarray, str, int, float, float]] = []
        if not self.flags["labels"]:
            return labels
        for index, name in enumerate("xyz"[: self.dim]):
            point = self.displayed @ (_basis(self.dim, index) * 2.6)
            point3 = np.zeros(3)
            point3[: self.dim] = point
            labels.append((name, point3, AXIS_COLOR, 11, 5.0, -5.0))
        if self.flags["basis"]:
            for index, name in enumerate(("i", "j", "k")[: self.dim]):
                positions = self.displayed @ (_basis(self.dim, index))
                point = np.zeros(3)
                point[: self.dim] = positions
                labels.append((name, point, core.BASIS_COLORS[index], 12, 7.0, -15.0))
        for spec in self.vectors:
            coords = np.asarray(spec.coords, dtype=float)[: self.dim]
            point = np.zeros(3)
            point[: self.dim] = self.displayed @ coords
            labels.append((spec.label or "v", point, spec.color, 12, 7.0, -15.0))
        for text, point, color in eigen_labels:
            labels.append((text, point, color, 11, 6.0, -6.0))
        return labels

    def _draw_labels(self, painter: QPainter) -> None:
        if not self._sample_labels:
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        fonts = self.__dict__.setdefault("_label_fonts", {})
        colors = self.__dict__.setdefault("_label_colors", {})
        for text, point, color, size, dx, dy in self._sample_labels:
            screen = self._project(point)
            if screen is None:
                continue
            if not (-80 <= screen.x() <= self.width() + 80 and -30 <= screen.y() <= self.height() + 30):
                continue
            italic = text in ("i", "j", "k", "x", "y", "z")
            key = (size, italic)
            font = fonts.get(key)
            if font is None:
                font = QFont("Segoe UI", size)
                font.setWeight(QFont.DemiBold)
                font.setItalic(italic)
                font.setHintingPreference(QFont.PreferFullHinting)
                fonts[key] = font
            painter.setFont(font)
            pen_color = colors.get(color)
            if pen_color is None:
                pen_color = QColor(color)
                colors[color] = pen_color
            position = QPointF(screen.x() + dx, screen.y() + dy)
            painter.setPen(QColor(5, 8, 12, 200))
            painter.drawText(QPointF(position.x() + 1.2, position.y() + 1.2), text)
            painter.setPen(pen_color)
            painter.drawText(position, text)

    # ----------------------------------------------------------- interaction
    def mousePressEvent(self, event) -> None:
        self._last_pos = event.position()
        if event.button() == Qt.LeftButton:
            if self.dim == 3 and not (event.modifiers() & Qt.ShiftModifier):
                self._drag_mode = "orbit"
            else:
                self._drag_mode = "pan"
        elif event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._drag_mode = "pan"
        else:
            self._drag_mode = None
        self.setCursor(Qt.ClosedHandCursor if self._drag_mode else Qt.CrossCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_mode is None or self._last_pos is None:
            return
        position = event.position()
        dx = position.x() - self._last_pos.x()
        dy = position.y() - self._last_pos.y()
        self._last_pos = position
        if self._drag_mode == "orbit":
            self.camera.orbit(dx, dy)
        else:
            self.camera.pan(dx, dy, self.height())
        self._dirty = True
        # Repaint immediately (Qt coalesces); the 10 ms watchdog alone makes a
        # fast drag update in coarse steps.
        self.update()
        self.cameraChanged.emit()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_mode = None
        self._last_pos = None
        self.setCursor(Qt.CrossCursor)
        self._dirty = True

    def mouseDoubleClickEvent(self, event) -> None:
        self.reset_camera()

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        self.camera.zoom(steps)
        self._dirty = True
        self.update()
        self.cameraChanged.emit()

    # --------------------------------------------------------------- helpers
    def snapshot(self):
        image = self.grabFramebuffer()
        return image
