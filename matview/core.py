"""Pure-math core for the live matrix visualizer.

Everything in this module is Qt-free so it can be unit-checked headlessly.
Matrices are column-vector convention: a point ``p`` maps to ``M @ p`` and a
stack ``[M1, M2]`` (applied in that order) composes to ``M2 @ M1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# scipy.linalg takes noticeable time to import, so it is loaded on first use
# (the first matrix that needs a logarithm) and warmed up in the background by
# the app. The visualizer starts without it.
_SCIPY_LINALG: object | None = None


def _scipy_linalg():
    global _SCIPY_LINALG
    if _SCIPY_LINALG is None:
        try:
            from scipy import linalg as _linalg

            _SCIPY_LINALG = _linalg
        except Exception:  # pragma: no cover - fallback path
            _SCIPY_LINALG = False
    return _SCIPY_LINALG

# Manim's canonical palette, reused for the live renderer so exports match.
MANIM_COLORS: tuple[str, ...] = (
    "#58C4DD",  # BLUE
    "#83C167",  # GREEN
    "#FC6255",  # RED
    "#F5D547",  # YELLOW
    "#9A72AC",  # PURPLE
    "#5CD0B3",  # TEAL
    "#FF862F",  # ORANGE
    "#D147BD",  # PINK
)
BASIS_COLORS: tuple[str, ...] = ("#83C167", "#FC6255", "#58C4DD")  # i, j, k


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def identity(n: int) -> np.ndarray:
    return np.eye(n, dtype=float)


def embed(matrix: np.ndarray, n: int) -> np.ndarray:
    """Pad (or truncate) a square matrix to ``n`` dimensions."""
    matrix = np.asarray(matrix, dtype=float)
    m = matrix.shape[0]
    if m == n:
        return matrix
    out = np.eye(n, dtype=float)
    k = min(m, n)
    out[:k, :k] = matrix[:k, :k]
    return out


def compose(stack: Sequence[np.ndarray], n: int) -> np.ndarray:
    """Composite of matrices applied left-to-right (newest on the left)."""
    total = identity(n)
    for matrix in stack:
        total = embed(matrix, n) @ total
    return total


def smoothstep(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * (3.0 - 2.0 * s)


def skew(axis: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    k = skew(axis)
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _expm(matrix: np.ndarray) -> np.ndarray:
    linalg = _scipy_linalg()
    if linalg:
        return np.real(linalg.expm(matrix))
    # Cayley-free Taylor fallback (dense, small matrices).
    result = np.eye(matrix.shape[0])
    term = np.eye(matrix.shape[0])
    for i in range(1, 24):
        term = term @ matrix / i
        result = result + term
    return result


def _is_rotation(matrix: np.ndarray) -> bool:
    n = matrix.shape[0]
    if n not in (2, 3):
        return False
    if not np.allclose(matrix.T @ matrix, np.eye(n), atol=1e-7):
        return False
    return np.linalg.det(matrix) > 0.0


def rotation_axis_angle(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Axis/angle of a proper 3D rotation, stable at 0 and pi."""
    cos_theta = (float(np.trace(matrix)) - 1.0) / 2.0
    cos_theta = min(1.0, max(-1.0, cos_theta))
    angle = math.acos(cos_theta)
    if angle < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if math.pi - angle < 1e-5:
        # R + I = 2 u u^T on the axis.
        a = (matrix + np.eye(3)) * 0.5
        axis = np.sqrt(np.maximum(np.diag(a), 0.0))
        # Fix signs using the off-diagonal entries.
        if axis[0] > 0 and a[0, 1] < 0:
            axis[1] = -axis[1]
        if axis[0] > 0 and a[0, 2] < 0:
            axis[2] = -axis[2]
        if axis[0] < 1e-8 and axis[1] > 0 and a[1, 2] < 0:
            axis[2] = -axis[2]
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            return np.array([0.0, 0.0, 1.0]), angle
        return axis / norm, angle
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    return axis / np.linalg.norm(axis), angle


class MatrixPath:
    """A continuous path from ``I`` to ``delta``.

    Priority order:

    * exact rotation paths (axis-angle) -- handles 180 degrees, where the real
      matrix logarithm does not exist;
    * the Lie-group geodesic ``exp(s log(delta))`` when its logarithm is real
      (scales, shears and most invertible matrices);
    * plain entry-wise interpolation for singular / reflecting matrices.

    ``at(0) == I`` and ``at(1) == delta`` for every path type.
    """

    def __init__(self, delta: np.ndarray):
        delta = np.asarray(delta, dtype=float)
        self.n = delta.shape[0]
        self.delta = delta
        self.kind, self.data = self._decompose(delta)

    def _decompose(self, matrix: np.ndarray) -> tuple[str, object]:
        if _is_rotation(matrix):
            if self.n == 2:
                return "rot2", math.atan2(matrix[1, 0], matrix[0, 0])
            return "rot3", rotation_axis_angle(matrix)
        if abs(float(np.linalg.det(matrix))) < 1e-9:
            # A singular map (projection / collapse) has no real logarithm;
            # interpolating entries shows the collapse cleanly.
            return "lerp", None
        linalg = _scipy_linalg()
        if linalg:
            try:
                with np.errstate(all="ignore"):
                    log = linalg.logm(matrix)
                log = np.asarray(log)
                if np.all(np.isfinite(log.real)) and np.max(np.abs(log.imag)) < 1e-7:
                    return "log", np.ascontiguousarray(log.real)
            except Exception:
                pass
        return "lerp", None

    def at(self, s: float) -> np.ndarray:
        s = float(s)
        if s <= 0.0:
            return np.eye(self.n)
        if self.kind == "rot2":
            angle = s * float(self.data)  # type: ignore[arg-type]
            c, t = math.cos(angle), math.sin(angle)
            return np.array([[c, -t], [t, c]])
        if self.kind == "rot3":
            axis, angle = self.data  # type: ignore[misc]
            return axis_angle_matrix(axis, s * angle)
        if self.kind == "log":
            return _expm(s * np.asarray(self.data))  # type: ignore[arg-type]
        return (1.0 - s) * np.eye(self.n) + s * self.delta

    def describe(self) -> str:
        return {"rot2": "rotation", "rot3": "rotation", "log": "geodesic", "lerp": "linear"}[
            self.kind
        ]


class Transition:
    """Animated path between two composite matrices.

    When the start matrix is invertible the motion is the geodesic of
    ``end @ start^-1``; otherwise (e.g. after a projection collapsed the
    space) it falls back to entry-wise interpolation, which always exists.
    """

    def __init__(self, start: np.ndarray, end: np.ndarray):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.path: MatrixPath | None = None
        try:
            delta = self.end @ np.linalg.inv(self.start)
            if np.all(np.isfinite(delta)):
                self.path = MatrixPath(delta)
        except np.linalg.LinAlgError:
            self.path = None

    def at(self, s: float) -> np.ndarray:
        if self.path is not None:
            return self.path.at(s) @ self.start
        return (1.0 - s) * self.start + s * self.end


class Timeline:
    """A sequence of transitions played back at a fixed duration per step."""

    def __init__(self, states: Sequence[np.ndarray], step: float):
        self.states = [np.asarray(state, dtype=float) for state in states]
        self.step = max(1e-4, float(step))
        self.transitions = [
            Transition(self.states[i], self.states[i + 1])
            for i in range(len(self.states) - 1)
        ]
        self.total = self.step * len(self.transitions)

    def at(self, t: float) -> tuple[np.ndarray, float, int]:
        if not self.transitions:
            return self.states[0], 0.0, 0
        t = min(max(t, 0.0), self.total)
        index = min(int(t // self.step), len(self.transitions) - 1)
        local = (t - index * self.step) / self.step
        return self.transitions[index].at(smoothstep(local)), local, index


# --------------------------------------------------------------------------- #
# presets
# --------------------------------------------------------------------------- #
def rotation_matrix(dim: int, degrees: float, axis: int = 2) -> np.ndarray:
    angle = math.radians(degrees)
    if dim == 2:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, -s], [s, c]])
    unit = np.zeros(3)
    unit[axis % 3] = 1.0
    return axis_angle_matrix(unit, angle)


def scale_matrix(dim: int, sx: float, sy: float, sz: float = 1.0) -> np.ndarray:
    values = [sx, sy] if dim == 2 else [sx, sy, sz]
    return np.diag(np.array(values, dtype=float))


def shear_matrix(dim: int, k_xy: float, k_xz: float = 0.0, k_yz: float = 0.0) -> np.ndarray:
    if dim == 2:
        return np.array([[1.0, k_xy], [0.0, 1.0]])
    return np.array(
        [
            [1.0, k_xy, k_xz],
            [0.0, 1.0, k_yz],
            [0.0, 0.0, 1.0],
        ]
    )


def reflection_matrix(dim: int, plane: str = "x") -> np.ndarray:
    index = {"x": 0, "y": 1, "z": 2}[plane]
    values = np.ones(dim)
    values[index] = -1.0
    return np.diag(values)


def projection_matrix(dim: int, onto: str = "x") -> np.ndarray:
    index = {"x": 0, "y": 1, "z": 2}[onto]
    values = np.zeros(dim)
    values[index] = 1.0
    return np.diag(values)


def random_matrix(dim: int, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    for _ in range(200):
        matrix = np.round(rng.uniform(-1.6, 1.6, size=(dim, dim)), 1)
        det = abs(np.linalg.det(matrix))
        if 0.15 < det < 4.5:
            return matrix
    return np.eye(dim)


# --------------------------------------------------------------------------- #
# spectral data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EigenDatum:
    value: complex
    vector: np.ndarray | None
    is_real: bool

    def value_text(self) -> str:
        if self.is_real:
            return f"{self.value.real:+.2f}"
        return f"{self.value.real:+.2f}{self.value.imag:+.2f}i"


def eigen_data(matrix: np.ndarray) -> list[EigenDatum]:
    values, vectors = np.linalg.eig(np.asarray(matrix, dtype=float))
    out: list[EigenDatum] = []
    for value, vector in zip(values, vectors.T):
        if abs(value.imag) < 1e-9:
            vec = np.real(vector)
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                continue
            out.append(EigenDatum(value=complex(float(value.real), 0.0), vector=vec / norm, is_real=True))
        else:
            out.append(EigenDatum(value=complex(value), vector=None, is_real=False))
    return out


def determinant(matrix: np.ndarray) -> float:
    return float(np.linalg.det(np.asarray(matrix, dtype=float)))


def geometric_series(matrix: np.ndarray) -> str:
    """Human label for the matrix family, used in the stack list."""
    det = determinant(matrix)
    if np.allclose(matrix, np.eye(matrix.shape[0]), atol=1e-9):
        return "identity"
    if _is_rotation(matrix):
        return "rotation"
    if np.allclose(matrix - np.diag(np.diag(matrix)), 0.0, atol=1e-9):
        return "scale"
    if np.allclose(matrix, matrix.T, atol=1e-9):
        if np.allclose(matrix @ matrix, np.eye(matrix.shape[0]), atol=1e-9):
            return "reflection"
        return "symmetric"
    if abs(det) < 1e-9:
        return "singular"
    return "linear map"


def pretty(matrix: np.ndarray, digits: int = 2) -> str:
    rows = []
    for row in np.asarray(matrix):
        rows.append(" ".join(f"{value:+.{digits}f}" for value in row))
    return "[" + " | ".join(rows) + "]"


def cumulative_states(stack: Sequence[np.ndarray], n: int) -> list[np.ndarray]:
    """[I, M1, M2*M1, ...] for replay animations."""
    states = [identity(n)]
    for matrix in stack:
        states.append(embed(matrix, n) @ states[-1])
    return states
