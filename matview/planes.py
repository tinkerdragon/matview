"""Plane geometry and Ax=b classification for the plane-intersection mode.

Each row of ``A`` is one plane ``a_i . x = b_i`` in R3.  The module is pure
NumPy: the viewport builds GL geometry from :class:`Plane` records and shows
the feasible set described by :class:`PlaneSolution`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Same palette as the live transform renderer.
PLANE_COLORS: tuple[str, ...] = ("#58C4DD", "#FC6255", "#83C167")
SOLUTION_COLOR = "#F5D547"

SolutionKind = Literal["unique", "line", "plane", "none", "all"]

DEFAULT_A = np.array(
    [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]], dtype=float
)
DEFAULT_B = np.array([3.0, 5.0, 4.0], dtype=float)


@dataclass(frozen=True)
class Plane:
    """One finite rendering window lying on an infinite plane."""

    index: int
    normal: np.ndarray          # raw row of A
    offset: float               # raw b_i
    unit_normal: np.ndarray
    distance: float             # signed offset from the origin along the normal
    foot: np.ndarray            # closest point on the plane to the origin
    u: np.ndarray               # first in-plane axis
    v: np.ndarray               # second in-plane axis
    color: str
    equation: str


@dataclass(frozen=True)
class PlaneSolution:
    kind: SolutionKind
    point: np.ndarray | None
    direction: np.ndarray | None
    basis: tuple[np.ndarray, np.ndarray] | None
    rank_a: int
    rank_aug: int
    det: float
    residual: float
    singular_values: np.ndarray
    ill_conditioned: bool
    reason: str
    planes: tuple[Plane, ...]


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-15 else vector.copy()


def _stable_sign(vector: np.ndarray) -> np.ndarray:
    """Orient a direction deterministically so labels do not flip sign."""
    if vector[np.argmax(np.abs(vector))] < 0.0:
        return -vector
    return vector.copy()


def format_linear_combination(row: np.ndarray, rhs: float) -> str:
    """Human-readable ``2x - y + 0.5z = 4``."""
    terms: list[str] = []
    for coefficient, variable in zip(row, "xyz"):
        value = float(coefficient)
        if abs(value) < 5e-4:
            continue
        magnitude = abs(value)
        magnitude_text = "" if abs(magnitude - 1.0) < 1e-9 else f"{magnitude:.3g}"
        sign = "-" if value < 0.0 else "+"
        if not terms:
            sign = "-" if value < 0.0 else ""
        terms.append(f"{sign}{magnitude_text}{variable}")
    left = " ".join(terms) if terms else "0"
    right = f"{float(rhs):.3g}"
    return f"{left} = {right}"


def format_point(point: np.ndarray) -> str:
    return "(" + ", ".join(f"{float(value):.2f}" for value in point) + ")"


def format_direction(direction: np.ndarray) -> str:
    return "(" + ", ".join(f"{float(value):+.2f}" for value in direction) + ")"


def _make_plane(index: int, row: np.ndarray, offset: float) -> Plane | None:
    norm = float(np.linalg.norm(row))
    if norm <= 1e-12:
        return None
    unit = row / norm
    distance = float(offset) / norm
    foot = unit * distance
    helper = np.eye(3)[int(np.argmin(np.abs(unit)))]
    u = _normalize(np.cross(unit, helper))
    v = _normalize(np.cross(unit, u))
    return Plane(
        index=index,
        normal=row.copy(),
        offset=float(offset),
        unit_normal=unit,
        distance=distance,
        foot=foot,
        u=u,
        v=v,
        color=PLANE_COLORS[index % len(PLANE_COLORS)],
        equation=format_linear_combination(row, offset),
    )


def classify(A, b) -> PlaneSolution:
    """Classify the intersection of the three rows of ``A x = b``."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(3)
    if A.shape != (3, 3):
        raise ValueError(f"A must be 3x3, got {A.shape}")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("A and b must be finite")

    _, singular, vh = np.linalg.svd(A)
    largest = float(singular[0]) if len(singular) else 0.0
    tolerance = 1e-9 * max(1.0, largest)
    rank_a = int(np.sum(singular > tolerance))

    augmented = np.column_stack([A, b])
    augmented_singular = np.linalg.svd(augmented, compute_uv=False)
    augmented_largest = float(augmented_singular[0])
    augmented_tolerance = 1e-9 * max(1.0, augmented_largest)
    rank_aug = int(np.sum(augmented_singular > augmented_tolerance))

    row_norms = np.linalg.norm(A, axis=1)
    plane_tolerance = 1e-9 * max(1.0, float(np.max(row_norms)))
    planes = tuple(
        plane
        for index in range(3)
        if (plane := _make_plane(index, A[index], float(b[index]))) is not None
        and row_norms[index] > plane_tolerance
    )
    zero_conflict = any(
        row_norms[index] <= plane_tolerance and abs(float(b[index])) > tolerance
        for index in range(3)
    )

    particular, *_ = np.linalg.lstsq(A, b, rcond=None)
    residual = float(np.linalg.norm(A @ particular - b))
    condition = (
        float(singular[-1] / singular[0])
        if len(singular) and singular[0] > 0.0
        else 0.0
    )

    kind: SolutionKind
    point: np.ndarray | None = None
    direction: np.ndarray | None = None
    basis: tuple[np.ndarray, np.ndarray] | None = None
    reason = ""

    if rank_a == 3:
        kind = "unique"
        point = np.linalg.solve(A, b)
        reason = "three planes meet at one point"
    elif rank_aug > rank_a:
        kind = "none"
        if zero_conflict:
            reason = "no solution — a zero row requires a nonzero value"
        elif rank_a == 2:
            reason = "no common point — the planes form a triangular prism"
        elif rank_a == 1:
            reason = "no common point — the planes are parallel"
        else:
            reason = "no solution"
    elif rank_a == 2:
        kind = "line"
        point = particular
        direction = _stable_sign(vh[-1])
        reason = "three planes share an intersection line"
    elif rank_a == 1:
        kind = "plane"
        point = particular
        basis = (_stable_sign(vh[-2]), _stable_sign(vh[-1]))
        reason = "all three equations describe the same plane"
    else:
        kind = "all"
        point = np.zeros(3)
        reason = "all equations are 0 = 0 — every point is a solution"

    return PlaneSolution(
        kind=kind,
        point=point,
        direction=direction,
        basis=basis,
        rank_a=rank_a,
        rank_aug=rank_aug,
        det=float(np.linalg.det(A)),
        residual=residual,
        singular_values=singular,
        ill_conditioned=bool(kind == "unique" and condition < 1e-6),
        reason=reason,
        planes=planes,
    )


def plane_containing(plane: Plane, tolerance: float = 1e-7) -> bool:
    """Small invariant helper used by tests and diagnostics."""
    return abs(float(np.dot(plane.unit_normal, plane.foot)) - plane.distance) <= tolerance
