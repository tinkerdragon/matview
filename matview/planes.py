"""Plane geometry and Ax=b classification for the plane-intersection mode.

Each row of ``A`` is one plane ``a_i . x = b_i`` in R3.  The module is pure
NumPy: the viewport builds GL geometry from :class:`Plane` records and shows
the feasible set described by :class:`PlaneSolution`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

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
class PlanePair:
    """The line where two non-parallel planes meet."""

    first: int
    second: int
    point: np.ndarray
    direction: np.ndarray
    color: str


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


def _mix_hex(first: str, second: str) -> str:
    def rgb(value: str) -> np.ndarray:
        value = value.lstrip("#")
        return np.array([int(value[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)

    mixed = 0.52 * rgb(first) + 0.48 * rgb(second)
    mixed = 0.84 * mixed + 0.16 * 255.0
    return "#" + "".join(f"{int(round(channel)):02X}" for channel in mixed)


def pairwise_intersections(planes: Sequence[Plane]) -> list[PlanePair]:
    """Every pairwise intersection line, skipping parallel/coincident pairs.

    Three planes in general position produce three lines that all pass through
    the unique solution.  In a prism they form the visible edges; parallel or
    coincident pairs have no line and are skipped.
    """
    pairs: list[PlanePair] = []
    for first_index in range(len(planes)):
        first = planes[first_index]
        for second in planes[first_index + 1 :]:
            cross = np.cross(first.unit_normal, second.unit_normal)
            length = float(np.linalg.norm(cross))
            if length < 1e-9:
                continue
            direction = _stable_sign(cross / length)
            point = np.linalg.lstsq(
                np.vstack([first.unit_normal, second.unit_normal]),
                np.array([first.distance, second.distance]),
                rcond=None,
            )[0]
            pairs.append(
                PlanePair(
                    first=first.index,
                    second=second.index,
                    point=point,
                    direction=direction,
                    color=_mix_hex(first.color, second.color),
                )
            )
    return pairs


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


def _random_nonzero_vector(rng: np.random.Generator) -> np.ndarray:
    while True:
        vector = rng.integers(-3, 4, size=3).astype(float)
        if np.linalg.norm(vector) >= 0.5:
            return vector


def _random_full_rank(rng: np.random.Generator) -> np.ndarray:
    for _ in range(500):
        matrix = rng.integers(-3, 4, size=(3, 3)).astype(float)
        if abs(float(np.linalg.det(matrix))) >= 1.0:
            return matrix
    return np.eye(3)


def random_system(kind: str, rng: np.random.Generator | None = None):
    """Generate a random system of the requested preset family."""
    rng = rng or np.random.default_rng()
    kind = str(kind)

    if kind == "unique":
        for _ in range(500):
            matrix = rng.integers(-3, 4, size=(3, 3)).astype(float)
            if abs(float(np.linalg.det(matrix))) >= 1.0:
                vector = rng.integers(-4, 5, size=3).astype(float)
                return matrix, vector
        return np.eye(3), np.zeros(3)

    if kind == "line":
        while True:
            first = _random_nonzero_vector(rng)
            second = _random_nonzero_vector(rng)
            if float(np.linalg.norm(np.cross(first, second))) < 0.5:
                continue
            point = rng.integers(-2, 3, size=3).astype(float)
            offset_first = float(first @ point)
            offset_second = float(second @ point)
            a, c = rng.choice([-3, -2, -1, 1, 2, 3], size=2)
            third = a * first + c * second
            offset_third = a * offset_first + c * offset_second
            return (
                np.vstack([first, second, third]),
                np.array([offset_first, offset_second, offset_third]),
            )

    if kind == "coincident":
        normal = _random_nonzero_vector(rng)
        point = rng.integers(-2, 3, size=3).astype(float)
        offset = float(normal @ point)
        scales = np.array(
            [1.0, float(rng.choice([-2, -1, 2, 3])), float(rng.choice([-2, -1, 2, 3]))]
        )
        return np.vstack([scale * normal for scale in scales]), scales * offset

    if kind == "parallel":
        normal = _random_nonzero_vector(rng)
        offsets = rng.integers(-5, 6, size=3).astype(float)
        while len(set(float(value) for value in offsets)) < 2:
            offsets = rng.integers(-5, 6, size=3).astype(float)
        return np.vstack([normal, normal, normal]), offsets

    if kind == "prism":
        while True:
            first = _random_nonzero_vector(rng)
            second = _random_nonzero_vector(rng)
            if float(np.linalg.norm(np.cross(first, second))) < 0.5:
                continue
            a, c = rng.choice([-3, -2, -1, 1, 2, 3], size=2)
            third = a * first + c * second
            offset_first, offset_second = rng.integers(-3, 4, size=2).astype(float)
            delta = float(rng.choice([-3, -2, -1, 1, 2, 3]))
            offset_third = a * offset_first + c * offset_second + delta
            return (
                np.vstack([first, second, third]),
                np.array([offset_first, offset_second, offset_third]),
            )

    if kind in ("near", "near-singular"):
        for _ in range(200):
            base = _random_full_rank(rng)
            solution = rng.integers(-2, 3, size=3).astype(float)
            base_vector = base @ solution
            a, c = rng.choice([-2, -1, 1, 2], size=2)
            epsilon = 2e-7
            matrix = base.copy()
            vector = base_vector.copy()
            matrix[2] = a * base[0] + c * base[1] + epsilon * base[2]
            vector[2] = a * base_vector[0] + c * base_vector[1] + epsilon * base_vector[2]
            solution_data = classify(matrix, vector)
            if solution_data.kind == "unique" and solution_data.ill_conditioned:
                return matrix, vector
        return (
            np.array(
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.000001], [1.0, 1.000001, 1.0]]
            ),
            np.array([3.0, 3.000001, 3.000001]),
        )

    return random_system("unique", rng)


def plane_containing(plane: Plane, tolerance: float = 1e-7) -> bool:
    """Small invariant helper used by tests and diagnostics."""
    return abs(float(np.dot(plane.unit_normal, plane.foot)) - plane.distance) <= tolerance
