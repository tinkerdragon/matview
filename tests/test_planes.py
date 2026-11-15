"""Classification tests for the Ax=b plane mode."""

from __future__ import annotations

import unittest

import numpy as np

from matview.planes import classify, pairwise_intersections, plane_containing, random_system

TOL = 1e-6


class PlaneClassificationTests(unittest.TestCase):
    def assert_on_planes(self, A, b, point, tol=TOL):
        np.testing.assert_allclose(A @ point, b, atol=tol)

    def test_unique_solution(self):
        A = [[2.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 1.0]]
        b = [4.0, 5.0, 3.0]
        solution = classify(A, b)
        self.assertEqual(solution.kind, "unique")
        self.assertEqual((solution.rank_a, solution.rank_aug), (3, 3))
        np.testing.assert_allclose(solution.point, [1.0, 0.0, 2.0], atol=TOL)
        self.assert_on_planes(np.asarray(A), np.asarray(b), solution.point)
        self.assertLess(solution.residual, TOL)
        self.assertEqual(len(solution.planes), 3)
        for plane in solution.planes:
            self.assertTrue(plane_containing(plane))
            point = solution.point
            self.assertAlmostEqual(
                float(np.dot(plane.unit_normal, point)), plane.distance, places=6
            )

    def test_intersection_line(self):
        A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 2.0, 1.0]])
        b = np.array([3.0, 5.0, 8.0])
        solution = classify(A, b)
        self.assertEqual(solution.kind, "line")
        self.assertEqual((solution.rank_a, solution.rank_aug), (2, 2))
        self.assert_on_planes(A, b, solution.point)
        np.testing.assert_allclose(A @ solution.direction, 0.0, atol=TOL)
        self.assertAlmostEqual(np.linalg.norm(solution.direction), 1.0, places=9)
        self.assertAlmostEqual(float(np.dot(solution.point, solution.direction)), 0.0, places=6)

    def test_coincident_planes(self):
        A = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]])
        b = np.array([3.0, 6.0, -3.0])
        solution = classify(A, b)
        self.assertEqual(solution.kind, "plane")
        self.assertEqual((solution.rank_a, solution.rank_aug), (1, 1))
        self.assert_on_planes(A, b, solution.point)
        self.assertEqual(len(solution.basis), 2)
        for direction in solution.basis:
            np.testing.assert_allclose(A @ direction, 0.0, atol=TOL)
            self.assertAlmostEqual(np.linalg.norm(direction), 1.0, places=9)
        self.assertEqual(len(solution.planes), 3)

    def test_parallel_planes_have_no_solution(self):
        A = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        b = np.array([1.0, 2.0, 3.0])
        solution = classify(A, b)
        self.assertEqual(solution.kind, "none")
        self.assertEqual((solution.rank_a, solution.rank_aug), (1, 2))
        self.assertIn("parallel", solution.reason)
        self.assertGreater(solution.residual, 0.1)

    def test_triangular_prism_has_no_solution(self):
        A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        b = np.array([0.0, 0.0, 1.0])
        solution = classify(A, b)
        self.assertEqual(solution.kind, "none")
        self.assertEqual((solution.rank_a, solution.rank_aug), (2, 3))
        self.assertIn("prism", solution.reason)

    def test_zero_row_is_ignored_when_consistent(self):
        A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        b = np.array([1.0, 2.0, 0.0])
        solution = classify(A, b)
        self.assertEqual(solution.kind, "line")
        self.assertEqual(len(solution.planes), 2)
        np.testing.assert_allclose(solution.point, [1.0, 2.0, 0.0], atol=TOL)
        np.testing.assert_allclose(np.abs(solution.direction), [0.0, 0.0, 1.0], atol=TOL)

    def test_zero_row_conflict_has_no_solution(self):
        A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        b = np.array([1.0, 2.0, 1.0])
        solution = classify(A, b)
        self.assertEqual(solution.kind, "none")
        self.assertIn("zero row", solution.reason)
        self.assertEqual(len(solution.planes), 2)

    def test_all_space(self):
        solution = classify(np.zeros((3, 3)), np.zeros(3))
        self.assertEqual(solution.kind, "all")
        self.assertEqual((solution.rank_a, solution.rank_aug), (0, 0))
        self.assertEqual(len(solution.planes), 0)

    def test_near_singular_is_unique_but_flagged(self):
        A = np.array(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.000001], [1.0, 1.000001, 1.0]]
        )
        b = np.array([3.0, 3.000001, 3.000001])
        solution = classify(A, b)
        self.assertEqual(solution.kind, "unique")
        self.assertTrue(solution.ill_conditioned)
        np.testing.assert_allclose(solution.point, [1.0, 1.0, 1.0], atol=1e-4)

    def test_pairwise_lines_for_unique_system(self):
        A = np.array([[2.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 1.0]])
        b = np.array([4.0, 5.0, 3.0])
        solution = classify(A, b)
        pairs = pairwise_intersections(solution.planes)
        self.assertEqual(len(pairs), 3)
        by_index = {plane.index: plane for plane in solution.planes}
        for pair in pairs:
            self.assertIn(pair.first, by_index)
            self.assertIn(pair.second, by_index)
            for index in (pair.first, pair.second):
                plane = by_index[index]
                self.assertAlmostEqual(
                    float(np.dot(plane.unit_normal, pair.point)),
                    plane.distance,
                    places=6,
                )
                self.assertAlmostEqual(
                    float(np.dot(plane.unit_normal, pair.direction)), 0.0, places=6
                )
            delta = solution.point - pair.point
            off_line = delta - np.dot(delta, pair.direction) * pair.direction
            self.assertLess(float(np.linalg.norm(off_line)), 1e-6)

    def test_pairwise_lines_skip_parallel_and_coincident(self):
        parallel = classify(
            np.array([[1.0, 0.0, 0.0]] * 3), np.array([1.0, 2.0, 3.0])
        )
        self.assertEqual(pairwise_intersections(parallel.planes), [])
        coincident = classify(
            np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]]),
            np.array([3.0, 6.0, -3.0]),
        )
        self.assertEqual(pairwise_intersections(coincident.planes), [])

    def test_pairwise_lines_for_prism(self):
        solution = classify(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
            np.array([0.0, 0.0, 1.0]),
        )
        pairs = pairwise_intersections(solution.planes)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(pair.direction.shape == (3,) for pair in pairs))

    def test_pairwise_lines_with_a_zero_row(self):
        solution = classify(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([1.0, 2.0, 0.0]),
        )
        pairs = pairwise_intersections(solution.planes)
        self.assertEqual(len(pairs), 1)
        self.assertEqual((pairs[0].first, pairs[0].second), (0, 1))

    def test_random_system_matches_each_preset(self):
        expected = {
            "unique": "unique",
            "line": "line",
            "coincident": "plane",
            "parallel": "none",
            "prism": "none",
            "near": "unique",
        }
        for kind, target in expected.items():
            for seed in range(10):
                A, b = random_system(kind, np.random.default_rng(seed))
                solution = classify(A, b)
                self.assertEqual(
                    solution.kind, target, f"{kind} seed {seed}: {solution.reason}"
                )
                self.assertTrue(np.all(np.isfinite(A)))
                self.assertTrue(np.all(np.isfinite(b)))
                if kind == "near":
                    self.assertTrue(solution.ill_conditioned, f"seed {seed}")
                if kind == "prism":
                    self.assertEqual(len(pairwise_intersections(solution.planes)), 3)


if __name__ == "__main__":
    unittest.main()
