"""Spherical orbit camera with z-up world space.

The 2D view is the same camera locked to a top-down orthographic view; only
pan and zoom are live there. The 3D view adds free orbit and a perspective
projection that can be toggled to orthographic.
"""

from __future__ import annotations

import math

import numpy as np

FOV = math.radians(38.0)
NEAR = 0.05
FAR = 900.0


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return vector.copy()
    return vector / norm


class Camera:
    def __init__(self) -> None:
        self.target = np.zeros(3)
        self.yaw = math.radians(-56.0)
        self.pitch = math.radians(26.0)
        self.distance = 21.0
        self.fov = FOV
        self.ortho = False
        self.ortho_height = 7.5
        self.near = NEAR
        self.far = FAR

    # -- view ---------------------------------------------------------------
    def set_mode(self, dim: int) -> None:
        self.target[:] = 0.0
        if dim == 2:
            self.ortho = True
            self.ortho_height = 7.5
            self.distance = 30.0
            self.yaw = -math.pi / 2.0
            self.pitch = math.pi / 2.0
        else:
            self.ortho = False
            self.ortho_height = 7.5
            self.distance = 13.0
            self.yaw = math.radians(-56.0)
            self.pitch = math.radians(26.0)

    def eye(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        direction = np.array(
            [cp * math.cos(self.yaw), cp * math.sin(self.yaw), math.sin(self.pitch)]
        )
        return self.target + self.distance * direction

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = _normalize(self.target - self.eye())
        up_hint = np.array([0.0, 1.0, 0.0]) if abs(forward[2]) > 0.999 else np.array([0.0, 0.0, 1.0])
        right = _normalize(np.cross(forward, up_hint))
        up = np.cross(right, forward)
        return right, up, forward

    def view_matrix(self) -> np.ndarray:
        right, up, forward = self.basis()
        eye = self.eye()
        rotation = np.array([right, up, -forward])
        view = np.eye(4)
        view[:3, :3] = rotation
        view[:3, 3] = -rotation @ eye
        return view

    def projection_matrix(self, aspect: float) -> np.ndarray:
        aspect = max(1e-4, aspect)
        if self.ortho:
            half_h = self.ortho_height * 0.5
            half_w = half_h * aspect
            proj = np.eye(4)
            proj[0, 0] = 1.0 / half_w
            proj[1, 1] = 1.0 / half_h
            proj[2, 2] = -2.0 / (self.far - self.near)
            proj[2, 3] = -(self.far + self.near) / (self.far - self.near)
            return proj
        t = 1.0 / math.tan(self.fov * 0.5)
        proj = np.zeros((4, 4))
        proj[0, 0] = t / aspect
        proj[1, 1] = t
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = 2.0 * self.far * self.near / (self.near - self.far)
        proj[3, 2] = -1.0
        return proj

    def world_per_pixel(self, viewport_height: float) -> float:
        height = max(1.0, float(viewport_height))
        if self.ortho:
            return self.ortho_height / height
        return 2.0 * self.distance * math.tan(self.fov * 0.5) / height

    # -- interaction --------------------------------------------------------
    def orbit(self, dx: float, dy: float) -> None:
        self.yaw -= dx * 0.009
        self.pitch = min(
            math.pi / 2.0 - 0.02, max(-math.pi / 2.0 + 0.02, self.pitch + dy * 0.009)
        )

    def pan(self, dx: float, dy: float, viewport_height: float) -> None:
        right, up, _ = self.basis()
        scale = self.world_per_pixel(viewport_height)
        self.target -= right * dx * scale
        self.target += up * dy * scale

    def zoom(self, steps: float) -> None:
        factor = math.exp(-steps * 0.13)
        if self.ortho:
            self.ortho_height = min(400.0, max(0.35, self.ortho_height * factor))
        else:
            self.distance = min(400.0, max(0.6, self.distance / factor))

    # -- presets ------------------------------------------------------------
    def set_view(self, name: str) -> None:
        if name == "iso":
            self.yaw, self.pitch = math.radians(-56.0), math.radians(26.0)
        elif name == "top":
            self.yaw, self.pitch = -math.pi / 2.0, math.pi / 2.0 - 0.001
        elif name == "front":
            self.yaw, self.pitch = -math.pi / 2.0, 0.0
        elif name == "right":
            self.yaw, self.pitch = 0.0, 0.0
        elif name == "back":
            self.yaw, self.pitch = math.pi / 2.0, 0.0

    def reset(self, dim: int, keep_projection: bool = False) -> None:
        ortho = self.ortho
        self.set_mode(dim)
        if keep_projection and dim == 3:
            self.ortho = ortho
