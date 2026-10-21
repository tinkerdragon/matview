"""Small OpenGL 3.3 core toolkit for the visualizer.

Two primitive types are batched:

* thick line segments -- expanded to screen-space quads in the vertex shader so
  stroke width stays constant in pixels at any zoom level;
* triangles -- flat fills for 2D arrow heads, translucent faces and 3D cones,
  optionally Lambert shaded.

Geometry is rebuilt on the CPU (tiny vertex counts) and uploaded to a single
reused buffer per batch.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np
from OpenGL import GL

LINE_STRIDE = 12  # pos3 tangent3 side1 extend1 color4
TRI_STRIDE = 10  # pos3 normal3 color4


# --------------------------------------------------------------------------- #
# colour helpers
# --------------------------------------------------------------------------- #
def hex_rgba(value: str, alpha: float = 1.0) -> np.ndarray:
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    rgb = np.array([int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=float)
    return np.array([rgb[0], rgb[1], rgb[2], alpha], dtype=float)


def model_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    out = np.eye(4)
    out[:n, :n] = matrix
    return out


# --------------------------------------------------------------------------- #
# shaders
# --------------------------------------------------------------------------- #
LINE_VS = """#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_tan;
layout(location = 2) in float a_side;
layout(location = 3) in float a_ext;
layout(location = 4) in vec4 a_color;

uniform mat4 u_viewproj;
uniform mat4 u_model;
uniform vec2 u_res;
uniform float u_width;
uniform float u_alpha;
uniform vec3 u_tint;

out vec4 v_color;

void main() {
    vec4 c0 = u_viewproj * u_model * vec4(a_pos, 1.0);
    vec4 c1 = u_viewproj * u_model * vec4(a_pos + a_tan, 1.0);
    vec2 s0 = (c0.xy / c0.w) * u_res * 0.5;
    vec2 s1 = (c1.xy / c1.w) * u_res * 0.5;
    vec2 d = s1 - s0;
    float len = length(d);
    vec2 dir = len > 1e-6 ? d / len : vec2(1.0, 0.0);
    vec2 nrm = vec2(-dir.y, dir.x);
    vec2 offset = (nrm * a_side + dir * a_ext) * (u_width * 0.5);
    c0.xy += offset * (2.0 / u_res) * c0.w;
    gl_Position = c0;
    v_color = vec4(a_color.rgb * u_tint, a_color.a * u_alpha);
}
"""

TRI_VS = """#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 a_color;

uniform mat4 u_viewproj;
uniform mat4 u_model;
uniform vec3 u_light;
uniform float u_lit;

out vec4 v_color;

void main() {
    vec4 world = u_model * vec4(a_pos, 1.0);
    gl_Position = u_viewproj * world;
    vec3 n = normalize(mat3(u_model) * a_normal);
    float diff = 0.45 + 0.55 * max(dot(n, normalize(u_light)), 0.0);
    v_color = vec4(a_color.rgb * mix(1.0, diff, u_lit), a_color.a);
}
"""

FRAG = """#version 330 core
in vec4 v_color;
out vec4 frag_color;
void main() {
    frag_color = v_color;
}
"""


def _compile(kind: int, source: str) -> int:
    shader = GL.glCreateShader(kind)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader)
        if isinstance(log, bytes):
            log = log.decode("utf-8", "replace")
        raise RuntimeError(f"shader compile failed:\n{log}\n{source}")
    return shader


class Shader:
    def __init__(self, vertex_source: str, fragment_source: str) -> None:
        program = GL.glCreateProgram()
        vertex = _compile(GL.GL_VERTEX_SHADER, vertex_source)
        fragment = _compile(GL.GL_FRAGMENT_SHADER, fragment_source)
        GL.glAttachShader(program, vertex)
        GL.glAttachShader(program, fragment)
        GL.glLinkProgram(program)
        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            log = GL.glGetProgramInfoLog(program)
            if isinstance(log, bytes):
                log = log.decode("utf-8", "replace")
            raise RuntimeError(f"program link failed:\n{log}")
        GL.glDeleteShader(vertex)
        GL.glDeleteShader(fragment)
        self.id = program
        self._uniforms: dict[str, int] = {}

    def use(self) -> None:
        GL.glUseProgram(self.id)

    def uniform(self, name: str) -> int:
        location = self._uniforms.get(name)
        if location is None:
            location = GL.glGetUniformLocation(self.id, name)
            self._uniforms[name] = location
        return location

    def mat4(self, name: str, matrix: np.ndarray) -> None:
        data = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32).T)
        GL.glUniformMatrix4fv(self.uniform(name), 1, GL.GL_FALSE, data)

    def vec2(self, name: str, x: float, y: float) -> None:
        GL.glUniform2f(self.uniform(name), float(x), float(y))

    def vec3(self, name: str, x: float, y: float, z: float) -> None:
        GL.glUniform3f(self.uniform(name), float(x), float(y), float(z))

    def float(self, name: str, value: float) -> None:
        GL.glUniform1f(self.uniform(name), float(value))

    def dispose(self) -> None:
        GL.glDeleteProgram(self.id)


# --------------------------------------------------------------------------- #
# geometry builders
# --------------------------------------------------------------------------- #
def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=1)
    lengths[lengths < 1e-12] = 1.0
    return vectors / lengths[:, None]


def segment_vertices(starts, ends, color: np.ndarray | str, alpha: float = 1.0) -> np.ndarray:
    """Triangles for one or more independent segments."""
    a = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(ends, dtype=np.float64).reshape(-1, 3)
    count = len(a)
    if count == 0:
        return np.zeros((0, LINE_STRIDE), dtype=np.float32)

    tangent = b - a
    tangent = _unit_rows(tangent)

    if isinstance(color, str):
        colors = np.broadcast_to(hex_rgba(color, alpha), (count, 4)).astype(np.float64)
    else:
        colors = np.asarray(color, dtype=np.float64)
        if colors.ndim == 1:
            colors = np.broadcast_to(colors, (count, 4))
        colors = colors.copy()
        colors[:, 3] *= alpha

    out = np.empty((count * 6, LINE_STRIDE), dtype=np.float32)
    positions = np.empty((count * 6, 3), dtype=np.float64)
    tangents = np.empty((count * 6, 3), dtype=np.float64)
    positions[0::6] = a
    positions[1::6] = a
    positions[2::6] = b
    positions[3::6] = a
    positions[4::6] = b
    positions[5::6] = b
    tangents[0::6] = tangent
    tangents[1::6] = tangent
    tangents[2::6] = tangent
    tangents[3::6] = tangent
    tangents[4::6] = tangent
    tangents[5::6] = tangent

    out[:, 0:3] = positions
    out[:, 3:6] = tangents
    out[:, 6] = np.tile(np.array([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0]), count)
    out[:, 7] = np.tile(np.array([-1.0, -1.0, 1.0, -1.0, 1.0, 1.0]), count)
    out[:, 8:12] = np.repeat(colors, 6, axis=0)
    return out


def polyline_vertices(points, color: np.ndarray | str, alpha: float = 1.0, closed: bool = False) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 2:
        return np.zeros((0, LINE_STRIDE), dtype=np.float32)
    if closed:
        points = np.vstack([points, points[:1]])
    return segment_vertices(points[:-1], points[1:], color, alpha)


def flat_triangles_vertices(triangles, color: np.ndarray | str, alpha: float = 1.0) -> np.ndarray:
    """``triangles`` is (n, 3, 3) -- three world points per triangle."""
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.size == 0:
        return np.zeros((0, TRI_STRIDE), dtype=np.float32)
    count = len(triangles)
    colors = np.broadcast_to(
        hex_rgba(color, alpha) if isinstance(color, str) else np.asarray(color, dtype=float),
        (count * 3, 4),
    )
    out = np.empty((count * 3, TRI_STRIDE), dtype=np.float32)
    out[:, 0:3] = triangles.reshape(-1, 3)
    out[:, 3:6] = np.tile(np.array([0.0, 0.0, 1.0]), (count * 3, 1))
    out[:, 6:10] = colors
    return out


def cone_vertices(
    tip,
    direction,
    length: float,
    radius: float,
    color: str,
    alpha: float = 1.0,
    segments: int = 18,
) -> np.ndarray:
    tip = np.asarray(tip, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-12 or length < 1e-9 or radius < 1e-9:
        return np.zeros((0, TRI_STRIDE), dtype=np.float32)
    direction = direction / norm
    base = tip - direction * length
    helper = np.array([0.0, 0.0, 1.0]) if abs(direction[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(direction, helper)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    ring = base + radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)

    rgba = hex_rgba(color, alpha)
    tri_list = []
    normal_list = []
    for i in range(segments):
        p0 = ring[i]
        p1 = ring[(i + 1) % segments]
        normal = np.cross(p0 - tip, p1 - tip)
        nlen = np.linalg.norm(normal)
        normal = normal / nlen if nlen > 1e-12 else direction
        if np.dot(normal, (p0 + p1) * 0.5 - base) < 0:
            normal = -normal
        tri_list.append([tip, p0, p1])
        normal_list.append([normal, normal, normal])
        tri_list.append([base, p1, p0])
        normal_list.append([-direction, -direction, -direction])

    tris = np.asarray(tri_list)
    normals = np.asarray(normal_list).reshape(-1, 3)
    out = np.empty((len(tris) * 3, TRI_STRIDE), dtype=np.float32)
    out[:, 0:3] = tris.reshape(-1, 3)
    out[:, 3:6] = normals
    out[:, 6:10] = np.broadcast_to(rgba, (len(tris) * 3, 4))
    return out


# --------------------------------------------------------------------------- #
# batches
# --------------------------------------------------------------------------- #
class _Batch:
    def __init__(self) -> None:
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        self.count = 0
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        self._describe()
        GL.glBindVertexArray(0)

    def _describe(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def upload(self, data: np.ndarray) -> None:
        data = np.ascontiguousarray(data, dtype=np.float32)
        self.count = 0 if data.size == 0 else int(data.shape[0])
        if self.count == 0:
            return
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_DYNAMIC_DRAW)
        GL.glBindVertexArray(0)

    def draw(self, count: int | None = None) -> None:
        total = self.count if count is None else min(self.count, int(count))
        if total <= 0:
            return
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, total)
        GL.glBindVertexArray(0)

    def dispose(self) -> None:
        GL.glDeleteBuffers(1, [self.vbo])
        GL.glDeleteVertexArrays(1, [self.vao])


class LineBatch(_Batch):
    def _describe(self) -> None:
        stride = LINE_STRIDE * 4
        specs = [(0, 3, 0), (1, 3, 12), (2, 1, 24), (3, 1, 28), (4, 4, 32)]
        for location, size, offset in specs:
            GL.glEnableVertexAttribArray(location)
            GL.glVertexAttribPointer(
                location, size, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(offset)
            )


class TriBatch(_Batch):
    def _describe(self) -> None:
        stride = TRI_STRIDE * 4
        specs = [(0, 3, 0), (1, 3, 12), (2, 4, 24)]
        for location, size, offset in specs:
            GL.glEnableVertexAttribArray(location)
            GL.glVertexAttribPointer(
                location, size, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(offset)
            )


def make_line_shader() -> Shader:
    return Shader(LINE_VS, FRAG)


def make_tri_shader() -> Shader:
    return Shader(TRI_VS, FRAG)
