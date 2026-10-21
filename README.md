# Matview

A live 2D/3D linear-transformation visualizer with a Manim export path.

Manim renders video; it cannot drive an interactive orbit camera. So the app
splits the job in two:

- **Live viewport** — a real-time OpenGL 3.3 renderer (PySide6 + PyOpenGL) that
  runs at the display refresh rate (165 Hz here) with free orbit/pan/zoom.
- **Manim export** — the current state sequence is rendered to an MP4 by a
  generated Manim scene that replays the *same* geodesic matrix paths from
  `matview/core.py`, with Manim's grid, vectors and typography.

## Install and run

Requires Python 3.11+ (developed on 3.13). No LaTeX install is needed — the
Manim export uses `Text`, not TeX.

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
run.bat
```

`run.bat` starts the app with `pythonw.exe` and returns immediately, so no
console window stays open. For debugging with a console:

```bat
.venv\Scripts\python.exe main.py
```

Command line (useful for reproducible screenshots):

```bat
python main.py --dim 3 --stack "rotz:35;scale:1.6,0.7,1" --vector "1.2,1.1,1.3"
python main.py --dim 2 --stack "rot:25;scale:1.8,0.6" --matrix "1,0.7,0,1" --shot out.png
```

`--stack` accepts `rot:30`, `rotx:45`, `scale:2,0.5`, `shear:1`,
`reflect:x`, `project:y`, `rand`, `eye`, or raw row-major entries.

## Features

- **2D / 3D spaces** — switch at any time; the state sequence is embedded/truncated
  so transforms survive the switch.
- **Matrix editor** — entries accept numbers or expressions (`sqrt(2)/2`,
  `pi/4`, `2**0.5`, `cos(pi/3)`); trig functions use radians and `^` is accepted
  as a power. Live determinant, transpose, inverse, identity. The pending matrix
  is drawn as a cyan ghost over the current space.
- **Rotation** — angle slider, with x/y/z axis selection in 3D; it writes the
  rotation into the matrix editor.
- **States** — the list is the sequence of *spaces*: the original identity
  state, then the space after each transformation. Clicking a row shows that
  state (view-only: no matrix is re-applied and the list is not modified).
  Apply appends a new state; Remove drops one; Undo/Clear return to the
  original. Replay walks the whole sequence from the identity.
- **Vectors** — add/edit/delete coloured vectors with labels; basis vectors and
  the unit square/cube are shown by default.
- **3D lattice skeleton** — 3D mode draws a full cubic lattice; every matrix
  warps the whole lattice, so volume change and shear are visible in space.
- **Analysis** — live composite matrix, determinant (area/volume factor,
  orientation), real eigenvalues and dashed eigenvector directions.
- **Camera** — drag orbit, Shift+drag pan, wheel zoom, double-click reset,
  isometric/top/front/right presets, orthographic/perspective toggle. 2D is a
  locked top-down view with pan/zoom. **Sync view** shares the 3D camera across
  render instances (2D tabs share pan/zoom).
- **Animation** — every apply animates along the matrix group geodesic
  (`exp(s log M)`), with exact rotation paths (including 180°) and a linear
  fallback for singular maps. Basis tips leave motion trails.
- **Multiple render instances** — a tab bar at the bottom left (`+` new, `×`
  close) holds independent viewports, each with its own space, state stack,
  camera and display settings. **Sync view** keeps the camera shared across
  instances (3D viewpoints fully; 2D tabs share pan/zoom).
- **Top-bar menus** — **Settings** holds all display toggles (grid, ghost grid
  showing the original space, basis, unit cell, eigenvectors, trails, labels). **Export** holds the Manim
  quality, Render/Cancel, PNG snapshot and Open-video actions.
- **Export** — render the current state sequence with Manim (480p/720p/1080p)
  or save a PNG snapshot of the viewport.

## Controls

| Key / input | Action |
|---|---|
| `1` / `2` | 2D / 3D |
| `Enter` (matrix cell) | Apply the pending matrix and jump back to the first cell |
| `Space` | Replay the full stack from the identity |
| `Ctrl+Z` | Remove the last state |
| `R` | Reset camera |
| `F` | Orthographic / perspective (3D) |
| `G` `B` `U` `E` `T` | Grid, basis, unit cell, eigenvectors, trails |
| Drag / Shift+drag / wheel | Orbit / pan / zoom (2D: pan / zoom) |

## Performance

Measured on this machine (RTX 3060, 1476x1220 framebuffer, 4x MSAA):

| View | GL render | + text overlay | Budget at 165 Hz |
|---|---|---|---|
| 2D | 1.4 ms | 3.1 ms | 6.06 ms |
| 3D | 3.8 ms | 4.1 ms | 6.06 ms |

The animation clock is `QElapsedTimer`-based; the render loop is paced by
`frameSwapped` (vsync), with a 10 ms watchdog only for occluded windows. Under
a 3 s transform the measured rate is 164.7–165.3 fps. Idle frames are not
redrawn.

Startup is ~1.1 s (`scipy.linalg`, used only for matrix logarithms, is imported
lazily and warmed in the background).

## Layout

```
main.py                  entry point + CLI state/screenshot helpers
matview/core.py        pure math: paths, transitions, eigenvalues, presets
matview/camera.py      orbit camera, perspective/orthographic projection
matview/gl.py          shaders, screen-space thick lines, batched meshes
matview/viewport.py    live renderer, animation timeline, interaction
matview/controls.py    matrix editor, presets, states, vectors
matview/export.py      Manim scene generation + QProcess render
matview/app.py         window, tabs, toolbar menus, shortcuts, theme
```

The export writes generated scenes and videos under `exports/`.
