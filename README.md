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

## Standalone Windows app

Matview can be shipped like a normal portable Windows app: a windowed app
folder, its own icon, and a per-user Start Menu shortcut so Windows Search
finds it. No installer or administrator rights are required.

Build from the repository:

```powershell
.venv\Scripts\python.exe -m pip install pyinstaller
.venv\Scripts\pyinstaller.exe --noconfirm --clean --onedir --windowed `
  --name Matview --icon assets\matview.ico `
  --version-file packaging\version_info.txt `
  --add-data "assets\matview.ico;assets" `
  --hidden-import scipy.linalg --collect-submodules OpenGL main.py
```

The result is `dist\Matview\Matview.exe`; keep the whole `dist\Matview`
folder together. Register it for the current user so it shows up in the Start
Menu and Windows Search:

```powershell
$ws = New-Object -ComObject WScript.Shell
$lnk = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Matview.lnk"
$s = $ws.CreateShortcut($lnk)
$s.TargetPath = (Resolve-Path "dist\Matview\Matview.exe").Path
$s.WorkingDirectory = (Resolve-Path "dist\Matview").Path
$s.IconLocation = "$($s.TargetPath),0"
$s.Description = "Live 2D/3D matrix and Ax=b plane visualizer"
$s.Save()
```

Type `Matview` in the Start Menu search to launch it. The executable carries
its own icon and sets an AppUserModelID so it groups under Matview on the
taskbar. The app is portable: delete the shortcut to remove the Start Menu
entry; no uninstaller is added, matching the reference ClickDeck setup.

In a frozen build, live rendering and PNG snapshots are self-contained. MP4
export uses the repository's `.venv` (Python + Manim) when the app sits in
`dist\Matview\`; if that source environment is absent, the app reports that
MP4 export needs the development environment.

## Features

- **2D / 3D spaces** — switch at any time; the state sequence is embedded/truncated
  so transforms survive the switch.
- **Ax=b plane mode** — a separate 3D tab type that draws each row of `A` as a
  plane `aᵢ·x = bᵢ` and classifies the intersection: a point, a line, coincident
  planes, all of R³, or no solution (parallel planes / triangular prism).
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
- **Multiple render instances** — a tab bar at the bottom left (`+` new
  transform instance, `+ Ax=b` plane instance, `×` close) holds independent
  viewports, each with its own space, state stack or plane system, camera and
  display settings. **Sync view** keeps the camera shared across instances
  (3D viewpoints fully; 2D tabs share pan/zoom).
- **Top-bar menus** — **Settings** holds all transform display toggles (grid,
  ghost grid showing the original space, basis, unit cell, eigenvectors, trails,
  labels); it is disabled in plane tabs. **Export** holds the Manim quality,
  Render/Cancel, PNG snapshot and Open-video actions.
- **Export** — render the current state sequence with Manim (480p/720p/1080p,
  transform tabs only) or save a PNG snapshot of any viewport.

## Ax=b plane mode

Use **+ Ax=b** in the bottom bar to open a separate 3D plane-intersection
instance. Each row of `A` is a plane; `b` is the right-hand side. The equation
fields accept the same expressions as the matrix editor, and edits update the
planes and classification live.

The **Solution** panel reports `det A`, `rank A`, `rank [A|b]`, the minimum
least-squares residual and the feasible set:

| Intersection | Condition | Display |
|---|---|---|
| One point | `rank A = 3` | yellow solution point and coordinates |
| A line | `rank A = rank [A\|b] = 2` | yellow line through `p + t d` |
| A plane | `rank A = rank [A\|b] = 1` | coincident planes; the whole plane solves the system |
| All of R³ | both ranks `0` | `0 = 0` equations; every point solves it |
| No solution | `rank [A\|b] > rank A` | parallel planes, a triangular prism, or a conflicting zero row |

Presets cover unique, line, coincident, parallel, prism and near-singular
systems. **Randomize selected preset** generates another system of whichever
family is currently selected. The viewport draws dashed pairwise-intersection
traces wherever two planes meet, and the display toggles control the plane
fills, in-plane grids, normal arrows, pairwise traces, the all-three
intersection marker and labels, and the optional spatial lattice. **Fit view to solution** reframes the camera; `R`
(double-click or **RESET VIEW**) does the same. Camera orbit, pan, zoom,
orthographic mode and the ISO/TOP/FRONT/RIGHT presets work as in 3D transform
tabs.

Plane tabs are always 3D: the 2D/3D switch and the transform Settings menu are
disabled there. **Render MP4** is transform-only; **Save PNG snapshot** works
for plane tabs.

## Controls

| Key / input | Action |
|---|---|
| `1` / `2` | 2D / 3D (transform tabs only) |
| `Enter` (matrix cell) | Apply the pending matrix and jump back to the first cell (transform tabs) |
| `Space` | Replay the full stack from the identity (transform tabs) |
| `Ctrl+Z` | Remove the last state (transform tabs) |
| `R` / double-click / RESET VIEW | Reset the camera; in a plane tab, fit the solution |
| `F` | Orthographic / perspective (3D) |
| `G` `B` `U` `E` `T` | Grid, basis, unit cell, eigenvectors, trails (transform tabs) |
| Drag / Shift+drag / wheel | Orbit / pan / zoom (2D: pan / zoom) |
| **+ Ax=b** (bottom bar) | Open a plane-intersection instance |
| Fit view to solution | Plane tab: reframe the camera on the current solution |

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
assets/matview.ico      Windows application icon
tools/make_icon.py      icon generator (development helper, needs Pillow)
packaging/version_info.txt  Windows EXE version resource
matview/core.py        pure math: paths, transitions, eigenvalues, presets
matview/planes.py      Ax=b classification, plane records, formatting
matview/camera.py      orbit camera, perspective/orthographic projection
matview/gl.py          shaders, screen-space thick lines, batched meshes
matview/viewport.py    live renderer, animation timeline, interaction
matview/plane_viewport.py  plane fills, intersection line/point, plane labels
matview/controls.py    matrix editor, presets, states, vectors
matview/plane_ui.py    Ax=b equation editor, solution panel, plane presets
matview/export.py      Manim scene generation + QProcess render
matview/app.py         window, tabs, toolbar menus, shortcuts, theme
tests/test_planes.py   classification invariants
```

The export writes generated scenes and videos under `exports/`.

Run the plane classification tests with:

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## License

MIT © 2026 Stanley Chen. See [LICENSE](LICENSE).
