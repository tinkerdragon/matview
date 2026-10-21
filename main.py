"""Entry point: python main.py [--dim 2|3] [--stack "..."] [--shot out.png]."""

from __future__ import annotations

import argparse
import sys

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont, QPalette, QColor, QSurfaceFormat
from PySide6.QtWidgets import QApplication

from matview import core
from matview.app import MainWindow
from matview.expr import evaluate
from matview.viewport import VectorSpec


def _configure_surface() -> None:
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)


def _palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#0A0E15"))
    palette.setColor(QPalette.WindowText, QColor("#C9D4E3"))
    palette.setColor(QPalette.Base, QColor("#080C12"))
    palette.setColor(QPalette.AlternateBase, QColor("#0C1119"))
    palette.setColor(QPalette.Text, QColor("#C9D4E3"))
    palette.setColor(QPalette.Button, QColor("#16202C"))
    palette.setColor(QPalette.ButtonText, QColor("#C9D4E3"))
    palette.setColor(QPalette.Highlight, QColor("#2B6C7E"))
    palette.setColor(QPalette.HighlightedText, QColor("#EAF6FA"))
    palette.setColor(QPalette.ToolTipBase, QColor("#16202C"))
    palette.setColor(QPalette.ToolTipText, QColor("#C9D4E3"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#4E5D70"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#4E5D70"))
    return palette


def _matrix_from_text(text: str, dim: int) -> np.ndarray:
    try:
        values = [evaluate(part) for part in text.replace(";", ",").split(",") if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid matrix expression: {exc}")
    if len(values) != dim * dim:
        raise SystemExit(f"matrix needs {dim * dim} entries, got {len(values)}")
    return np.array(values, dtype=float).reshape(dim, dim)


def _stack_from_spec(spec: str, dim: int) -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for token in spec.split(";"):
        token = token.strip()
        if not token:
            continue
        name, _, rest = token.partition(":")
        name = name.lower()
        rest = rest.strip()
        if name in ("rot", "rotx", "roty", "rotz"):
            axis = {"rot": 2, "rotx": 0, "roty": 1, "rotz": 2}[name]
            matrices.append(core.rotation_matrix(dim, float(rest or 0), axis))
        elif name in ("scale", "s"):
            values = [float(part) for part in rest.split(",") if part.strip()]
            if len(values) == 1:
                values = values * dim
            matrices.append(core.scale_matrix(dim, *values))
        elif name in ("shear", "sh"):
            matrices.append(core.shear_matrix(dim, float(rest or 0)))
        elif name in ("reflect", "ref"):
            matrices.append(core.reflection_matrix(dim, (rest or "x").lower()))
        elif name in ("project", "proj"):
            matrices.append(core.projection_matrix(dim, (rest or "x").lower()))
        elif name in ("rand", "random"):
            matrices.append(core.random_matrix(dim))
        elif name in ("eye", "identity"):
            matrices.append(np.eye(dim))
        else:
            matrices.append(_matrix_from_text(token, dim))
    return matrices


def _apply_cli_state(window: MainWindow, args) -> None:
    dim = 2 if args.dim == 2 else 3
    if dim != window.viewport.dim:
        window.set_mode(dim)
    viewport = window.viewport
    if args.stack:
        viewport.stack = _stack_from_spec(args.stack, dim)
        viewport.active = len(viewport.stack)
        viewport.displayed = core.compose(viewport.stack, dim)
        viewport.timeline = None
    if args.matrix:
        viewport.set_editor(_matrix_from_text(args.matrix, dim))
    for raw in args.vector or []:
        values = [float(part) for part in raw.split(",") if part.strip()]
        if len(values) < dim:
            continue
        coords = np.zeros(3)
        coords[:dim] = values[:dim]
        viewport.vectors.append(
            VectorSpec(
                f"v{len(viewport.vectors) + 1}",
                coords,
                core.MANIM_COLORS[(len(viewport.vectors) + 3) % len(core.MANIM_COLORS)],
            )
        )
    viewport.stateChanged.emit()
    window.panel.refresh()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live 2D/3D matrix transformation visualizer")
    parser.add_argument("--dim", type=int, choices=(2, 3), default=2)
    parser.add_argument("--stack", default="", help="e.g. 'rot:30;scale:2,0.5'")
    parser.add_argument("--matrix", default="", help="pending matrix entries, row major")
    parser.add_argument("--vector", action="append", help="add a vector, e.g. '2,1'")
    parser.add_argument("--shot", default="", help="save a screenshot and exit")
    parser.add_argument("--delay", type=int, default=1400, help="screenshot delay in ms")
    args, qt_args = parser.parse_known_args(argv)

    _configure_surface()
    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("Matview")
    app.setStyle("Fusion")
    app.setPalette(_palette())
    app.setFont(QFont("Segoe UI", 9))

    window = MainWindow()
    _apply_cli_state(window, args)
    window.show()
    window.raise_()
    window.activateWindow()
    QApplication.processEvents()

    if args.shot:
        def capture() -> None:
            window.viewport.repaint()
            QApplication.processEvents()
            framebuffer = window.viewport.grabFramebuffer()
            framebuffer.save(args.shot.rsplit(".", 1)[0] + "-gl.png")
            window.repaint()
            QApplication.processEvents()
            image = window.grab()
            image.save(args.shot)
            app.quit()

        QTimer.singleShot(args.delay, capture)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
