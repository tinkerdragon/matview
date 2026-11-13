"""Generate the Matview Windows icon (development helper, needs Pillow).

The mark is matrix brackets around a sheared 2x2 grid with a vector arrow:
matrix, transformed space and a vector in one simple silhouette.  The 16, 20
and 24 px frames use a supersampled bracket-and-arrow silhouette so the small
Start Menu/list icon stays clean.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

BG_TOP = (12, 18, 27)
BG_BOTTOM = (5, 9, 14)
CYAN = (88, 196, 221)
YELLOW = (245, 213, 71)
BORDER = (38, 62, 84)


def _point(size: int, x: float, y: float) -> tuple[int, int]:
    return int(round(x * size)), int(round(y * size))


def _rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=max(1, radius), fill=255
    )
    return mask


def _background(size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gradient = Image.new("RGB", (1, size))
    pixels = gradient.load()
    for y in range(size):
        t = y / max(1, size - 1)
        pixels[0, y] = tuple(
            int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3)
        )
    gradient = gradient.resize((size, size))
    image.paste(gradient, (0, 0), _rounded_mask(size, int(size * 0.22)))
    return image


def _draw_brackets(
    draw: ImageDraw.ImageDraw,
    size: int,
    detailed: bool,
    width: int | None = None,
    arm: int | None = None,
) -> None:
    if width is None:
        width = max(1, int(size * (0.026 if detailed else 0.030)))
    if arm is None:
        arm = max(1, int(size * (0.060 if detailed else 0.100)))
    top, bottom = int(size * 0.255), int(size * 0.745)
    left, right = int(size * 0.205), int(size * 0.795)
    color = (*CYAN, 255)
    draw.line((left, top, left, bottom), fill=color, width=width)
    draw.line((left, top, left + arm, top), fill=color, width=width)
    draw.line((left, bottom, left + arm, bottom), fill=color, width=width)
    draw.line((right, top, right, bottom), fill=color, width=width)
    draw.line((right, top, right - arm, top), fill=color, width=width)
    draw.line((right, bottom, right - arm, bottom), fill=color, width=width)


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    size: int,
    width: int,
    head: int,
    half: int,
) -> None:
    origin = (0.315, 0.665)
    tip = (0.700, 0.315)
    shaft_end = (
        origin[0] + (tip[0] - origin[0]) * 0.76,
        origin[1] + (tip[1] - origin[1]) * 0.76,
    )
    draw.line(
        [_point(size, *origin), _point(size, *shaft_end)],
        fill=(*YELLOW, 255),
        width=width,
    )
    dx = tip[0] - shaft_end[0]
    dy = tip[1] - shaft_end[1]
    length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip_px = _point(size, *tip)
    p1 = (
        int(round(tip_px[0] - ux * head + px * half)),
        int(round(tip_px[1] - uy * head + py * half)),
    )
    p2 = (
        int(round(tip_px[0] - ux * head - px * half)),
        int(round(tip_px[1] - uy * head - py * half)),
    )
    draw.polygon([tip_px, p1, p2], fill=(*YELLOW, 255))


def render(size: int) -> Image.Image:
    """Detailed frame used at 32 px and above."""
    image = _background(size)
    if size >= 64:
        glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(glow)
        gdraw.ellipse(
            (
                int(-0.25 * size),
                int(-0.35 * size),
                int(0.75 * size),
                int(0.55 * size),
            ),
            fill=(88, 196, 221, 26),
        )
        glow = glow.filter(ImageFilter.GaussianBlur(max(1, int(size * 0.10))))
        image = Image.alpha_composite(
            image,
            Image.composite(
                glow,
                Image.new("RGBA", image.size),
                _rounded_mask(size, int(size * 0.22)),
            ),
        )
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle(
        (
            int(size * 0.025),
            int(size * 0.025),
            int(size * 0.975),
            int(size * 0.975),
        ),
        radius=max(1, int(size * 0.20)),
        outline=(*BORDER, 190),
        width=max(1, int(size * 0.006)),
    )
    _draw_brackets(draw, size, detailed=True)

    # A unit square sheared to the right: the visual signature of a linear map.
    a = (0.315, 0.665)
    b = (0.500, 0.665)
    c = (0.675, 0.470)
    d = (0.490, 0.470)
    quad = [_point(size, *p) for p in (a, b, c, d)]
    draw.polygon(quad, fill=(*CYAN, 28))
    draw.line(
        quad + [quad[0]],
        fill=(*CYAN, 180),
        width=max(1, int(size * 0.008)),
        joint="curve",
    )
    if size >= 48:
        def mid(p, q):
            return (p[0] + (q[0] - p[0]) * 0.5, p[1] + (q[1] - p[1]) * 0.5)

        grid = (*CYAN, 90)
        width = max(1, int(size * 0.004))
        draw.line(
            [_point(size, *mid(a, b)), _point(size, *mid(d, c))],
            fill=grid,
            width=width,
        )
        draw.line(
            [_point(size, *mid(a, d)), _point(size, *mid(b, c))],
            fill=grid,
            width=width,
        )

    _draw_arrow(
        draw,
        size,
        width=max(1, int(size * 0.030)),
        head=max(2, int(size * 0.085)),
        half=max(1, int(size * 0.050)),
    )
    if size >= 48:
        ox, oy = _point(size, *a)
        dot = max(1, int(size * 0.010))
        draw.ellipse(
            (ox - dot, oy - dot, ox + dot, oy + dot), fill=(235, 246, 255, 240)
        )
    return image


def render_small(target: int, factor: int = 4) -> Image.Image:
    """Antialiased bracket-and-arrow frame for 16-24 px icons."""
    size = target * factor
    image = _background(size)
    draw = ImageDraw.Draw(image)
    # Small frames drop the outer hairline; at 16 px it only adds a grey ring.
    # Widths are authored in target pixels, then rendered at 4x and downsampled.
    _draw_brackets(
        draw,
        size,
        detailed=False,
        width=max(1, round(target * 0.030)) * factor,
        arm=max(1, round(target * 0.100)) * factor,
    )
    _draw_arrow(
        draw,
        size,
        width=max(1, round(target * 0.040)) * factor,
        head=max(2, round(target * 0.100)) * factor,
        half=max(1, round(target * 0.050)) * factor,
    )
    return image.resize((target, target), Image.Resampling.LANCZOS)


def build_ico() -> None:
    base = render(256)
    extras = [render_small(size) for size in (16, 20, 24)]
    extras += [render(size) for size in (32, 40, 48, 64, 128)]
    base.save(
        ASSETS / "matview.ico",
        format="ICO",
        sizes=[
            (16, 16),
            (20, 20),
            (24, 24),
            (32, 32),
            (40, 40),
            (48, 48),
            (64, 64),
            (128, 128),
            (256, 256),
        ],
        append_images=extras,
    )


if __name__ == "__main__":
    build_ico()
    render(512).save(ASSETS / "matview.png")
    print(f"Wrote {ASSETS / 'matview.ico'} and matview.png")
