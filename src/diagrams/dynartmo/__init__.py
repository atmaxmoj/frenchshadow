"""DYNARTmo-backed midsagittal articulation diagrams for French phonemes.

Renders directly to SVG polylines from the model's contour coordinates (no
matplotlib at runtime), themed to the Lucerne palette.  Because the underlying
model is parametric, every French phoneme is drawable — see :mod:`.phone_params`.

Model © Bernd J. Kröger, CC-BY 4.0 (arXiv:2507.20343); see :mod:`.model`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .model import build_polylines
from .phone_params import params_for

# Lucerne palette (SVGs are standalone; they cannot read page CSS variables).
_INK = "#4a3826"       # main contour
_MUTED = "#a08a6c"     # static bones/palate
_TONGUE = "#2f6db0"    # the active articulator (blue — colorblind-safe)
_LABEL = "#5b4636"

_TARGET_W = 240.0      # rendered width in px; height follows the aspect ratio
_PAD = 12.0


def has_sagittal(phone: str) -> bool:
    """True if *phone* can be drawn by the DYNARTmo renderer."""
    return params_for(phone) is not None


def _points_to_str(pts: np.ndarray, minx: float, maxy: float, scale: float) -> str:
    # Flip Y (model y points up, SVG y points down) and scale into px.
    xs = (pts[:, 0] - minx) * scale + _PAD
    ys = (maxy - pts[:, 1]) * scale + _PAD
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


@lru_cache(maxsize=256)
def render_sagittal(phone: str) -> str | None:
    """Return an SVG string for *phone*, or None if unsupported."""
    params = params_for(phone)
    if params is None:
        return None

    structure, contour = build_polylines(params)
    all_lines = structure + contour

    pts_all = np.vstack([p for p, _ in all_lines])
    minx, maxx = float(pts_all[:, 0].min()), float(pts_all[:, 0].max())
    miny, maxy = float(pts_all[:, 1].min()), float(pts_all[:, 1].max())
    scale = _TARGET_W / max(maxx - minx, 1e-6)
    w = (maxx - minx) * scale + 2 * _PAD
    h = (maxy - miny) * scale + 2 * _PAD

    parts: list[str] = []
    for pts, _ in structure:
        parts.append(
            f'<polyline points="{_points_to_str(pts, minx, maxy, scale)}" '
            f'fill="none" stroke="{_MUTED}" stroke-width="1.2" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
    for pts, is_tongue in contour:
        color = _TONGUE if is_tongue else _INK
        width = 2.6 if is_tongue else 1.7
        parts.append(
            f'<polyline points="{_points_to_str(pts, minx, maxy, scale)}" '
            f'fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    safe = phone.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    label = (
        f'<text x="{w - _PAD:.1f}" y="{_PAD + 14:.1f}" text-anchor="end" '
        f'fill="{_LABEL}" font-size="18" font-weight="bold" '
        f'font-family="Georgia, serif">/{safe}/</text>'
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w:.1f} {h:.1f}" width="{w:.0f}" height="{h:.0f}">'
        f'{"".join(parts)}{label}</svg>'
    )
