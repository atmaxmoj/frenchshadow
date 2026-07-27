"""SVG renderer for declarative phone diagrams.

Knows how to draw the shared mouth-profile canvas and how to map config values
(place, manner, tongue position, rounding) to visual elements.  It does not
know which phones exist; that comes from config files.

The colour palette matches the Lucerne sepia/light theme used by the sibling
vocabulary and grammar apps.
"""

from __future__ import annotations

from .config import ConsonantConfig, VowelConfig

# Lucerne light-theme palette (hard-coded because these SVGs are served as
# standalone image files and cannot read the page CSS variables).
_INK = "#2c2722"
_MUTED = "#8a7f6c"
_LINE = "#d8cfb8"
_CARD = "#ece6d6"
_BLUE = "#2b6cb0"
_ORANGE = "#c1600a"
_TEAL = "#2f7d6b"
_BROWN = "#b05a2c"
_PURPLE = "#9c4f86"
_OLIVE = "#7d6f2e"
_SLATE = "#6a6f8a"
_TAUPE = "#6f6457"
_GOLD = "#b8860b"
_RED = "#c0492f"
_CHESTNUT = "#8a5a44"
_INK_2 = "#5b4636"

_PLACE_STYLE: dict[str, dict[str, object]] = {
    "bilabial": {"x": 35, "y": 60, "color": _BLUE},
    "labiodental": {"x": 50, "y": 60, "color": _SLATE},
    "dental": {"x": 60, "y": 62, "color": _TEAL},
    "alveolar": {"x": 82, "y": 62, "color": _BROWN},
    "postalveolar": {"x": 105, "y": 68, "color": _PURPLE},
    "retroflex": {"x": 120, "y": 72, "color": _OLIVE},
    "alveolopalatal": {"x": 115, "y": 75, "color": _TAUPE},
    "palatal": {"x": 130, "y": 82, "color": _GOLD},
    "velar": {"x": 155, "y": 95, "color": _CHESTNUT},
    "uvular": {"x": 175, "y": 115, "color": _RED},
    "pharyngeal": {"x": 195, "y": 140, "color": _INK_2},
    "glottal": {"x": 220, "y": 165, "color": _INK_2},
}

_PLACE_CN: dict[str, str] = {
    "bilabial": "双唇",
    "labiodental": "唇齿",
    "dental": "齿/齿龈",
    "alveolar": "齿龈",
    "postalveolar": "齿龈后",
    "retroflex": "卷舌",
    "alveolopalatal": "龈腭",
    "palatal": "硬腭",
    "velar": "软腭",
    "uvular": "小舌",
    "pharyngeal": "咽",
    "glottal": "声门",
}

_MANNER_CN: dict[str, str] = {
    "plosive": "塞音",
    "fricative": "擦音",
    "sibilant_fricative": "咝擦音",
    "affricate": "塞擦音",
    "nasal": "鼻音",
    "approximant": "近音",
    "lateral_approximant": "边近音",
}

_HEAD_OUTLINE = (
    "M 20,180 "
    "C 20,120 40,80 70,60 "
    "L 75,55 "
    "C 85,45 100,40 115,45 "
    "L 130,50 "
    "C 160,55 185,70 205,95 "
    "C 225,120 235,150 235,180 "
    "L 235,195 "
    "L 20,195 Z"
)

_GUIDES = (
    f'<path d="M 75,55 Q 110,48 140,58 T 200,95" fill="none" stroke="{_MUTED}" stroke-width="1.5"/>'
    f'<path d="M 200,95 Q 215,110 220,130" fill="none" stroke="{_MUTED}" stroke-width="1.5"/>'
    f'<path d="M 70,60 Q 90,90 120,110 T 180,140" fill="none" stroke="{_MUTED}" stroke-width="1.5"/>'
    f'<line x1="235" y1="150" x2="220" y2="150" stroke="{_MUTED}" stroke-width="1.5"/>'
)


def _safe(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _consonant_manner_svg(config: ConsonantConfig, x: float, y: float, color: str) -> str:
    manner = config.manner
    if manner == "plosive":
        return f'<line x1="{x - 10}" y1="{y}" x2="{x + 10}" y2="{y}" stroke="{color}" stroke-width="4"/>'
    if manner == "fricative":
        return "".join(
            f'<line x1="{x - 8}" y1="{y + o}" x2="{x + 8}" y2="{y - o}" stroke="{color}" stroke-width="1.5"/>'
            for o in (-4, 0, 4)
        )
    if manner == "sibilant_fricative":
        return "".join(
            f'<line x1="{x - 8}" y1="{y + o}" x2="{x + 8}" y2="{y - o}" stroke="{color}" stroke-width="2"/>'
            for o in (-5, 0, 5)
        )
    if manner == "nasal":
        return (
            f'<line x1="{x - 8}" y1="{y}" x2="{x + 8}" y2="{y}" stroke="{color}" stroke-width="3"/>'
            f'<path d="M {x} {y} L {x - 6} {y - 18}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#arrow)"/>'
        )
    if manner == "lateral_approximant":
        return (
            f'<circle cx="{x}" cy="{y}" r="8" fill="none" stroke="{color}" stroke-width="3"/>'
            f'<path d="M {x} {y + 8} L {x + 14} {y + 14}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#arrow)"/>'
        )
    # approximant / affricate
    return (
        f'<circle cx="{x}" cy="{y}" r="5" fill="none" stroke="{color}" stroke-width="2"/>'
        f'<path d="M {x - 10} {y} L {x + 10} {y}" fill="none" stroke="{color}" stroke-width="1.5"/>'
    )


def _consonant_legend(config: ConsonantConfig, color: str) -> str:
    """Return a small legend explaining the diagram's visual encoding."""
    if config.voiced:
        voicing_sample = (
            f'<path d="M 4,52 Q 7,48 10,52 T 16,52" fill="none" stroke="{_BLUE}" stroke-width="1.2"/>'
            f'<text x="18" y="55" fill="{_INK}" font-size="9" font-family="sans-serif">浊音</text>'
        )
    else:
        voicing_sample = f'<text x="4" y="55" fill="{_INK}" font-size="9" font-family="sans-serif">— 清音</text>'

    manner_sample = (
        f'<line x1="4" y1="36" x2="16" y2="36" stroke="{color}" stroke-width="3"/>'
        f'<text x="18" y="39" fill="{_INK}" font-size="9" font-family="sans-serif">发音方法</text>'
    )

    return f'''<g transform="translate(238, 40)">
  <rect x="-4" y="-10" width="64" height="72" rx="6" fill="{_CARD}" stroke="{_LINE}" stroke-width="1" opacity="0.98"/>
  <circle cx="9" cy="6" r="4" fill="{color}"/>
  <text x="16" y="9" fill="{_INK}" font-size="9" font-family="sans-serif">发音部位</text>
  {manner_sample}
  {voicing_sample}
</g>'''


def render_consonant(config: ConsonantConfig, label: str | None = None) -> str:
    """Render a consonant mouth-profile schematic from config."""
    style = _PLACE_STYLE[config.place]
    x, y, color = style["x"], style["y"], style["color"]
    display = label if label is not None else config.phone

    marker = f'<circle cx="{x}" cy="{y}" r="6" fill="{color}" />'
    manner_svg = _consonant_manner_svg(config, x, y, color)
    legend = _consonant_legend(config, color)

    if config.voiced:
        voicing_svg = (
            f'<g transform="translate(225, 170)">'
            f'<path d="M 0,0 Q 5,-5 10,0 T 20,0" fill="none" stroke="{_BLUE}" stroke-width="1.5"/>'
            f'<path d="M 0,6 Q 5,1 10,6 T 20,6" fill="none" stroke="{_BLUE}" stroke-width="1.5"/>'
            f'<text x="22" y="6" fill="{_MUTED}" font-size="9" font-family="sans-serif">voiced</text>'
            '</g>'
        )
    else:
        voicing_svg = f'<text x="225" y="178" fill="{_MUTED}" font-size="9" font-family="sans-serif">voiceless</text>'

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 210" width="300" height="210">
  <defs>
    <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6" fill="{color}"/>
    </marker>
  </defs>
  <text x="150" y="20" text-anchor="middle" fill="{_INK}" font-size="16" font-weight="bold" font-family="sans-serif">/{_safe(display)}/</text>
  <path d="{_HEAD_OUTLINE}" fill="none" stroke="{_INK}" stroke-width="2"/>
  {_GUIDES}
  {marker}
  {manner_svg}
  {voicing_svg}
  {legend}
  <text x="150" y="205" text-anchor="middle" fill="{_MUTED}" font-size="11" font-family="sans-serif">{_PLACE_CN[config.place]} · {_MANNER_CN[config.manner]} · {"浊" if config.voiced else "清"}</text>
</svg>'''


def _vowel_quality_text(config: VowelConfig) -> str:
    parts = []
    if config.y < 88:
        parts.append("高元音")
    elif config.y > 118:
        parts.append("低元音")
    else:
        parts.append("中元音")
    if config.x < 132:
        parts.append("前舌")
    elif config.x > 158:
        parts.append("后舌")
    else:
        parts.append("中央")
    parts.append("圆唇" if config.rounded else "不圆唇")
    return " · ".join(parts)


def _vowel_legend(rounded: bool) -> str:
    """Return a small legend explaining the vowel diagram's visual encoding."""
    if rounded:
        lips_sample = (
            f'<circle cx="9" cy="34" r="5" fill="none" stroke="{_ORANGE}" stroke-width="2"/>'
            f'<text x="18" y="37" fill="{_INK}" font-size="9" font-family="sans-serif">圆唇</text>'
        )
    else:
        lips_sample = (
            f'<line x1="4" y1="34" x2="14" y2="34" stroke="{_ORANGE}" stroke-width="2" stroke-linecap="round"/>'
            f'<text x="18" y="37" fill="{_INK}" font-size="9" font-family="sans-serif">展唇</text>'
        )

    return f'''<g transform="translate(238, 45)">
  <rect x="-4" y="-10" width="64" height="58" rx="6" fill="{_CARD}" stroke="{_LINE}" stroke-width="1" opacity="0.98"/>
  <ellipse cx="9" cy="6" rx="7" ry="5" fill="{_BLUE}"/>
  <text x="18" y="9" fill="{_INK}" font-size="9" font-family="sans-serif">舌位</text>
  {lips_sample}
  <text x="4" y="55" fill="{_MUTED}" font-size="8" font-family="sans-serif">轮廓 = 口腔侧视</text>
</g>'''


def render_vowel(config: VowelConfig, label: str | None = None) -> str:
    """Render a vowel mouth-profile schematic from config."""
    display = label if label is not None else config.phone
    tongue_rx = 22 if config.rounded else 18

    tongue = (
        f'<ellipse cx="{config.x}" cy="{config.y}" rx="{tongue_rx}" ry="16" fill="{_BLUE}" opacity="0.85"/>'
        f'<ellipse cx="{config.x}" cy="{config.y}" rx="{tongue_rx}" ry="16" fill="none" stroke="{_BLUE}" stroke-width="2"/>'
    )

    lip_x = 35
    if config.rounded:
        lips = (
            f'<circle cx="{lip_x}" cy="62" r="10" fill="none" stroke="{_ORANGE}" stroke-width="2.5"/>'
            f'<ellipse cx="{lip_x}" cy="62" rx="6" ry="8" fill="{_ORANGE}" opacity="0.35"/>'
        )
    else:
        lips = (
            f'<line x1="{lip_x - 8}" y1="62" x2="{lip_x + 8}" y2="62" stroke="{_ORANGE}" stroke-width="2.5" stroke-linecap="round"/>'
            f'<line x1="{lip_x - 5}" y1="66" x2="{lip_x + 5}" y2="66" stroke="{_ORANGE}" stroke-width="1.5" stroke-linecap="round"/>'
        )

    legend = _vowel_legend(config.rounded)

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 210" width="300" height="210">
  <text x="150" y="20" text-anchor="middle" fill="{_INK}" font-size="16" font-weight="bold" font-family="sans-serif">/{_safe(display)}/</text>
  <path d="{_HEAD_OUTLINE}" fill="none" stroke="{_INK}" stroke-width="2"/>
  {_GUIDES}
  {tongue}
  {lips}
  {legend}
  <text x="150" y="205" text-anchor="middle" fill="{_MUTED}" font-size="11" font-family="sans-serif">{_vowel_quality_text(config)}</text>
</svg>'''
