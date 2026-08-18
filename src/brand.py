"""Generate an accessible brand kit for a new nonprofit."""

import colorsys
import hashlib
import html
import re

# Cause area -> base hue (degrees on the color wheel)
CAUSE_HUE = {
    "education": 220, "science technology": 265, "youth development": 25,
    "recreation sports": 30, "environment": 145, "animals": 130,
    "health": 190, "mental health": 195, "disease research": 185,
    "medical research": 185, "arts culture humanities": 285,
    "human services": 15, "housing": 20, "food agriculture": 35,
    "crime legal": 250, "civil rights": 255, "public safety disaster relief": 240,
    "employment": 205, "community development": 200, "international": 210,
    "philanthropy": 215, "public benefit": 215, "religion": 270,
    "social science": 260, "mutual benefit": 215,
}

TONES = {
    "Calm":    {"sat": 0.34, "light": 0.42, "shift": 0},
    "Warm":    {"sat": 0.56, "light": 0.46, "shift": 18},
    "Bold":    {"sat": 0.72, "light": 0.40, "shift": 0},
    "Classic": {"sat": 0.26, "light": 0.30, "shift": -10},
}

FONTS = {
    "Calm":    ("Lora", "Source Sans 3"),
    "Warm":    ("Fraunces", "Nunito Sans"),
    "Bold":    ("Space Grotesk", "Inter"),
    "Classic": ("Libre Baskerville", "Inter"),
}

SHAPES = ["circle", "rounded square", "hexagon"]


def _hex(h, s, l):
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360, l, s)
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))


def _rgb(code):
    code = code.lstrip("#")
    return tuple(int(code[i:i + 2], 16) for i in (0, 2, 4))


def _channel(v):
    v /= 255
    return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4


def luminance(code):
    r, g, b = (_channel(c) for c in _rgb(code))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a, b):
    """WCAG 2.1 contrast ratio, from 1 (identical) to 21 (black on white)."""
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def readable_on(bg):
    """Whichever of black or white is legible against this background."""
    return "#FFFFFF" if contrast_ratio(bg, "#FFFFFF") >= contrast_ratio(bg, "#111111") else "#111111"


def wcag_level(ratio, large=False):
    if ratio >= (4.5 if large else 7):
        return "AAA"
    if ratio >= (3 if large else 4.5):
        return "AA"
    return "Fails"


def palette(name, cause, tone):
    """Deterministic palette — same name and settings always give the same colors."""
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    t = TONES[tone]
    base = (CAUSE_HUE.get(cause, 215) + t["shift"] + seed % 12) % 360

    primary = _hex(base, t["sat"], t["light"])
    return {
        "primary": primary,
        "dark": _hex(base, t["sat"] + 0.08, max(t["light"] - 0.22, 0.10)),
        "accent": _hex(base + 152, min(t["sat"] + 0.14, 0.85), t["light"] + 0.10),
        "surface": _hex(base, 0.16, 0.955),
        "ink": _hex(base, 0.22, 0.13),
        "on_primary": readable_on(primary),
    }


def initials(name, limit=2):
    """Up to two letters, stripped to characters that are safe inside SVG."""
    words = re.findall(r"[A-Za-z0-9]+", name)
    letters = "".join(w[0] for w in words if w)[:limit].upper()
    return letters or "N"


def logo_svg(name, colors, shape="circle", size=180):
    """A geometric monogram mark. Text is escaped before it enters the markup."""
    c, r = size / 2, size / 2 - 4
    text = html.escape(initials(name))

    if shape == "circle":
        figure = f'<circle cx="{c}" cy="{c}" r="{r}" fill="{colors["primary"]}"/>'
    elif shape == "hexagon":
        pts = " ".join(
            f"{c + r * __import__('math').cos(__import__('math').radians(60 * i - 30)):.1f},"
            f"{c + r * __import__('math').sin(__import__('math').radians(60 * i - 30)):.1f}"
            for i in range(6))
        figure = f'<polygon points="{pts}" fill="{colors["primary"]}"/>'
    else:
        figure = (f'<rect x="4" y="4" width="{size-8}" height="{size-8}" '
                  f'rx="{size*0.22:.0f}" fill="{colors["primary"]}"/>')

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}"
     width="{size}" height="{size}" role="img" aria-label="Logo mark">
  {figure}
  <circle cx="{c}" cy="{c}" r="{r*0.62:.1f}" fill="none"
          stroke="{colors["accent"]}" stroke-width="{size*0.028:.1f}" opacity="0.9"/>
  <text x="{c}" y="{c}" text-anchor="middle" dominant-baseline="central"
        font-family="Helvetica, Arial, sans-serif" font-weight="700"
        font-size="{size*0.34:.0f}" fill="{colors["on_primary"]}">{text}</text>
</svg>'''


def contrast_report(colors):
    """Every pairing a nonprofit will actually use, checked against WCAG."""
    pairs = [
        ("Body text on background", colors["ink"], colors["surface"]),
        ("Logo text on brand color", colors["on_primary"], colors["primary"]),
        ("Brand color on white", colors["primary"], "#FFFFFF"),
        ("Accent on dark", colors["accent"], colors["dark"]),
    ]
    return [{"use": u, "fg": f, "bg": b,
             "ratio": round(contrast_ratio(f, b), 2),
             "level": wcag_level(contrast_ratio(f, b))}
            for u, f, b in pairs]


def brand_sheet(name, cause, city, tone, colors, checks):
    heading, body = FONTS[tone]
    rows = "\n".join(f"| {c['use']} | {c['ratio']}:1 | {c['level']} |" for c in checks)
    return f"""# {name} — brand sheet

**Cause area:** {cause}  ·  **Location:** {city}  ·  **Tone:** {tone}

## Colors

| Role | Hex | Use for |
|---|---|---|
| Primary | `{colors['primary']}` | Logo, buttons, headers |
| Dark | `{colors['dark']}` | Footers, hover states |
| Accent | `{colors['accent']}` | Highlights, calls to action |
| Surface | `{colors['surface']}` | Page background |
| Ink | `{colors['ink']}` | Body text |

## Type

- Headings: **{heading}**
- Body: **{body}**

Both are open-licensed and free on Google Fonts, so there's no licensing cost
and no restriction on using them in printed materials.

## Accessibility

| Pairing | Contrast | WCAG |
|---|---|---|
{rows}

WCAG AA requires 4.5:1 for body text and 3:1 for large text. AAA requires 7:1.
Roughly one in twelve people has some form of color vision deficiency, so
never use color alone to convey meaning — pair it with text or an icon.

## Usage

- Keep clear space around the logo equal to the height of one letter
- Minimum logo size: 32px on screen, 0.5 inch in print
- Don't recolor, stretch, or add effects to the mark
"""