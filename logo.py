from __future__ import annotations

import argparse
import os
import re
from typing import Optional, Tuple, Union

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFilter
except ModuleNotFoundError:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageChops = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFilter = None  # type: ignore[assignment]

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

SrgbColor = Tuple[float, float, float]
BackgroundSpec = Union[str, SrgbColor]

_HEX_COLOR_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


def _require_deps() -> None:
    missing = []
    if np is None:
        missing.append("numpy")
    if Image is None:
        missing.append("Pillow")
    if missing:
        raise ModuleNotFoundError(
            "Missing dependencies for logo generation: "
            + ", ".join(missing)
            + ". Install with: python3 -m pip install numpy Pillow"
        )


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _srgb_from_hex(hex_color: str) -> SrgbColor:
    hex_color = hex_color.strip().lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color!r}")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (_clamp01(r), _clamp01(g), _clamp01(b))


def _normalize_srgb(color: SrgbColor) -> SrgbColor:
    r, g, b = (float(color[0]), float(color[1]), float(color[2]))
    if max(r, g, b) > 1.0:
        r, g, b = (r / 255.0, g / 255.0, b / 255.0)
    return (_clamp01(r), _clamp01(g), _clamp01(b))


def _parse_background(background: BackgroundSpec) -> Tuple[str, Optional[SrgbColor]]:
    if isinstance(background, str):
        raw = background.strip().lower()
        name = re.sub(r"[\s_-]+", "", raw)
        if name in {"dark", "black"}:
            return "dark", None
        if name in {"light", "lightgrey", "lightgray", "grey", "gray"}:
            return "light", None
        if name in {"transparent", "none"}:
            return "transparent", None
        if _HEX_COLOR_RE.match(raw):
            return "solid", _srgb_from_hex(raw)
        raise ValueError(
            f"Unknown background {background!r}. Use 'dark', 'light', 'transparent', or '#RRGGBB'."
        )

    return "solid", _normalize_srgb(background)


def _rounded_rect_mask(size: Tuple[int, int], radius_px: float, antialias: int = 4) -> Image.Image:
    w, h = size
    radius_px = max(0.0, min(float(radius_px), min(w, h) / 2.0))
    if radius_px <= 0:
        return Image.new("L", (w, h), 255)

    scale = max(1, int(antialias))
    ww, hh = w * scale, h * scale
    mask = Image.new("L", (ww, hh), 0)
    draw = ImageDraw.Draw(mask)
    r = int(round(radius_px * scale))
    draw.rounded_rectangle([0, 0, ww - 1, hh - 1], radius=r, fill=255)
    return mask.resize((w, h), resample=Image.Resampling.LANCZOS)


def _apply_corner_radius(im: Image.Image, radius_px: float) -> Image.Image:
    if radius_px <= 0:
        return im
    if im.mode != "RGBA":
        im = im.convert("RGBA")

    mask = _rounded_rect_mask(im.size, radius_px, antialias=4)
    a = im.getchannel("A")
    a = ImageChops.multiply(a, mask)
    im.putalpha(a)
    return im

def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)

def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(c <= 0.0031308, 12.92 * c, (1 + a) * (c ** (1 / 2.4)) - a)

def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + (b - a) * t

def gradient_stops(t: np.ndarray, stops) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    out = np.zeros(t.shape + (3,), dtype=np.float32)
    for i in range(len(stops) - 1):
        p0, c0 = stops[i]
        p1, c1 = stops[i + 1]
        m = (t >= p0) & (t <= p1)
        if np.any(m):
            tt = (t[m] - p0) / (p1 - p0 + 1e-12)
            out[m] = lerp(np.array(c0, np.float32), np.array(c1, np.float32), tt[..., None])
    out[t <= stops[0][0]] = np.array(stops[0][1], np.float32)
    out[t >= stops[-1][0]] = np.array(stops[-1][1], np.float32)
    return out

def make_beam_logo_corner_tip(
    size: int = 1024,
    supersample: int = 3,
    end_width_px: float = 260.0,     # desired half-width at the far end (in final pixels)
    blur_strength: float = 0.85,
    background: BackgroundSpec = "dark",
    corner_radius_px: float = 0.0,
    rgba_mode: str = "beam",         # "beam" (original) or "tile" (opaque background)
):
    """
    Beam starts EXACTLY at top-left (0,0) with width=0 at that point.
    Straight edges: width grows linearly with distance along the beam.
    background: 'dark' | 'light' | 'transparent' | '#RRGGBB' | (r,g,b).
    rgba_mode:
      - 'beam': RGB contains background; alpha follows beam/halo/core (original behavior).
      - 'tile': alpha is opaque (optionally with rounded corners via corner_radius_px).
    Returns (PIL_RGB, PIL_RGBA).
    """
    _require_deps()
    W = H = int(size * supersample)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    p = np.stack([xx, yy], axis=-1)

    # Direction (top-left -> bottom-right)
    v = np.array([0.596, 0.803], dtype=np.float32)
    v /= np.linalg.norm(v)
    n = np.array([-v[1], v[0]], dtype=np.float32)

    origin = np.array([0.0, 0.0], dtype=np.float32)  # EXACT corner

    q = p - origin
    t = q[..., 0] * v[0] + q[..., 1] * v[1]  # along
    s = q[..., 0] * n[0] + q[..., 1] * n[1]  # across

    tpos = np.maximum(t, 0.0)
    tmax = np.sqrt(W * W + H * H)

    # End width (in supersampled pixels)
    w1 = end_width_px * supersample
    tn = np.clip(tpos / (0.98 * tmax), 0.0, 1.0)

    # LINEAR widening starting from 0 => straight edges through the corner
    width = w1 * tn  # width==0 at t=0

    dist = np.abs(s)

    # Beam profile
    boundary = 1.0 / (1.0 + np.exp((dist - width) / (8.0 * supersample)))
    soft = np.exp(- (dist / (width * 0.90 + 1e-6)) ** 2 * 0.55)
    beam = np.clip(boundary * soft, 0.0, 1.0)

    halo = np.exp(- (dist / (width * 1.55 + 1e-6)) ** 2 * 0.85) * 0.55

    # Along-beam fade
    along = np.exp(-tpos / (0.95 * tmax) * 0.55)
    beam *= along
    halo *= along

    # Use a small epsilon width near the tip to avoid crazy division when width~0
    w_eps = 8.0 * supersample
    u = np.clip((s / (width + w_eps)) * 0.5 + 0.5, 0.0, 1.0)

    stops = [
        (0.00, (0.10, 0.35, 1.00)),  # blue edge
        (0.22, (0.08, 0.86, 1.00)),  # cyan
        (0.45, (0.25, 0.98, 0.70)),  # green/cyan
        (0.62, (1.00, 0.96, 0.42)),  # yellow
        (0.80, (1.00, 0.60, 0.14)),  # orange
        (1.00, (1.00, 0.20, 0.20)),  # red edge
    ]
    col = srgb_to_linear(gradient_stops(u, stops))

    # White-hot core (starts at the corner)
    core_w = width * 0.22 + 6.0 * supersample
    core = np.exp(- (s / (core_w + 1e-6)) ** 2 * 3.0) * np.exp(-tpos / (0.20 * tmax))
    warm_white = srgb_to_linear(np.array([1.0, 0.99, 0.96], dtype=np.float32))

    ambience = np.exp(- (dist / (width * 3.2 + w_eps)) ** 2 * 0.70) * along

    # Small corner bloom (subtle)
    r2 = (xx - 0.0) ** 2 + (yy - 0.0) ** 2
    bloom = np.exp(-r2 / (2 * (40.0 * supersample) ** 2))

    bg_kind, bg_srgb = _parse_background(background)
    if bg_kind == "dark":
        bg = srgb_to_linear(np.zeros((H, W, 3), dtype=np.float32))
        bg += srgb_to_linear(np.array([0.008, 0.008, 0.012], dtype=np.float32))

        ambience_col = srgb_to_linear(np.array([0.02, 0.06, 0.18], dtype=np.float32))
        bg += ambience[..., None] * ambience_col * 0.85

        bg += bloom[..., None] * srgb_to_linear(np.array([0.02, 0.04, 0.08], dtype=np.float32)) * 0.7
    elif bg_kind == "light":
        base = srgb_to_linear(np.array([0.961, 0.961, 0.969], dtype=np.float32))  # ~ #f5f5f7
        bg = np.ones((H, W, 3), dtype=np.float32) * base

        ambience_col = srgb_to_linear(np.array([0.09, 0.14, 0.30], dtype=np.float32))
        bg += ambience[..., None] * ambience_col * 0.20

        dx = xx / max(1.0, (W - 1)) - 0.5
        dy = yy / max(1.0, (H - 1)) - 0.5
        vignette = np.clip(dx * dx + dy * dy, 0.0, 0.5)
        bg *= 1.0 - vignette[..., None] * 0.12

        bg += bloom[..., None] * srgb_to_linear(np.array([0.12, 0.14, 0.18], dtype=np.float32)) * 0.25
    elif bg_kind == "solid":
        base = srgb_to_linear(np.array(bg_srgb, dtype=np.float32))
        bg = np.ones((H, W, 3), dtype=np.float32) * base
    elif bg_kind == "transparent":
        bg = srgb_to_linear(np.zeros((H, W, 3), dtype=np.float32))
    else:
        raise RuntimeError(f"Unhandled background kind: {bg_kind}")

    # Composite in linear space
    img_lin = bg
    img_lin += col * (beam[..., None] * 2.1)
    img_lin += col * (halo[..., None] * 0.65)
    img_lin += warm_white * (core[..., None] * 1.7)

    img_lin = np.clip(img_lin, 0.0, 1.0)
    img = np.clip(linear_to_srgb(img_lin), 0.0, 1.0)

    if rgba_mode == "tile":
        alpha = np.ones((H, W), dtype=np.float32)
    elif rgba_mode == "beam":
        alpha = np.clip(beam * 1.0 + halo * 0.80 + core * 0.45, 0.0, 1.0)
    else:
        raise ValueError("rgba_mode must be 'beam' or 'tile'")

    img8 = (img * 255).astype(np.uint8)
    a8 = (alpha * 255).astype(np.uint8)

    im_rgb = Image.fromarray(img8)
    im_rgba = Image.fromarray(np.dstack([img8, a8]))

    # Mild blur (keeps edges visually straight)
    r = blur_strength * supersample
    im_rgb = im_rgb.filter(ImageFilter.GaussianBlur(radius=r))
    im_rgba = im_rgba.filter(ImageFilter.GaussianBlur(radius=r))

    if supersample != 1:
        im_rgb = im_rgb.resize((size, size), resample=Image.Resampling.LANCZOS)
        im_rgba = im_rgba.resize((size, size), resample=Image.Resampling.LANCZOS)

    if corner_radius_px > 0:
        im_rgba = _apply_corner_radius(im_rgba, corner_radius_px)

    return im_rgb, im_rgba

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate RayX beam logo PNGs.")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--supersample", type=int, default=3)
    parser.add_argument("--end-width", type=float, default=260.0)
    parser.add_argument("--blur", type=float, default=0.85)
    parser.add_argument(
        "--background",
        default="dark",
        help="dark | light | transparent | #RRGGBB (or 3-tuple via code)",
    )
    parser.add_argument("--corner-radius", type=float, default=0.0, help="Rounded corner radius (px).")
    parser.add_argument("--rgba-mode", choices=["beam", "tile"], default="tile")
    parser.add_argument("--out-rgb", default="data/rayx_beam_logo_rgb.png")
    parser.add_argument("--out-rgba", default="data/rayx_beam_logo_rgba.png")
    args = parser.parse_args(argv)

    im_rgb, im_rgba = make_beam_logo_corner_tip(
        size=args.size,
        supersample=args.supersample,
        end_width_px=args.end_width,
        blur_strength=args.blur,
        background=args.background,
        corner_radius_px=args.corner_radius,
        rgba_mode=args.rgba_mode,
    )

    for out_path, im in ((args.out_rgb, im_rgb), (args.out_rgba, im_rgba)):
        out_dir = os.path.dirname(out_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        im.save(out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
