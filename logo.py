from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

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
):
    """
    Beam starts EXACTLY at top-left (0,0) with width=0 at that point.
    Straight edges: width grows linearly with distance along the beam.
    Returns (PIL_RGB, PIL_RGBA).
    """
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

    # Background: deep black with cool haze around beam
    bg = srgb_to_linear(np.zeros((H, W, 3), dtype=np.float32))
    bg += srgb_to_linear(np.array([0.008, 0.008, 0.012], dtype=np.float32))

    ambience = np.exp(- (dist / (width * 3.2 + w_eps)) ** 2 * 0.70) * along
    ambience_col = srgb_to_linear(np.array([0.02, 0.06, 0.18], dtype=np.float32))
    bg += ambience[..., None] * ambience_col * 0.85

    # Small corner bloom (subtle)
    r2 = (xx - 0.0) ** 2 + (yy - 0.0) ** 2
    bloom = np.exp(-r2 / (2 * (40.0 * supersample) ** 2))
    bg += bloom[..., None] * srgb_to_linear(np.array([0.02, 0.04, 0.08], dtype=np.float32)) * 0.7

    # Composite in linear space
    img_lin = bg
    img_lin += col * (beam[..., None] * 2.1)
    img_lin += col * (halo[..., None] * 0.65)
    img_lin += warm_white * (core[..., None] * 1.7)

    img_lin = np.clip(img_lin, 0.0, 1.0)
    img = np.clip(linear_to_srgb(img_lin), 0.0, 1.0)

    alpha = np.clip(beam * 1.0 + halo * 0.80 + core * 0.45, 0.0, 1.0)

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

    return im_rgb, im_rgba

im_rgb6, im_rgba6 = make_beam_logo_corner_tip(size=1024, supersample=3, end_width_px=260.0, blur_strength=0.85)

out_rgb6 = "data/rayx_beam_logo_rgb.png"
out_rgba6 = "data/rayx_beam_logo_rgba.png"
im_rgb6.save(out_rgb6)
im_rgba6.save(out_rgba6)
