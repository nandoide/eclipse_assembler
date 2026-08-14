#!/usr/bin/env python3
"""
High-Resolution Solar Eclipse Composite & Mosaic Generator (UHD Squared 3840x3840)
===================================================================================
Generates ultra-high-resolution composite artwork capturing the complete
progression of the 2026 total solar eclipse on a clean aesthetic canvas.

Supported Layout Modes:
  1. 'sinusoid' (or 's-curve', 's'):
     Graceful S-shaped sinusoidal wave sweeping across the square canvas with
     grand totality at the center inflection.
  2. 'circle' (or 'ring'):
     Circular orbit wreath with symmetric totality apex and trimmed partials.
  3. 'diagonal':
     Progressing diagonally from bottom-left to top-right with expansive corona streamers.
  4. 'horizontal':
     Linear progression across the horizontal midline with laser-aligned centers.
  5. 'arc':
     Graceful parabolic arc mirroring the Sun's celestial trajectory.

Totality Sequence & Symmetry:
  - Ingress beads (frame ~30): Diamond Ring / point sources on left limb.
  - Ingress chromosphere (frame ~70): Red H-alpha arc and prominence on left limb.
  - Mid totality (frame ~600): Chromosphere prominence and inner corona.
  - Grand Corona (frame ~1200): Wide outer corona streamers at peak exposure.
  - Egress chromosphere (frame ~2700): Red H-alpha arc on right limb.
  - Egress beads (frame ~2850): C3 Diamond Ring on right limb.

Alignment & Spacing:
  - Disks use the stabilized video center (640, 360) ensuring 100% true geometric alignment.
  - Default disk_scale_factor = 0.88 prevents circular disk overlap across all layouts,
    providing clean margins while letting corona streamers flow into the dark space.

Authors: Fernando (nandoide) & Antigravity (Google Deepmind)
Workspace: eclipse_assembler
"""

import os
import sys
import argparse
import math
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def equalize_solar_color(
    frame,
    target_gr=0.619,
    target_br=0.507,
    target_peak_r=230.0
):
    """
    Normalizes solar filter color and photosphere brightness for all partial frames.
    Achieves consistent warm solar orange (R/B ~1.97, G/R ~0.62).
    Skips totality frames (black disk with corona - no red dominant area to sample).
    """
    r = frame[:, :, 2].astype(np.float32)
    b = frame[:, :, 0].astype(np.float32)
    mask = (r > 40) & (r > b * 1.05)
    if np.count_nonzero(mask) < 200:
        return frame

    peak_r = float(np.percentile(r[mask], 90))
    peak_g = float(np.percentile(frame[:, :, 1].astype(np.float32)[mask], 90))
    peak_b = float(np.percentile(b[mask], 90))

    if peak_r < 10:
        return frame

    current_gr = peak_g / peak_r
    current_br = peak_b / peak_r
    lum_gain = float(np.clip(target_peak_r / peak_r, 0.7, 1.4))

    out = frame.copy().astype(np.float32)
    out[:, :, 2] = np.clip(out[:, :, 2] * lum_gain, 0, 255)
    out[:, :, 1] = np.clip(out[:, :, 1] * lum_gain * (target_gr / max(0.01, current_gr)), 0, 255)
    out[:, :, 0] = np.clip(out[:, :, 0] * lum_gain * (target_br / max(0.01, current_br)), 0, 255)
    return out.astype(np.uint8)


def clean_and_crop_square(frame, crop_size=1280, is_totality=False):
    """
    Crops square patch around the true stabilized center (w//2, h//2),
    applying soft noise floor and smooth border feathering.
    Using the stabilized video frame center ensures all solar/lunar disks are
    geometrically aligned without drift.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2  # Locked stabilized center (640, 360)

    f_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) if frame.ndim == 3 else frame.astype(np.float32)
    floor_w = np.clip((f_gray - 2.5) / 6.0, 0.0, 1.0)

    if is_totality:
        # Elliptical smooth taper matching 1280x720 aspect ratio
        Y_grid, X_grid = np.ogrid[:h, :w]
        dx = (X_grid - cx).astype(np.float32)
        dy = (Y_grid - cy).astype(np.float32)
        rx, ry = 540.0, 340.0
        ellip_dist = np.sqrt((dx / rx)**2 + (dy / ry)**2)
        ramp = np.clip((1.0 - ellip_dist) / (1.0 - 0.70), 0.0, 1.0)
        shape_w = 0.5 - 0.5 * np.cos(np.pi * ramp)
        actual_crop = crop_size
    else:
        # 4-edge border feather for partial frames
        Y_grid, X_grid = np.ogrid[:h, :w]
        w_x = np.clip(X_grid / 20.0, 0.0, 1.0) * np.clip((w - 1 - X_grid) / 20.0, 0.0, 1.0)
        w_y = np.clip(Y_grid / 20.0, 0.0, 1.0) * np.clip((h - 1 - Y_grid) / 20.0, 0.0, 1.0)
        shape_w = w_x * w_y
        actual_crop = min(crop_size, 720)

    cleaned = (frame.astype(np.float32) * (shape_w * floor_w)[:, :, np.newaxis]).astype(np.uint8)

    # Square crop around (cx, cy)
    half = actual_crop // 2
    padded = cv2.copyMakeBorder(cleaned, half, half, half, half, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    pcx, pcy = cx + half, cy + half
    cropped = padded[pcy - half : pcy + half, pcx - half : pcx + half]
    return cropped


# ─────────────────────────────────────────────────────────────────────────────
# ECLIPSE SEQUENCE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def resolve_output_dir(out_dir):
    if os.path.isabs(out_dir) and os.path.exists(out_dir):
        return out_dir
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(repo_root, out_dir)
    if os.path.exists(candidate):
        return candidate
    return out_dir


def sample_eclipse_sequence(
    out_dir="040_out",
    crop_size=1280,
):
    """
    Extracts keyframes with trimmed extreme partial ends and full symmetric totality entry/exit:
      - Ingress: 2 well-defined crescents (omits first full sun).
      - Pre-totality: 2 thin crescents approaching C2.
      - Totality: 6 symmetric keyframes (f30, f70, f600, f1200, f2700, f2850).
      - Egress: 4 crescents exiting C3 (omits last full sun).
    Total: 14 balanced, physically aligned keyframes.
    """
    resolved = resolve_output_dir(out_dir)
    paths = {
        "ingress":  os.path.join(resolved, "partial_ingress.mp4"),
        "pre_tot":  os.path.join(resolved, "pre_totality.mp4"),
        "totality": os.path.join(resolved, "totality.mp4"),
        "egress":   os.path.join(resolved, "partial_egress.mp4"),
    }

    caps = {k: cv2.VideoCapture(v) for k, v in paths.items()}
    counts = {k: int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for k, c in caps.items()}

    def read_at(cap, idx, total):
        if total <= 0:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total - 1, int(idx))))
        ret, f = cap.read()
        return f if ret else None

    samples = []

    # 1. Ingress Partials (omitting the first full sun, starting from ~38% bite)
    for frac in [0.38, 0.78]:
        f = read_at(caps["ingress"], frac * (counts["ingress"] - 1), counts["ingress"])
        if f is not None:
            f = equalize_solar_color(f)
            samples.append({"phase": "partial", "label": "ingress", "img": clean_and_crop_square(f, crop_size=crop_size, is_totality=False)})

    # 2. Pre-Totality Thin Crescents
    for frac in [0.38, 0.78]:
        f = read_at(caps["pre_tot"], frac * (counts["pre_tot"] - 1), counts["pre_tot"])
        if f is not None:
            f = equalize_solar_color(f)
            samples.append({"phase": "partial", "label": "pre_totality", "img": clean_and_crop_square(f, crop_size=crop_size, is_totality=False)})

    # 3. Totality Keyframes (symmetric entrance f30, f70 -> grand corona -> egress f2700, f2850)
    tot_indices = [
        (30,   "ingress_beads"),
        (70,   "ingress_chromosphere"),
        (600,  "inner_corona"),
        (1200, "grand_corona"),
        (2700, "egress_chromosphere"),
        (2850, "egress_beads"),
    ]
    for fi, label in tot_indices:
        f = read_at(caps["totality"], fi, counts["totality"])
        if f is not None:
            samples.append({"phase": "totality", "label": label, "img": clean_and_crop_square(f, crop_size=crop_size, is_totality=True)})

    # 4. Egress Partials (starting right after C3, ending at ~65% bite without last full sun)
    for frac in [0.15, 0.35, 0.55, 0.75]:
        f = read_at(caps["egress"], frac * (counts["egress"] - 1), counts["egress"])
        if f is not None:
            f = equalize_solar_color(f)
            samples.append({"phase": "partial", "label": "egress", "img": clean_and_crop_square(f, crop_size=crop_size, is_totality=False)})

    for c in caps.values():
        c.release()

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# RENDERING WITH UNIFIED DISK SCALE & EXPANDED CORONA
# ─────────────────────────────────────────────────────────────────────────────

def render_consistent_scale(samples, positions, canvas_size=3840, disk_scale_factor=0.88):
    """
    Renders all sequence samples with 1:1 consistent disk diameters on the canvas.
    Totality crops naturally extend their wide corona streamers across the canvas
    and blend onto neighboring black background seamlessly with np.maximum.
    disk_scale_factor = 0.88 ensures no overlap between adjacent disk circles.
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    n = len(samples)
    if n <= 1:
        return canvas

    pts = np.array(positions, dtype=np.float32)
    min_dist = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if d < min_dist:
                min_dist = d

    target_disk_diameter = min_dist * disk_scale_factor
    scale = target_disk_diameter / 475.0  # Disk diameter in raw crops is ~475px

    for i, s in enumerate(samples):
        px, py = positions[i]
        img = s["img"]
        ch, cw = img.shape[:2]
        pw, ph = int(round(cw * scale)), int(round(ch * scale))
        resized = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_LANCZOS4)

        half_w, half_h = pw // 2, ph // 2
        x1, y1 = px - half_w, py - half_h
        x2, y2 = x1 + pw, y1 + ph

        src_x1 = max(0, -x1); src_y1 = max(0, -y1)
        src_x2 = pw - max(0, x2 - canvas_size)
        src_y2 = ph - max(0, y2 - canvas_size)
        dst_x1 = max(0, x1); dst_y1 = max(0, y1)
        dst_x2 = min(canvas_size, x2); dst_y2 = min(canvas_size, y2)

        if dst_x2 > dst_x1 and dst_y2 > dst_y1:
            roi = resized[src_y1:src_y2, src_x1:src_x2]
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = np.maximum(
                canvas[dst_y1:dst_y2, dst_x1:dst_x2],
                roi
            )

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT ENGINES
# ─────────────────────────────────────────────────────────────────────────────

def generate_circular_composite(
    samples,
    canvas_size=3840,
    orbit_radius=None,
    direction="ccw",
    start_angle_deg=270.0,
    disk_scale_factor=0.88,
):
    """Circular wreath layout with clean non-overlapping disks and wide corona."""
    cx, cy = canvas_size // 2, canvas_size // 2
    if orbit_radius is None:
        orbit_radius = int(canvas_size * 0.365)

    n = len(samples)
    sign = 1.0 if direction.lower() == "ccw" else -1.0
    start_rad = math.radians(start_angle_deg)

    positions = []
    for i in range(n):
        theta = start_rad + sign * (2 * math.pi * i) / n
        px = int(round(cx + orbit_radius * math.cos(theta)))
        py = int(round(cy - orbit_radius * math.sin(theta)))
        positions.append((px, py))

    return render_consistent_scale(samples, positions, canvas_size=canvas_size, disk_scale_factor=disk_scale_factor)


def generate_sinusoid_composite(
    samples,
    canvas_size=3840,
    margin=350,
    amplitude=900,
    disk_scale_factor=0.88,
):
    """
    Sinusoidal S-curve layout sweeping across the canvas.
    Ingress rises into an upper crest, passes through grand totality at the center,
    dips into a lower trough, and egress rises back to the right margin.
    """
    n = len(samples)
    cy = canvas_size // 2

    positions = []
    for i in range(n):
        t = i / max(1, n - 1)
        px = int(round(margin + t * (canvas_size - 2 * margin)))
        py = int(round(cy - amplitude * math.sin(2 * math.pi * t)))
        positions.append((px, py))

    return render_consistent_scale(samples, positions, canvas_size=canvas_size, disk_scale_factor=disk_scale_factor)


def generate_diagonal_composite(
    samples,
    canvas_size=3840,
    direction="bottom_left_to_top_right",
    margin=380,
    disk_scale_factor=0.88,
):
    """Diagonal progression from bottom-left to top-right with non-overlapping disks."""
    n = len(samples)
    if direction == "bottom_left_to_top_right":
        x_start, y_start = margin, canvas_size - margin
        x_end,   y_end   = canvas_size - margin, margin
    else:
        x_start, y_start = margin, margin
        x_end,   y_end   = canvas_size - margin, canvas_size - margin

    positions = []
    for i in range(n):
        t  = i / max(1, n - 1)
        px = int(round(x_start + t * (x_end - x_start)))
        py = int(round(y_start + t * (y_end - y_start)))
        positions.append((px, py))

    return render_consistent_scale(samples, positions, canvas_size=canvas_size, disk_scale_factor=disk_scale_factor)


def generate_horizontal_composite(
    samples,
    canvas_size=3840,
    margin=350,
    disk_scale_factor=0.88,
):
    """Horizontal left-to-right progression with laser-aligned centers and clean spacing."""
    n  = len(samples)
    cy = canvas_size // 2

    positions = [
        (int(round(margin + (i / max(1, n - 1)) * (canvas_size - 2 * margin))), cy)
        for i in range(n)
    ]

    return render_consistent_scale(samples, positions, canvas_size=canvas_size, disk_scale_factor=disk_scale_factor)


def generate_arc_composite(
    samples,
    canvas_size=3840,
    margin_x=350,
    margin_bot=350,
    margin_top=420,
    disk_scale_factor=0.88,
):
    """
    Celestial parabolic arc progression reaching high into the upper frame,
    utilizing the full vertical and horizontal canvas expanse with grand totality crowning the apex.
    """
    n = len(samples)
    positions = []
    y_bot = canvas_size - margin_bot
    for i in range(n):
        t      = i / max(1, n - 1)
        px     = int(round(margin_x + t * (canvas_size - 2 * margin_x)))
        t_norm = (t - 0.5) * 2.0
        py     = int(round(y_bot - (1.0 - t_norm ** 2) * (y_bot - margin_top)))
        positions.append((px, py))

    return render_consistent_scale(samples, positions, canvas_size=canvas_size, disk_scale_factor=disk_scale_factor)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER BUILD FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_composite(
    layout="sinusoid",
    canvas_size=3840,
    orbit_radius=None,
    direction=None,
    disk_scale_factor=0.88,
    out_dir="040_out",
    output_path=None,
):
    resolved_out = resolve_output_dir(out_dir)
    os.makedirs(resolved_out, exist_ok=True)

    print("=" * 65)
    print(f"ECLIPSE COMPOSITE GENERATOR (UHD Squared {canvas_size}x{canvas_size})")
    print("=" * 65)
    print(f"  Layout Mode       : {layout.upper()}")
    print(f"  Resolution        : {canvas_size}x{canvas_size} px")
    print(f"  Disk Scale Factor : {disk_scale_factor:.2f}")

    samples = sample_eclipse_sequence(
        out_dir=resolved_out,
        crop_size=1280,
    )
    n_tot_got  = sum(1 for s in samples if s["phase"] == "totality")
    n_part_got = sum(1 for s in samples if s["phase"] == "partial")
    print(f"  Frames loaded     : {len(samples)} total ({n_part_got} partial, {n_tot_got} totality)")

    # Layout
    if layout in ["circle", "ring"]:
        canvas = generate_circular_composite(
            samples, canvas_size=canvas_size,
            orbit_radius=orbit_radius,
            direction=direction or "ccw",
            disk_scale_factor=disk_scale_factor,
        )
        suffix = "circle"

    elif layout in ["sinusoid", "s-curve", "s", "sinusoidal"]:
        canvas = generate_sinusoid_composite(
            samples, canvas_size=canvas_size,
            disk_scale_factor=disk_scale_factor,
        )
        suffix = "sinusoid"

    elif layout == "diagonal":
        canvas = generate_diagonal_composite(
            samples, canvas_size=canvas_size,
            direction=direction or "bottom_left_to_top_right",
            disk_scale_factor=disk_scale_factor,
        )
        suffix = "diagonal"

    elif layout == "horizontal":
        canvas = generate_horizontal_composite(
            samples, canvas_size=canvas_size,
            disk_scale_factor=disk_scale_factor,
        )
        suffix = "horizontal"

    elif layout == "arc":
        canvas = generate_arc_composite(
            samples, canvas_size=canvas_size,
            disk_scale_factor=disk_scale_factor,
        )
        suffix = "arc"

    else:
        raise ValueError(f"Unknown layout: '{layout}'. Choose sinusoid/circle/diagonal/horizontal/arc.")

    # Save
    default_name = f"eclipse_composite_{suffix}_{canvas_size}p.png"
    final_out = output_path or os.path.join(resolved_out, default_name)
    if not os.path.isabs(final_out):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        final_out = os.path.join(repo_root, final_out)

    print(f"\nSaving UHD Composite Artwork...")
    cv2.imwrite(final_out, canvas)
    jpg_out = os.path.splitext(final_out)[0] + ".jpg"
    cv2.imwrite(jpg_out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

    size_png = os.path.getsize(final_out) / (1024 * 1024)
    size_jpg = os.path.getsize(jpg_out)   / (1024 * 1024)
    print("=" * 65)
    print(f"SUCCESS: Composite generated!")
    print(f"  PNG (Lossless) : {final_out} ({size_png:.2f} MB)")
    print(f"  JPG (High-Q)   : {jpg_out} ({size_jpg:.2f} MB)")
    print("=" * 65)
    return final_out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 3840x3840 UHD solar eclipse composite artwork."
    )
    parser.add_argument("--layout", "-l", type=str,
                        choices=["sinusoid", "s-curve", "circle", "ring", "diagonal", "horizontal", "arc", "all"],
                        default="sinusoid",
                        help="Composition layout (default: sinusoid)")
    parser.add_argument("--size", "-s", type=int, default=3840,
                        help="Square canvas size in pixels (default: 3840)")
    parser.add_argument("--scale-factor", type=float, default=0.88,
                        help="Disk scale factor relative to separation distance (default: 0.88)")
    parser.add_argument("--orbit-radius", type=int, default=None,
                        help="Circle orbit radius in pixels (default: auto)")
    parser.add_argument("--direction", type=str, default=None,
                        help="Progression direction (ccw/cw for circle; bottom_left_to_top_right for diagonal)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Custom output path (default: 040_out/eclipse_composite_<layout>_<size>p.png)")
    args = parser.parse_args()

    layouts = ["sinusoid", "circle", "diagonal", "horizontal", "arc"] if args.layout == "all" else [args.layout]
    for lay in layouts:
        build_composite(
            layout=lay,
            canvas_size=args.size,
            orbit_radius=args.orbit_radius,
            direction=args.direction,
            disk_scale_factor=args.scale_factor,
            out_dir="040_out",
            output_path=args.output if args.layout != "all" else None,
        )
