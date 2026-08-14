#!/usr/bin/env python3
"""
High-Resolution Solar Eclipse Composite & Mosaic Generator (UHD Squared)
========================================================================
Generates ultra-high-resolution composite artwork capturing the complete
progression of the 2026 total solar eclipse on a clean aesthetic canvas.

Supported Layout Modes:
  1. 'circle' (or 'ring'):
     Solar phases arranged along a circular orbit wreath (as seen in iconic astrophotography).
     Options for totality at the apex or as the centerpiece in the middle of the ring.
  2. 'diagonal':
     Progressing diagonally from left to right (e.g. bottom-left to top-right)
     with totality anchored at the center.
  3. 'horizontal':
     Linear progression across the horizontal axis from left to right.
  4. 'arc':
     Graceful parabolic arc mirroring the Sun's trajectory across the sky.

Features:
  - Default UHD Squared 3840x3840 resolution (customizable).
  - Subpixel Lanczos-4 antialiasing and soft feather alpha-masking.
  - Automatic solar filter chromaticity equalization for uniform color warmth.
  - Intelligent phase pacing across ingress, totality, and egress.
  - Lossless PNG and high-quality JPEG exports.

Authors: Fernando (nandoide) & Antigravity (Google Gemini 3.6 Flash High)
Workspace: eclipse_assembler
"""

import os
import sys
import argparse
import math
import cv2
import numpy as np
from tqdm import tqdm


def equalize_solar_color(frame: np.ndarray, target_gr: float = 0.619, target_br: float = 0.507, target_peak_r: float = 230.0) -> np.ndarray:
    """
    Normalizes solar filter color and photosphere brightness across all partial frames
    to achieve a rich, consistent warm solar orange (R/B ≈ 1.97, G/R ≈ 0.62) with uniform illumination.
    """
    r = frame[:, :, 2].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    b = frame[:, :, 0].astype(np.float32)

    mask = (r > 40) & (r > b * 1.05)
    if np.count_nonzero(mask) < 200:
        return frame

    peak_r = np.percentile(r[mask], 90)
    peak_g = np.percentile(g[mask], 90)
    peak_b = np.percentile(b[mask], 90)

    if peak_r > 10:
        current_gr = peak_g / peak_r
        current_br = peak_b / peak_r

        # Luminance gain
        lum_gain = np.clip(target_peak_r / peak_r, 0.7, 1.4)

        scale_r = lum_gain
        scale_g = lum_gain * (target_gr / max(0.01, current_gr))
        scale_b = lum_gain * (target_br / max(0.01, current_br))

        out = frame.copy().astype(np.float32)
        out[:, :, 2] = np.clip(out[:, :, 2] * scale_r, 0, 255)
        out[:, :, 1] = np.clip(out[:, :, 1] * scale_g, 0, 255)
        out[:, :, 0] = np.clip(out[:, :, 0] * scale_b, 0, 255)
        return out.astype(np.uint8)
    return frame


def crop_solar_disk(frame: np.ndarray, crop_size: int = 560) -> np.ndarray:
    """
    Crops a square region centered on (640, 360) where the solar/lunar disk is stabilized.
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    half = crop_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    cropped = frame[y1:y2, x1:x2]
    if cropped.shape[0] != crop_size or cropped.shape[1] != crop_size:
        cropped = cv2.resize(cropped, (crop_size, crop_size), interpolation=cv2.INTER_LANCZOS4)
    return cropped


def sample_eclipse_sequence(
    out_dir: str = "040_out",
    num_phases: int = 15,
    include_totality: bool = True,
    equalize_color: bool = True,
    totality_fraction: float = 0.45
) -> list:
    """
    Extracts representative keyframes covering the full eclipse timeline:
    Ingress -> Totality -> Egress.
    """
    ingress_path = os.path.join(out_dir, "partial_ingress.mp4")
    pre_total_path = os.path.join(out_dir, "pre_totality.mp4")
    totality_path = os.path.join(out_dir, "totality.mp4")
    egress_path = os.path.join(out_dir, "partial_egress.mp4")

    # If stabilized clips are in parent or root, resolve them
    for path in [ingress_path, pre_total_path, totality_path, egress_path]:
        if not os.path.exists(path):
            alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
            if os.path.exists(alt_path):
                ingress_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "040_out", "partial_ingress.mp4")
                pre_total_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "040_out", "pre_totality.mp4")
                totality_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "040_out", "totality.mp4")
                egress_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "040_out", "partial_egress.mp4")
                break

    cap_in = cv2.VideoCapture(ingress_path)
    cap_pre = cv2.VideoCapture(pre_total_path)
    cap_tot = cv2.VideoCapture(totality_path)
    cap_egr = cv2.VideoCapture(egress_path)

    n_in = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
    n_pre = int(cap_pre.get(cv2.CAP_PROP_FRAME_COUNT))
    n_tot = int(cap_tot.get(cv2.CAP_PROP_FRAME_COUNT))
    n_egr = int(cap_egr.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(cap, idx):
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(count - 1, idx)))
        ret, f = cap.read()
        return f if ret else None

    num_side = (num_phases - 1) // 2 if include_totality else num_phases // 2
    n_in_early = max(1, num_side // 3)
    n_in_late = num_side - n_in_early

    samples = []

    # 1. Ingress early (partial_ingress.mp4)
    for i in range(n_in_early):
        frac = i / max(1, n_in_early)
        idx = int(frac * (n_in - 1)) if n_in > 0 else 0
        f = read_frame(cap_in, idx)
        if f is not None:
            if equalize_color:
                f = equalize_solar_color(f)
            samples.append(("ingress", crop_solar_disk(f)))

    # 2. Ingress late (pre_totality.mp4)
    for i in range(n_in_late):
        frac = (i + 0.5) / n_in_late
        curv_frac = frac ** 1.3
        idx = min(n_pre - 1, int(curv_frac * (n_pre - 1))) if n_pre > 0 else 0
        f = read_frame(cap_pre, idx)
        if f is not None:
            if equalize_color:
                f = equalize_solar_color(f)
            samples.append(("pre_totality", crop_solar_disk(f)))

    # 3. Totality (totality.mp4)
    if include_totality and n_tot > 0:
        idx = int(n_tot * np.clip(totality_fraction, 0.0, 1.0))
        f = read_frame(cap_tot, idx)
        if f is not None:
            samples.append(("totality", crop_solar_disk(f)))

    # 4. Egress (partial_egress.mp4)
    for i in range(num_side):
        frac = i / max(1, num_side - 1)
        curv_frac = (1.0 - frac) ** 1.3
        idx = min(n_egr - 1, int((1.0 - curv_frac) * (n_egr - 1))) if n_egr > 0 else 0
        f = read_frame(cap_egr, idx)
        if f is not None:
            if equalize_color:
                f = equalize_solar_color(f)
            samples.append(("egress", crop_solar_disk(f)))

    cap_in.release()
    cap_pre.release()
    cap_tot.release()
    cap_egr.release()

    return samples


def render_disk_patch(
    canvas: np.ndarray,
    img: np.ndarray,
    center_x: int,
    center_y: int,
    disk_size: int,
    feather_px: float = 15.0
):
    """
    Renders a single solar disk image onto the canvas with soft feathered alpha blending.
    """
    canvas_h, canvas_w = canvas.shape[:2]
    resized = cv2.resize(img, (disk_size, disk_size), interpolation=cv2.INTER_LANCZOS4)

    half = disk_size // 2
    Y, X = np.ogrid[:disk_size, :disk_size]
    dist_from_center = np.sqrt((X - half)**2 + (Y - half)**2)
    mask = np.clip((half - dist_from_center) / max(1.0, feather_px), 0.0, 1.0)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    x1 = center_x - half
    y1 = center_y - half
    x2 = x1 + disk_size
    y2 = y1 + disk_size

    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    src_x2 = disk_size - max(0, x2 - canvas_w)
    src_y2 = disk_size - max(0, y2 - canvas_h)

    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(canvas_w, x2)
    dst_y2 = min(canvas_h, y2)

    if dst_x2 > dst_x1 and dst_y2 > dst_y1:
        patch = resized[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
        m = mask[src_y1:src_y2, src_x1:src_x2]
        target = canvas[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)

        # Maximum intensity blend ensures seamless fusion with zero black rectangular seams
        blended = np.maximum(target, patch * m)
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(blended, 0, 255).astype(np.uint8)


def generate_circular_composite(
    samples: list,
    canvas_size: int = 3840,
    orbit_radius: int = 1350,
    disk_size: int = 740,
    direction: str = "ccw",
    start_angle_deg: float = -90.0,
    center_totality: bool = False
) -> np.ndarray:
    """
    Generates a circular wreath composite image (3840x3840).
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    cx, cy = canvas_size // 2, canvas_size // 2

    totality_img = None
    ring_samples = []

    for label, img in samples:
        if label == "totality" and center_totality:
            totality_img = img
        else:
            ring_samples.append((label, img))

    n = len(ring_samples)
    start_rad = math.radians(start_angle_deg)
    step_sign = 1.0 if direction.lower() == "ccw" else -1.0

    for i, (label, img) in enumerate(ring_samples):
        theta = start_rad + step_sign * (2 * math.pi * i) / n
        px = int(round(cx + orbit_radius * math.cos(theta)))
        py = int(round(cy - orbit_radius * math.sin(theta)))

        render_disk_patch(canvas, img, px, py, disk_size)

    # If totality is placed in center
    if center_totality and totality_img is not None:
        center_disk_size = int(disk_size * 1.25)
        render_disk_patch(canvas, totality_img, cx, cy, center_disk_size, feather_px=25.0)

    return canvas


def generate_diagonal_composite(
    samples: list,
    canvas_size: int = 3840,
    disk_size: int = 620,
    direction: str = "bottom_left_to_top_right",
    margin: int = 400
) -> np.ndarray:
    """
    Generates a diagonal progression composite image (3840x3840).
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    n = len(samples)

    if direction == "bottom_left_to_top_right":
        x_start, y_start = margin, canvas_size - margin
        x_end, y_end = canvas_size - margin, margin
    else:  # top_left_to_bottom_right
        x_start, y_start = margin, margin
        x_end, y_end = canvas_size - margin, canvas_size - margin

    for i, (label, img) in enumerate(samples):
        t = i / max(1, n - 1)
        px = int(round(x_start + t * (x_end - x_start)))
        py = int(round(y_start + t * (y_end - y_start)))

        # Slightly enlarge totality frame for dramatic emphasis
        current_disk_size = int(disk_size * 1.15) if label == "totality" else disk_size
        render_disk_patch(canvas, img, px, py, current_disk_size)

    return canvas


def generate_horizontal_composite(
    samples: list,
    canvas_size: int = 3840,
    disk_size: int = 600,
    margin: int = 350
) -> np.ndarray:
    """
    Generates a horizontal progression composite across the canvas.
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    n = len(samples)
    cy = canvas_size // 2

    for i, (label, img) in enumerate(samples):
        t = i / max(1, n - 1)
        px = int(round(margin + t * (canvas_size - 2 * margin)))
        py = cy

        current_disk_size = int(disk_size * 1.15) if label == "totality" else disk_size
        render_disk_patch(canvas, img, px, py, current_disk_size)

    return canvas


def generate_arc_composite(
    samples: list,
    canvas_size: int = 3840,
    disk_size: int = 620,
    margin: int = 380,
    arc_height: float = 0.5
) -> np.ndarray:
    """
    Generates a celestial arc progression composite across the sky.
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    n = len(samples)

    for i, (label, img) in enumerate(samples):
        t = i / max(1, n - 1)
        px = int(round(margin + t * (canvas_size - 2 * margin)))
        
        # Parabolic arc: normalized t in [-1, 1]
        t_norm = (t - 0.5) * 2.0
        py = int(round((canvas_size - margin) - (1.0 - t_norm**2) * (canvas_size * arc_height)))

        current_disk_size = int(disk_size * 1.15) if label == "totality" else disk_size
        render_disk_patch(canvas, img, px, py, current_disk_size)

    return canvas


def build_composite(
    layout: str = "circle",
    canvas_size: int = 3840,
    num_phases: int = 15,
    disk_size: int = None,
    orbit_radius: int = None,
    direction: str = None,
    center_totality: bool = False,
    out_dir: str = "040_out",
    output_path: str = None,
    totality_fraction: float = 0.45
):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    resolved_out_dir = os.path.join(repo_root, out_dir) if not os.path.isabs(out_dir) else out_dir
    os.makedirs(resolved_out_dir, exist_ok=True)

    print("=" * 65)
    print(f"ECLIPSE COMPOSITE GENERATOR (UHD Squared {canvas_size}x{canvas_size})")
    print("=" * 65)
    print(f"  Layout Mode     : {layout.upper()}")
    print(f"  Total Phases    : {num_phases}")
    print(f"  Resolution      : {canvas_size}x{canvas_size} px")
    print(f"  Center Totality : {center_totality}")

    samples = sample_eclipse_sequence(
        out_dir=resolved_out_dir,
        num_phases=num_phases,
        include_totality=True,
        equalize_color=True,
        totality_fraction=totality_fraction
    )
    print(f"  Extracted       : {len(samples)} stabilized keyframes")

    if disk_size is None:
        disk_size = int(round(canvas_size * (740.0 / 3840.0)))

    if orbit_radius is None:
        orbit_radius = int(round(canvas_size * (1350.0 / 3840.0)))

    if layout in ["circle", "ring"]:
        dir_val = direction or "ccw"
        canvas = generate_circular_composite(
            samples,
            canvas_size=canvas_size,
            orbit_radius=orbit_radius,
            disk_size=disk_size,
            direction=dir_val,
            center_totality=center_totality
        )
        default_name = f"eclipse_composite_circle_{canvas_size}p.png"
    elif layout == "diagonal":
        dir_val = direction or "bottom_left_to_top_right"
        canvas = generate_diagonal_composite(
            samples,
            canvas_size=canvas_size,
            disk_size=int(disk_size * 0.85),
            direction=dir_val
        )
        default_name = f"eclipse_composite_diagonal_{canvas_size}p.png"
    elif layout == "horizontal":
        canvas = generate_horizontal_composite(
            samples,
            canvas_size=canvas_size,
            disk_size=int(disk_size * 0.80)
        )
        default_name = f"eclipse_composite_horizontal_{canvas_size}p.png"
    elif layout == "arc":
        canvas = generate_arc_composite(
            samples,
            canvas_size=canvas_size,
            disk_size=int(disk_size * 0.85)
        )
        default_name = f"eclipse_composite_arc_{canvas_size}p.png"
    else:
        raise ValueError(f"Unknown layout mode: '{layout}'. Choose 'circle', 'diagonal', 'horizontal', or 'arc'.")

    final_out = output_path or os.path.join(resolved_out_dir, default_name)
    if not os.path.isabs(final_out):
        final_out = os.path.join(repo_root, final_out)

    print(f"\nSaving UHD Composite Artwork...")
    cv2.imwrite(final_out, canvas)
    
    # Also export high-quality JPG
    jpg_out = os.path.splitext(final_out)[0] + ".jpg"
    cv2.imwrite(jpg_out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

    size_png_mb = os.path.getsize(final_out) / (1024 * 1024)
    size_jpg_mb = os.path.getsize(jpg_out) / (1024 * 1024)

    print("=" * 65)
    print(f"SUCCESS: Composite image generated successfully!")
    print(f"  PNG (Lossless) : {final_out} ({size_png_mb:.2f} MB)")
    print(f"  JPG (High-Q)   : {jpg_out} ({size_jpg_mb:.2f} MB)")
    print("=" * 65)
    return final_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate high-resolution (3840x3840 UHD) composite images of the solar eclipse progression")
    parser.add_argument("--layout", "-l", type=str, choices=["circle", "ring", "diagonal", "horizontal", "arc", "all"], default="circle",
                        help="Composition layout: 'circle' (wreath ring), 'diagonal' (bottom-left to top-right), 'horizontal', 'arc', or 'all'. Default: circle")
    parser.add_argument("--size", "-s", type=int, default=3840,
                        help="Square canvas size in pixels (default: 3840 for UHD squared)")
    parser.add_argument("--phases", "-n", type=int, default=15,
                        help="Number of eclipse phase steps to render (default: 15)")
    parser.add_argument("--disk-size", type=int, default=None,
                        help="Diameter of individual solar disks in pixels (default: auto-scaled)")
    parser.add_argument("--orbit-radius", type=int, default=None,
                        help="Radius of circle orbit in pixels if layout is circle (default: auto-scaled)")
    parser.add_argument("--direction", type=str, default=None,
                        help="Direction of progression: for circle ('ccw' or 'cw'), for diagonal ('bottom_left_to_top_right' or 'top_left_to_bottom_right')")
    parser.add_argument("--center-totality", action="store_true",
                        help="Place totality in the center of the canvas in circular mode")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Custom output image path (default: 040_out/eclipse_composite_<layout>_<size>p.png)")
    args = parser.parse_args()

    if args.layout == "all":
        for lay in ["circle", "diagonal", "horizontal", "arc"]:
            build_composite(
                layout=lay,
                canvas_size=args.size,
                num_phases=args.phases,
                disk_size=args.disk_size,
                orbit_radius=args.orbit_radius,
                direction=args.direction,
                center_totality=args.center_totality,
                output_path=args.output
            )
    else:
        build_composite(
            layout=args.layout,
            canvas_size=args.size,
            num_phases=args.phases,
            disk_size=args.disk_size,
            orbit_radius=args.orbit_radius,
            direction=args.direction,
            center_totality=args.center_totality,
            output_path=args.output
        )
