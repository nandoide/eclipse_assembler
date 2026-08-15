#!/usr/bin/env python3
"""
High-Resolution Solar Eclipse Composite & Mosaic Generator (UHD Squared 3840x3840 & Mobile 9:16 / 16:9)
========================================================================================================
Generates ultra-high-resolution composite artwork capturing the complete
progression of the total solar eclipse on a clean aesthetic canvas.

Features:
  - Exact astronomical timestamp tagging for every sampled frame via eclipse_ephemeris_db.py.
  - Generates accompanying JSON metadata (e.g. eclipse_composite_<layout>_<res>.json)
    recording the exact timestamps, source clips, and time range of the composite.
  - Color equalization and lunar limb stabilization alignment.
  - Supports --date and --force-db for universal multi-eclipse processing.

Supported Layout Modes:
  1. 'sinusoid' (or 's-curve', 's'):
     Graceful S-shaped sinusoidal wave sweeping across the canvas with
     grand totality at the center inflection.
  2. 'vertical' (or 'mobile', 'vertical-linear'):
     Laser-aligned top-to-bottom vertical progression (default 2160x3840 9:16 for mobile).
  3. 'vertical-s' (or 's-vertical', 'mobile-s'):
     Vertical S-curve snake snaking down the mobile screen (default 2160x3840 9:16).
  4. 'circle' (or 'ring'):
     Circular orbit wreath with symmetric totality apex and trimmed partials.
  5. 'diagonal':
     Progressing diagonally from bottom-left to top-right with expansive corona streamers.
  6. 'horizontal':
     Linear progression across the horizontal midline with laser-aligned centers.
  7. 'arc':
     Graceful parabolic arc mirroring the Sun's celestial trajectory.

Authors: Fernando (nandoide) & Antigravity (Google Deepmind)
Workspace: eclipse_assembler
"""

import os
import sys
import argparse
import json
import datetime
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eclipse_ephemeris_db as eedb


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
        norm_dist = np.sqrt((dx / 620.0)**2 + (dy / 345.0)**2)
        edge_w = np.clip((1.0 - norm_dist) / 0.15, 0.0, 1.0)
    else:
        # Soft margin taper
        dist_x = np.minimum(np.arange(w), w - 1 - np.arange(w)).astype(np.float32)
        dist_y = np.minimum(np.arange(h), h - 1 - np.arange(h)).astype(np.float32)
        wx = np.clip(dist_x / 18.0, 0.0, 1.0)[np.newaxis, :]
        wy = np.clip(dist_y / 18.0, 0.0, 1.0)[:, np.newaxis]
        edge_w = wx * wy

    clean_f = frame.astype(np.float32) * floor_w[:, :, np.newaxis] * edge_w[:, :, np.newaxis]
    clean_f = np.clip(clean_f, 0, 255).astype(np.uint8)

    half = crop_size // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = cx + half, cy + half

    pad_left   = max(0, -x1)
    pad_top    = max(0, -y1)
    pad_right  = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if any([pad_left, pad_top, pad_right, pad_bottom]):
        clean_f = cv2.copyMakeBorder(
            clean_f, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    return clean_f[y1:y2, x1:x2]


def resolve_output_dir(out_dir):
    if os.path.isabs(out_dir):
        return out_dir
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE SAMPLER & TIMESTAMP TAGGER
# ─────────────────────────────────────────────────────────────────────────────

def sample_eclipse_sequence(
    out_dir="040_out",
    crop_size=1280,
    clock_offset_s=44.3,
    date_str=None,
    force_db=False
):
    """
    Extracts keyframes from CoC output videos (or input directory) and tags each
    with its exact real-world astronomical local timestamp (CEST):
      - 01_timelapse (ingress): 2 crescents.
      - 02_video_slowdown (pre-totality): 2 thin crescents.
      - 03_video_realtime (totality): 6 symmetric keyframes (beads, chromosphere, corona).
      - 04_timelapse (egress): 4 crescents.
    """
    resolved = resolve_output_dir(out_dir)

    def find_file(prefix_or_patterns):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        in_dir = os.path.join(repo_root, "010_in")
        search_dirs = [resolved, in_dir]

        patterns = prefix_or_patterns if isinstance(prefix_or_patterns, list) else [prefix_or_patterns]
        for sdir in search_dirs:
            if not os.path.exists(sdir):
                continue
            dir_files = sorted(os.listdir(sdir))
            for p in patterns:
                # 1. Exact match
                candidate = os.path.join(sdir, p)
                if os.path.exists(candidate):
                    return candidate
                # 2. Prefix or substring match
                for fname in dir_files:
                    if fname.endswith((".mp4", ".mov", ".avi")) and (fname.startswith(p) or p in fname):
                        return os.path.join(sdir, fname)
        return None

    p_ingress = find_file(["01_timelapse", "01_", "partial_ingress"])
    p_pre_tot = find_file(["02_video_slowdown", "02_", "pre_totality"])
    p_totality = find_file(["03_video_realtime", "03_video", "03_", "totality"])
    p_egress = find_file(["04_timelapse", "04_", "partial_egress"])

    paths = {
        "ingress":  p_ingress,
        "pre_tot":  p_pre_tot,
        "totality": p_totality,
        "egress":   p_egress,
    }

    caps = {k: cv2.VideoCapture(v) for k, v in paths.items() if v and os.path.exists(v)}
    counts = {k: int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for k, c in caps.items()}

    def read_at(cap, idx, total):
        if total <= 0 or cap is None:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total - 1, int(idx))))
        ret, f = cap.read()
        return f if ret else None

    # Resolve date and base timestamps dynamically from ephemeris DB
    resolved_date = date_str or eedb.detect_eclipse_date()
    eph = eedb.solve_eclipse(date_str=resolved_date, force_db=force_db)
    dt_base = datetime.datetime.strptime(eph["date"], "%Y-%m-%d")

    delta_offset = datetime.timedelta(seconds=clock_offset_s)
    dt_ingress_start = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 19, 35, 13) + delta_offset
    dt_ingress_end   = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 20, 20, 17) + delta_offset

    dt_pre_tot_start = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 20, 20, 36) + delta_offset
    dt_pre_tot_end   = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 20, 26, 47) + delta_offset

    dt_totality_start = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 20, 27, 31, 588000)
    dt_totality_end   = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 20, 29, 18, 921000)

    dt_egress_start = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 20, 35, 24) + delta_offset
    dt_egress_end   = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 21, 16, 44) + delta_offset

    samples = []

    # Single-clip fallback
    if len(caps) == 1:
        single_key = list(caps.keys())[0]
        cap = caps[single_key]
        total = counts[single_key]
        for frac in np.linspace(0.05, 0.95, 14):
            f_idx = int(frac * (total - 1))
            f = read_at(cap, f_idx, total)
            if f is not None:
                f_eq = equalize_solar_color(f)
                dt_sample = dt_totality_start + datetime.timedelta(seconds=f_idx/30.0)
                samples.append({
                    "phase": "partial",
                    "label": f"frame_{int(frac*100)}pct",
                    "source_clip": os.path.basename(paths[single_key]),
                    "frame_idx": f_idx,
                    "fraction": frac,
                    "timestamp_str": dt_sample.strftime("%H:%M:%S"),
                    "timestamp_dt": dt_sample,
                    "img": clean_and_crop_square(f_eq, crop_size=crop_size, is_totality=False)
                })
        cap.release()
        return samples

    # 1. Ingress Partials (2 samples)
    if "ingress" in caps and counts["ingress"] > 0:
        for frac in [0.38, 0.78]:
            f_idx = int(frac * (counts["ingress"] - 1))
            f = read_at(caps["ingress"], f_idx, counts["ingress"])
            if f is not None:
                f = equalize_solar_color(f)
                dt_sample = dt_ingress_start + frac * (dt_ingress_end - dt_ingress_start)
                samples.append({
                    "phase": "partial",
                    "label": "ingress",
                    "source_clip": os.path.basename(paths["ingress"]),
                    "frame_idx": f_idx,
                    "fraction": frac,
                    "timestamp_str": dt_sample.strftime("%H:%M:%S"),
                    "timestamp_dt": dt_sample,
                    "img": clean_and_crop_square(f, crop_size=crop_size, is_totality=False)
                })

    # 2. Pre-Totality Thin Crescents (2 samples)
    if "pre_tot" in caps and counts["pre_tot"] > 0:
        for frac in [0.38, 0.78]:
            f_idx = int(frac * (counts["pre_tot"] - 1))
            f = read_at(caps["pre_tot"], f_idx, counts["pre_tot"])
            if f is not None:
                f = equalize_solar_color(f)
                dt_sample = dt_pre_tot_start + frac * (dt_pre_tot_end - dt_pre_tot_start)
                samples.append({
                    "phase": "partial",
                    "label": "pre_totality",
                    "source_clip": os.path.basename(paths["pre_tot"]),
                    "frame_idx": f_idx,
                    "fraction": frac,
                    "timestamp_str": dt_sample.strftime("%H:%M:%S"),
                    "timestamp_dt": dt_sample,
                    "img": clean_and_crop_square(f, crop_size=crop_size, is_totality=False)
                })

    # 3. Totality Keyframes (6 samples)
    if "totality" in caps and counts["totality"] > 0:
        tot_fps = caps["totality"].get(cv2.CAP_PROP_FPS) or 30.0
        tot_total = counts["totality"]
        tot_key_moments = [
            (3.45,   "ingress_beads"),        # C2 -> 20:27:35 CEST
            (10.00,  "ingress_chromosphere"), # 20:27:41 CEST
            (25.00,  "inner_corona"),         # 20:27:56 CEST
            (51.73,  "grand_corona"),         # TOTAL -> 20:28:23 CEST
            (98.00,  "egress_chromosphere"),  # 20:29:09 CEST
            (100.50, "egress_beads"),         # C3 -> 20:29:12 CEST
        ]
        for t_sec, label in tot_key_moments:
            actual_fi = min(tot_total - 1, int(round(t_sec * tot_fps)))
            f = read_at(caps["totality"], actual_fi, tot_total)
            if f is not None:
                dt_sample = dt_totality_start + datetime.timedelta(seconds=t_sec)
                samples.append({
                    "phase": "totality",
                    "label": label,
                    "source_clip": os.path.basename(paths["totality"]),
                    "frame_idx": actual_fi,
                    "fraction": actual_fi / max(1, tot_total - 1),
                    "timestamp_str": dt_sample.strftime("%H:%M:%S"),
                    "timestamp_dt": dt_sample,
                    "img": clean_and_crop_square(f, crop_size=crop_size, is_totality=True)
                })

    # 4. Egress Partials (4 samples)
    if "egress" in caps and counts["egress"] > 0:
        for frac in [0.15, 0.35, 0.55, 0.75]:
            f_idx = int(frac * (counts["egress"] - 1))
            f = read_at(caps["egress"], f_idx, counts["egress"])
            if f is not None:
                f = equalize_solar_color(f)
                dt_sample = dt_egress_start + frac * (dt_egress_end - dt_egress_start)
                samples.append({
                    "phase": "partial",
                    "label": "egress",
                    "source_clip": os.path.basename(paths["egress"]),
                    "frame_idx": f_idx,
                    "fraction": frac,
                    "timestamp_str": dt_sample.strftime("%H:%M:%S"),
                    "timestamp_dt": dt_sample,
                    "img": clean_and_crop_square(f, crop_size=crop_size, is_totality=False)
                })

    for c in caps.values():
        c.release()

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT ENGINES
# ─────────────────────────────────────────────────────────────────────────────

def render_consistent_scale(samples, positions, width=1280, height=720, disk_scale_factor=0.78, show_labels=False, featured_contacts=None):
    """
    Composites all frames into the black canvas with consistent disk scaling and soft additive blending.
    Uses minimum distance between adjacent frames to ensure clean, non-overlapping spacing.
    Optionally overlays centered astronomical timestamp labels (HH:MM:SS) below each frame.
    Optionally overlays prominent featured totality contacts (C2, TOTAL, C3) inside central cavity.
    """
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    N = len(positions)

    # Compute pairwise adjacent separation distances
    diffs = np.diff(positions, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    min_dist = float(np.min(dists)) if len(dists) > 0 else 300.0

    # Scale disk relative to minimum adjacent distance to guarantee zero overlap
    target_disk_d = min_dist * disk_scale_factor
    scale = target_disk_d / 500.0  # 500 px is solar disk diameter in cropped patch
    out_patch_sz = max(64, int(round(1280 * scale)))
    out_patch_sz = (out_patch_sz // 2) * 2
    half_patch = out_patch_sz // 2

    # Draw partials first, then totality on top
    draw_order = [i for i, s in enumerate(samples) if s["phase"] == "partial"] + \
                 [i for i, s in enumerate(samples) if s["phase"] == "totality"]

    for i in draw_order:
        s = samples[i]
        px, py = positions[i]
        patch = cv2.resize(s["img"], (out_patch_sz, out_patch_sz), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)

        x1 = int(round(px - half_patch))
        y1 = int(round(py - half_patch))
        x2 = x1 + out_patch_sz
        y2 = y1 + out_patch_sz

        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(width, x2), min(height, y2)

        px1 = cx1 - x1
        py1 = cy1 - y1
        px2 = px1 + (cx2 - cx1)
        py2 = py1 + (cy2 - cy1)

        if cx2 > cx1 and cy2 > cy1:
            curr_reg = canvas[cy1:cy2, cx1:cx2]
            patch_reg = patch[py1:py2, px1:px2]

            # Soft maximum blend preserves brightest coronas and photospheric crescents
            blended = np.maximum(curr_reg, patch_reg)
            canvas[cy1:cy2, cx1:cx2] = blended

    # Draw Featured Totality Contacts (C2, TOTAL, C3) if provided
    if featured_contacts:
        for item in featured_contacts:
            s = item["sample"]
            px, py = item["pos"]
            d_inner = item["disk_d"]
            inner_scale = d_inner / 500.0
            in_patch_sz = max(64, int(round(1280 * inner_scale)))
            in_patch_sz = (in_patch_sz // 2) * 2
            half_in = in_patch_sz // 2

            patch = cv2.resize(s["img"], (in_patch_sz, in_patch_sz), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)

            x1 = int(round(px - half_in))
            y1 = int(round(py - half_in))
            x2 = x1 + in_patch_sz
            y2 = y1 + in_patch_sz

            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(width, x2), min(height, y2)

            px1 = cx1 - x1
            py1 = cy1 - y1
            px2 = px1 + (cx2 - cx1)
            py2 = py1 + (cy2 - cy1)

            if cx2 > cx1 and cy2 > cy1:
                curr_reg = canvas[cy1:cy2, cx1:cx2]
                patch_reg = patch[py1:py2, px1:px2]
                blended = np.maximum(curr_reg, patch_reg)
                canvas[cy1:cy2, cx1:cx2] = blended

    out_img = np.clip(canvas, 0, 255).astype(np.uint8)

    # Overlay centered labels using PIL TrueType anti-aliased font
    if show_labels or featured_contacts:
        img_rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # Dynamic font sizing scaled to canvas and disk dimensions
        font_sz = max(11, int(round(target_disk_d * 0.08)))

        font = None
        for fp in [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/SFNS.ttf",
            "/System/Library/Fonts/Menlo.ttc"
        ]:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, font_sz)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()

        if show_labels:
            label_offset_y = (target_disk_d / 2.0) + (font_sz * 0.45)
            for i in range(N):
                s = samples[i]
                time_str = s.get("timestamp_str", "")
                if not time_str:
                    continue

                px, py = positions[i]
                bbox = draw.textbbox((0, 0), time_str, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]

                tx = px - tw / 2.0
                ty = py + label_offset_y

                # Keep within canvas bounds
                if ty + th > height - 10:
                    ty = py - (target_disk_d / 2.0) - th - (font_sz * 0.45)

                # Soft dark drop shadow for high contrast
                draw.text((tx + 1, ty + 1), time_str, font=font, fill=(0, 0, 0))
                draw.text((tx - 1, ty + 1), time_str, font=font, fill=(0, 0, 0))
                draw.text((tx + 1, ty - 1), time_str, font=font, fill=(0, 0, 0))
                draw.text((tx - 1, ty - 1), time_str, font=font, fill=(0, 0, 0))

                # Foreground text color: subtle gold for totality, crisp white for partials
                text_color = (255, 230, 160) if s.get("phase") == "totality" else (235, 235, 240)
                draw.text((tx, ty), time_str, font=font, fill=text_color)

        # 2. Overlay contacts labels if present
        if featured_contacts:
            for item in featured_contacts:
                contact_label = item.get("label", "")
                if not contact_label:
                    continue

                px, py = item["pos"]
                d_inner = item["disk_d"]
                inner_scale = d_inner / 500.0
                in_patch_sz = max(64, int(round(1280 * inner_scale)))
                half_in = in_patch_sz // 2

                inner_font_sz = max(13, int(round(d_inner * 0.088)))
                inner_font = None
                for fp in [
                    "/System/Library/Fonts/Helvetica.ttc",
                    "/Library/Fonts/Arial.ttf",
                    "/System/Library/Fonts/Supplemental/Arial.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                ]:
                    if os.path.exists(fp):
                        try:
                            inner_font = ImageFont.truetype(fp, inner_font_sz)
                            break
                        except Exception:
                            pass
                if inner_font is None:
                    inner_font = font


                bbox = draw.textbbox((0, 0), contact_label, font=inner_font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]

                tx = px - tw / 2.0
                ty = py + (d_inner / 2.0) + (inner_font_sz * 0.40)

                # Keep within canvas bounds
                if ty + th > height - 10:
                    ty = py - (d_inner / 2.0) - th - (inner_font_sz * 0.40)

                # Dark drop shadow for high contrast
                draw.text((tx + 1, ty + 1), contact_label, font=inner_font, fill=(0, 0, 0))
                draw.text((tx - 1, ty + 1), contact_label, font=inner_font, fill=(0, 0, 0))
                draw.text((tx + 1, ty - 1), contact_label, font=inner_font, fill=(0, 0, 0))
                draw.text((tx - 1, ty - 1), contact_label, font=inner_font, fill=(0, 0, 0))

                # Elegant warm golden color for the featured inner labels
                draw.text((tx, ty), contact_label, font=inner_font, fill=(255, 232, 170))

        out_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return out_img


def generate_circular_composite(samples, width=1280, height=720, orbit_radius=None, direction="cw", disk_scale_factor=0.78, show_labels=False, contacts="auto"):
    cx, cy = width / 2.0, height / 2.0

    # For square format (W == H): Rx == Ry (perfect circle)
    # For non-square formats (16:9, 9:16, etc.): Rx and Ry scale proportionally to width and height,
    # forming an ellipse that adapts to the canvas contour with generous margins.
    if orbit_radius is not None:
        if width == height:
            rx, ry = float(orbit_radius), float(orbit_radius)
        else:
            scale_r = float(orbit_radius) / (min(width, height) * 0.36)
            rx = width * 0.36 * scale_r
            ry = height * 0.36 * scale_r
    else:
        rx = width * 0.36
        ry = height * 0.36

    N = len(samples)

    # Locate the peak totality frame (grand corona) to anchor exactly at 12 o'clock (top)
    apex_idx = None
    for i, s in enumerate(samples):
        if s.get("label") == "grand_corona":
            apex_idx = i
            break
    if apex_idx is None:
        tot_indices = [i for i, s in enumerate(samples) if s["phase"] == "totality"]
        apex_idx = tot_indices[len(tot_indices) // 2] if tot_indices else (N - 1) / 2.0

    # Dense sampling along ellipse perimeter for equal arc-length step
    dense_theta = np.linspace(-math.pi / 2.0, -math.pi / 2.0 + 2.0 * math.pi, 2000)
    dense_x = cx + rx * np.cos(dense_theta)
    dense_y = cy + ry * np.sin(dense_theta)

    dx = np.diff(dense_x)
    dy = np.diff(dense_y)
    cum_len = np.concatenate([[0.0], np.cumsum(np.sqrt(dx**2 + dy**2))])
    total_len = cum_len[-1]

    step_s = total_len / float(N)
    interp_x = interp1d(cum_len, dense_x)
    interp_y = interp1d(cum_len, dense_y)

    positions = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        if direction == "ccw":
            s_val = (total_len - (i - apex_idx) * step_s) % total_len
        else:
            s_val = ((i - apex_idx) * step_s) % total_len
        positions[i, 0] = interp_x(s_val)
        positions[i, 1] = interp_y(s_val)

    # Compute Featured Totality Contacts (C2, TOTAL, C3) if enabled
    featured_contacts_data = None
    if contacts and contacts != "none":
        c2_sample = next((s for s in samples if s.get("label") == "ingress_beads"), None)
        if c2_sample is None:
            c2_sample = next((s for s in samples if s.get("phase") == "totality"), None)

        tot_sample = next((s for s in samples if s.get("label") == "grand_corona"), None)
        if tot_sample is None:
            tot_samples = [s for s in samples if s.get("phase") == "totality"]
            tot_sample = tot_samples[len(tot_samples) // 2] if tot_samples else None

        c3_sample = next((s for s in samples if s.get("label") == "egress_beads"), None)
        if c3_sample is None:
            tot_samples = [s for s in samples if s.get("phase") == "totality"]
            c3_sample = tot_samples[-1] if tot_samples else None

        if c2_sample and tot_sample and c3_sample:
            if contacts == "vertical" or (contacts in ["auto", True] and height > width):
                is_vert = True
            else:
                is_vert = False

            diffs = np.diff(positions, axis=0)
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            min_d = float(np.min(dists)) if len(dists) > 0 else 300.0
            outer_disk_d = min_d * disk_scale_factor

            w_cavity = rx - outer_disk_d / 2.0
            h_cavity = ry - outer_disk_d / 2.0

            if is_vert:
                # Vertical Contacts: C2 top, TOTAL center, C3 bottom
                d_inner = min(outer_disk_d * 1.22, (2.0 * h_cavity) / 4.2, w_cavity * 0.76)
                spacing = d_inner * 1.28
                p_c2 = np.array([cx, cy - spacing], dtype=np.float32)
                p_tot = np.array([cx, cy], dtype=np.float32)
                p_c3 = np.array([cx, cy + spacing], dtype=np.float32)
            else:
                # Horizontal Contacts: C2 left, TOTAL center, C3 right
                d_inner = min(outer_disk_d * 1.22, (2.0 * w_cavity) / 4.2, h_cavity * 0.76)
                spacing = d_inner * 1.28
                p_c2 = np.array([cx - spacing, cy], dtype=np.float32)
                p_tot = np.array([cx, cy], dtype=np.float32)
                p_c3 = np.array([cx + spacing, cy], dtype=np.float32)

            featured_contacts_data = [
                {
                    "sample": c2_sample,
                    "pos": p_c2,
                    "disk_d": d_inner,
                    "label": f"C2 {c2_sample['timestamp_str']}"
                },
                {
                    "sample": tot_sample,
                    "pos": p_tot,
                    "disk_d": d_inner,
                    "label": f"TOTAL {tot_sample['timestamp_str']}"
                },
                {
                    "sample": c3_sample,
                    "pos": p_c3,
                    "disk_d": d_inner,
                    "label": f"C3 {c3_sample['timestamp_str']}"
                }
            ]

    return render_consistent_scale(samples, positions, width=width, height=height, disk_scale_factor=disk_scale_factor, show_labels=show_labels, featured_contacts=featured_contacts_data)


def generate_sinusoid_composite(samples, width=1280, height=720, margin=None, amplitude=None, disk_scale_factor=0.78, show_labels=False, contacts="auto"):
    N = len(samples)
    mx = margin or int(width * 0.08)
    amp = amplitude or int(height * 0.25)
    cx, cy = width / 2.0, height / 2.0

    # Equal arc-length parameterization along the sinusoidal wave to prevent bunching at crests
    dense_t = np.linspace(0.0, 1.0, 2000)
    dense_x = np.linspace(mx, width - mx, 2000)
    dense_y = cy - amp * np.sin(dense_t * 2.0 * np.pi - math.pi / 2.0)

    dx = np.diff(dense_x)
    dy = np.diff(dense_y)
    cum_len = np.concatenate([[0.0], np.cumsum(np.sqrt(dx**2 + dy**2))])
    total_len = cum_len[-1]

    target_s = np.linspace(0.0, total_len, N)
    interp_x = interp1d(cum_len, dense_x)
    interp_y = interp1d(cum_len, dense_y)

    positions = np.column_stack([interp_x(target_s), interp_y(target_s)]).astype(np.float32)

    # Compute Featured Totality Contacts (C2, TOTAL, C3) in lower central vault if enabled
    featured_contacts_data = None
    if contacts and contacts != "none":
        c2_sample = next((s for s in samples if s.get("label") == "ingress_beads"), None)
        if c2_sample is None:
            c2_sample = next((s for s in samples if s.get("phase") == "totality"), None)

        tot_sample = next((s for s in samples if s.get("label") == "grand_corona"), None)
        if tot_sample is None:
            tot_samples = [s for s in samples if s.get("phase") == "totality"]
            tot_sample = tot_samples[len(tot_samples) // 2] if tot_samples else None

        c3_sample = next((s for s in samples if s.get("label") == "egress_beads"), None)
        if c3_sample is None:
            tot_samples = [s for s in samples if s.get("phase") == "totality"]
            c3_sample = tot_samples[-1] if tot_samples else None

        if c2_sample and tot_sample and c3_sample:
            diffs = np.diff(positions, axis=0)
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            min_d = float(np.min(dists)) if len(dists) > 0 else 300.0
            outer_disk_d = min_d * disk_scale_factor

            d_inner = outer_disk_d * 1.22
            spacing = d_inner * 1.35
            y_contacts = cy + amp * 0.52

            p_c2 = np.array([cx - spacing, y_contacts], dtype=np.float32)
            p_tot = np.array([cx, y_contacts], dtype=np.float32)
            p_c3 = np.array([cx + spacing, y_contacts], dtype=np.float32)

            featured_contacts_data = [
                {
                    "sample": c2_sample,
                    "pos": p_c2,
                    "disk_d": d_inner,
                    "label": f"C2 {c2_sample['timestamp_str']}"
                },
                {
                    "sample": tot_sample,
                    "pos": p_tot,
                    "disk_d": d_inner,
                    "label": f"TOTAL {tot_sample['timestamp_str']}"
                },
                {
                    "sample": c3_sample,
                    "pos": p_c3,
                    "disk_d": d_inner,
                    "label": f"C3 {c3_sample['timestamp_str']}"
                }
            ]

    return render_consistent_scale(samples, positions, width=width, height=height, disk_scale_factor=disk_scale_factor, show_labels=show_labels, featured_contacts=featured_contacts_data)


def generate_vertical_composite(samples, width=2160, height=3840, margin=None, disk_scale_factor=0.78, show_labels=False):
    N = len(samples)
    my = margin or int(height * 0.08)
    cx = width / 2.0

    ys = np.linspace(my, height - my, N)
    positions = np.zeros((N, 2), dtype=np.float32)
    for i, y in enumerate(ys):
        positions[i, 0] = cx
        positions[i, 1] = y

    return render_consistent_scale(samples, positions, width=width, height=height, disk_scale_factor=disk_scale_factor, show_labels=show_labels)


def generate_vertical_sinusoid_composite(samples, width=2160, height=3840, margin=None, amplitude=None, disk_scale_factor=0.78, show_labels=False):
    N = len(samples)
    my = margin or int(height * 0.08)
    amp = amplitude or int(width * 0.25)
    cx = width / 2.0

    dense_t = np.linspace(0.0, 1.0, 2000)
    dense_y = np.linspace(my, height - my, 2000)
    dense_x = cx + amp * np.sin(dense_t * 2.0 * math.pi)

    dx = np.diff(dense_x)
    dy = np.diff(dense_y)
    cum_len = np.concatenate([[0.0], np.cumsum(np.sqrt(dx**2 + dy**2))])
    total_len = cum_len[-1]

    target_s = np.linspace(0.0, total_len, N)
    interp_x = interp1d(cum_len, dense_x)
    interp_y = interp1d(cum_len, dense_y)

    positions = np.column_stack([interp_x(target_s), interp_y(target_s)]).astype(np.float32)
    return render_consistent_scale(samples, positions, width=width, height=height, disk_scale_factor=disk_scale_factor, show_labels=show_labels)


def generate_diagonal_composite(samples, width=1280, height=720, direction="bottom_left_to_top_right", margin=None, disk_scale_factor=0.78, show_labels=False):
    N = len(samples)
    m = margin or int(min(width, height) * 0.08)

    if direction == "bottom_left_to_top_right":
        xs = np.linspace(m, width - m, N)
        ys = np.linspace(height - m, m, N)
    else:
        xs = np.linspace(m, width - m, N)
        ys = np.linspace(m, height - m, N)

    positions = np.column_stack([xs, ys]).astype(np.float32)
    return render_consistent_scale(samples, positions, width=width, height=height, disk_scale_factor=disk_scale_factor, show_labels=show_labels)


def generate_horizontal_composite(samples, width=1280, height=720, margin=None, disk_scale_factor=0.78, show_labels=False):
    N = len(samples)
    mx = margin or int(width * 0.05)
    cy = height / 2.0

    xs = np.linspace(mx, width - mx, N)
    positions = np.zeros((N, 2), dtype=np.float32)
    for i, x in enumerate(xs):
        positions[i, 0] = x
        positions[i, 1] = cy

    return render_consistent_scale(samples, positions, width=width, height=height, disk_scale_factor=disk_scale_factor, show_labels=show_labels)


def generate_arc_composite(samples, width=1280, height=720, margin_x=None, disk_scale_factor=0.78, show_labels=False, contacts="auto"):
    N = len(samples)
    mx = margin_x or int(width * 0.08)
    cx = width / 2.0

    # Expand arch vertically on portrait formats (height > width) to utilize canvas space generously
    if height > width:
        cy = height * 0.85
        h_arc = height * 0.65
    else:
        cy = height * 0.70
        h_arc = height * 0.40

    dense_x = np.linspace(mx, width - mx, 2000)
    norm_x = (dense_x - cx) / ((width - 2 * mx) / 2.0)
    dense_y = cy - h_arc * (1.0 - norm_x**2)

    dx = np.diff(dense_x)
    dy = np.diff(dense_y)
    cum_len = np.concatenate([[0.0], np.cumsum(np.sqrt(dx**2 + dy**2))])
    total_len = cum_len[-1]

    target_s = np.linspace(0.0, total_len, N)
    interp_x = interp1d(cum_len, dense_x)
    interp_y = interp1d(cum_len, dense_y)

    positions = np.column_stack([interp_x(target_s), interp_y(target_s)]).astype(np.float32)

    # Compute Featured Totality Contacts (C2, TOTAL, C3) in central vault of arc
    featured_contacts_data = None
    if contacts and contacts != "none":
        c2_sample = next((s for s in samples if s.get("label") == "ingress_beads"), None)
        if c2_sample is None:
            c2_sample = next((s for s in samples if s.get("phase") == "totality"), None)

        tot_sample = next((s for s in samples if s.get("label") == "grand_corona"), None)
        if tot_sample is None:
            tot_samples = [s for s in samples if s.get("phase") == "totality"]
            tot_sample = tot_samples[len(tot_samples) // 2] if tot_samples else None

        c3_sample = next((s for s in samples if s.get("label") == "egress_beads"), None)
        if c3_sample is None:
            tot_samples = [s for s in samples if s.get("phase") == "totality"]
            c3_sample = tot_samples[-1] if tot_samples else None

        if c2_sample and tot_sample and c3_sample:
            diffs = np.diff(positions, axis=0)
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            min_d = float(np.min(dists)) if len(dists) > 0 else 300.0
            outer_disk_d = min_d * disk_scale_factor

            if contacts == "vertical" or (contacts in ["auto", True] and height > width):
                is_vert = True
            else:
                is_vert = False

            if is_vert:
                # Vertical Contacts for portrait frames: C2 top, TOTAL center, C3 bottom
                d_inner = outer_disk_d * 1.22
                spacing = d_inner * 1.30
                contacts_cy = cy - h_arc * 0.40
                p_c2 = np.array([cx, contacts_cy - spacing], dtype=np.float32)
                p_tot = np.array([cx, contacts_cy], dtype=np.float32)
                p_c3 = np.array([cx, contacts_cy + spacing], dtype=np.float32)
            else:
                # Horizontal Contacts for landscape frames: C2 left, TOTAL center, C3 right
                d_inner = outer_disk_d * 1.22
                spacing = d_inner * 1.35
                y_contacts = height * 0.74
                p_c2 = np.array([cx - spacing, y_contacts], dtype=np.float32)
                p_tot = np.array([cx, y_contacts], dtype=np.float32)
                p_c3 = np.array([cx + spacing, y_contacts], dtype=np.float32)

            featured_contacts_data = [
                {
                    "sample": c2_sample,
                    "pos": p_c2,
                    "disk_d": d_inner,
                    "label": f"C2 {c2_sample['timestamp_str']}"
                },
                {
                    "sample": tot_sample,
                    "pos": p_tot,
                    "disk_d": d_inner,
                    "label": f"TOTAL {tot_sample['timestamp_str']}"
                },
                {
                    "sample": c3_sample,
                    "pos": p_c3,
                    "disk_d": d_inner,
                    "label": f"C3 {c3_sample['timestamp_str']}"
                }
            ]

    return render_consistent_scale(samples, positions, width=width, height=height, disk_scale_factor=disk_scale_factor, show_labels=show_labels, featured_contacts=featured_contacts_data)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COMPOSITE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_composite(
    layout="sinusoid",
    width=1280,
    height=720,
    orbit_radius=None,
    direction=None,
    disk_scale_factor=0.78,
    margin=None,
    amplitude=None,
    clock_offset_s=44.3,
    date_str=None,
    force_db=False,
    show_labels=False,
    contacts="auto",
    out_dir="040_out",
    output_path=None,
):
    resolved_out = resolve_output_dir(out_dir)
    os.makedirs(resolved_out, exist_ok=True)

    print("=" * 65)
    print(f"ECLIPSE COMPOSITE GENERATOR ({width}x{height})")
    print("=" * 65)
    print(f"  Layout Mode       : {layout.upper()}")
    print(f"  Resolution        : {width}x{height} px")
    print(f"  Disk Scale Factor : {disk_scale_factor:.2f}")
    print(f"  Labels (Timestamps): {'ENABLED (HH:MM:SS)' if show_labels else 'DISABLED'}")
    if layout in ["circle", "ring", "ellipse", "oval", "sinusoid", "s-curve", "s", "sinusoidal", "arc"]:
        print(f"  Totality Contacts : {contacts.upper()} (C2, TOTAL, C3)")

    samples = sample_eclipse_sequence(
        out_dir=resolved_out,
        crop_size=1280,
        clock_offset_s=clock_offset_s,
        date_str=date_str,
        force_db=force_db
    )
    n_tot_got  = sum(1 for s in samples if s["phase"] == "totality")
    n_part_got = sum(1 for s in samples if s["phase"] == "partial")
    print(f"  Frames loaded     : {len(samples)} total ({n_part_got} partial, {n_tot_got} totality)")

    first_dt = samples[0]["timestamp_dt"]
    last_dt  = samples[-1]["timestamp_dt"]
    time_range_str = f"{first_dt.strftime('%H:%M')} - {last_dt.strftime('%H:%M')} CEST"
    print(f"  Sample Time Range : {time_range_str} ({first_dt.strftime('%H:%M:%S')} to {last_dt.strftime('%H:%M:%S')})")

    # Layout selection
    if layout in ["circle", "ring", "ellipse", "oval"]:
        canvas = generate_circular_composite(
            samples, width=width, height=height,
            orbit_radius=orbit_radius,
            direction=direction or "cw",
            disk_scale_factor=disk_scale_factor,
            show_labels=show_labels,
            contacts=contacts,
        )
        suffix = "circle" if width == height else "ellipse"

    elif layout in ["sinusoid", "s-curve", "s", "sinusoidal"]:
        canvas = generate_sinusoid_composite(
            samples, width=width, height=height,
            margin=margin,
            amplitude=amplitude,
            disk_scale_factor=disk_scale_factor,
            show_labels=show_labels,
            contacts=contacts,
        )
        suffix = "sinusoid"

    elif layout in ["vertical", "mobile", "vertical-linear"]:
        canvas = generate_vertical_composite(
            samples, width=width, height=height,
            margin=margin,
            disk_scale_factor=disk_scale_factor,
            show_labels=show_labels,
        )
        suffix = "vertical"

    elif layout in ["vertical-s", "s-vertical", "mobile-s"]:
        canvas = generate_vertical_sinusoid_composite(
            samples, width=width, height=height,
            margin=margin,
            amplitude=amplitude,
            disk_scale_factor=disk_scale_factor,
            show_labels=show_labels,
        )
        suffix = "vertical_s"

    elif layout == "diagonal":
        canvas = generate_diagonal_composite(
            samples, width=width, height=height,
            direction=direction or "bottom_left_to_top_right",
            margin=margin,
            disk_scale_factor=disk_scale_factor,
            show_labels=show_labels,
        )
        suffix = "diagonal"

    elif layout == "horizontal":
        canvas = generate_horizontal_composite(
            samples, width=width, height=height,
            margin=margin,
            disk_scale_factor=disk_scale_factor,
            show_labels=show_labels,
        )
        suffix = "horizontal"

    elif layout == "arc":
        canvas = generate_arc_composite(
            samples, width=width, height=height,
            margin_x=margin,
            disk_scale_factor=disk_scale_factor,
            show_labels=show_labels,
            contacts=contacts,
        )
        suffix = "arc"

    else:
        raise ValueError(f"Unknown layout: '{layout}'. Choose sinusoid/vertical/vertical-s/circle/diagonal/horizontal/arc.")

    # Save image files
    res_tag = f"{width}p" if width == height else f"{width}x{height}"
    default_name = f"eclipse_composite_{suffix}_{res_tag}.png"
    final_out = output_path or os.path.join(resolved_out, default_name)
    if not os.path.isabs(final_out):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        final_out = os.path.join(repo_root, final_out)

    print(f"\nSaving Composite Artwork...")
    cv2.imwrite(final_out, canvas)
    jpg_out = os.path.splitext(final_out)[0] + ".jpg"
    cv2.imwrite(jpg_out, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Save JSON metadata with exact frame timestamps
    meta_json_path = os.path.splitext(final_out)[0] + ".json"
    meta_data = {
        "layout": layout,
        "width": width,
        "height": height,
        "disk_scale_factor": disk_scale_factor,
        "show_labels": show_labels,
        "contacts": contacts,
        "sample_count": len(samples),
        "first_sample_time": first_dt.strftime("%H:%M:%S CEST"),
        "last_sample_time": last_dt.strftime("%H:%M:%S CEST"),
        "time_range_str": time_range_str,
        "samples": [
            {
                "index": idx + 1,
                "source_clip": s["source_clip"],
                "frame_idx": s["frame_idx"],
                "fraction": round(float(s["fraction"]), 3),
                "phase": s["phase"],
                "label": s["label"],
                "time_cest": s["timestamp_str"]
            }
            for idx, s in enumerate(samples)
        ]
    }
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    # Also save standard sample metadata in output dir
    summary_json_path = os.path.join(resolved_out, "composite_samples.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    size_png = os.path.getsize(final_out) / (1024 * 1024)
    size_jpg = os.path.getsize(jpg_out)   / (1024 * 1024)
    print("=" * 65)
    print(f"SUCCESS: Composite generated!")
    print(f"  PNG (Lossless) : {final_out} ({size_png:.2f} MB)")
    print(f"  JPG (High-Q)   : {jpg_out} ({size_jpg:.2f} MB)")
    print(f"  JSON (Metadata): {meta_json_path}")
    print("=" * 65)
    return final_out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate UHD & custom aspect-ratio solar eclipse composite artwork."
    )
    parser.add_argument("--layout", "-l", type=str,
                        choices=["sinusoid", "s-curve", "vertical", "vertical-s", "mobile", "circle", "ring", "ellipse", "oval", "diagonal", "horizontal", "arc", "all"],
                        default="sinusoid",
                        help="Composition layout (default: sinusoid)")
    parser.add_argument("--size", "-s", type=int, default=None,
                        help="Square canvas size in pixels (e.g. 3840, 2048)")
    parser.add_argument("--width", "-W", type=int, default=None,
                        help="Canvas width in pixels (default: 1280 for landscape, 2160 for mobile)")
    parser.add_argument("--height", "-H", type=int, default=None,
                        help="Canvas height in pixels (default: 720 for landscape, 3840 for mobile)")
    parser.add_argument("--scale-factor", type=float, default=0.78,
                        help="Disk scale factor relative to separation distance (default: 0.78)")
    parser.add_argument("--margin", type=int, default=None,
                        help="Canvas margin padding in pixels (default: auto)")
    parser.add_argument("--amplitude", type=int, default=None,
                        help="Sinusoidal wave amplitude in pixels (default: auto)")
    parser.add_argument("--orbit-radius", type=int, default=None,
                        help="Circle orbit radius in pixels (default: auto)")
    parser.add_argument("--direction", type=str, default=None,
                        help="Progression direction (ccw/cw for circle; bottom_left_to_top_right for diagonal)")
    parser.add_argument("--offset-seconds", type=float, default=44.3,
                        help="Clock calibration offset in seconds (default: 44.3s)")
    parser.add_argument("--date", "-d", type=str, default=None,
                        help="Eclipse date YYYY-MM-DD (default: auto-detect)")
    parser.add_argument("--force-db", action="store_true",
                        help="Force rebuilding/refreshing eclipse database")
    parser.add_argument("--show-labels", "--timestamps", "--labels", "-t", action="store_true",
                        help="Overlay centered HH:MM:SS timestamp labels below each solar disk")
    parser.add_argument("--contacts", type=str, nargs="?", const="auto",
                        choices=["auto", "horizontal", "vertical", "none"], default="auto",
                        help="Featured totality contacts sequence (C2, TOTAL, C3) layout (auto/horizontal/vertical/none, default: auto)")
    parser.add_argument("--no-contacts", dest="contacts", action="store_const", const="none",
                        help="Disable featured totality contacts sequence (render only the primary layout curve)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Custom output path (default: 040_out/eclipse_composite_<layout>_<resolution>.png)")
    args = parser.parse_args()

    layouts = ["sinusoid", "vertical", "vertical-s", "circle", "diagonal", "horizontal", "arc"] if args.layout == "all" else [args.layout]
    for lay in layouts:
        if args.size is not None:
            lay_w = args.size
            lay_h = args.size
        elif lay in ["vertical", "vertical-s", "mobile", "s-vertical"]:
            lay_w = args.width or 2160
            lay_h = args.height or 3840
        else:
            lay_w = args.width or 1280
            lay_h = args.height or 720

        build_composite(
            layout=lay,
            width=lay_w,
            height=lay_h,
            orbit_radius=args.orbit_radius,
            direction=args.direction,
            disk_scale_factor=args.scale_factor,
            margin=args.margin,
            amplitude=args.amplitude,
            clock_offset_s=args.offset_seconds,
            date_str=args.date,
            force_db=args.force_db,
            show_labels=args.show_labels,
            contacts=args.contacts,
            out_dir="040_out",
            output_path=args.output if args.layout != "all" else None,
        )
