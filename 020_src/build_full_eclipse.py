#!/usr/bin/env python3
"""
Automated Pipeline for Solar Eclipse Processing & Master Film Assembly (CoC Architecture)
========================================================================================
Processes and assembles solar eclipse and astronomical footage using Convention over
Configuration (CoC). File names in '010_in/' define sequence ordering, asset types,
and temporal duration parameters.

Supported CoC Filename Conventions in '010_in/':
  - '01_timelapse.mp4':
      Timelapse capture at 1:1 native frame rate, with subpixel solar limb stabilization.
  - '02_video_slowdown_10.mp4' (or '02_video_slowdown.mp4', default 10s):
      Intervalometer / dense burst video with automatic corrupt/black frame filtering,
      temporal resampling to specified seconds, solar limb stabilization, and luminance smoothing.
  - '03_video_realtime.mp4' (or '03_video.mp4'):
      Continuous 1x real-time video (30 fps) with lunar silhouette, corona, and Baily's beads tracking.
  - '04_timelapse.mp4':
      Timelapse capture at 1:1 native frame rate with solar limb stabilization.
  - '05_photo_6.jpg' (or '05_photo.png', default 10s):
      Still photo scaled to fit master project canvas (preserving aspect ratio) for specified duration.
  - '06_composite_sinusoid_10' (or '06_composite_circle_3840x2160_10', default 10s):
      Generates composite mosaic artwork on-the-fly, fits to master project canvas, and inserts for specified duration.

Dynamic Resolution Detection:
  - Master project resolution (W, H) and optical center (W/2, H/2) are automatically extracted
    from the first video/timelapse or photo asset in '010_in/'.

Authors: Fernando (nandoide) & Antigravity (Google Deepmind)
Workspace: eclipse_assembler (branch: multi)
"""

import os
import sys
import re
import argparse
import subprocess
import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d

# Import composite engine for on-the-fly composite asset generation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import create_eclipse_composite as cec


# ─────────────────────────────────────────────────────────────────────────────
# COC PARSER & ASSET DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def parse_coc_filename(filepath):
    """
    Parses CoC metadata from a filename in 010_in/.
    Returns dict:
      {
        'index': int,
        'raw_name': str,
        'path': str,
        'asset_type': 'timelapse' | 'video_slowdown' | 'video_realtime' | 'photo' | 'composite',
        'duration': float (default: 10.0 for photo/composite/slowdown),
        'layout': str (for composite, e.g. 'sinusoid', 'circle', etc.),
        'comp_width': int (for composite, default 3840),
        'comp_height': int (for composite, default 3840),
      }
    or None if non-compliant.
    """
    basename = os.path.basename(filepath)
    name_no_ext, ext = os.path.splitext(basename)
    ext_lower = ext.lower()

    # Pattern: ^(?P<idx>\d+)_(?P<rest>.+)$
    m = re.match(r"^(\d+)_(.+)$", name_no_ext)
    if not m:
        return None

    idx = int(m.group(1))
    tokens = m.group(2).split('_')

    primary_token = tokens[0].lower()

    # Defaults
    asset_type = None
    duration = 10.0
    layout = "sinusoid"
    comp_w = 3840
    comp_h = 3840

    if primary_token == "timelapse":
        asset_type = "timelapse"

    elif primary_token == "video":
        sub_type = tokens[1].lower() if len(tokens) > 1 else "realtime"
        if sub_type.startswith("slowdown"):
            asset_type = "video_slowdown"
            # Parse optional duration e.g. video_slowdown_10
            for tok in tokens[2:]:
                if tok.replace('.', '', 1).isdigit():
                    duration = float(tok)
                    break
        elif sub_type.startswith("realtime"):
            asset_type = "video_realtime"
        else:
            # Default generic 'video' is realtime
            asset_type = "video_realtime"

    elif primary_token == "photo":
        asset_type = "photo"
        for tok in tokens[1:]:
            if tok.replace('.', '', 1).isdigit():
                duration = float(tok)
                break

    elif primary_token == "composite":
        asset_type = "composite"
        # Token format can be: composite_<layout>_<RES>_<duration>
        # e.g., composite_sinusoid_10, composite_circle_3840x2160_10
        for tok in tokens[1:]:
            tok_l = tok.lower()
            if tok_l in ["sinusoid", "circle", "diagonal", "horizontal", "vertical"]:
                layout = tok_l
            elif "x" in tok_l:
                res_parts = tok_l.split("x")
                if len(res_parts) == 2 and res_parts[0].isdigit() and res_parts[1].isdigit():
                    comp_w = int(res_parts[0])
                    comp_h = int(res_parts[1])
            elif tok_l.replace('.', '', 1).isdigit():
                duration = float(tok_l)

    if asset_type is None:
        return None

    return {
        'index': idx,
        'raw_name': basename,
        'path': filepath,
        'asset_type': asset_type,
        'duration': duration,
        'layout': layout,
        'comp_width': comp_w,
        'comp_height': comp_h,
    }


def discover_coc_assets(in_dir="010_in"):
    """
    Scans in_dir for files matching CoC rules, sorting strictly by index ascending.
    """
    if not os.path.exists(in_dir):
        return []

    assets = []
    for fname in os.listdir(in_dir):
        if fname.startswith('.') or fname.startswith('tmp_') or fname.endswith('.tmp'):
            continue
        full_path = os.path.join(in_dir, fname)
        item = parse_coc_filename(full_path)
        if item is not None:
            assets.append(item)

    assets.sort(key=lambda x: x['index'])
    return assets


def detect_project_resolution(assets, fallback=(1280, 720)):
    """
    Extracts native resolution from the first video/timelapse or photo asset.
    """
    for item in assets:
        t = item['asset_type']
        p = item['path']
        if t in ['timelapse', 'video_slowdown', 'video_realtime']:
            cap = cv2.VideoCapture(p)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if w > 0 and h > 0:
                    return w, h
        elif t == 'photo':
            img = cv2.imread(p)
            if img is not None:
                h, w = img.shape[:2]
                return w, h

    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY & CIRCLE FITTING CORE
# ─────────────────────────────────────────────────────────────────────────────

def optimize_circle_center(points, r_fixed, initial_guess=None, num_steps=20):
    """
    Gradient descent to minimize sum((||p_i - (cx,cy)|| - r_fixed)^2).
    Fast subpixel circle fitting for convex solar/lunar limbs.
    """
    pts = points.astype(np.float32)
    if initial_guess is None:
        cx, cy = np.mean(pts, axis=0)
    else:
        cx, cy = float(initial_guess[0]), float(initial_guess[1])

    lr = 0.5
    for _ in range(num_steps):
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        dist = np.sqrt(dx * dx + dy * dy)
        valid = dist > 1e-6
        if not np.any(valid):
            break
        err = dist[valid] - r_fixed
        grad_x = -np.mean(err * (dx[valid] / dist[valid]))
        grad_y = -np.mean(err * (dy[valid] / dist[valid]))
        cx -= lr * grad_x
        cy -= lr * grad_y
        lr *= 0.95
    return cx, cy


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE READERS & HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def read_video_frames_pipe(in_path):
    """
    Reliably extracts all raw BGR frames from any video container via FFmpeg image2pipe.
    Guarantees 100% reliable frame decoding across macOS OpenCV container quirks.
    """
    cmd_probe = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=s=x:p=0',
        in_path
    ]
    res = subprocess.run(cmd_probe, capture_output=True, text=True)
    dims = res.stdout.strip().split('x')
    w, h = int(dims[0]), int(dims[1])

    cmd_stream = [
        'ffmpeg', '-i', in_path,
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo',
        '-'
    ]
    proc = subprocess.Popen(cmd_stream, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_size = w * h * 3
    frames = []
    while True:
        raw_bytes = proc.stdout.read(frame_size)
        if len(raw_bytes) < frame_size:
            break
        frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3))
        frames.append(frame)

    proc.stdout.close()
    proc.wait()
    return frames, w, h


# ─────────────────────────────────────────────────────────────────────────────
# TIMELAPSE ASSET ENGINE (1:1 SPEED, LIMB STABILIZATION)
# ─────────────────────────────────────────────────────────────────────────────

def stabilize_timelapse_asset(
    in_path: str,
    out_path: str,
    master_w: int = 1280,
    master_h: int = 720,
    r_fixed: float = None,
    shift_offset=(12.5, 7.5),
    fps: float = 30.0,
    crf: int = 16,
    preset: str = "fast"
):
    """
    Stabilizes solar limb time-lapse sequence to dynamic center (master_w/2, master_h/2)
    with subpixel precision and color equalization.
    """
    cap = cv2.VideoCapture(in_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    cx_target = master_w / 2.0
    cy_target = master_h / 2.0
    if r_fixed is None:
        r_fixed = min(master_w, master_h) * 0.331  # ~238.5 for 720p

    # Pass 1: Measure raw centers
    cap = cv2.VideoCapture(in_path)
    raw_centers = []
    pbar = tqdm(total=total_frames, desc=f"Stabilizing {os.path.basename(in_path)}")

    last_valid = (in_w / 2.0, in_h / 2.0)
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        max_val = float(np.max(gray))
        th_val = max(20, min(80, int(0.35 * max_val)))
        _, binary = cv2.threshold(gray, th_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            raw_centers.append(last_valid)
            pbar.update(1)
            continue

        cnt = max(contours, key=cv2.contourArea)
        pts = cnt.reshape(-1, 2)
        hull = cv2.convexHull(pts).reshape(-1, 2)

        if len(hull) < 20:
            raw_centers.append(last_valid)
            pbar.update(1)
            continue

        cx_fit, cy_fit = optimize_circle_center(hull, r_fixed)
        cx_corr = cx_fit - shift_offset[0]
        cy_corr = cy_fit - shift_offset[1]
        last_valid = (cx_corr, cy_corr)
        raw_centers.append(last_valid)
        pbar.update(1)

    cap.release()
    pbar.close()

    # Pass 2: Warp and encode
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{master_w}x{master_h}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx265',
        '-crf', str(crf),
        '-preset', preset,
        '-tag:v', 'hvc1',
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-color_range', '1',
        '-colorspace', 'bt709',
        '-color_trc', 'bt709',
        '-color_primaries', 'bt709',
        out_path
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cap = cv2.VideoCapture(in_path)
    for idx in range(len(raw_centers)):
        ret, frame = cap.read()
        if not ret:
            break
        cx_c, cy_c = raw_centers[idx]
        dx = cx_target - cx_c
        dy = cy_target - cy_c
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

        # Equalize solar color
        frame_eq = cec.equalize_solar_color(frame)
        warped = cv2.warpAffine(frame_eq, M, (master_w, master_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        proc.stdin.write(warped.tobytes())

    cap.release()
    proc.stdin.close()
    proc.wait()


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO SLOWDOWN / RESAMPLE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def process_video_slowdown_asset(
    in_path: str,
    out_path: str,
    target_duration_s: float = 10.0,
    master_w: int = 1280,
    master_h: int = 720,
    r_fixed: float = None,
    shift_offset=(12.5, 7.5),
    fps: float = 30.0,
    crf: int = 16,
    preset: str = "fast"
):
    """
    Filters corrupt/black frames, resamples to target_duration_s at fps (30fps),
    equalizes solar color, smooths exposure steps, and stabilizes solar limb.
    """
    cap = cv2.VideoCapture(in_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cx_target = master_w / 2.0
    cy_target = master_h / 2.0
    if r_fixed is None:
        r_fixed = min(master_w, master_h) * 0.331  # ~238.5 for 720p

    # 1. Filter black frames
    valid_frames = []
    cap = cv2.VideoCapture(in_path)
    pbar = tqdm(total=total_frames, desc=f"Scanning valid frames ({os.path.basename(in_path)})")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.max(gray) >= 30:
            valid_frames.append(frame)
    cap.release()
    pbar.close()

    if not valid_frames:
        raise ValueError(f"No valid bright frames found in {in_path}")

    # 2. Resample to target frame count
    out_num_frames = int(round(target_duration_s * fps))
    indices = np.linspace(0, len(valid_frames) - 1, out_num_frames)
    sampled_frames = [valid_frames[int(round(idx))] for idx in indices]

    # 3. Track solar limb
    centers = []
    last_valid = (cx_target, cy_target)
    for frame in tqdm(sampled_frames, desc="Tracking solar limb"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        max_val = float(np.max(gray))
        if max_val < 30:
            centers.append(last_valid)
            continue
        th_val = max(20, min(80, int(0.35 * max_val)))
        _, binary = cv2.threshold(gray, th_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            centers.append(last_valid)
            continue
        cnt = max(contours, key=cv2.contourArea)
        pts = cnt.reshape(-1, 2)
        hull = cv2.convexHull(pts).reshape(-1, 2)
        if len(hull) < 20:
            centers.append(last_valid)
            continue
        cx_fit, cy_fit = optimize_circle_center(hull, r_fixed)
        cx_corr = cx_fit - shift_offset[0]
        cy_corr = cy_fit - shift_offset[1]
        last_valid = (cx_corr, cy_corr)
        centers.append(last_valid)

    # 4. Exposure smoothing
    p90_vals = np.array([np.percentile(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 90) for f in sampled_frames], dtype=np.float32)
    cutoff_fade = max(0, len(sampled_frames) - int(2.0 * fps))
    robust_p90 = p90_vals.copy()
    for idx in range(cutoff_fade):
        local_med = float(np.median(p90_vals[max(0, idx-20) : min(cutoff_fade, idx+21)]))
        if p90_vals[idx] < 0.75 * local_med or p90_vals[idx] > 1.35 * local_med:
            robust_p90[idx] = local_med

    smooth_lum_target = uniform_filter1d(robust_p90, size=15)
    smooth_lum_target[cutoff_fade:] = p90_vals[cutoff_fade:]

    lum_scales = np.ones(len(sampled_frames), dtype=np.float32)
    for idx in range(len(sampled_frames)):
        if p90_vals[idx] > 20 and idx < cutoff_fade:
            lum_scales[idx] = smooth_lum_target[idx] / max(p90_vals[idx], 1.0)

    # 5. Warp and encode
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{master_w}x{master_h}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx265',
        '-crf', str(crf),
        '-preset', preset,
        '-tag:v', 'hvc1',
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-color_range', '1',
        '-colorspace', 'bt709',
        '-color_trc', 'bt709',
        '-color_primaries', 'bt709',
        out_path
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for idx, (frame, (cx, cy)) in enumerate(zip(sampled_frames, centers)):
        corr_frame = frame.astype(np.float32)
        if lum_scales[idx] != 1.0:
            corr_frame *= lum_scales[idx]

        frame_to_warp = np.clip(corr_frame, 0, 255).astype(np.uint8)
        frame_to_warp = cec.equalize_solar_color(frame_to_warp)

        dx = cx_target - cx
        dy = cy_target - cy
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        warped = cv2.warpAffine(frame_to_warp, M, (master_w, master_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        proc.stdin.write(warped.tobytes())

    proc.stdin.close()
    proc.wait()


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO REALTIME ENGINE (TOTALITY & CORONA TRACKING)
# ─────────────────────────────────────────────────────────────────────────────

def process_video_realtime_asset(
    in_path: str,
    out_path: str,
    master_w: int = 1280,
    master_h: int = 720,
    lunar_radius: float = None,
    shift_offset=(12.5, 7.5),
    fps: float = 30.0,
    crf: int = 16,
    preset: str = "fast"
):
    """
    Stabilizes real-time continuous video (totality/corona/chromosphere) at 1x speed.
    """
    cap = cv2.VideoCapture(in_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cx_target = master_w / 2.0
    cy_target = master_h / 2.0
    if lunar_radius is None:
        lunar_radius = min(master_w, master_h) * 0.3416  # ~246.0 for 720p

    # Pass 1: Track centers
    cap = cv2.VideoCapture(in_path)
    centers = []
    last_valid = (cx_target, cy_target)
    pbar = tqdm(total=total_frames, desc=f"Stabilizing Realtime {os.path.basename(in_path)}")

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        max_val = float(np.max(gray))

        if max_val < 30:
            centers.append(last_valid)
            pbar.update(1)
            continue

        th_val = max(15, min(60, int(0.25 * max_val)))
        _, binary = cv2.threshold(gray, th_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            centers.append(last_valid)
            pbar.update(1)
            continue

        cnt = max(contours, key=cv2.contourArea)
        pts = cnt.reshape(-1, 2)
        hull = cv2.convexHull(pts).reshape(-1, 2)

        if len(hull) < 20:
            centers.append(last_valid)
            pbar.update(1)
            continue

        cx_fit, cy_fit = optimize_circle_center(hull, lunar_radius)
        cx_corr = cx_fit - shift_offset[0]
        cy_corr = cy_fit - shift_offset[1]
        last_valid = (cx_corr, cy_corr)
        centers.append(last_valid)
        pbar.update(1)

    cap.release()
    pbar.close()

    # Pass 2: Warp and encode
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{master_w}x{master_h}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx265',
        '-crf', str(crf),
        '-preset', preset,
        '-tag:v', 'hvc1',
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-color_range', '1',
        '-colorspace', 'bt709',
        '-color_trc', 'bt709',
        '-color_primaries', 'bt709',
        out_path
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cap = cv2.VideoCapture(in_path)
    for idx in range(len(centers)):
        ret, frame = cap.read()
        if not ret:
            break
        cx_c, cy_c = centers[idx]
        dx = cx_target - cx_c
        dy = cy_target - cy_c
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        warped = cv2.warpAffine(frame, M, (master_w, master_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        proc.stdin.write(warped.tobytes())

    cap.release()
    proc.stdin.close()
    proc.wait()


# ─────────────────────────────────────────────────────────────────────────────
# STILL PHOTO & COMPOSITE ARTWORK ENGINES (NO OPTICAL TRACKING)
# ─────────────────────────────────────────────────────────────────────────────

def scale_and_pad_to_canvas(image, target_w, target_h):
    """
    Scales image preserving aspect ratio and places it on a (target_w, target_h)
    canvas with clean black letterbox/pillarbox padding. No optical tracking.
    """
    ih, iw = image.shape[:2]
    scale = min(target_w / float(iw), target_h / float(ih))
    new_w = int(round(iw * scale))
    new_h = int(round(ih * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


def process_photo_asset(
    in_path: str,
    out_path: str,
    duration_s: float = 10.0,
    master_w: int = 1280,
    master_h: int = 720,
    fps: float = 30.0,
    crf: int = 16,
    preset: str = "fast"
):
    """
    Creates a static video clip of a photo padded with black borders to project resolution.
    No optical tracking applied.
    """
    img = cv2.imread(in_path)
    if img is None:
        raise FileNotFoundError(f"Could not read photo asset at {in_path}")

    padded_frame = scale_and_pad_to_canvas(img, master_w, master_h)
    total_frames = int(round(duration_s * fps))

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{master_w}x{master_h}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx265',
        '-crf', str(crf),
        '-preset', preset,
        '-tag:v', 'hvc1',
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-color_range', '1',
        '-colorspace', 'bt709',
        '-color_trc', 'bt709',
        '-color_primaries', 'bt709',
        out_path
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frame_bytes = padded_frame.tobytes()

    for _ in range(total_frames):
        proc.stdin.write(frame_bytes)

    proc.stdin.close()
    proc.wait()


def process_composite_asset(
    asset_meta: dict,
    out_path: str,
    out_dir: str = "040_out",
    master_w: int = 1280,
    master_h: int = 720,
    fps: float = 30.0,
    crf: int = 16,
    preset: str = "fast"
):
    """
    Generates high-resolution composite artwork on-the-fly via create_eclipse_composite,
    scales and pads it to master project canvas, and produces a video clip of duration_s.
    """
    layout = asset_meta.get('layout', 'sinusoid')
    comp_w = asset_meta.get('comp_width', 3840)
    comp_h = asset_meta.get('comp_height', 3840)
    duration_s = asset_meta.get('duration', 10.0)
    idx = asset_meta.get('index', 6)

    # 1. Sample keyframes from sequence
    sampled_keyframes = cec.sample_eclipse_sequence()
    if not sampled_keyframes:
        raise RuntimeError("Failed to sample keyframes for composite generation.")

    # 2. Render composite artwork image
    composite_img = cec.render_eclipse_composite(
        sampled_keyframes=sampled_keyframes,
        layout_mode=layout,
        out_w=comp_w,
        out_h=comp_h
    )

    # 3. Save high-res composite image artifacts
    png_path = os.path.join(out_dir, f"composite_artwork_{idx}_{layout}.png")
    jpg_path = os.path.join(out_dir, f"composite_artwork_{idx}_{layout}.jpg")
    cv2.imwrite(png_path, composite_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cv2.imwrite(jpg_path, composite_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    print(f"=================================================================")
    print(f"SUCCESS: Composite generated!")
    print(f"  PNG (Lossless) : {png_path} ({os.path.getsize(png_path)/(1024*1024):.2f} MB)")
    print(f"  JPG (High-Q)   : {jpg_path} ({os.path.getsize(jpg_path)/(1024*1024):.2f} MB)")
    print(f"=================================================================")

    # 4. Scale and pad to master project resolution
    canvas_frame = scale_and_pad_to_canvas(composite_img, master_w, master_h)
    total_frames = int(round(duration_s * fps))

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{master_w}x{master_h}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx265',
        '-crf', str(crf),
        '-preset', preset,
        '-tag:v', 'hvc1',
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-color_range', '1',
        '-colorspace', 'bt709',
        '-color_trc', 'bt709',
        '-color_primaries', 'bt709',
        out_path
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frame_bytes = canvas_frame.tobytes()

    for _ in range(total_frames):
        proc.stdin.write(frame_bytes)

    proc.stdin.close()
    proc.wait()


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FILM MULTI-ASSET ASSEMBLY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def assemble_master_film(
    clip_paths: list,
    output_path: str,
    master_w: int = 1280,
    master_h: int = 720,
    transition_type: str = "fade_to_black",
    freeze_before: float = 1.0,
    fade_out: float = 0.5,
    black_duration: float = 0.0,
    fade_in: float = 0.5,
    fade_duration: float = 1.0,
    freeze_after: float = 1.0,
    fps: float = 30.0,
    crf: int = 16,
    preset: str = "fast"
):
    """
    Dynamically stitches N clips in sequential order with uniform transitions:
      - 'fade_to_black': freeze_before -> fade_out -> black_duration -> fade_in -> freeze_after
      - 'crossfade': linear dissolve between clips over fade_duration
      - 'hard': immediate cut between clips
    """
    print("=================================================================")
    print(f"ASSEMBLING MASTER FILM: {output_path}")
    print(f"  Resolution : {master_w}x{master_h} @ {fps:.2f} fps")
    print(f"  Clips count: {len(clip_paths)}")
    print(f"  Transition : '{transition_type}'")
    print("=================================================================")

    if len(clip_paths) == 0:
        raise ValueError("No clips to assemble.")

    if len(clip_paths) == 1:
        # Single clip mode: copy/re-encode directly without transitions
        single_clip = clip_paths[0]
        if os.path.abspath(single_clip) != os.path.abspath(output_path):
            subprocess.run(['cp', single_clip, output_path], check=True)
        print(f"Single-clip master film exported directly: {output_path}")
        return

    # Load all clips into memory using robust rawvideo pipe
    all_clips_frames = []
    for idx, cpath in enumerate(clip_paths):
        print(f"  • Reading clip {idx+1}/{len(clip_paths)}: {os.path.basename(cpath)}")
        frames, w, h = read_video_frames_pipe(cpath)
        all_clips_frames.append(frames)

    # Build master frame sequence
    master_frames = []
    n_freeze_b = int(round(freeze_before * fps))
    n_fade_o = int(round(fade_out * fps))
    n_black = int(round(black_duration * fps))
    n_fade_i = int(round(fade_in * fps))
    n_freeze_a = int(round(freeze_after * fps))
    n_crossfade = int(round(fade_duration * fps))

    for idx, clip in enumerate(all_clips_frames):
        # Append main clip body
        master_frames.extend(clip)

        # Apply transition if not the last clip
        if idx < len(all_clips_frames) - 1:
            last_frame = clip[-1]
            next_first = all_clips_frames[idx + 1][0]

            if transition_type == "fade_to_black":
                # 1. Freeze on last frame
                for _ in range(n_freeze_b):
                    master_frames.append(last_frame)

                # 2. Fade out to black
                for i in range(1, n_fade_o + 1):
                    alpha = 1.0 - (i / float(n_fade_o))
                    faded = np.clip(last_frame.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
                    master_frames.append(faded)

                # 3. Pure black pause
                if n_black > 0:
                    black_frame = np.zeros((master_h, master_w, 3), dtype=np.uint8)
                    for _ in range(n_black):
                        master_frames.append(black_frame)

                # 4. Fade in from black to next clip's first frame
                for i in range(1, n_fade_i + 1):
                    alpha = i / float(n_fade_i)
                    faded = np.clip(next_first.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
                    master_frames.append(faded)

                # 5. Freeze on next clip's first frame
                for _ in range(n_freeze_a):
                    master_frames.append(next_first)

            elif transition_type == "crossfade":
                # Crossfade dissolve
                for i in range(1, n_crossfade + 1):
                    alpha = i / float(n_crossfade)
                    blended = cv2.addWeighted(last_frame, 1.0 - alpha, next_first, alpha, 0.0)
                    master_frames.append(blended)

            elif transition_type == "hard":
                pass  # Direct cut

    # Encode master video
    total_master_frames = len(master_frames)
    duration_total_s = total_master_frames / fps
    print(f"\nEncoding Final Master Film ({total_master_frames} frames, {duration_total_s:.2f}s)...")

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{master_w}x{master_h}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx265',
        '-crf', str(crf),
        '-preset', preset,
        '-tag:v', 'hvc1',
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-color_range', '1',
        '-colorspace', 'bt709',
        '-color_trc', 'bt709',
        '-color_primaries', 'bt709',
        output_path
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for frame in tqdm(master_frames, desc="Writing Master Film"):
        proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

    sz_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("=================================================================")
    print(f"SUCCESS: Master film generated at: {output_path}")
    print(f"  Resolution: {master_w}x{master_h} @ {fps:.2f} fps")
    print(f"  Total frames: {total_master_frames} ({duration_total_s:.2f}s)")
    print(f"  File size: {sz_mb:.2f} MB")
    print("=================================================================")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

def run_coc_pipeline(
    in_dir: str = "010_in",
    out_dir: str = "040_out",
    output_film: str = "040_out/full_eclipse.mp4",
    transition_type: str = "fade_to_black",
    freeze_before: float = 1.0,
    fade_out: float = 0.5,
    black_duration: float = 0.0,
    fade_in: float = 0.5,
    fade_duration: float = 1.0,
    freeze_after: float = 1.0,
    force_all: bool = False
):
    """
    Executes the end-to-end CoC Pipeline.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Discover CoC assets
    assets = discover_coc_assets(in_dir)
    if not assets:
        print(f"ERROR: No valid CoC assets found in '{in_dir}'.")
        print("Expected formats e.g.: 01_timelapse.mp4, 02_video_slowdown_10.mp4, 03_video_realtime.mp4, 05_photo_6.jpg, 06_composite_sinusoid_10")
        sys.exit(1)

    # 2. Auto-detect project master resolution
    master_w, master_h = detect_project_resolution(assets, fallback=(1280, 720))
    cx_opt = master_w / 2.0
    cy_opt = master_h / 2.0

    print("=================================================================")
    print("SOLAR ECLIPSE PIPELINE (Convention-Over-Configuration)")
    print("=================================================================")
    print(f"  Input Directory    : {in_dir}")
    print(f"  Master Resolution  : {master_w}x{master_h} px (Auto-detected)")
    print(f"  Optical Center     : ({cx_opt:.1f}, {cy_opt:.1f})")
    print(f"  Discovered Assets  : {len(assets)} items")
    for a in assets:
        print(f"    [{a['index']:02d}] {a['raw_name']:30s} -> Type: {a['asset_type']} (Duration: {a['duration']}s)")
    print("=================================================================\n")

    # 3. Process each asset individually
    processed_clip_paths = []

    for a in assets:
        idx = a['index']
        a_type = a['asset_type']
        in_path = a['path']
        raw_name = a['raw_name']
        name_no_ext, _ = os.path.splitext(raw_name)
        out_clip_path = os.path.join(out_dir, f"{name_no_ext}.mp4")

        print(f"Processing Asset [{idx:02d}]: {raw_name} ({a_type.upper()})...")

        if os.path.exists(out_clip_path) and not force_all:
            print(f"  • Asset output exists ({out_clip_path}), skipping (use --force-all to rebuild).")
            processed_clip_paths.append(out_clip_path)
            continue

        if a_type == "timelapse":
            stabilize_timelapse_asset(
                in_path=in_path,
                out_path=out_clip_path,
                master_w=master_w,
                master_h=master_h,
                fps=30.0, crf=16, preset="fast"
            )
        elif a_type == "video_slowdown":
            process_video_slowdown_asset(
                in_path=in_path,
                out_path=out_clip_path,
                target_duration_s=a['duration'],
                master_w=master_w,
                master_h=master_h,
                fps=30.0, crf=16, preset="fast"
            )
        elif a_type == "video_realtime":
            process_video_realtime_asset(
                in_path=in_path,
                out_path=out_clip_path,
                master_w=master_w,
                master_h=master_h,
                fps=30.0, crf=16, preset="fast"
            )
        elif a_type == "photo":
            print(f"Generating Still Photo Video: {raw_name} -> {out_clip_path} ({a['duration']}s)")
            process_photo_asset(
                in_path=in_path,
                out_path=out_clip_path,
                duration_s=a['duration'],
                master_w=master_w,
                master_h=master_h,
                fps=30.0, crf=16, preset="fast"
            )
        elif a_type == "composite":
            print(f"Generating On-The-Fly Composite Artwork: Layout={a['layout'].upper()} ({a['comp_width']}x{a['comp_height']}) -> {out_clip_path} ({a['duration']}s)")
            process_composite_asset(
                asset_meta=a,
                out_path=out_clip_path,
                out_dir=out_dir,
                master_w=master_w,
                master_h=master_h,
                fps=30.0, crf=16, preset="fast"
            )

        processed_clip_paths.append(out_clip_path)
        print()

    # 4. Assemble master film
    assemble_master_film(
        clip_paths=processed_clip_paths,
        output_path=output_film,
        master_w=master_w,
        master_h=master_h,
        transition_type=transition_type,
        freeze_before=freeze_before,
        fade_out=fade_out,
        black_duration=black_duration,
        fade_in=fade_in,
        fade_duration=fade_duration,
        freeze_after=freeze_after,
        fps=30.0, crf=16, preset="fast"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoC Automated Solar Eclipse Processing & Master Film Assembly Pipeline")
    parser.add_argument("--in-dir", "-i", type=str, default="010_in",
                        help="Input directory with CoC named assets (default: 010_in)")
    parser.add_argument("--out-dir", type=str, default="040_out",
                        help="Intermediate and master output directory (default: 040_out)")
    parser.add_argument("--output", "-o", type=str, default="040_out/full_eclipse.mp4",
                        help="Path to final master film (default: 040_out/full_eclipse.mp4)")
    parser.add_argument("--transition-type", type=str, choices=["fade_to_black", "hard", "crossfade"], default="fade_to_black",
                        help="Transition style: 'fade_to_black', 'hard', or 'crossfade'. Default: fade_to_black")
    parser.add_argument("--freeze-before", type=float, default=1.0,
                        help="Duration in seconds of freeze on the last frame of preceding clip (default: 1.0s)")
    parser.add_argument("--fade-out", type=float, default=0.5,
                        help="Duration in seconds of fade out to black (default: 0.5s)")
    parser.add_argument("--black-duration", type=float, default=0.0,
                        help="Duration in seconds of pure black pause (default: 0.0s)")
    parser.add_argument("--fade-in", type=float, default=0.5,
                        help="Duration in seconds of fade in from black (default: 0.5s)")
    parser.add_argument("--fade-duration", type=float, default=1.0,
                        help="Duration in seconds of crossfade dissolve (default: 1.0s)")
    parser.add_argument("--freeze-after", type=float, default=1.0,
                        help="Duration in seconds of freeze on the first frame of subsequent clip (default: 1.0s)")
    parser.add_argument("--force-all", action="store_true",
                        help="Force re-processing and re-stabilizing all assets from scratch")
    args = parser.parse_args()

    run_coc_pipeline(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        output_film=args.output,
        transition_type=args.transition_type,
        freeze_before=args.freeze_before,
        fade_out=args.fade_out,
        black_duration=args.black_duration,
        fade_in=args.fade_in,
        fade_duration=args.fade_duration,
        freeze_after=args.freeze_after,
        force_all=args.force_all
    )
