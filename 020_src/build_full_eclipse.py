#!/usr/bin/env python3
"""
Automated Pipeline for Solar Eclipse Processing & Master Film Assembly (CoC Architecture)
========================================================================================
Processes and assembles solar eclipse and astronomical footage using Convention over
Configuration (CoC). File names in '010_in/' define sequence ordering, asset types,
and temporal duration parameters.

Supported CoC Filename Conventions in '010_in/':
  - '01_timelapse.mp4':
      Timelapse capture at 1:1 native frame rate, with subpixel outer convex limb
      segmentation, Nelder-Mead Huber loss circle fitting, and chromaticity normalization.
  - '02_video_slowdown_10.mp4' (or '02_video_slowdown.mp4', default 10s):
      Intervalometer / dense burst video with automatic corrupt/black frame filtering,
      temporal resampling to specified seconds, subpixel solar limb Nelder-Mead stabilization,
      dynamic white balance equalization, and 90th-percentile temporal luminance smoothing.
  - '03_video_realtime.mp4' (or '03_video.mp4'):
      Continuous 1x real-time video (30 fps) with Canny-edge lunar silhouette tracking,
      annular distance filtering, and C2 transition boundary inheritance (+12.5 px, +7.5 px).
  - '04_timelapse.mp4':
      Egress timelapse capture at 1:1 native frame rate with subpixel outer limb stabilization
      and C3 transition boundary inheritance (+35.0 px, +24.0 px).
  - '05_photo_6.jpg' (or '05_photo.png', default 10s):
      Still photo scaled to fit master project canvas (preserving aspect ratio with black letterbox)
      for specified duration. No optical limb tracking.
  - '06_composite_sinusoid_10' (or '06_composite_circle_3840x2160_10', default 10s):
      Generates composite mosaic artwork on-the-fly, fits to master project canvas, and inserts
      for specified duration. No optical limb tracking.

Authors: Fernando (nandoide) & Antigravity (Google Deepmind)
Workspace: eclipse_assembler (branch: multi)
"""

import os
import sys
import re
import json
import argparse
import subprocess
import cv2
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter1d

# Import modular external engines
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import create_title_card as ctc
import create_end_titles as cet
import create_eclipse_composite as cec
import generate_eclipse_subtitles as ges
import eclipse_ephemeris_db as eedb
import create_camera_dive as ccd
import add_audio_track as aat
from stabilize_eclipse import extract_pure_solar_limb, find_circle_center_fixed_r



# ─────────────────────────────────────────────────────────────────────────────
# COC PARSER & ASSET DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def parse_coc_filename(filepath, is_project_preprocessed=False):
    """
    Parses CoC metadata from a filename in 010_in/.
    """
    basename = os.path.basename(filepath)
    name_no_ext, ext = os.path.splitext(basename)

    m = re.match(r"^(\d+)_(.+)$", name_no_ext)
    if not m:
        return None

    idx = int(m.group(1))
    tokens = m.group(2).split('_')

    primary_token = tokens[0].lower()

    asset_type = None
    duration = 10.0
    layout = "sinusoid"
    comp_w = None
    comp_h = None
    music_title = None
    music_author = None

    interval = None
    is_preprocessed = is_project_preprocessed
    if primary_token == "timelapse":
        asset_type = "timelapse"
        interval = 10.0
        for tok in tokens[1:]:
            tok_l = tok.lower()
            if tok_l in ["prep", "preprocessed", "restored"]:
                is_preprocessed = True
            elif tok_l.startswith("i") and tok_l[1:].replace('.', '', 1).isdigit():
                interval = float(tok_l[1:])
            elif tok_l.endswith("s") and tok_l[:-1].replace('.', '', 1).isdigit():
                interval = float(tok_l[:-1])
            elif tok_l.replace('.', '', 1).isdigit():
                interval = float(tok_l)

    elif primary_token == "video":
        sub_type = tokens[1].lower() if len(tokens) > 1 else "realtime"
        if sub_type.startswith("slowdown"):
            asset_type = "video_slowdown"
            for tok in tokens[2:]:
                if tok.replace('.', '', 1).isdigit():
                    duration = float(tok)
                    break
        elif sub_type.startswith("realtime"):
            asset_type = "video_realtime"
        else:
            asset_type = "video_realtime"

    elif primary_token in ["photo", "totality", "hdr"]:
        asset_type = "photo"
        for tok in tokens[1:]:
            if tok.replace('.', '', 1).isdigit():
                duration = float(tok)
                break

    elif primary_token == "composite":
        asset_type = "composite"
        for tok in tokens[1:]:
            tok_l = tok.lower()
            if tok_l in ["sinusoid", "circle", "diagonal", "horizontal", "vertical", "arc", "ellipse", "vertical-s", "spiral"]:
                layout = tok_l
            elif tok_l in ["4k", "uhd"]:
                comp_w = 3840
                comp_h = 2160
            elif tok_l in ["square", "1:1"]:
                comp_w = 3840
                comp_h = 3840
            elif "x" in tok_l:
                res_parts = tok_l.split("x")
                if len(res_parts) == 2 and res_parts[0].isdigit() and res_parts[1].isdigit():
                    comp_w = int(res_parts[0])
                    comp_h = int(res_parts[1])
            elif tok_l.replace('.', '', 1).isdigit():
                duration = float(tok_l)

    elif primary_token in ["endtitles", "endtitle", "credits", "end"]:
        asset_type = "endtitles"
        duration = 6.0
        for tok in tokens[1:]:
            if tok.replace('.', '', 1).isdigit():
                duration = float(tok)
                break

    elif primary_token in ["music", "audio", "soundtrack", "song"]:
        asset_type = "music"
        # Format: [INDEX]_music_[TITLE]_[AUTHOR].[ext]
        # Example: 01_music_corrubedo_nandoide.wav
        music_title = tokens[1].replace('-', ' ').replace('_', ' ').title() if len(tokens) > 1 else "Original Score"
        music_author = tokens[2].replace('-', ' ').replace('_', ' ').title() if len(tokens) > 2 else "Unknown"

    if asset_type is None:
        return None

    return {
        'index': idx,
        'raw_name': basename,
        'path': filepath,
        'asset_type': asset_type,
        'is_preprocessed': is_preprocessed,
        'duration': duration,
        'interval': interval,
        'layout': layout,
        'comp_width': comp_w,
        'comp_height': comp_h,
        'music_title': music_title,
        'music_author': music_author,
    }


def discover_coc_assets(in_dir="010_in", is_project_preprocessed=False):
    """
    Scans in_dir for files matching CoC rules, separating visual sequence assets
    from audio tracks. Returns (visual_assets, music_asset).
    """
    if not os.path.exists(in_dir):
        return [], None

    raw_visual_map = {}
    music_asset = None

    for fname in sorted(os.listdir(in_dir)):
        if fname.startswith('.') or fname.startswith('tmp_') or fname.endswith('.tmp'):
            continue
        full_path = os.path.join(in_dir, fname)
        if os.path.isdir(full_path):
            continue
        item = parse_coc_filename(full_path, is_project_preprocessed=is_project_preprocessed)
        if item is not None:
            if item['asset_type'] == 'music':
                if music_asset is None:
                    music_asset = item
            else:
                idx = item['index']
                if idx in raw_visual_map:
                    existing = raw_visual_map[idx]
                    sz_new = os.path.getsize(full_path) if os.path.exists(full_path) else 0
                    sz_old = os.path.getsize(existing['path']) if os.path.exists(existing['path']) else 0
                    if is_project_preprocessed:
                        if item['is_preprocessed'] and not existing['is_preprocessed']:
                            raw_visual_map[idx] = item
                        elif item['is_preprocessed'] == existing['is_preprocessed'] and sz_new > sz_old:
                            raw_visual_map[idx] = item
                    else:
                        if sz_new > 0 and sz_old == 0:
                            raw_visual_map[idx] = item
                        elif not item['is_preprocessed'] and existing['is_preprocessed']:
                            raw_visual_map[idx] = item
                else:
                    raw_visual_map[idx] = item

    visual_assets = sorted(raw_visual_map.values(), key=lambda x: x['index'])
    return visual_assets, music_asset



def resolve_preprocessed_timelapse(
    asset_item,
    repo_root=None,
    raw_dir=None,
    prep_dir=None,
    force=False,
    extrapolate_c1=True,
    extrapolate_c2=True,
    extrapolate_c3=True,
    extrapolate_c4=True
):
    """
    Resolves preprocessed restored timelapses from 005_raw_preprocessed/<observation>/.
    If missing or force is True, automatically runs preprocess_raw_eclipse_timelapses.py.
    """
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if raw_dir is None:
        raw_obs = os.path.join(repo_root, "000_raw", "solar26")
        raw_dir = raw_obs if os.path.exists(raw_obs) else os.path.join(repo_root, "000_raw")
    if prep_dir is None:
        prep_obs = os.path.join(repo_root, "005_raw_preprocessed", "solar26")
        prep_dir = prep_obs if os.path.exists(prep_obs) else os.path.join(repo_root, "005_raw_preprocessed")
    os.makedirs(prep_dir, exist_ok=True)

    raw_tls = sorted([f for f in os.listdir(raw_dir) if f.endswith(".mp4") and "_TL_" in f])
    if not raw_tls:
        raise FileNotFoundError(f"No raw timelapse files (*_TL_*.mp4) found in {raw_dir}")

    prep_tls = sorted([f for f in os.listdir(prep_dir) if f.endswith(".mp4") and "_TL_" in f])
    idx = asset_item['index']

    if len(prep_tls) < 2 or force:
        print(f"\n[CoC Preprocessing] Preprocessed videos missing or rebuild forced in {prep_dir}")
        print("Launching Autonomous Raw Eclipse Preprocessing Pipeline...")
        from preprocess_raw_eclipse_timelapses import run_master_preprocessing_pipeline
        run_master_preprocessing_pipeline(
            raw_dir=raw_dir,
            preprocessed_dir=prep_dir,
            out_dir=os.path.join(repo_root, "040_out"),
            ingress_filename=raw_tls[0],
            egress_filename=raw_tls[1] if len(raw_tls) > 1 else raw_tls[0],
            extrapolate_c1=extrapolate_c1,
            extrapolate_c2=extrapolate_c2,
            extrapolate_c3=extrapolate_c3,
            extrapolate_c4=extrapolate_c4
        )
        prep_tls = sorted([f for f in os.listdir(prep_dir) if f.endswith(".mp4") and "_TL_" in f])

    if not prep_tls:
        raise FileNotFoundError(f"Failed to find any preprocessed timelapses in {prep_dir}")

    chosen_file = prep_tls[0] if idx <= 2 else prep_tls[-1]
    prep_video_path = os.path.join(prep_dir, chosen_file)

    if not os.path.exists(prep_video_path):
        raise FileNotFoundError(f"Failed to find preprocessed video at: {prep_video_path}")

    return prep_video_path


def detect_project_resolution(assets, repo_root=None, fallback=(1280, 720)):
    """
    Extracts native resolution from the first video/timelapse or photo asset.
    """
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for item in assets:
        t = item['asset_type']
        p = item['path']
        if item.get('is_preprocessed'):
            try:
                p = resolve_preprocessed_timelapse(item, repo_root=repo_root, force=False)
            except Exception:
                p = item['path']

        if t in ['timelapse', 'video_slowdown', 'video_realtime']:
            if os.path.exists(p) and not os.path.isdir(p):
                cap = cv2.VideoCapture(p)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    if w > 0 and h > 0:
                        return w, h
        elif t == 'photo':
            if os.path.exists(p) and not os.path.isdir(p):
                img = cv2.imread(p)
                if img is not None:
                    h, w = img.shape[:2]
                    return w, h

    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO FRAME PIPE READER
# ─────────────────────────────────────────────────────────────────────────────

def read_video_frames_pipe(in_path):
    """
    Reliably extracts all raw BGR frames from any video container via FFmpeg image2pipe.
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


def stream_video_frames_pipe(in_path):
    """
    Memory-efficient generator that yields raw BGR frames sequentially from any video container.
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
    while True:
        raw_bytes = proc.stdout.read(frame_size)
        if len(raw_bytes) < frame_size:
            break
        yield np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3)), w, h

    proc.stdout.close()
    proc.wait()


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-PRECISION SOLAR LIMB STABILIZATION (TIMELAPSE)
# ─────────────────────────────────────────────────────────────────────────────

def stabilize_timelapse_asset(
    in_path: str,
    out_path: str,
    master_w: int = 1280,
    master_h: int = 720,
    r_fixed: float = None,
    shift_offset=(0.0, 0.0),
    fps: float = 30.0,
    crf: int = 16,
    preset: str = "fast"
):
    """
    Stabilizes solar limb time-lapse sequence to dynamic center (master_w/2, master_h/2)
    using exact outer convex limb filtering and Nelder-Mead Huber loss circle fitting.
    """
    cap = cv2.VideoCapture(in_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cx_target = master_w / 2.0
    cy_target = master_h / 2.0
    scale_factor = master_h / 720.0
    if r_fixed is None:
        r_fixed = 238.0 * scale_factor

    # Pass 1: Extract pure solar limb & find exact oblate center with trimmed inlier optimization
    cap = cv2.VideoCapture(in_path)
    frames = []
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    N = len(frames)
    centers = []
    last_valid = (cx_target, cy_target)

    for idx, frame in enumerate(tqdm(frames, desc=f"Stabilizing {os.path.basename(in_path)}")):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        max_val = float(np.max(gray))
        if max_val < 20:
            centers.append(last_valid)
            continue

        # Multi-threshold ensemble to capture true outer limb across cloud transits
        all_pts = []
        for pct in [0.20, 0.35, 0.50]:
            th_val = max(15, min(90, int(pct * max_val)))
            _, binary = cv2.threshold(gray, th_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
                hull = cv2.convexHull(cnt, returnPoints=True).reshape(-1, 2)
                tree = cv2.BFMatcher(cv2.NORM_L2)
                matches = tree.match(hull.astype(np.float32), cnt.astype(np.float32))
                pure = np.array([hull[m.queryIdx] for m in matches if m.distance < 1.5], dtype=np.float32)
                if len(pure) >= 5:
                    all_pts.append(pure)

        if not all_pts:
            centers.append(last_valid)
            continue

        pts = np.vstack(all_pts)
        progress = idx / float(max(1, N - 1))
        # Atmospheric refraction compresses vertical semi-axis progressively near horizon
        ry_target = r_fixed * (1.0 - 0.042 * progress)

        curr_pts = pts
        curr_c = last_valid
        for iter_step in range(3):
            def loss(c):
                cx, cy = c
                d = np.sqrt(((curr_pts[:, 0] - cx) / r_fixed)**2 + ((curr_pts[:, 1] - cy) / ry_target)**2)
                err = np.abs(d - 1.0) * r_fixed
                return np.sum(np.where(err < 2.0, 0.5 * err**2, 2.0 * (err - 1.0)))

            res = minimize(loss, [curr_c[0], curr_c[1]], method='Nelder-Mead', options={'xatol': 0.0005, 'fatol': 0.005})
            if res.success:
                curr_c = (float(res.x[0]), float(res.x[1]))
                d = np.sqrt(((curr_pts[:, 0] - curr_c[0]) / r_fixed)**2 + ((curr_pts[:, 1] - curr_c[1]) / ry_target)**2)
                err = np.abs(d - 1.0) * r_fixed
                inliers = err < 2.5
                if np.sum(inliers) >= 15:
                    curr_pts = curr_pts[inliers]

        last_valid = curr_c
        centers.append(last_valid)

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

    off_x, off_y = shift_offset
    for frame, (cx_c, cy_c) in zip(frames, centers):
        dx = cx_target - cx_c + (off_x * scale_factor)
        dy = cy_target - cy_c + (off_y * scale_factor)
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

        frame_eq = cec.equalize_solar_color(frame)
        warped = cv2.warpAffine(frame_eq, M, (master_w, master_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        proc.stdin.write(warped.tobytes())

    proc.stdin.close()
    proc.wait()


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-PRECISION PRE-TOTALITY ENGINE (VIDEO SLOWDOWN)
# ─────────────────────────────────────────────────────────────────────────────

def process_video_slowdown_asset(
    in_path: str,
    out_path: str,
    target_duration_s: float = 10.0,
    master_w: int = 1280,
    master_h: int = 720,
    solar_radius: float = None,
    fps: float = 30.0,
    crf: int = 16,
    preset: str = "fast"
):
    """
    Filters corrupt/black intervalometer frames, resamples to target_duration_s,
    stabilizes solar limb via Nelder-Mead Huber circle optimization, equalizes
    camera white balance drift, and applies temporal luminance continuity smoothing.
    """
    cap = cv2.VideoCapture(in_path)
    total_in_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cx_target = master_w / 2.0
    cy_target = master_h / 2.0
    scale_factor = master_h / 720.0
    if solar_radius is None:
        solar_radius = 238.0 * scale_factor

    # 1. Scan valid frames (discarding black frames)
    valid_indices = []
    idx = 0
    cap = cv2.VideoCapture(in_path)
    pbar = tqdm(total=total_in_frames, desc=f"1/3 Scanning valid frames ({os.path.basename(in_path)})")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if np.max(frame) >= 25:
            valid_indices.append(idx)
        idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()

    print(f"Valid bright frames detected: {len(valid_indices)} of {total_in_frames} (Discarded {total_in_frames - len(valid_indices)} black frames).")

    if not valid_indices:
        raise ValueError(f"No valid bright frames found in {in_path}")

    # 2. Extract selected sampled frames
    target_frames = int(round(target_duration_s * fps))
    sample_pos = np.linspace(0, len(valid_indices) - 1, target_frames)
    sampled_raw_indices = [valid_indices[int(round(p))] for p in sample_pos]
    sampled_set = set(sampled_raw_indices)

    cap = cv2.VideoCapture(in_path)
    sampled_frames_dict = {}
    curr = 0
    pbar = tqdm(total=sampled_raw_indices[-1] + 1, desc="2/3 Sampling clean frames")
    while curr <= sampled_raw_indices[-1]:
        if curr in sampled_set:
            ret, frame = cap.read()
            if not ret:
                break
            sampled_frames_dict[curr] = frame
        else:
            ret = cap.grab()
            if not ret:
                break
        curr += 1
        pbar.update(1)
    cap.release()
    pbar.close()

    sampled_frames = [sampled_frames_dict[i] for i in sampled_raw_indices]

    # 3. Track solar limb center with Iterative Trimmed Inlier Estimator (Fixed Oblate Geometry)
    rx_fixed = solar_radius
    centers = []
    last_valid_center = (cx_target, cy_target)
    n_frames = len(sampled_frames)

    for idx, frame in enumerate(tqdm(sampled_frames, desc="3/3 Tracking solar limb")):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        max_val = float(np.max(gray))
        if max_val < 20:
            centers.append(last_valid_center)
            continue

        # Multi-threshold ensemble to capture true outer limb across cloud transits
        all_limb_pts = []
        for pct in [0.20, 0.35, 0.50]:
            th_val = max(15, min(90, int(pct * max_val)))
            _, binary = cv2.threshold(gray, th_val, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
                hull = cv2.convexHull(cnt, returnPoints=True).reshape(-1, 2)
                tree = cv2.BFMatcher(cv2.NORM_L2)
                matches = tree.match(hull.astype(np.float32), cnt.astype(np.float32))
                pure = np.array([hull[m.queryIdx] for m in matches if m.distance < 1.5], dtype=np.float32)
                if len(pure) >= 5:
                    all_limb_pts.append(pure)

        if not all_limb_pts:
            centers.append(last_valid_center)
            continue

        pts = np.vstack(all_limb_pts)
        progress = idx / float(max(1, n_frames - 1))
        # Atmospheric refraction compresses the vertical semi-axis progressively near the horizon
        ry_target = rx_fixed * (1.0 - 0.042 * progress)

        # 3-iteration trimmed inlier optimization: ONLY (cx, cy)
        curr_pts = pts
        curr_c = last_valid_center
        for iter_step in range(3):
            def loss(c):
                cx, cy = c
                d = np.sqrt(((curr_pts[:, 0] - cx) / rx_fixed)**2 + ((curr_pts[:, 1] - cy) / ry_target)**2)
                err = np.abs(d - 1.0) * rx_fixed
                return np.sum(np.where(err < 2.0, 0.5 * err**2, 2.0 * (err - 1.0)))

            res = minimize(loss, [curr_c[0], curr_c[1]], method='Nelder-Mead', options={'xatol': 0.0005, 'fatol': 0.005})
            if res.success:
                curr_c = (float(res.x[0]), float(res.x[1]))
                # Trim outlier points (cloud artifacts / seeing distortion / chord points)
                d = np.sqrt(((curr_pts[:, 0] - curr_c[0]) / rx_fixed)**2 + ((curr_pts[:, 1] - curr_c[1]) / ry_target)**2)
                err = np.abs(d - 1.0) * rx_fixed
                inliers = err < 2.5
                if np.sum(inliers) >= 15:
                    curr_pts = curr_pts[inliers]

        last_valid_center = curr_c
        centers.append(last_valid_center)

    centers = np.array(centers, dtype=np.float32)

    # 4. White Balance & Temporal Luminance Continuity Equalization
    r_means, g_means, b_means, p90_vals = [], [], [], []
    for f in sampled_frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        mask = gray > 20
        if np.sum(mask) > 50:
            b_means.append(float(np.mean(f[:, :, 0][mask])))
            g_means.append(float(np.mean(f[:, :, 1][mask])))
            r_means.append(float(np.mean(f[:, :, 2][mask])))
            p90_vals.append(float(np.percentile(f[:, :, 2][mask], 90)))
        else:
            b_means.append(0.0)
            g_means.append(0.0)
            r_means.append(0.0)
            p90_vals.append(0.0)

    r_means = np.array(r_means)
    g_means = np.array(g_means)
    b_means = np.array(b_means)
    p90_vals = np.array(p90_vals)

    gr_ratio = g_means / np.maximum(r_means, 1.0)
    br_ratio = b_means / np.maximum(r_means, 1.0)

    valid_color_mask = (gr_ratio < 0.75) & (br_ratio < 0.65) & (r_means > 10)
    x_valid_color = np.where(valid_color_mask)[0]

    if len(x_valid_color) > 10:
        target_gr = np.interp(np.arange(len(sampled_frames)), x_valid_color, gr_ratio[x_valid_color])
        target_br = np.interp(np.arange(len(sampled_frames)), x_valid_color, br_ratio[x_valid_color])
    else:
        target_gr = gr_ratio
        target_br = br_ratio

    cutoff_fade = int(0.95 * len(sampled_frames))
    robust_p90 = p90_vals.copy()
    for idx in range(10, cutoff_fade):
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

        # 1. White balance chromaticity correction
        if not valid_color_mask[idx] and r_means[idx] > 10:
            curr_g_scale = target_gr[idx] / max(gr_ratio[idx], 1e-4)
            curr_b_scale = target_br[idx] / max(br_ratio[idx], 1e-4)
            corr_frame[:, :, 0] *= curr_b_scale
            corr_frame[:, :, 1] *= curr_g_scale

        # 2. Temporal Luminance correction
        if lum_scales[idx] != 1.0:
            corr_frame *= lum_scales[idx]

        frame_to_warp = np.clip(corr_frame, 0, 255).astype(np.uint8)

        dx = cx_target - cx
        dy = cy_target - cy
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        warped = cv2.warpAffine(frame_to_warp, M, (master_w, master_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        proc.stdin.write(warped.tobytes())

    proc.stdin.close()
    proc.wait()


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-PRECISION TOTALITY ENGINE (VIDEO REALTIME)
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
    Stabilizes totality phase by tracking lunar silhouette edge points using Canny edge detection,
    annular distance masking, and Nelder-Mead Huber loss optimization.
    Inherits C2 shift offset (+12.5 px, +7.5 px) for seamless continuity.
    Uses memory-efficient two-pass streaming.
    """
    cap = cv2.VideoCapture(in_path)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    cx_target = master_w / 2.0
    cy_target = master_h / 2.0
    scale_factor = master_h / 720.0
    if lunar_radius is None:
        lunar_radius = 246.0 * scale_factor

    cap = cv2.VideoCapture(in_path)
    centers = []
    last_valid_center = (622.0 * scale_factor, 377.0 * scale_factor)

    pbar = tqdm(total=cnt, desc=f"Analyzing totality lunar limb ({os.path.basename(in_path)})")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 20, 60)

        y_indices, x_indices = np.where(edges > 0)
        dists = np.sqrt((x_indices - last_valid_center[0])**2 + (y_indices - last_valid_center[1])**2)
        valid_mask = (dists >= 200 * scale_factor) & (dists <= 285 * scale_factor)
        edge_pts = np.column_stack([x_indices[valid_mask], y_indices[valid_mask]])

        if len(edge_pts) < 100:
            centers.append(last_valid_center)
        else:
            def cost(params):
                cx, cy = params
                d = np.sqrt((edge_pts[:, 0] - cx)**2 + (edge_pts[:, 1] - cy)**2)
                err = np.abs(d - lunar_radius)
                return np.sum(np.where(err < 4.0, 0.5 * err**2, 4.0 * (err - 2.0)))

            res = minimize(cost, [last_valid_center[0], last_valid_center[1]], method='Nelder-Mead')
            if res.success and (cx_target - 150) < res.x[0] < (cx_target + 150) and (cy_target - 150) < res.x[1] < (cy_target + 150):
                last_valid_center = (res.x[0], res.x[1])
                centers.append(last_valid_center)
            else:
                centers.append(last_valid_center)
        pbar.update(1)

    cap.release()
    pbar.close()

    # Pass 2: Warp and encode with streaming (preserving exact 1:1 real-time duration)
    cap = cv2.VideoCapture(in_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 27.307
    real_duration_s = len(centers) / native_fps
    target_total_frames = int(round(real_duration_s * fps))

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

    off_x, off_y = shift_offset
    pbar = tqdm(total=target_total_frames, desc="Warping & encoding totality (1x Real-Time)")
    
    current_src_idx = -1
    current_frame = None

    for out_i in range(target_total_frames):
        src_idx = min(int(round((out_i / float(target_total_frames)) * (len(centers) - 1))), len(centers) - 1)
        
        while current_src_idx < src_idx:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = frame
            current_src_idx += 1

        if current_frame is None:
            break

        c_x, c_y = centers[src_idx]
        dx = cx_target - c_x + (off_x * scale_factor)
        dy = cy_target - c_y + (off_y * scale_factor)
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        warped = cv2.warpAffine(current_frame, M, (master_w, master_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        proc.stdin.write(warped.tobytes())
        pbar.update(1)

    cap.release()
    pbar.close()
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
    project: str = None,
    in_dir: str = "010_in",
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
    comp_w = asset_meta.get('comp_width') or master_w
    comp_h = asset_meta.get('comp_height') or master_h
    duration_s = asset_meta.get('duration', 10.0)
    idx = asset_meta.get('index', 6)

    # 1. Generate high-res composite artwork image via build_composite (clean without timestamps, without metadata block, with contacts)
    composite_png = cec.build_composite(
        layout=layout,
        width=comp_w,
        height=comp_h,
        show_labels=False,
        show_info=False,
        contacts="auto",
        project=project,
        in_dir=in_dir,
        out_dir=out_dir
    )
    composite_img = cv2.imread(composite_png)
    if composite_img is None:
        raise RuntimeError(f"Failed to read composite artwork from {composite_png}")

    # 2. Save named composite artwork image artifacts and metadata JSON
    png_path = os.path.join(out_dir, f"composite_artwork_{idx}_{layout}.png")
    jpg_path = os.path.join(out_dir, f"composite_artwork_{idx}_{layout}.jpg")
    json_src = os.path.splitext(composite_png)[0] + ".json"
    json_dst = os.path.splitext(out_path)[0] + ".json"
    cv2.imwrite(png_path, composite_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cv2.imwrite(jpg_path, composite_img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    if os.path.exists(json_src):
        import shutil
        shutil.copy2(json_src, json_dst)
        shutil.copy2(json_src, os.path.join(out_dir, f"composite_artwork_{idx}_{layout}.json"))

    # 3. Scale and pad to master project resolution
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
    Uses pure FFmpeg image2pipe streaming with zero RAM footprint and no OpenCV HEVC decoding issues.
    """
    print("=" * 65)
    print(f"ASSEMBLING MASTER FILM: {output_path}")
    print(f"  Resolution : {master_w}x{master_h} @ {fps:.2f} fps")
    print(f"  Clips count: {len(clip_paths)}")
    print(f"  Transition : '{transition_type}'")
    print("=" * 65)

    if len(clip_paths) == 0:
        raise ValueError("No clips to assemble.")

    if len(clip_paths) == 1:
        single_clip = clip_paths[0]
        if os.path.abspath(single_clip) != os.path.abspath(output_path):
            subprocess.run(['cp', single_clip, output_path], check=True)
        print(f"Single-clip master film exported directly: {output_path}")
        return

    # Calculate transition frames
    n_freeze_b = int(round(freeze_before * fps))
    n_fade_o = int(round(fade_out * fps))
    n_black = int(round(black_duration * fps))
    n_fade_i = int(round(fade_in * fps))
    n_freeze_a = int(round(freeze_after * fps))
    n_crossfade = int(round(fade_duration * fps))

    # Temporary intermediate raw container
    out_dir = os.path.dirname(os.path.abspath(output_path))
    temp_raw = os.path.join(out_dir, "temp_master_raw.mp4")

    # Initialize FFmpeg encoder pipe
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
        '-pix_fmt', 'yuv420p',
        '-color_range', '1',
        '-colorspace', 'bt709',
        '-color_trc', 'bt709',
        '-color_primaries', 'bt709',
        temp_raw
    ]
    proc_out = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    total_frames_written = 0
    prev_last_frame = None

    for idx, cpath in enumerate(clip_paths):
        print(f"  • Streaming [{idx+1}/{len(clip_paths)}]: {os.path.basename(cpath)}")
        clip_frames_written = 0
        first_frame_of_clip = None
        last_frame_of_clip = None

        for frame, w, h in stream_video_frames_pipe(cpath):
            if w != master_w or h != master_h:
                frame = cv2.resize(frame, (master_w, master_h), interpolation=cv2.INTER_LANCZOS4)

            # When starting a new clip (and we had a previous clip), write the transition frames first
            if first_frame_of_clip is None:
                first_frame_of_clip = frame.copy()
                if prev_last_frame is not None:
                    if transition_type == "fade_to_black":
                        for _ in range(n_freeze_b):
                            proc_out.stdin.write(prev_last_frame.tobytes())
                            total_frames_written += 1

                        for i in range(1, n_fade_o + 1):
                            alpha = 1.0 - (i / float(n_fade_o))
                            faded = np.clip(prev_last_frame.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
                            proc_out.stdin.write(faded.tobytes())
                            total_frames_written += 1

                        if n_black > 0:
                            black_frame = np.zeros((master_h, master_w, 3), dtype=np.uint8)
                            for _ in range(n_black):
                                proc_out.stdin.write(black_frame.tobytes())
                                total_frames_written += 1

                        for i in range(1, n_fade_i + 1):
                            alpha = i / float(n_fade_i)
                            faded = np.clip(first_frame_of_clip.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
                            proc_out.stdin.write(faded.tobytes())
                            total_frames_written += 1

                        for _ in range(n_freeze_a):
                            proc_out.stdin.write(first_frame_of_clip.tobytes())
                            total_frames_written += 1

                    elif transition_type == "crossfade":
                        for i in range(1, n_crossfade + 1):
                            alpha = i / float(n_crossfade)
                            blended = cv2.addWeighted(prev_last_frame, 1.0 - alpha, first_frame_of_clip, alpha, 0.0)
                            proc_out.stdin.write(blended.tobytes())
                            total_frames_written += 1

                    elif transition_type == "hard":
                        pass

            proc_out.stdin.write(frame.tobytes())
            last_frame_of_clip = frame.copy()
            clip_frames_written += 1
            total_frames_written += 1

        prev_last_frame = last_frame_of_clip

    proc_out.stdin.close()
    proc_out.wait()

    # Move moov atom to beginning with copy faststart
    cmd_fast = ['ffmpeg', '-y', '-loglevel', 'error', '-i', temp_raw, '-c', 'copy', '-movflags', '+faststart', output_path]
    subprocess.run(cmd_fast, check=True)
    if os.path.exists(temp_raw):
        os.remove(temp_raw)

    duration_total_s = total_frames_written / fps
    sz_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("=" * 65)
    print(f"SUCCESS: Master film generated at: {output_path}")
    print(f"  Resolution: {master_w}x{master_h} @ {fps:.2f} fps")
    print(f"  Total frames: {total_frames_written} ({duration_total_s:.2f}s)")
    print(f"  File size: {sz_mb:.2f} MB")
    print("=" * 65)
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ART FILM ASSEMBLY (CAMERA DIVE & NARRATIVE MONTAGE)
# ─────────────────────────────────────────────────────────────────────────────

def assemble_art_film(
    project: str = None,
    in_dir: str = "010_in",
    out_dir: str = "040_out",
    output_path: str = "040_out/full_eclipse.mp4",
    master_w: int = 1280,
    master_h: int = 720,
    visual_assets: list = None,
    processed_clip_paths: list = None,
    no_title: bool = False,
    title_duration: float = 5.0,
    freeze_before: float = 1.0,
    fade_out: float = 0.5,
    black_duration: float = 0.0,
    fade_in: float = 0.5,
    fade_duration: float = 0.8,
    freeze_after: float = 1.0,
    fps: float = 30.0,
    crf: int = 16,
    preset: str = "fast",
    music_asset: dict = None,
    date_str: str = None,
    force_all: bool = False,
    force_prep: bool = False,
    extrapolate_c1: bool = False
):
    """
    Assembles the film in 'art' cinematic style:
      1. Title Card (00_title.mp4) [fade_to_black]
      2. Full Composite Overview & Ingress Zoom-In Dive (art_01_dive_in.mp4) [hard]
      3. Ingress Timelapse (01_timelapse) [fade_to_black]
      4. Pre-totality Slowdown [fade_to_black]
      5. Real-time Totality Part 1 [crossfade 0.8s]
      6. Multi-Exposure Totality HDR Artwork [crossfade 0.8s]
      7. Real-time Totality Part 2 [fade_to_black]
      8. Egress Timelapse [hard]
      9. Egress Zoom-Out Dive & Full Composite Outro [fade_to_black]
      10. Closing Credits & End Titles (07_endtitles.mp4)
    """
    if processed_clip_paths is None:
        processed_clip_paths = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".mp4")]

    print("=" * 65)
    print(f"ASSEMBLING CINEMATIC 'ART' FILM: {output_path}")
    print(f"  Resolution : {master_w}x{master_h} @ {fps:.2f} fps")
    print(f"  Transitions: Dynamic (Camera Dives, Crossfades, Fades to Black)")
    print("=" * 65)

    # 1. Ensure Title Card exists if requested
    title_path = os.path.join(out_dir, "00_title.mp4")
    if not no_title and (not os.path.exists(title_path) or force_all):
        ctc.generate_title_card(duration_s=title_duration, width=master_w, height=master_h, out_dir=out_dir, output_mp4=title_path)

    # 2. Locate / generate composite 4k artwork for camera dive
    comp_asset = next((a for a in (visual_assets or []) if a['asset_type'] == 'composite'), None)
    comp_layout = comp_asset.get('layout', 'circle') if comp_asset else 'circle'
    art_comp_png = os.path.join(out_dir, f"art_composite_4k_{comp_layout}.png")
    art_comp_json = os.path.join(out_dir, f"art_composite_4k_{comp_layout}.json")
    art_comp_video = os.path.join(out_dir, f"art_composite_4k_{comp_layout}.mp4")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prep_dir = os.path.join(repo_root, "005_raw_preprocessed")
    has_prep_raw = os.path.exists(prep_dir) and any(f.endswith(".mp4") for f in os.listdir(prep_dir))
    prep_newer = False
    if has_prep_raw and os.path.exists(art_comp_png):
        mtime_comp = os.path.getmtime(art_comp_png)
        for f in os.listdir(prep_dir):
            if f.endswith(".mp4") and os.path.getmtime(os.path.join(prep_dir, f)) > mtime_comp:
                prep_newer = True
                break

    comp_regenerated = False
    if not os.path.exists(art_comp_png) or not os.path.exists(art_comp_json) or force_all or force_prep or prep_newer:
        print(f"Generating 4K Ultra-HD Composite Canvas for Camera Dives ({comp_layout})...")
        cec.build_composite(
            layout=comp_layout,
            width=3840,
            height=2160,
            show_labels=False,
            show_info=False,
            contacts="auto",
            project=project,
            in_dir=in_dir,
            out_dir=out_dir,
            output_path=art_comp_png
        )
        comp_regenerated = True

    # Read composite metadata for sample endpoints
    in_start_frame = 0
    eg_end_frame = -1
    if os.path.exists(art_comp_json):
        try:
            with open(art_comp_json, "r", encoding="utf-8") as f:
                cm = json.load(f)
            in_start_frame = int(cm["samples"][0].get("frame_idx", 0))
            eg_end_frame = int(cm["samples"][-1].get("frame_idx", -1))
        except Exception as e:
            print(f"Warning reading composite json: {e}")

    # Locate Ingress and Egress timelapse processed clips dynamically (prefer preprocessed)
    prep_in = [p for p in processed_clip_paths if os.path.basename(p).startswith("01_") and "prep" in os.path.basename(p) and p.endswith(".mp4")]
    tl_in_candidates = prep_in if prep_in else [p for p in processed_clip_paths if os.path.basename(p).startswith("01_") and p.endswith(".mp4")]
    tl_in_path = tl_in_candidates[0] if tl_in_candidates else os.path.join(out_dir, "01_timelapse_prep_i10.mp4")

    prep_eg = [p for p in processed_clip_paths if os.path.basename(p).startswith("04_") and "prep" in os.path.basename(p) and p.endswith(".mp4")]
    tl_eg_candidates = prep_eg if prep_eg else [p for p in processed_clip_paths if os.path.basename(p).startswith("04_") and p.endswith(".mp4")]
    tl_eg_path = tl_eg_candidates[0] if tl_eg_candidates else os.path.join(out_dir, "04_timelapse_prep_i10.mp4")

    art_tl_in_path = os.path.join(out_dir, "art_01_timelapse.mp4")
    need_tl_in = (
        not os.path.exists(art_tl_in_path)
        or force_all
        or force_prep
        or comp_regenerated
        or (os.path.exists(tl_in_path) and os.path.getmtime(art_tl_in_path) < os.path.getmtime(tl_in_path))
    )
    if need_tl_in:
        print(f"Trimming Ingress Timelapse from {os.path.basename(tl_in_path)} (starting at sample 0 frame {in_start_frame})...")
        cap = cv2.VideoCapture(tl_in_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(art_tl_in_path, fourcc, fps, (master_w, master_h))
        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if f_idx >= in_start_frame: vw.write(frame)
            f_idx += 1
        cap.release()
        vw.release()

    dive_in_path = os.path.join(out_dir, "art_01_dive_in.mp4")
    need_dive_in = (
        not os.path.exists(dive_in_path)
        or force_all
        or force_prep
        or comp_regenerated
        or (os.path.exists(art_comp_png) and os.path.getmtime(dive_in_path) < os.path.getmtime(art_comp_png))
        or (os.path.exists(art_tl_in_path) and os.path.getmtime(dive_in_path) < os.path.getmtime(art_tl_in_path))
    )
    if need_dive_in:
        print("Generating Ingress Camera Dive (art_01_dive_in.mp4)...")
        ref_in = ccd.extract_first_frame(art_tl_in_path)
        ccd.generate_camera_dive_clip(
            composite_img_path=art_comp_png,
            composite_meta_path=art_comp_json,
            target_sample_idx=0,
            output_video_path=dive_in_path,
            direction="zoom_in",
            hold_duration=2.0,
            dive_duration=3.0,
            fps=fps,
            out_width=master_w,
            out_height=master_h,
            target_disk_diameter_px=476.0 * (master_h / 720.0),
            reference_video_frame=ref_in
        )

    art_tl_eg_path = os.path.join(out_dir, "art_04_timelapse.mp4")
    need_tl_eg = (
        not os.path.exists(art_tl_eg_path)
        or force_all
        or force_prep
        or comp_regenerated
        or (os.path.exists(tl_eg_path) and os.path.getmtime(art_tl_eg_path) < os.path.getmtime(tl_eg_path))
    )
    if need_tl_eg:
        print(f"Trimming Egress Timelapse from {os.path.basename(tl_eg_path)} (ending at last sample frame {eg_end_frame})...")
        cap = cv2.VideoCapture(tl_eg_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(art_tl_eg_path, fourcc, fps, (master_w, master_h))
        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if eg_end_frame < 0 or f_idx <= eg_end_frame:
                vw.write(frame)
            if eg_end_frame >= 0 and f_idx >= eg_end_frame:
                break
            f_idx += 1
        cap.release()
        vw.release()

    dive_out_path = os.path.join(out_dir, "art_02_dive_out.mp4")
    need_dive_out = (
        not os.path.exists(dive_out_path)
        or force_all
        or force_prep
        or comp_regenerated
        or (os.path.exists(art_comp_png) and os.path.getmtime(dive_out_path) < os.path.getmtime(art_comp_png))
        or (os.path.exists(art_tl_eg_path) and os.path.getmtime(dive_out_path) < os.path.getmtime(art_tl_eg_path))
    )
    if need_dive_out:
        print("Generating Egress Camera Dive (art_02_dive_out.mp4)...")
        ref_out = ccd.extract_last_frame(art_tl_eg_path)
        ccd.generate_camera_dive_clip(
            composite_img_path=art_comp_png,
            composite_meta_path=art_comp_json,
            target_sample_idx=-1,
            output_video_path=dive_out_path,
            direction="zoom_out",
            hold_duration=2.0,
            dive_duration=3.0,
            fps=fps,
            out_width=master_w,
            out_height=master_h,
            target_disk_diameter_px=476.0 * (master_h / 720.0),
            reference_video_frame=ref_out
        )

    # 5. Split Real-time Totality Clip at Totality Max (frame 1552)
    vt_path = os.path.join(out_dir, "03_video_realtime.mp4")
    tot_p1_path = os.path.join(out_dir, "art_03_totality_p1.mp4")
    tot_p2_path = os.path.join(out_dir, "art_03_totality_p2.mp4")
    max_totality_frame = 1552

    if (not os.path.exists(tot_p1_path) or not os.path.exists(tot_p2_path)) or force_all:
        print("Splitting real-time totality video at Totality Max frame...")
        cap = cv2.VideoCapture(vt_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw1 = cv2.VideoWriter(tot_p1_path, fourcc, fps, (master_w, master_h))
        vw2 = cv2.VideoWriter(tot_p2_path, fourcc, fps, (master_w, master_h))
        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx <= max_totality_frame:
                vw1.write(frame)
            if f_idx >= max_totality_frame:
                vw2.write(frame)
            f_idx += 1
        cap.release()
        vw1.release()
        vw2.release()

    # 6. Locate Totality HDR Artwork and End Titles
    tot_hdr_path = os.path.join(out_dir, "05_totality_6.mp4")
    if not os.path.exists(tot_hdr_path):
        tot_hdr_path = os.path.join(out_dir, "05_photo_6.mp4")

    slow_path = os.path.join(out_dir, "02_video_slowdown_10.mp4")
    end_path = os.path.join(out_dir, "07_endtitles.mp4")

    # Ordered Assembly Sequence
    sequence = []
    if not no_title and os.path.exists(title_path):
        sequence.append({"path": title_path, "trans_after": "fade_to_black"})

    sequence.append({"path": dive_in_path, "trans_after": "hard"})
    sequence.append({"path": art_tl_in_path, "trans_after": "fade_to_black"})

    has_slow = any(
        f.startswith("02_") or "slowdown" in f.lower() for f in os.listdir(in_dir) if not f.startswith(".")
    ) if os.path.exists(in_dir) else False
    if has_slow and os.path.exists(slow_path):
        sequence.append({"path": slow_path, "trans_after": "fade_to_black"})

    sequence.append({"path": tot_p1_path, "trans_after": "crossfade", "cross_duration": 0.8})
    sequence.append({"path": tot_hdr_path, "trans_after": "crossfade", "cross_duration": 0.8})
    sequence.append({"path": tot_p2_path, "trans_after": "fade_to_black"})
    sequence.append({"path": art_tl_eg_path, "trans_after": "hard"})
    sequence.append({"path": dive_out_path, "trans_after": "fade_to_black"})

    if os.path.exists(end_path):
        sequence.append({"path": end_path, "trans_after": None})

    # Assemble into temp_raw
    out_parent = os.path.dirname(os.path.abspath(output_path))
    temp_raw = os.path.join(out_parent, "temp_master_raw_art.mp4")

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
        '-pix_fmt', 'yuv420p',
        '-color_range', '1',
        '-colorspace', 'bt709',
        '-color_trc', 'bt709',
        '-color_primaries', 'bt709',
        temp_raw
    ]
    proc_out = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    total_frames_written = 0
    prev_last_frame = None
    prev_trans_after = None
    prev_cross_dur = 0.8

    n_freeze_b = int(round(freeze_before * fps))
    n_fade_o = int(round(fade_out * fps))
    n_black = int(round(black_duration * fps))
    n_fade_i = int(round(fade_in * fps))
    n_freeze_a = int(round(freeze_after * fps))

    for item_idx, item in enumerate(sequence):
        cpath = item['path']
        trans_type = prev_trans_after
        cross_dur = prev_cross_dur
        n_crossfade = int(round(cross_dur * fps))

        print(f"  • Streaming [{item_idx+1}/{len(sequence)}]: {os.path.basename(cpath)}")
        first_frame_of_clip = None
        last_frame_of_clip = None

        for frame, w, h in stream_video_frames_pipe(cpath):
            if w != master_w or h != master_h:
                frame = cv2.resize(frame, (master_w, master_h), interpolation=cv2.INTER_LANCZOS4)

            if first_frame_of_clip is None:
                first_frame_of_clip = frame.copy()
                if prev_last_frame is not None and trans_type is not None:
                    if trans_type == "fade_to_black":
                        for _ in range(n_freeze_b):
                            proc_out.stdin.write(prev_last_frame.tobytes())
                            total_frames_written += 1
                        for i in range(1, n_fade_o + 1):
                            alpha = 1.0 - (i / float(n_fade_o))
                            faded = np.clip(prev_last_frame.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
                            proc_out.stdin.write(faded.tobytes())
                            total_frames_written += 1
                        if n_black > 0:
                            black_frame = np.zeros((master_h, master_w, 3), dtype=np.uint8)
                            for _ in range(n_black):
                                proc_out.stdin.write(black_frame.tobytes())
                                total_frames_written += 1
                        for i in range(1, n_fade_i + 1):
                            alpha = i / float(n_fade_i)
                            faded = np.clip(first_frame_of_clip.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
                            proc_out.stdin.write(faded.tobytes())
                            total_frames_written += 1
                        for _ in range(n_freeze_a):
                            proc_out.stdin.write(first_frame_of_clip.tobytes())
                            total_frames_written += 1

                    elif trans_type == "crossfade":
                        for i in range(1, n_crossfade + 1):
                            alpha = i / float(n_crossfade)
                            blended = cv2.addWeighted(prev_last_frame, 1.0 - alpha, first_frame_of_clip, alpha, 0.0)
                            proc_out.stdin.write(blended.tobytes())
                            total_frames_written += 1

                    elif trans_type == "hard":
                        pass

            proc_out.stdin.write(frame.tobytes())
            last_frame_of_clip = frame.copy()
            total_frames_written += 1

        prev_last_frame = last_frame_of_clip
        prev_trans_after = item.get('trans_after')
        prev_cross_dur = item.get('cross_duration', 0.8)

    proc_out.stdin.close()
    proc_out.wait()

    cmd_fast = ['ffmpeg', '-y', '-loglevel', 'error', '-i', temp_raw, '-c', 'copy', '-movflags', '+faststart', output_path]
    subprocess.run(cmd_fast, check=True)
    if os.path.exists(temp_raw):
        os.remove(temp_raw)

    duration_total_s = total_frames_written / fps
    sz_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("=" * 65)
    print(f"SUCCESS: Art film generated at: {output_path}")
    print(f"  Resolution: {master_w}x{master_h} @ {fps:.2f} fps")
    print(f"  Total frames: {total_frames_written} ({duration_total_s:.2f}s)")
    print(f"  File size: {sz_mb:.2f} MB")
    print("=" * 65)
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

def run_coc_pipeline(
    project=None,
    in_dir=None,
    out_dir=None,
    output_film=None,
    film_style=None,
    transition_type="fade_to_black",
    freeze_before=1.0,
    fade_out=0.5,
    black_duration=0.0,
    fade_in=0.5,
    fade_duration=1.0,
    freeze_after=1.0,
    force_all=False,
    force_prep=False,
    extrapolate_c1=False,
    no_subtitles=False,
    no_title=False,
    title_duration=5.0,
    force_db=False,
    date_str=None,
    no_music=False,
    music_mode="arrange"
):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from project_manager import resolve_project
    proj = resolve_project(project_arg=project, repo_root=repo_root, in_base=in_dir or "010_in", out_base=out_dir or "040_out")
    in_dir = proj["in_dir"]
    out_dir = proj["out_dir"]
    temp_dir = proj["temp_dir"]
    project_id = proj["project_id"]
    raw_dir = proj["raw_dir"]
    prep_dir = proj["prep_dir"]
    is_project_preprocessed = proj["is_preprocessed"]
    if film_style is None:
        film_style = proj["film_style"]

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    if output_film is None:
        output_film = os.path.join(temp_dir, f"full_eclipse_{film_style}_video.mp4")

    final_subtitled_master = os.path.join(out_dir, f"{project_id}.mp4")

    # Discover and separate visual assets from audio
    visual_assets, music_asset = discover_coc_assets(in_dir, is_project_preprocessed=is_project_preprocessed)
    if not visual_assets:
        print(f"[ERROR] No valid CoC assets found in '{in_dir}'!")
        return

    master_w, master_h = detect_project_resolution(visual_assets, repo_root=repo_root, fallback=(1280, 720))

    print("=" * 65)
    print("ECLIPSE ASSEMBLER: CONVENTION-OVER-CONFIGURATION MASTER PIPELINE")
    print("=" * 65)
    print(f"  Project            : {proj['name']} (ID: {project_id})")
    print(f"  Film Assembly Style: {film_style.upper()}")
    print(f"  Input Directory    : {in_dir}")
    print(f"  Output Directory   : {out_dir}")
    print(f"  Temp Directory     : {temp_dir}")
    print(f"  Clean Video Output : {output_film}")
    if not no_subtitles:
        print(f"  Subtitled Master   : {final_subtitled_master}")
    print(f"  Master Resolution  : {master_w}x{master_h} px")
    print(f"  Visual Clips       : {len(visual_assets)} discovered")
    if music_asset is not None and not no_music:
        print(f"  Music Track        : {music_asset['raw_name']} (Title: '{music_asset['music_title']}', Author: '{music_asset['music_author']}')")
    print("=" * 65)
    print()

    has_totality = any(a['asset_type'] in ['video_realtime', 'photo'] and 'totality' in a['raw_name'].lower() for a in visual_assets)
    processed_clip_paths = []

    # 1. Generate opening dynamic title card if requested
    if not no_title:
        title_clip_path = os.path.join(temp_dir, "00_title.mp4")
        if not os.path.exists(title_clip_path) or force_all:
            print(f"Generating Dynamic Opening Title Card ({title_duration}s)...")
            ctc.generate_title_card(
                duration_s=title_duration,
                width=master_w,
                height=master_h,
                out_dir=temp_dir,
                output_mp4=title_clip_path,
                fps=30.0,
                force_db=force_db,
                date_str=date_str
            )
        processed_clip_paths.append(title_clip_path)

    # 2. Process all visual CoC assets in order
    for a in visual_assets:
        idx = a['index']
        a_type = a['asset_type']
        in_path = a['path']
        raw_name = a['raw_name']
        name_no_ext, _ = os.path.splitext(raw_name)
        out_clip_path = os.path.join(temp_dir, f"{name_no_ext}.mp4")

        print(f"Processing Asset [{idx:02d}]: {raw_name} ({a_type.upper()})...")

        is_prep = a.get('is_preprocessed', False)
        is_comp = (a_type == "composite")
        skip_cache = (is_prep and force_prep) or (is_comp and force_prep)

        if os.path.exists(out_clip_path) and os.path.getsize(out_clip_path) > 1024 and not force_all and not skip_cache:
            print(f"  • Asset output exists ({out_clip_path}), skipping (use --force-all to rebuild).")
            processed_clip_paths.append(out_clip_path)
            continue

        if a_type == "timelapse":
            if a.get('is_preprocessed'):
                prep_src = resolve_preprocessed_timelapse(
                    a, repo_root=repo_root, raw_dir=raw_dir, prep_dir=prep_dir,
                    force=force_all or force_prep, extrapolate_c1=extrapolate_c1
                )
                print(f"  • Using Preprocessed Restored Timelapse: {prep_src} (stabilization skipped)")
                cmd_prep = [
                    'ffmpeg', '-y', '-loglevel', 'error',
                    '-i', prep_src,
                    '-vf', f"scale={master_w}:{master_h}:force_original_aspect_ratio=decrease,pad={master_w}:{master_h}:(ow-iw)/2:(oh-ih)/2:black",
                    '-c:v', 'libx264', '-crf', '16', '-preset', 'fast',
                    '-pix_fmt', 'yuv420p',
                    out_clip_path
                ]
                subprocess.run(cmd_prep, check=True)
            else:
                if has_totality and idx >= 4:
                    shift_offset = (35.0, 24.0)
                    r_fixed = 237.5 * (master_h / 720.0)
                else:
                    shift_offset = (0.0, 0.0)
                    r_fixed = 238.5 * (master_h / 720.0)

                stabilize_timelapse_asset(
                    in_path=in_path,
                    out_path=out_clip_path,
                    master_w=master_w,
                    master_h=master_h,
                    r_fixed=r_fixed,
                    shift_offset=shift_offset,
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
            shift_offset = (12.5, 7.5) if idx >= 3 else (0.0, 0.0)
            process_video_realtime_asset(
                in_path=in_path,
                out_path=out_clip_path,
                master_w=master_w,
                master_h=master_h,
                shift_offset=shift_offset,
                fps=30.0, crf=16, preset="fast"
            )

        elif a_type == "photo":
            process_photo_asset(
                in_path=in_path,
                out_path=out_clip_path,
                duration_s=a['duration'],
                master_w=master_w,
                master_h=master_h,
                fps=30.0, crf=16, preset="fast"
            )

        elif a_type == "composite":
            process_composite_asset(
                asset_meta=a,
                out_path=out_clip_path,
                project=proj['name'],
                in_dir=in_dir,
                out_dir=temp_dir,
                master_w=master_w,
                master_h=master_h,
                fps=30.0, crf=16, preset="fast"
            )

        elif a_type == "endtitles":
            print(f"Generating Closing Credits & End Titles: {raw_name} -> {out_clip_path} ({a['duration']}s)")
            music_info = None
            if music_asset is not None:
                music_info = {
                    'title': music_asset.get('music_title', 'Original Score'),
                    'author': music_asset.get('music_author', 'Unknown')
                }
            cet.generate_end_titles(
                md_path=in_path,
                duration_s=a['duration'],
                width=master_w,
                height=master_h,
                out_dir=temp_dir,
                output_mp4=out_clip_path,
                music_info=music_info,
                eclipse_date_str=date_str
            )

        processed_clip_paths.append(out_clip_path)
        print()

    # 3. Assemble master film based on film_style
    if film_style.lower() == "art":
        assemble_art_film(
            project=proj['name'],
            in_dir=in_dir,
            out_dir=temp_dir,
            output_path=output_film,
            master_w=master_w,
            master_h=master_h,
            visual_assets=visual_assets,
            processed_clip_paths=processed_clip_paths,
            no_title=no_title,
            title_duration=title_duration,
            freeze_before=freeze_before,
            fade_out=fade_out,
            black_duration=black_duration,
            fade_in=fade_in,
            fade_duration=fade_duration,
            freeze_after=freeze_after,
            fps=30.0, crf=16, preset="fast",
            music_asset=music_asset,
            date_str=date_str,
            force_all=force_all,
            force_prep=force_prep,
            extrapolate_c1=extrapolate_c1
        )
    else:
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

    # 4. Invoke standalone subtitle generator
    if not no_subtitles:
        srt_master_path = os.path.join(temp_dir, f"full_eclipse_{film_style}.srt")
        ges.generate_eclipse_subtitles_pipeline(
            video_path=output_film,
            output_srt_path=srt_master_path,
            interval_s=5.0,
            include_phase=True,
            lang="both",
            embed=True,
            date_str=date_str,
            include_title=not no_title,
            title_duration=title_duration,
            force_db=force_db,
            project=proj['name'],
            in_dir=in_dir,
            out_dir=out_dir,
            film_style=film_style
        )

    # 5. Audio / Music Track Synchronization and Muxing
    if music_asset is not None and not no_music:
        print("=" * 65)
        print(f"INTEGRATING AUDIO TRACK: {music_asset['raw_name']}")
        print(f"  Title  : {music_asset['music_title']}")
        print(f"  Author : {music_asset['music_author']}")
        print("=" * 65)

        targets_to_mux = []
        if not no_subtitles:
            if os.path.exists(final_subtitled_master):
                targets_to_mux.append(final_subtitled_master)
        if os.path.exists(output_film) and output_film not in targets_to_mux:
            targets_to_mux.append(output_film)

        for tgt in targets_to_mux:
            temp_out = os.path.join(temp_dir, f"tmp_mux_{os.path.basename(tgt)}")
            aat.add_audio_to_video(
                video_path=tgt,
                audio_path=music_asset['path'],
                output_path=temp_out,
                mode=music_mode
            )
            if os.path.exists(temp_out):
                os.replace(temp_out, tgt)
                print(f"  -> Successfully synchronized audio track into {tgt}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoC Automated Solar Eclipse Processing & Master Film Assembly Pipeline")
    parser.add_argument("--project", "-p", type=str, default=None,
                        help="Project folder name in 010_in/ (default: most recently modified)")
    parser.add_argument("--film-style", type=str, choices=["standard", "art"], default=None,
                        help="Film assembly style: 'standard' (linear sequence) or 'art' (camera dive & narrative montage). Default: deduced from project")
    parser.add_argument("--in-dir", "-i", type=str, default=None,
                        help="Input directory with CoC named assets (default: 010_in)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Intermediate and master output directory (default: 040_out)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to final master clean video (default: 040_out/<project>/full_eclipse_<film_style>_video.mp4)")
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
    parser.add_argument("--no-subtitles", action="store_true",
                        help="Skip automatic subtitle generation and QuickTime embedding")
    parser.add_argument("--no-title", action="store_true",
                        help="Skip automatic opening title card generation and concatenation")
    parser.add_argument("--title-duration", type=float, default=5.0,
                        help="Duration in seconds of opening title card (default: 5.0s)")
    parser.add_argument("--date", "-d", type=str, default=None,
                        help="Eclipse date YYYY-MM-DD (default: auto-detect)")
    parser.add_argument("--force-db", action="store_true",
                        help="Force refresh eclipse database")
    parser.add_argument("--force-all", action="store_true",
                        help="Force re-processing and re-stabilizing all assets from scratch")
    parser.add_argument("--force-prep", action="store_true",
                        help="Force regeneration of preprocessed raw timelapses in 005_raw_preprocessed/")
    parser.add_argument("--extrapolate-c1", "--extrapolate-C1", action="store_true",
                        help="Extrapolate Ingress timelapse backward to First Contact (C1)")
    parser.add_argument("--no-music", action="store_true",
                        help="Skip automatic musical audio track synchronization and muxing")
    parser.add_argument("--music-mode", type=str, choices=["arrange", "stretch", "cut", "auto"], default="arrange",
                        help="Audio sync mode: 'arrange' (musical phrase edit at 100%% tempo, default), 'stretch', 'cut', or 'auto'")
    args = parser.parse_args()

    run_coc_pipeline(
        project=args.project,
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        output_film=args.output,
        film_style=args.film_style,
        transition_type=args.transition_type,
        freeze_before=args.freeze_before,
        fade_out=args.fade_out,
        black_duration=args.black_duration,
        fade_in=args.fade_in,
        fade_duration=args.fade_duration,
        freeze_after=args.freeze_after,
        force_all=args.force_all,
        force_prep=args.force_prep,
        extrapolate_c1=args.extrapolate_c1,
        no_subtitles=args.no_subtitles,
        no_title=args.no_title,
        title_duration=args.title_duration,
        force_db=args.force_db,
        date_str=args.date,
        no_music=args.no_music,
        music_mode=args.music_mode
    )
