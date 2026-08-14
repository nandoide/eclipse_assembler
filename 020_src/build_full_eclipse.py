#!/usr/bin/env python3
"""
Automated Pipeline for Solar Eclipse Processing & Master Film Assembly
========================================================================
Processes and assembles the 2026 total solar eclipse footage into a single,
perfectly stabilized and centered cinematic master film (full_eclipse.mp4).

Pipeline Steps:
  1. Partial Ingress (partial_in.mp4 / partial_ingress.mp4):
     Subpixel solar limb stabilization (R ≈ 238.5 px) and optical centering.
  2. Pre-totality Approach (pretotal.mp4 / pre_totality.mp4):
     Automatic corrupt/black intervalometer frame filtering, time-lapse acceleration
     to target duration (default: 30.0s), solar limb stabilization (R ≈ 238.0 px),
     dynamic White Balance (chromaticity) equalization, and temporal luminance continuity smoothing.
  3. Totality (total.mp4 / totality.mp4):
     Lunar silhouette and solar corona tracking (R ≈ 246.0 px) at 30 fps,
     inheriting exact C2 transition boundary alignment (+12.5 px, +7.5 px).
  4. Partial Egress (partial_out.mp4 / partial_egress.mp4):
     Subpixel solar limb stabilization (R ≈ 237.5 px) with adaptive thresholding,
     inheriting exact C3 transition boundary alignment (+35.0 px, +24.0 px).
  5. Master Film Assembly (full_eclipse.mp4):
     Seamless concatenation with customizable cinematic transitions:
     - 'fade_to_black': Freeze A -> Smooth fade out to black -> Pure black pause -> Smooth fade in -> Freeze B
     - 'hard': Freeze A -> Instant cut -> Freeze B
     - 'crossfade': Freeze A -> Smooth cross-dissolve -> Freeze B

Authors: Fernando (nandoide) & Antigravity (Google Gemini 3.6 Flash High)
Workspace: eclipse_assembler
"""

import os
import sys
import argparse
import subprocess
import cv2
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.ndimage import uniform_filter1d

from stabilize_eclipse import stabilize_video


def stabilize_and_accelerate_pre_totality(
    in_path: str,
    out_path: str,
    target_seconds: float = 30.0,
    target_fps: float = 30.0,
    solar_radius: float = 238.0,
    crf: int = 16,
    preset: str = "fast"
):
    """
    Filters out black/corrupted intervalometer frames, resamples the pre-totality
    sequence to target_seconds, stabilizes the solar limb to center (640, 360),
    equalizes camera white balance drift, and applies temporal luminance smoothing
    to eliminate exposure dips and flicker.
    """
    print("=" * 60)
    print(f"Step 2/5: Accelerating & stabilizing pre-totality sequence to {target_seconds:.1f}s...")
    print("  (Filtering out black/corrupted intervalometer frames)")
    print("=" * 60)

    cap = cv2.VideoCapture(in_path)
    total_in_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 1. Scan valid frames (discarding black frames)
    valid_indices = []
    idx = 0
    pbar = tqdm(total=total_in_frames, desc="1/3 Scanning valid frames")
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

    target_frames = int(round(target_seconds * target_fps))
    sample_pos = np.linspace(0, len(valid_indices) - 1, target_frames)
    sampled_raw_indices = [valid_indices[int(round(p))] for p in sample_pos]
    sampled_set = set(sampled_raw_indices)

    # 2. Extract selected sampled frames
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
    print(f"Sampled {len(sampled_frames)} clean frames across pre-totality.")

    # 3. Track solar limb center
    centers = []
    last_valid_center = (640.0, 360.0)

    for idx, frame in enumerate(tqdm(sampled_frames, desc="3/3 Tracking solar limb")):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        max_val = float(np.max(gray))
        th_val = max(20, min(80, int(0.35 * max_val)))
        _, binary = cv2.threshold(gray, th_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours or idx >= len(sampled_frames) - 10:
            centers.append(last_valid_center)
            continue

        cnt = max(contours, key=cv2.contourArea)
        pts = cnt.reshape(-1, 2)
        hull = cv2.convexHull(pts).reshape(-1, 2)

        if len(hull) < 20:
            centers.append(last_valid_center)
            continue

        def cost(center):
            cx, cy = center
            d = np.sqrt((hull[:, 0] - cx)**2 + (hull[:, 1] - cy)**2)
            err = np.abs(d - solar_radius)
            return np.sum(np.where(err < 4.0, 0.5 * err**2, 4.0 * (err - 2.0)))

        init_cx, init_cy = last_valid_center
        res = minimize(cost, [init_cx, init_cy], method='Nelder-Mead')
        if res.success and 550 < res.x[0] < 730 and 270 < res.x[1] < 450:
            last_valid_center = (res.x[0], res.x[1])
            centers.append(last_valid_center)
        else:
            centers.append(last_valid_center)

    # 4. Automatic White Balance & Temporal Luminance Continuity Equalization
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

    # Valid baseline reference frames (outside white balance shift where G/R < 0.75 and B/R < 0.65)
    valid_color_mask = (gr_ratio < 0.75) & (br_ratio < 0.65) & (r_means > 10)
    x_valid_color = np.where(valid_color_mask)[0]

    if len(x_valid_color) > 10:
        target_gr = np.interp(np.arange(len(sampled_frames)), x_valid_color, gr_ratio[x_valid_color])
        target_br = np.interp(np.arange(len(sampled_frames)), x_valid_color, br_ratio[x_valid_color])
    else:
        target_gr = gr_ratio
        target_br = br_ratio

    # Temporal Luminance Smoothing (eliminates sudden exposure dips/jumps)
    cutoff_fade = int(0.95 * len(sampled_frames))  # preserve natural diamond ring fade at very end
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

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'bgr24',
        '-r', str(target_fps),
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

        # 2. Temporal Luminance Continuity correction
        if lum_scales[idx] != 1.0:
            corr_frame *= lum_scales[idx]

        frame_to_warp = np.clip(corr_frame, 0, 255).astype(np.uint8)

        dx = 640.0 - cx
        dy = 360.0 - cy
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        warped = cv2.warpAffine(frame_to_warp, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        proc.stdin.write(warped.tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"Successfully generated: {out_path}")


# Backwards compatibility alias
stabilize_and_accelerate_pretotal = stabilize_and_accelerate_pre_totality


def stabilize_totality(
    in_path: str,
    out_path: str,
    lunar_radius: float = 246.0,
    shift_offset=(12.5, 7.5),
    fps: float = 30.0,
    crf: int = 16,
    preset: str = "fast"
):
    """
    Stabilizes totality phase by tracking lunar silhouette edge points,
    applying the inherited C2 shift offset (+12.5 px, +7.5 px) for seamless continuity.
    """
    print("\n" + "=" * 60)
    print(f"Step 3/5: Stabilizing totality (with C2 transition center inheritance: {shift_offset})...")
    print("=" * 60)

    cap = cv2.VideoCapture(in_path)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    centers = []
    last_valid_center = (622.0, 377.0)

    pbar = tqdm(total=cnt, desc="1/2 Analyzing totality lunar limb")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 20, 60)

        y_indices, x_indices = np.where(edges > 0)
        dists = np.sqrt((x_indices - last_valid_center[0])**2 + (y_indices - last_valid_center[1])**2)
        valid_mask = (dists >= 200) & (dists <= 285)
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
            if res.success and 580 < res.x[0] < 700 and 320 < res.x[1] < 420:
                last_valid_center = (res.x[0], res.x[1])
                centers.append(last_valid_center)
            else:
                centers.append(last_valid_center)
        pbar.update(1)

    cap.release()
    pbar.close()

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
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
    for frame, (c_x, c_y) in zip(tqdm(frames, desc="2/2 Warping & encoding totality"), centers):
        dx = 640.0 - c_x + off_x
        dy = 360.0 - c_y + off_y
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        warped = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        proc.stdin.write(warped.tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"Successfully generated: {out_path}")


def read_all_frames(video_path: str):
    """Reads all frames from a video file into memory."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames


def assemble_master_film(
    v1_path: str,
    v2_path: str,
    v3_path: str,
    v4_path: str,
    out_path: str,
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
    Assembles the 4 stabilized eclipse sequences into the master film
    with customizable transitions between pre-totality, totality, and partial egress.
    """
    n_freeze_a = int(round(freeze_before * fps))
    n_freeze_b = int(round(freeze_after * fps))

    if transition_type in ["fade_to_black", "dip_to_black", "black"]:
        n_fade_out = int(round(fade_out * fps))
        n_black = int(round(black_duration * fps))
        n_fade_in = int(round(fade_in * fps))
        total_trans = n_freeze_a + n_fade_out + n_black + n_fade_in + n_freeze_b
        trans_desc = (
            f"Fade to Black: {freeze_before:.2f}s freeze A + {fade_out:.2f}s fade out "
            f"+ {black_duration:.2f}s black + {fade_in:.2f}s fade in + {freeze_after:.2f}s freeze B"
        )
    elif transition_type == "crossfade":
        n_fade = int(round(fade_duration * fps))
        total_trans = n_freeze_a + n_fade + n_freeze_b
        trans_desc = f"Crossfade: {freeze_before:.2f}s freeze A + {fade_duration:.2f}s dissolve + {freeze_after:.2f}s freeze B"
    else:  # hard
        total_trans = n_freeze_a + n_freeze_b
        trans_desc = f"Hard Cut: {freeze_before:.2f}s freeze A + instant cut + {freeze_after:.2f}s freeze B"

    print("\n" + "=" * 60)
    print(f"Step 5/5: Assembling master film ({total_trans/fps:.2f}s per transition)...")
    print(f"  Type: {trans_desc} ({total_trans} frames)")
    print("=" * 60)

    v1_frames = read_all_frames(v1_path)
    v2_frames = read_all_frames(v2_path)
    v3_frames = read_all_frames(v3_path)
    v4_frames = read_all_frames(v4_path)

    def make_transition(last_frame_a, first_frame_b):
        trans = []
        # 1. Freeze on last frame of clip A
        for _ in range(n_freeze_a):
            trans.append(last_frame_a.copy())

        # 2. Intermediate transition
        if transition_type in ["fade_to_black", "dip_to_black", "black"]:
            black_frame = np.zeros_like(last_frame_a)
            # Fade out to black
            for i in range(n_fade_out):
                alpha = (i + 1) / (n_fade_out + 1)
                blended = cv2.addWeighted(last_frame_a, 1.0 - alpha, black_frame, alpha, 0.0)
                trans.append(blended)
            # Pure black pause
            for _ in range(n_black):
                trans.append(black_frame.copy())
            # Fade in from black
            for i in range(n_fade_in):
                alpha = (i + 1) / (n_fade_in + 1)
                blended = cv2.addWeighted(black_frame, 1.0 - alpha, first_frame_b, alpha, 0.0)
                trans.append(blended)
        elif transition_type == "crossfade":
            for i in range(n_fade):
                alpha = (i + 1) / (n_fade + 1)
                blended = cv2.addWeighted(last_frame_a, 1.0 - alpha, first_frame_b, alpha, 0.0)
                trans.append(blended)
        elif transition_type == "hard":
            pass

        # 3. Freeze on first frame of clip B
        for _ in range(n_freeze_b):
            trans.append(first_frame_b.copy())

        return trans

    master = []
    master.extend(v1_frames)
    master.extend(v2_frames)

    # Transition 1: pre-totality -> totality
    print(f"Generating Transition 1 (pre-totality -> totality, {total_trans} frames)...")
    trans1 = make_transition(v2_frames[-1], v3_frames[0])
    master.extend(trans1)

    # Totality
    master.extend(v3_frames)

    # Transition 2: totality -> partial egress
    print(f"Generating Transition 2 (totality -> partial egress, {total_trans} frames)...")
    trans2 = make_transition(v3_frames[-1], v4_frames[0])
    master.extend(trans2)

    # Partial egress
    master.extend(v4_frames)

    total_frames = len(master)
    total_dur = total_frames / fps
    print(f"\nFinal master sequence: {total_frames} frames ({total_dur:.2f}s = {int(total_dur//60)}m {int(total_dur%60)}s)")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', '1280x720',
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
    for frame in tqdm(master, desc="Encoding final master film"):
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()

    if not os.path.exists(out_path):
        print(f"Error: Output file {out_path} not found.", file=sys.stderr)
        return

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print("\n" + "=" * 60)
    print(f"SUCCESS: Master film generated at: {out_path}")
    print(f"  Resolution: 1280x720")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_dur:.2f}s ({int(total_dur//60)}m {int(total_dur%60)}s)")
    print(f"  File size: {size_mb:.2f} MB")
    print("=" * 60)


def is_valid_video(path: str) -> bool:
    """Checks if a video exists, has valid dimensions, and contains playable frames."""
    if not os.path.exists(path):
        return False
    try:
        cap = cv2.VideoCapture(path)
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        return cnt > 10 and w > 0
    except Exception:
        return False


def resolve_input_file(in_dir: str, primary_name: str, alt_names: list[str]) -> str:
    """Finds the input file using primary name or alternative aliases."""
    for name in [primary_name] + alt_names:
        p = os.path.join(in_dir, name)
        if os.path.exists(p):
            return p
    return os.path.join(in_dir, primary_name)


def build_pipeline(
    transition_type: str = "fade_to_black",
    freeze_before: float = 1.0,
    fade_out: float = 0.5,
    black_duration: float = 0.0,
    fade_in: float = 0.5,
    fade_duration: float = 1.0,
    freeze_after: float = 1.0,
    output_path: str = "040_out/full_eclipse.mp4",
    force_all: bool = False
):
    in_dir = "010_in"
    out_dir = "040_out"
    os.makedirs(out_dir, exist_ok=True)

    partial_ingress_src = resolve_input_file(in_dir, "partial_ingress.mp4", ["partial_in.mp4"])
    pre_totality_src = resolve_input_file(in_dir, "pre_totality.mp4", ["pretotal.mp4"])
    totality_src = resolve_input_file(in_dir, "totality.mp4", ["total.mp4"])
    partial_egress_src = resolve_input_file(in_dir, "partial_egress.mp4", ["partial_out.mp4"])

    partial_ingress_out = os.path.join(out_dir, "partial_ingress.mp4")
    pre_totality_out = os.path.join(out_dir, "pre_totality.mp4")
    totality_out = os.path.join(out_dir, "totality.mp4")
    partial_egress_out = os.path.join(out_dir, "partial_egress.mp4")
    final_film = output_path

    # Step 1: Partial Ingress
    if os.path.exists(partial_ingress_src) and (force_all or not is_valid_video(partial_ingress_out)):
        print("=" * 60)
        print(f"Step 1/5: Stabilizing and centering partial ingress ({os.path.basename(partial_ingress_src)})...")
        print("=" * 60)
        stabilize_video(partial_ingress_src, partial_ingress_out, r_fixed=238.5, crf=16, preset="fast")
    else:
        print(f"Step 1/5: Using cached partial ingress video at {partial_ingress_out}")

    # Step 2: Pre-totality
    if os.path.exists(pre_totality_src) and (force_all or not is_valid_video(pre_totality_out)):
        stabilize_and_accelerate_pre_totality(pre_totality_src, pre_totality_out, target_seconds=30.0, target_fps=30.0, solar_radius=238.0)
    else:
        print(f"Step 2/5: Using cached pre-totality video at {pre_totality_out}")

    # Step 3: Totality (inherits C2 transition alignment)
    if os.path.exists(totality_src) and (force_all or not is_valid_video(totality_out)):
        stabilize_totality(totality_src, totality_out, lunar_radius=246.0, shift_offset=(12.5, 7.5), fps=30.0)
    else:
        print(f"Step 3/5: Using cached totality video at {totality_out}")

    # Step 4: Partial Egress (inherits C3 transition alignment)
    if os.path.exists(partial_egress_src) and (force_all or not is_valid_video(partial_egress_out)):
        print("\n" + "=" * 60)
        print(f"Step 4/5: Stabilizing and centering partial egress ({os.path.basename(partial_egress_src)})...")
        print("=" * 60)
        stabilize_video(partial_egress_src, partial_egress_out, shift_offset=(35.0, 24.0), r_fixed=237.5, crf=16, preset="fast")
    else:
        print(f"Step 4/5: Using cached partial egress video at {partial_egress_out}")

    # Step 5: Master Film Assembly
    assemble_master_film(
        partial_ingress_out, pre_totality_out, totality_out, partial_egress_out,
        final_film,
        transition_type=transition_type,
        freeze_before=freeze_before,
        fade_out=fade_out,
        black_duration=black_duration,
        fade_in=fade_in,
        fade_duration=fade_duration,
        freeze_after=freeze_after,
        fps=30.0, crf=16, preset="fast"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Solar Eclipse 2026 Processing & Assembly Pipeline")
    parser.add_argument("--transition-type", type=str, choices=["fade_to_black", "hard", "crossfade"], default="fade_to_black",
                        help="Transition style: 'fade_to_black' (freeze A + dip to black + freeze B), 'hard' (instant cut with freezes), or 'crossfade' (dissolve with freezes). Default: fade_to_black")
    parser.add_argument("--freeze-before", type=float, default=1.0,
                        help="Duration in seconds of the freeze on the last frame of the preceding clip (default: 1.0s)")
    parser.add_argument("--fade-out", type=float, default=0.5,
                        help="Duration in seconds of the fade out to black (default: 0.5s)")
    parser.add_argument("--black-duration", type=float, default=0.0,
                        help="Duration in seconds of pure black pause (default: 0.0s)")
    parser.add_argument("--fade-in", type=float, default=0.5,
                        help="Duration in seconds of the fade in from black (default: 0.5s)")
    parser.add_argument("--fade-duration", type=float, default=1.0,
                        help="Duration in seconds of crossfade dissolve if transition-type is crossfade (default: 1.0s)")
    parser.add_argument("--freeze-after", type=float, default=1.0,
                        help="Duration in seconds of the freeze on the first frame of the subsequent clip (default: 1.0s)")
    parser.add_argument("--output", "-o", type=str, default="040_out/full_eclipse.mp4",
                        help="Path to final master film (default: 040_out/full_eclipse.mp4)")
    parser.add_argument("--force-all", action="store_true",
                        help="Force re-processing and re-stabilizing all phases from scratch")
    args = parser.parse_args()

    build_pipeline(
        transition_type=args.transition_type,
        freeze_before=args.freeze_before,
        fade_out=args.fade_out,
        black_duration=args.black_duration,
        fade_in=args.fade_in,
        fade_duration=args.fade_duration,
        freeze_after=args.freeze_after,
        output_path=args.output,
        force_all=args.force_all
    )
