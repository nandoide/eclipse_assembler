#!/usr/bin/env python3
"""
Camera Dive & Zoom Transition Engine for 'Art' Style Eclipse Layout
===================================================================
Generates seamless, continuous camera zoom-in and zoom-out transitions
between high-resolution solar eclipse composite mosaic artwork and
the celestial telescope video footage.

Key Features:
  - Perceptually linear exponential scale interpolation (log-scale zoom).
  - Quintic ease-in-out acceleration curve for cinematic smoothness.
  - Subpixel affine transformation with Lanczos-4 sinc resampling.
  - Soft boundary crossfade matching the first/last telescope video frame.
"""

import os
import sys
import json
import math
import cv2
import numpy as np


def quintic_ease_in_out(t: float) -> float:
    """Smoothest 5th-order polynomial ease-in-out curve with zero jerk at boundaries."""
    t = float(np.clip(t, 0.0, 1.0))
    return 6.0 * (t ** 5) - 15.0 * (t ** 4) + 10.0 * (t ** 3)


def cosine_ease_in_out(t: float) -> float:
    """Classic cosine smoothstep ease-in-out curve."""
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 * (1.0 - math.cos(math.pi * t))


def generate_camera_dive_clip(
    composite_img_path: str,
    composite_meta_path: str,
    target_sample_idx: int,  # 0 for first sample (zoom-in), -1 for last sample (zoom-out)
    output_video_path: str,
    direction: str = "zoom_in",  # "zoom_in" or "zoom_out"
    hold_duration: float = 2.0,   # Seconds holding the full composite
    dive_duration: float = 3.0,   # Seconds during continuous zoom movement
    fps: float = 30.0,
    out_width: int = 1280,
    out_height: int = 720,
    target_disk_diameter_px: float = 500.0,
    crossfade_tail_s: float = 0.6,
    reference_video_frame: np.ndarray = None  # Exact telescope frame to dissolve into
):
    """
    Renders a continuous zoom video clip between full composite overview and
    a specific solar disk sample.
    """
    if not os.path.exists(composite_img_path):
        raise FileNotFoundError(f"Composite image not found: {composite_img_path}")
    if not os.path.exists(composite_meta_path):
        raise FileNotFoundError(f"Composite metadata JSON not found: {composite_meta_path}")

    # 1. Load composite image and metadata
    comp_img = cv2.imread(composite_img_path, cv2.IMREAD_COLOR)
    if comp_img is None:
        raise ValueError(f"Could not read composite image at {composite_img_path}")

    comp_h, comp_w = comp_img.shape[:2]

    with open(composite_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    samples = meta.get("samples", [])
    if not samples:
        raise ValueError(f"No samples found in composite metadata: {composite_meta_path}")

    sample = samples[target_sample_idx]
    orig_w = meta.get("width", comp_w)
    orig_h = meta.get("height", comp_h)

    scale_x = comp_w / float(orig_w)
    scale_y = comp_h / float(orig_h)

    # Sample position on composite canvas
    sample_cx = float(sample.get("center_x", orig_w / 2.0)) * scale_x
    sample_cy = float(sample.get("center_y", orig_h / 2.0)) * scale_y
    sample_d = float(sample.get("disk_diameter", 60.0)) * ((scale_x + scale_y) / 2.0)

    # 2. Compute Camera Viewport States
    # State A: Full composite overview centered in 16:9 output canvas
    scale_fit_w = out_width / float(comp_w)
    scale_fit_h = out_height / float(comp_h)
    s_full = min(scale_fit_w, scale_fit_h)
    cx_full = comp_w / 2.0
    cy_full = comp_h / 2.0

    # State B: Magnified camera centered on sample disk with target solar disk diameter
    s_sample = target_disk_diameter_px / max(sample_d, 1.0)
    cx_sample = sample_cx
    cy_sample = sample_cy

    # 3. Time parameterization
    n_hold = int(round(hold_duration * fps))
    n_dive = int(round(dive_duration * fps))
    total_frames = n_hold + n_dive

    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

    for frame_idx in range(total_frames):
        if direction == "zoom_in":
            if frame_idx < n_hold:
                # Holding full overview
                tau = 0.0
            else:
                # Diving in
                tau = (frame_idx - n_hold) / float(max(n_dive - 1, 1))
        else:  # "zoom_out"
            if frame_idx < n_dive:
                # Diving out from sample to full overview
                tau = 1.0 - (frame_idx / float(max(n_dive - 1, 1)))
            else:
                # Holding full overview at the end
                tau = 0.0

        ease = quintic_ease_in_out(tau)

        # Perceptual logarithmic scale zoom
        log_s_full = math.log(s_full)
        log_s_sample = math.log(s_sample)
        curr_log_s = (1.0 - ease) * log_s_full + ease * log_s_sample
        curr_s = math.exp(curr_log_s)

        # Center position
        curr_cx = (1.0 - ease) * cx_full + ease * cx_sample
        curr_cy = (1.0 - ease) * cy_full + ease * cy_sample

        # Affine Matrix: Map (curr_cx, curr_cy) on composite to (out_width/2, out_height/2)
        tx = (out_width / 2.0) - curr_s * curr_cx
        ty = (out_height / 2.0) - curr_s * curr_cy

        M = np.array([
            [curr_s, 0.0, tx],
            [0.0, curr_s, ty]
        ], dtype=np.float32)

        warped = cv2.warpAffine(
            comp_img, M, (out_width, out_height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # Crossfade tail with reference video frame if provided
        if reference_video_frame is not None:
            n_cross = int(round(crossfade_tail_s * fps))
            if direction == "zoom_in":
                # Tail end of dive dissolves into reference frame
                if frame_idx >= (total_frames - n_cross):
                    alpha = (frame_idx - (total_frames - n_cross)) / float(max(n_cross - 1, 1))
                    alpha_e = cosine_ease_in_out(alpha)
                    ref_resized = cv2.resize(reference_video_frame, (out_width, out_height), interpolation=cv2.INTER_LANCZOS4)
                    warped = cv2.addWeighted(warped, 1.0 - alpha_e, ref_resized, alpha_e, 0.0)
            else:  # zoom_out
                # Start of dive dissolves from reference frame
                if frame_idx < n_cross:
                    alpha = 1.0 - (frame_idx / float(max(n_cross - 1, 1)))
                    alpha_e = cosine_ease_in_out(alpha)
                    ref_resized = cv2.resize(reference_video_frame, (out_width, out_height), interpolation=cv2.INTER_LANCZOS4)
                    warped = cv2.addWeighted(warped, 1.0 - alpha_e, ref_resized, alpha_e, 0.0)

        writer.write(warped)

    writer.release()
    print(f"Generated camera dive ({direction}): {output_video_path} ({total_frames} frames, {total_frames/fps:.2f}s)")
    return output_video_path


def extract_first_frame(video_path: str) -> np.ndarray:
    """Extracts first valid frame from video."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return frame


def extract_last_frame(video_path: str) -> np.ndarray:
    """Extracts last valid frame from video."""
    cap = cv2.VideoCapture(video_path)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_f > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_f - 1)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    return frame
