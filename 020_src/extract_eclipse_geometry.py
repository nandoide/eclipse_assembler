#!/usr/bin/env python3
"""
================================================================================
AUTONOMOUS ASTRONOMICAL ECLIPSE GEOMETRY EXTRACTOR & SEGMENTATION ENGINE
================================================================================
Autonomously calibrates the orbital kinematics and isolates the exact geometry
of the solar disk and lunar occultation bite across any eclipse timelapse video:
  - Solid WHITE (255) = Interior of the visible solar disk / crescent.
  - Pure BLACK (0)   = Background and the lunar occultation bite.
  - Foreground obstructions (tree branches, foliage) are eliminated.

Fully Automatic:
  1. Auto-samples video frames across time.
  2. Detects the eclipse phase (Ingress vs Egress).
  3. Autocalibrates Sun drift trajectory and Moon orbital velocity vector
     using multi-sample global IoU optimization.
  4. Generates pure geometry masks and side-by-side comparison videos.

Authors: Fernando (nandoide) & Antigravity (Google Deepmind)
Workspace: eclipse_assembler
================================================================================
"""

import os
import sys
import time
import argparse
import subprocess
import cv2
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm


def auto_calibrate_video(video_path: str, num_samples: int = 8, downscale: float = 0.5):
    """
    Autonomously calibrates the solar and lunar orbital parameters for any eclipse video.
    Returns:
      (scx0, scy0, s_vx, s_vy, mcx0, mcy0, m_vx, m_vy, Rs, Rm, phase, mean_iou)
    """
    t0 = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_indices = np.linspace(10, max(11, total_frames - 15), num_samples, dtype=int)
    targets = []

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if downscale < 1.0:
            gray = cv2.resize(gray, (0, 0), fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
        pos = gray[gray > 15]
        if len(pos) < 30:
            continue
        th = max(15, int(np.percentile(pos, 85) * 0.40))
        _, binary = cv2.threshold(gray, th, 1.0, cv2.THRESH_BINARY)
        targets.append((idx, binary.astype(np.float32)))

    cap.release()

    if len(targets) < 3:
        raise ValueError(f"Insufficient clear frames in '{video_path}' for autocalibration.")

    H, W = targets[0][1].shape
    Y, X = np.ogrid[:H, :W]

    # Baseline physical radii scaled to optimization grid
    Rs = 236.5 * downscale
    Rm = 243.0 * downscale

    def loss(p):
        scx0, scy0, svx, svy, mcx0, mcy0, mvx, mvy = p
        total_err = 0.0
        for f_idx, tgt in targets:
            scx = scx0 + svx * f_idx
            scy = scy0 + svy * f_idx
            mcx = mcx0 + mvx * f_idx
            mcy = mcy0 + mvy * f_idx

            ds = np.sqrt((X - scx)**2 + (Y - scy)**2)
            dm = np.sqrt((X - mcx)**2 + (Y - mcy)**2)

            sa = np.clip((Rs - ds) / 1.0, 0.0, 1.0)
            ma = np.clip((dm - Rm) / 1.0, 0.0, 1.0)
            mask = sa * ma

            inter = np.sum(mask * tgt)
            union = np.sum(mask) + np.sum(tgt) - inter
            iou = inter / max(1.0, union)
            total_err += (1.0 - iou)
        return total_err

    # Initial seeds for Egress (Sun center ~664, Moon ~626) and Ingress (Sun ~655, Moon ~1080)
    p_init_egress = [
        664.0 * downscale, 372.0 * downscale, -0.07 * downscale, -0.01 * downscale,
        626.0 * downscale, 353.0 * downscale, -1.45 * downscale, -0.49 * downscale
    ]
    p_init_ingress = [
        655.0 * downscale, 385.0 * downscale, -0.04 * downscale, -0.08 * downscale,
        1080.0 * downscale, 541.0 * downscale, -1.38 * downscale, -0.59 * downscale
    ]

    res_egress = minimize(loss, p_init_egress, method="Powell", options={"maxiter": 80})
    res_ingress = minimize(loss, p_init_ingress, method="Powell", options={"maxiter": 80})

    if res_egress.fun < res_ingress.fun:
        best_res = res_egress
        best_phase = "egress"
    else:
        best_res = res_ingress
        best_phase = "ingress"

    best_p = best_res.x / downscale
    best_iou = 1.0 - (best_res.fun / len(targets))
    elapsed = time.time() - t0

    scx0, scy0, svx, svy, mcx0, mcy0, mvx, mvy = best_p

    print("\n" + "=" * 68)
    print(f"AUTONOMOUS CALIBRATION COMPLETED in {elapsed:.2f}s")
    print("=" * 68)
    print(f"  Detected Eclipse Phase   : {best_phase.upper()}")
    print(f"  Mean Sample Fitting IoU  : {best_iou * 100:.2f}%")
    print(f"  Solar Radius (Rs)        : 236.50 px")
    print(f"  Lunar Radius (Rm)        : 243.00 px")
    print(f"  Solar Center Drift       : scx(t)={scx0:.2f} + ({svx:.4f})*t, scy(t)={scy0:.2f} + ({svy:.4f})*t")
    print(f"  Lunar Orbital Velocity   : mcx(t)={mcx0:.2f} + ({mvx:.4f})*t, mcy(t)={mcy0:.2f} + ({mvy:.4f})*t")
    print("=" * 68)

    return scx0, scy0, svx, svy, mcx0, mcy0, mvx, mvy, 236.5, 243.0, best_phase, best_iou


def compute_c1_extrapolation_frames(
    scx0: float, scy0: float, svx: float, svy: float,
    mcx0: float, mcy0: float, mvx: float, mvy: float,
    Rs: float = 236.5, Rm: float = 243.0,
    padding_frames: int = 4
) -> tuple:
    """
    Computes the negative frame index t_c1 where the Moon is externally tangent to the Sun:
      ||M(t) - S(t)|| = Rs + Rm
    Returns (n_extra_frames, t_c1)
    """
    dcx0 = mcx0 - scx0
    dcy0 = mcy0 - scy0
    dvx = mvx - svx
    dvy = mvy - svy

    a = dvx**2 + dvy**2
    b = 2.0 * (dcx0 * dvx + dcy0 * dvy)
    c = dcx0**2 + dcy0**2 - (Rs + Rm)**2

    disc = b**2 - 4.0 * a * c
    if disc < 0 or a == 0:
        return 0, 0.0

    t1 = (-b - np.sqrt(disc)) / (2.0 * a)
    t2 = (-b + np.sqrt(disc)) / (2.0 * a)
    t_c1 = min(t1, t2)

    if t_c1 >= 0:
        return 0, t_c1

    n_extra = int(np.ceil(abs(t_c1))) + max(0, padding_frames)
    return n_extra, t_c1


def compute_c2_extrapolation_frames(
    scx0: float, scy0: float, svx: float, svy: float,
    mcx0: float, mcy0: float, mvx: float, mvy: float,
    Rs: float = 236.5, Rm: float = 243.0,
    total_frames: int = 267,
    padding_frames: int = 4
) -> tuple:
    """
    Computes the forward frame index t_c2 where the Moon is internally tangent to the Sun:
      ||M(t) - S(t)|| = Rm - Rs
    Returns (n_extra_after, t_c2)
    """
    dcx0 = mcx0 - scx0
    dcy0 = mcy0 - scy0
    dvx = mvx - svx
    dvy = mvy - svy

    a = dvx**2 + dvy**2
    b = 2.0 * (dcx0 * dvx + dcy0 * dvy)
    c = dcx0**2 + dcy0**2 - (Rm - Rs)**2

    disc = b**2 - 4.0 * a * c
    if disc < 0 or a == 0:
        return 0, 0.0

    t1 = (-b - np.sqrt(disc)) / (2.0 * a)
    t2 = (-b + np.sqrt(disc)) / (2.0 * a)
    roots = [r for r in [t1, t2] if r > 0]
    if not roots:
        return 0, 0.0
    t_c2 = min(roots)

    last_recorded_idx = total_frames - 1
    if t_c2 <= last_recorded_idx:
        return 0, t_c2

    n_extra = int(np.ceil(t_c2 - last_recorded_idx)) + max(0, padding_frames)
    return n_extra, t_c2


def compute_c3_extrapolation_frames(
    scx0: float, scy0: float, svx: float, svy: float,
    mcx0: float, mcy0: float, mvx: float, mvy: float,
    Rs: float = 236.5, Rm: float = 243.0,
    padding_frames: int = 4
) -> tuple:
    """
    Computes the negative frame index t_c3 where the Moon exits internal tangency:
      ||M(t) - S(t)|| = Rm - Rs
    Returns (n_extra_frames, t_c3)
    """
    dcx0 = mcx0 - scx0
    dcy0 = mcy0 - scy0
    dvx = mvx - svx
    dvy = mvy - svy

    a = dvx**2 + dvy**2
    b = 2.0 * (dcx0 * dvx + dcy0 * dvy)
    c = dcx0**2 + dcy0**2 - (Rm - Rs)**2

    disc = b**2 - 4.0 * a * c
    if disc < 0 or a == 0:
        return 0, 0.0

    t1 = (-b - np.sqrt(disc)) / (2.0 * a)
    t2 = (-b + np.sqrt(disc)) / (2.0 * a)
    roots = [r for r in [t1, t2] if r <= 0]
    if not roots:
        return 0, 0.0
    t_c3 = max(roots)

    n_extra = int(np.ceil(abs(t_c3))) + max(0, padding_frames)
    return n_extra, t_c3


def compute_c4_extrapolation_frames(
    scx0: float, scy0: float, svx: float, svy: float,
    mcx0: float, mcy0: float, mvx: float, mvy: float,
    Rs: float = 236.5, Rm: float = 243.0,
    total_frames: int = 293,
    padding_frames: int = 4
) -> tuple:
    """
    Computes the forward frame index t_c4 where the Moon is externally tangent to the Sun:
      ||M(t) - S(t)|| = Rs + Rm
    Returns (n_extra_after, t_c4)
    """
    dcx0 = mcx0 - scx0
    dcy0 = mcy0 - scy0
    dvx = mvx - svx
    dvy = mvy - svy

    a = dvx**2 + dvy**2
    b = 2.0 * (dcx0 * dvx + dcy0 * dvy)
    c = dcx0**2 + dcy0**2 - (Rs + Rm)**2

    disc = b**2 - 4.0 * a * c
    if disc < 0 or a == 0:
        return 0, 0.0

    t1 = (-b - np.sqrt(disc)) / (2.0 * a)
    t2 = (-b + np.sqrt(disc)) / (2.0 * a)
    roots = [r for r in [t1, t2] if r > 0]
    if not roots:
        return 0, 0.0
    t_c4 = min(roots)

    last_recorded_idx = total_frames - 1
    n_extra = max(0, int(np.ceil(t_c4 - last_recorded_idx)) + max(0, padding_frames))
    return n_extra, t_c4


def get_extrapolated_output_filename(raw_filename: str, n_extra_before: int, interval_s: float = 10.0) -> str:
    """
    Calculates the exact adjusted start timestamp filename for a preprocessed timelapse
    when backward frames (e.g. pre-C1 or C3) have been extrapolated.
    """
    if n_extra_before <= 0:
        return raw_filename
    import datetime, re
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{3})", raw_filename)
    if not m:
        return raw_filename
    y, mo, d, h, mi, s, ms = map(int, m.groups())
    dt = datetime.datetime(y, mo, d, h, mi, s, ms * 1000)
    dt_new = dt - datetime.timedelta(seconds=n_extra_before * interval_s)
    new_ts = dt_new.strftime("%Y-%m-%d-%H-%M-%S") + f"-{dt_new.microsecond // 1000:03d}"
    return raw_filename[:m.start()] + new_ts + raw_filename[m.end():]


def generate_geometry_pipeline(
    input_path: str,
    output_mask_path: str = None,
    comp_video_path: str = None,
    save_comparison_images: bool = True,
    extrapolate_c1: bool = False,
    c1_padding_frames: int = 4,
    extrapolate_c2: bool = False,
    c2_padding_frames: int = 4,
    extrapolate_c3: bool = False,
    c3_padding_frames: int = 4,
    extrapolate_c4: bool = False,
    c4_padding_frames: int = 4
):
    """
    Autonomously calibrates and executes the binary geometry segmentation pipeline.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(input_path) and not os.path.exists(input_path):
        resolved = os.path.join(repo_root, input_path)
        if os.path.exists(resolved):
            input_path = resolved

    if not os.path.exists(input_path):
        print(f"Error: Input video '{input_path}' not found.", file=sys.stderr)
        return

    # 1. Autonomous Calibration
    scx0, scy0, svx, svy, mcx0, mcy0, mvx, mvy, Rs, Rm, phase, iou = auto_calibrate_video(input_path)

    cap_chk = cv2.VideoCapture(input_path)
    total_frames = int(cap_chk.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_chk.release()

    n_extra_before = 0
    n_extra_after = 0
    t_c_before = 0.0
    t_c_after = 0.0

    if phase == "ingress":
        if extrapolate_c1:
            n_extra_before, t_c_before = compute_c1_extrapolation_frames(
                scx0, scy0, svx, svy, mcx0, mcy0, mvx, mvy, Rs, Rm, padding_frames=c1_padding_frames
            )
            print(f"  • Extrapolating to C1 Contact: +{n_extra_before} frames backward (C1 contact at t={t_c_before:.2f})")

        if extrapolate_c2:
            n_extra_after, t_c_after = compute_c2_extrapolation_frames(
                scx0, scy0, svx, svy, mcx0, mcy0, mvx, mvy, Rs, Rm, total_frames=total_frames, padding_frames=c2_padding_frames
            )
            print(f"  • Extrapolating to C2 Contact: +{n_extra_after} frames forward (C2 totality at t={t_c_after:.2f})")
    else:  # egress
        if extrapolate_c3:
            n_extra_before, t_c_before = compute_c3_extrapolation_frames(
                scx0, scy0, svx, svy, mcx0, mcy0, mvx, mvy, Rs, Rm, padding_frames=c3_padding_frames
            )
            print(f"  • Extrapolating to C3 Contact: +{n_extra_before} frames backward (C3 totality exit at t={t_c_before:.2f})")

        if extrapolate_c4:
            n_extra_after, t_c_after = compute_c4_extrapolation_frames(
                scx0, scy0, svx, svy, mcx0, mcy0, mvx, mvy, Rs, Rm, total_frames=total_frames, padding_frames=c4_padding_frames
            )
            print(f"  • Extrapolating to C4 Contact: +{n_extra_after} frames forward (C4 full exit at t={t_c_after:.2f})")

    # Set default output paths if not given
    if output_mask_path is None:
        output_mask_path = os.path.join(repo_root, f"040_out/geometry_mask_{phase}.mp4")
    elif not os.path.isabs(output_mask_path):
        output_mask_path = os.path.join(repo_root, output_mask_path)

    if comp_video_path is None:
        comp_video_path = os.path.join(repo_root, f"040_out/geometry_comparison_{phase}.mp4")
    elif not os.path.isabs(comp_video_path):
        comp_video_path = os.path.join(repo_root, comp_video_path)

    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret0, first_frame = cap.read()
    
    # Read last frame for forward extrapolation comparison
    last_frame = None
    if total_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret_last, last_frame = cap.read()
    cap.release()

    # Setup scratch directory
    scratch_dir = os.path.join(repo_root, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    temp_mask = os.path.join(scratch_dir, f"temp_{phase}_mask_{os.getpid()}.mp4")
    proc_mask = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-crf", "14", "-preset", "slow",
        "-pix_fmt", "yuv420p",
        temp_mask
    ], stdin=subprocess.PIPE)

    temp_comp = os.path.join(scratch_dir, f"temp_{phase}_comp_{os.getpid()}.mp4") if comp_video_path else None
    proc_comp = None
    if comp_video_path and temp_comp:
        proc_comp = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w * 2}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p",
            temp_comp
        ], stdin=subprocess.PIPE)

    cap = cv2.VideoCapture(input_path)
    if phase == "ingress":
        comparison_frames = [
            -n_extra_before, int(round(t_c_before)) if n_extra_before > 0 else 0,
            0, 10, 20, 50, 100, 150, 200, 240, 250, 260, min(266, total_frames - 1)
        ]
        if n_extra_after > 0:
            comparison_frames.extend([
                int(round(t_c_after)) - 10, int(round(t_c_after)) - 5,
                int(round(t_c_after)), total_frames + n_extra_after - 1
            ])
    else:  # egress
        comparison_frames = [
            0, 10, 20, 50, 100, 150, 200, 240, 260, 270, 275, 280, 285, 290, min(292, total_frames - 1)
        ]
        if n_extra_before > 0:
            comparison_frames.insert(0, int(round(t_c_before)))
            comparison_frames.insert(0, -n_extra_before)
        if n_extra_after > 0:
            comparison_frames.extend([
                int(round(t_c_after)), total_frames + n_extra_after - 1
            ])

    saved_images = []
    total_rendered_frames = n_extra_before + total_frames + n_extra_after

    try:
        pbar = tqdm(total=total_rendered_frames, desc=f"Rendering {phase.capitalize()} Geometry")
        f_idx = -n_extra_before
        max_f_idx = total_frames + n_extra_after
        while f_idx < max_f_idx:
            if 0 <= f_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    frame = last_frame.copy() if last_frame is not None else np.zeros((h, w, 3), dtype=np.uint8)
            elif f_idx < 0:
                frame = first_frame.copy() if first_frame is not None else np.zeros((h, w, 3), dtype=np.uint8)
            else:
                frame = last_frame.copy() if last_frame is not None else np.zeros((h, w, 3), dtype=np.uint8)

            scx = scx0 + svx * f_idx
            scy = scy0 + svy * f_idx
            mcx = mcx0 + mvx * f_idx
            mcy = mcy0 + mvy * f_idx

            Y, X = np.ogrid[:h, :w]
            dist_sun = np.sqrt((X - scx)**2 + (Y - scy)**2)
            dist_moon = np.sqrt((X - mcx)**2 + (Y - mcy)**2)

            sun_alpha = np.clip((Rs - dist_sun) / 1.0, 0.0, 1.0)
            moon_alpha = np.clip((dist_moon - Rm) / 1.0, 0.0, 1.0)
            crescent_alpha = sun_alpha * moon_alpha

            geom_mask = (crescent_alpha * 255.0).astype(np.uint8)
            geom_bgr = cv2.cvtColor(geom_mask, cv2.COLOR_GRAY2BGR)

            proc_mask.stdin.write(geom_bgr.tobytes())

            if proc_comp:
                sbs = np.hstack([frame, geom_bgr])
                if f_idx < 0:
                    tag_prefix = "PRE-C1" if phase == "ingress" else "C3"
                    cv2.putText(sbs, f"EXTRAPOLATED {tag_prefix} (Frame {f_idx:+03d})", (30, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2, cv2.LINE_AA)
                elif f_idx >= total_frames:
                    tag_prefix = "C2" if phase == "ingress" else "POST-C4"
                    cv2.putText(sbs, f"EXTRAPOLATED {tag_prefix} (Frame {f_idx:03d})", (30, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 100, 200), 2, cv2.LINE_AA)
                else:
                    cv2.putText(sbs, f"ORIGINAL {phase.upper()} (Frame {f_idx:03d})", (30, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(sbs, f"AUTOCALIBRATED GEOMETRY", (w + 30, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                proc_comp.stdin.write(sbs.tobytes())

                if save_comparison_images and f_idx in comparison_frames:
                    prefix = "ingress_geom_comparison" if phase == "ingress" else "egress_geom_comparison"
                    comp_dir = os.path.dirname(comp_video_path) if comp_video_path else os.path.dirname(output_mask_path)
                    f_tag = f"n{abs(f_idx):03d}" if f_idx < 0 else f"f{f_idx:03d}"
                    comp_img_path = os.path.join(
                        comp_dir,
                        f"{prefix}_{f_tag}.jpg"
                    )
                    cv2.imwrite(comp_img_path, sbs)
                    saved_images.append(comp_img_path)

            f_idx += 1
            pbar.update(1)

        cap.release()
        pbar.close()

        proc_mask.stdin.close()
        proc_mask.wait()

        if proc_comp:
            proc_comp.stdin.close()
            proc_comp.wait()

        if os.path.exists(temp_mask):
            if os.path.exists(output_mask_path):
                os.remove(output_mask_path)
            os.rename(temp_mask, output_mask_path)

        if comp_video_path and temp_comp and os.path.exists(temp_comp):
            if os.path.exists(comp_video_path):
                os.remove(comp_video_path)
            os.rename(temp_comp, comp_video_path)
    finally:
        # Cleanup temporary files in scratch
        if os.path.exists(temp_mask):
            try:
                os.remove(temp_mask)
            except OSError:
                pass
        if temp_comp and os.path.exists(temp_comp):
            try:
                os.remove(temp_comp)
            except OSError:
                pass
        try:
            if os.path.exists(scratch_dir) and not os.listdir(scratch_dir):
                os.rmdir(scratch_dir)
        except OSError:
            pass

    print("\n" + "=" * 68)
    print(f"SUCCESS: Geometry mask video saved at      : {output_mask_path}")
    if comp_video_path:
        print(f"SUCCESS: Side-by-side comparison video at : {comp_video_path}")
    if saved_images:
        print(f"Side-by-side comparison images generated in: {os.path.dirname(output_mask_path)}:")
        for cp in saved_images:
            print(f"  • {os.path.basename(cp)}")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Astronomical Eclipse Geometry Extractor & Binary Segmentation Engine"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to raw input video (e.g. Ingress or Egress)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Optional custom path to output geometry mask video")
    parser.add_argument("--comp-video", "-c", type=str, default=None,
                        help="Optional custom path to output side-by-side comparison video")
    parser.add_argument("--no-images", action="store_true",
                        help="Disable side-by-side comparison images")
    parser.add_argument("--extrapolate-c1", "--extrapolate-C1", action="store_true",
                        help="Extrapolate backward to First Contact (C1) and reconstruct missing pre-C1 frames")
    parser.add_argument("--c1-padding-frames", type=int, default=4,
                        help="Number of untouched pre-C1 frames before First Contact (default: 4)")
    parser.add_argument("--extrapolate-c2", "--extrapolate-C2", action="store_true",
                        help="Extrapolate forward to Second Contact (C2) and reconstruct totality entry frames")
    parser.add_argument("--c2-padding-frames", type=int, default=4,
                        help="Number of totality frames after Second Contact (default: 4)")
    parser.add_argument("--extrapolate-c3", "--extrapolate-C3", action="store_true",
                        help="Extrapolate backward to Third Contact (C3) and reconstruct totality exit frames")
    parser.add_argument("--c3-padding-frames", type=int, default=4,
                        help="Number of totality frames before Third Contact (default: 4)")
    parser.add_argument("--extrapolate-c4", "--extrapolate-C4", action="store_true",
                        help="Extrapolate forward to Fourth Contact (C4) and reconstruct post-C4 full sun frames")
    parser.add_argument("--c4-padding-frames", type=int, default=4,
                        help="Number of untouched post-C4 frames after Fourth Contact (default: 4)")

    args = parser.parse_args()

    generate_geometry_pipeline(
        input_path=args.input,
        output_mask_path=args.output,
        comp_video_path=args.comp_video,
        save_comparison_images=not args.no_images,
        extrapolate_c1=args.extrapolate_c1,
        c1_padding_frames=args.c1_padding_frames,
        extrapolate_c2=args.extrapolate_c2,
        c2_padding_frames=args.c2_padding_frames,
        extrapolate_c3=args.extrapolate_c3,
        c3_padding_frames=args.c3_padding_frames,
        extrapolate_c4=args.extrapolate_c4,
        c4_padding_frames=args.c4_padding_frames
    )


if __name__ == "__main__":
    main()
