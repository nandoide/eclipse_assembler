#!/usr/bin/env python3
"""
Solar Eclipse Video Stabilization & Centering Tool
===================================================
Stabilizes solar eclipse time-lapse videos affected by wind oscillation, telescope jitter,
tracking drift, and eclipse phase progression.

Algorithm:
1. Adaptive Outer Solar Limb Segmentation:
   Extracts exclusively the convex outer boundary points of the Sun facing the sky,
   strictly ignoring the inner concave lunar arc, faint corona flares, and chords across cusps.
2. Exact Geometric Center via Fixed-Radius Circle Fitting:
   For every frame, finds the exact solar center (cx, cy) by non-linear least squares
   optimization against the true solar radius (R ≈ 237.5 - 238.5 px), keeping the solar disk
   fixed at the image center (640, 360) across all eclipse phases with zero drift.
3. Lanczos-4 Spatial Warping & HEVC/H.265 Direct Video Encoding.

Authors: Fernando (nandoide) & Antigravity (Google Gemini 3.6 Flash High)
Workspace: eclipse_assembler
"""

import argparse
import glob
import os
import subprocess
import sys
import numpy as np
import cv2
from scipy.optimize import minimize
from tqdm import tqdm


def extract_pure_solar_limb(gray_img: np.ndarray, thresh_val: int = None) -> np.ndarray:
    """
    Extracts purely the convex outer boundary points of the Sun facing the black sky.
    Excludes the concave lunar arc, faint residual corona halo, and the chord bridging the two cusps.
    """
    if thresh_val is None:
        max_val = float(np.max(gray_img))
        thresh_val = max(20, min(80, int(0.35 * max_val)))

    _, thresh = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)

    cnt = max(contours, key=cv2.contourArea)
    cnt_pts = cnt.reshape(-1, 2)
    hull = cv2.convexHull(cnt, returnPoints=True).reshape(-1, 2)

    # Filter hull points that lie on the true contour
    tree = cv2.BFMatcher(cv2.NORM_L2)
    matches = tree.match(hull.astype(np.float32), cnt_pts.astype(np.float32))

    limb_pts = []
    for m in matches:
        if m.distance < 1.5:
            limb_pts.append(hull[m.queryIdx])

    if len(limb_pts) < 5:
        return hull.astype(np.float32)

    return np.array(limb_pts, dtype=np.float32)


def find_circle_center_fixed_r(pts: np.ndarray, r_fixed: float = 237.5, ry_fixed: float = None) -> tuple[float, float]:
    """
    Finds the exact (cx, cy) center on circular or oblate arc points given fixed semi-axes.
    Minimizes trimmed Huber loss to reject cloud artifacts and chord outliers.
    """
    if len(pts) < 5:
        return 640.0, 360.0

    if ry_fixed is None:
        ry_fixed = r_fixed

    best_c = (640.0, 360.0)
    curr_pts = pts
    for iter_step in range(3):
        def loss(c):
            cx, cy = c
            d = np.sqrt(((curr_pts[:, 0] - cx) / r_fixed)**2 + ((curr_pts[:, 1] - cy) / ry_fixed)**2)
            err = np.abs(d - 1.0) * r_fixed
            return np.sum(np.where(err < 2.0, 0.5 * err**2, 2.0 * (err - 1.0)))

        opt = minimize(loss, [best_c[0], best_c[1]], method='Nelder-Mead', options={'xatol': 0.0005, 'fatol': 0.005})
        if opt.success:
            best_c = (float(opt.x[0]), float(opt.x[1]))
            d = np.sqrt(((curr_pts[:, 0] - best_c[0]) / r_fixed)**2 + ((curr_pts[:, 1] - best_c[1]) / ry_fixed)**2)
            err = np.abs(d - 1.0) * r_fixed
            inliers = err < 2.5
            if np.sum(inliers) >= 15:
                curr_pts = curr_pts[inliers]

    return best_c[0], best_c[1]


def track_video_trajectory(video_path: str, r_fixed: float = 237.5):
    """
    Tracks the exact solar center (cx, cy) for every frame with oblate atmospheric correction.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    centers = []

    print(f"Tracking solar center across {total_frames} frames in {video_path}...")

    for idx in tqdm(range(total_frames), desc="Analyzing solar limb"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        max_val = float(np.max(gray))
        
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
        
        progress = idx / float(max(1, total_frames - 1))
        ry_target = r_fixed * (1.0 - 0.042 * progress)
        
        if all_pts:
            pts = np.vstack(all_pts)
            cx, cy = find_circle_center_fixed_r(pts, r_fixed=r_fixed, ry_fixed=ry_target)
            centers.append((cx, cy))
        else:
            centers.append(centers[-1] if centers else (640.0, 360.0))

    cap.release()
    return frames, np.array(centers, dtype=np.float32)


def stabilize_video(
    input_path: str,
    output_path: str,
    target_center=(640.0, 360.0),
    shift_offset=(0.0, 0.0),
    r_fixed: float = 237.5,
    crf: int = 16,
    preset: str = "slow"
):
    """
    Stabilizes and centers a video file, encoding to FFmpeg HEVC/H.265.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    frames, centers = track_video_trajectory(input_path, r_fixed=r_fixed)
    total_frames = len(frames)

    print(f"Encoding stabilized video to {output_path} (HEVC/H.265, CRF {crf}, {w}x{h} @ {fps:.2f} fps)...")

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'bgr24',
        '-r', f'{fps}',
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

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    t_cx, t_cy = target_center
    off_x, off_y = shift_offset

    for i in tqdm(range(total_frames), desc="Warping & Encoding"):
        frame = frames[i]
        c_x, c_y = centers[i]
        dx = t_cx - c_x + off_x
        dy = t_cy - c_y + off_y

        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        warped = cv2.warpAffine(
            frame,
            M,
            (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        proc.stdin.write(warped.tobytes())

    proc.stdin.close()
    proc.wait()

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Successfully created: {output_path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Solar Eclipse Video Stabilization & Centering Tool")
    parser.add_argument("--input", "-i", default="010_in", help="Input file or directory (default: 010_in)")
    parser.add_argument("--output", "-o", default="040_out", help="Output file or directory (default: 040_out)")
    parser.add_argument("--radius", "-r", type=float, default=237.5, help="Fixed solar radius (default: 237.5)")
    parser.add_argument("--crf", type=int, default=16, help="HEVC CRF quality parameter (default: 16)")
    parser.add_argument("--preset", default="slow", help="x265 preset (default: slow)")

    args = parser.parse_args()

    if os.path.isfile(args.input):
        stabilize_video(args.input, args.output, r_fixed=args.radius, crf=args.crf, preset=args.preset)
    else:
        video_files = sorted(glob.glob(os.path.join(args.input, "*.mp4")))
        for v in video_files:
            bname = os.path.basename(v)
            out_v = os.path.join(args.output, bname)
            stabilize_video(v, out_v, r_fixed=args.radius, crf=args.crf, preset=args.preset)


if __name__ == "__main__":
    main()
