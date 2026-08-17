#!/usr/bin/env python3
"""
=============================================================================
TOTALITY HDR MULTI-EXPOSURE COMPOSITE ENGINE & AI PROMPT ADAPTER
=============================================================================
Extracts key temporal phases of solar totality from video footage, computes
an authentic high-fidelity local HDR composite via OpenCV using Druckmüller
radial normalization, ruby H-alpha prominence enhancement, discrete PSF pearl
bead rendering, and pure black lunar masking, and dynamically constructs an
optimized, feature-adapted prompt for web-based multi-modal AI generation
(Nano Banana / Gemini / ChatGPT).

Key Phases Extracted:
  1. Ingress Baily's Beads (t ≈ 0.8s)  - Diamond sparks & western limb
  2. C2 Chromosphere & Prominence (t ≈ 8.0s) - Ruby H-alpha western flare
  3. Mid-Totality Corona (t ≈ 51.7s)   - Soft, natural dipolar solar corona
  4. C3 Chromosphere & Prominence (t ≈ 98.0s) - Carmine eastern limb features
  5. Egress Baily's Beads (t ≈ 102.2s) - Sparkling diamond necklace along southeast limb

Author: nandoide / eclipse_assembler
=============================================================================
"""

import os
import sys
import argparse
import cv2
import numpy as np
from scipy.signal import find_peaks


def fit_lunar_limb_raytracing(
    img: np.ndarray,
    approx_center: tuple = (640.0, 360.0),
    r_min: float = 180.0,
    r_max: float = 320.0,
    num_rays: int = 360,
    num_samples_per_ray: int = 200
) -> tuple:
    """Finds the sub-pixel center (cx, cy) and radius r using radial ray-tracing."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    radii = np.linspace(r_min, r_max, num_samples_per_ray)

    limb_pts = []
    h, w = gray.shape

    for ang in angles:
        cos_a = np.cos(ang)
        sin_a = np.sin(ang)
        xs = np.clip(approx_center[0] + radii * cos_a, 0, w - 1).astype(np.float32)
        ys = np.clip(approx_center[1] + radii * sin_a, 0, h - 1).astype(np.float32)

        samples = cv2.remap(gray, xs[:, np.newaxis], ys[:, np.newaxis], cv2.INTER_LINEAR).flatten()
        grad = np.gradient(samples)
        max_idx = np.argmax(grad)

        if grad[max_idx] > 2.0:
            r_edge = radii[max_idx]
            limb_pts.append((approx_center[0] + r_edge * cos_a, approx_center[1] + r_edge * sin_a))

    if len(limb_pts) < 10:
        return approx_center, 246.0

    pts = np.array(limb_pts, dtype=np.float32)
    (cx, cy), r = cv2.minEnclosingCircle(pts)
    return (float(cx), float(cy)), float(r)


def align_to_master_geometry(
    img: np.ndarray,
    target_center: tuple = (640.0, 360.0),
    target_radius: float = 246.0
) -> tuple:
    """Warps input image with Lanczos4 to align lunar disk to target_center and radius."""
    (cx, cy), r = fit_lunar_limb_raytracing(img, approx_center=target_center)
    scale = target_radius / max(r, 1e-4)

    M = np.array([
        [scale, 0.0, target_center[0] - scale * cx],
        [0.0, scale, target_center[1] - scale * cy]
    ], dtype=np.float32)

    aligned = cv2.warpAffine(
        img, M, (img.shape[1], img.shape[0]),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return aligned, (cx, cy), r


def extract_aligned_frame_at(cap, sec: float, fps: float, target_center=(640.0, 360.0), target_radius=246.0):
    """Fetches a frame at timestamp sec and returns its sub-pixel aligned version."""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_idx = max(0, min(total_frames - 1, int(sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret:
        return None, None
    aligned, (cx, cy), r = align_to_master_geometry(frame, target_center, target_radius)
    return frame, aligned


def angle_to_clock_position(radians: float) -> str:
    """Converts mathematical angle (0=East, pi/2=South, pi=West, -pi/2=North) to clock position."""
    deg = np.degrees(radians) % 360.0
    clock_hour = int(round((deg / 30.0) + 3.0)) % 12
    if clock_hour == 0:
        clock_hour = 12
    return f"{clock_hour} o'clock"


def detect_prominence_features(img_aligned: np.ndarray, target_center=(640.0, 360.0), target_radius=246.0, is_east=False) -> str:
    """Detects primary angular sectors of H-alpha prominences and returns clean clock position."""
    h, w = img_aligned.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - target_center[0])**2 + (Y - target_center[1])**2)
    angles = np.arctan2(Y - target_center[1], X - target_center[0])

    b, g, r = img_aligned[:, :, 0].astype(float), img_aligned[:, :, 1].astype(float), img_aligned[:, :, 2].astype(float)
    redness = np.maximum(0.0, r - np.maximum(g, b))
    in_ring = (dist >= (target_radius - 2.0)) & (dist <= (target_radius + 25.0))
    valid_red = redness * in_ring.astype(float)

    # 24 sectors (15 deg each)
    num_sectors = 24
    sector_edges = np.linspace(-np.pi, np.pi, num_sectors + 1)
    sector_sums = []
    for i in range(num_sectors):
        in_sec = in_ring & (angles >= sector_edges[i]) & (angles < sector_edges[i+1])
        sector_sums.append(np.sum(valid_red[in_sec]))

    top_indices = np.argsort(sector_sums)[::-1]
    max_val = max(1e-4, sector_sums[top_indices[0]])
    active_sectors = [i for i in top_indices if sector_sums[i] > 0.40 * max_val]

    clocks = []
    for idx in active_sectors:
        mid_ang = 0.5 * (sector_edges[idx] + sector_edges[idx+1])
        deg = (np.degrees(mid_ang) + 360.0) % 360.0
        clock = int(round((deg / 30.0) + 3.0)) % 12
        if clock == 0:
            clock = 12
        clocks.append(clock)

    clocks = sorted(list(set(clocks)))
    if is_east:
        # Eastern hemisphere (12 to 6 o'clock)
        east_clocks = [c for c in clocks if c in [1, 2, 3, 4, 5]]
        if east_clocks:
            return f"{min(east_clocks)} to {max(east_clocks)} o'clock position"
        return "3 to 4 o'clock position"
    else:
        # Western hemisphere (6 to 12 o'clock)
        west_clocks = [c for c in clocks if c in [7, 8, 9, 10, 11]]
        if west_clocks:
            return f"{min(west_clocks)} to {max(west_clocks)} o'clock position (centering at 9 o'clock)"
        return "9 o'clock position"


def build_dynamic_ai_prompt(c2_pos: str, c3_pos: str) -> str:
    """Constructs the tailored, programmatically adapted prompt for AI generation."""
    prompt = (
        "Exact scale astrophotographic composite of the 5 total solar eclipse reference photos. "
        "Maintain the exact true proportion, size, and position of all solar features as captured in the camera: "
        f"do not enlarge the red prominence on the left (it must remain a small, delicate ruby-red feature at {c2_pos} "
        "along the lunar limb, exactly matching the reference photo). "
        "The solar corona must be soft, diffuse, and natural, matching the mid-totality photo. "
        f"On the eastern limb ({c3_pos}), include the small real chromospheric red points. "
        "Along the western edge and the lower-right crescent (around 5 o'clock), include the sparkling white Baily's Beads "
        "from the ingress and egress frames. Center is a pitch black circular Moon. "
        "Direct faithful overlay of the 5 exposures without exaggeration."
    )
    return prompt


def generate_local_opencv_composite(
    aligned_frames: dict,
    target_center=(640.0, 360.0),
    target_radius=246.0
) -> np.ndarray:
    """
    Generates an authentic, high-fidelity local mathematical HDR composite using OpenCV:
      1. Druckmüller Adaptive Radial Normalization for natural dipolar coronal streamers.
      2. Authentic ruby-red H-alpha prominence layer with smooth angular sector transitions.
      3. Point Spread Function (PSF) diamond pearl Baily's beads rendering.
      4. Anti-aliased pure zero-noise black Moon disk.
    """
    h, w = 720, 1280
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - target_center[0])**2 + (Y - target_center[1])**2)
    angles_rad = np.arctan2(Y - target_center[1], X - target_center[0])
    angles_deg = np.rad2deg(angles_rad)
    r_diff = np.maximum(0.0, dist - target_radius)

    f_in = aligned_frames['1_baily_in']
    f_c2 = aligned_frames['2_c2_prom']
    f_mid = aligned_frames['3_mid_corona']
    f_c3 = aligned_frames['4_c3_prom']
    f_eg = aligned_frames['5_baily_eg']

    # -------------------------------------------------------------------------
    # 1. Base Corona: Druckmüller Adaptive Radial Normalization
    # -------------------------------------------------------------------------
    gray_mid = cv2.cvtColor(f_mid, cv2.COLOR_BGR2GRAY).astype(np.float32)

    r_int = np.clip(np.round(r_diff).astype(int), 0, 450)
    mean_r = np.zeros(451, dtype=np.float32)
    std_r  = np.zeros(451, dtype=np.float32)

    for ri in range(451):
        mask = (r_int == ri) & (dist >= target_radius)
        if np.any(mask):
            mean_r[ri] = np.mean(gray_mid[mask])
            std_r[ri]  = np.std(gray_mid[mask])
        else:
            mean_r[ri] = mean_r[max(0, ri - 1)]
            std_r[ri]  = std_r[max(0, ri - 1)]

    mean_2d = mean_r[r_int]
    std_2d  = std_r[r_int]

    # Normalized high-frequency coronal streamer signal
    norm_streamers = (gray_mid - mean_2d) / (std_2d + 3.5)

    # Reference target mean and std profiles
    ref_target_mean = 145.0 * np.exp(-((r_diff / 72.0) ** 0.88))
    ref_target_std  = 40.0  * np.exp(-((r_diff / 78.0) ** 0.88))

    corona_druck = np.clip(ref_target_mean + norm_streamers * ref_target_std, 0.0, 255.0)

    # Smooth space fade
    space_fade = 1.0 / (1.0 + np.exp((dist - 375.0) / 18.0))
    corona_druck = corona_druck * space_fade

    # Astrophotographic warm platinum color grading
    corona_r = np.clip(corona_druck * 1.05, 0, 255)
    corona_g = np.clip(corona_druck * 0.99, 0, 255)
    corona_b = np.clip(corona_druck * 0.91, 0, 255)
    corona_bgr = np.stack([corona_b, corona_g, corona_r], axis=2)

    # Soft transition at lunar limb
    limb_trans = np.clip((dist - (target_radius - 0.5)) / 1.5, 0.0, 1.0)[:, :, np.newaxis]
    corona_bgr = corona_bgr * limb_trans

    # -------------------------------------------------------------------------
    # 2. Western Ruby H-alpha Prominences (C2 @ 8.0s)
    # -------------------------------------------------------------------------
    b2, g2, r2 = f_c2[:, :, 0].astype(np.float32), f_c2[:, :, 1].astype(np.float32), f_c2[:, :, 2].astype(np.float32)
    excess2 = np.maximum(0.0, r2 - 0.90 * np.maximum(g2, b2))

    ang_diff2 = np.abs(np.abs(angles_deg) - 180.0)
    west_smooth = np.clip(1.0 - ang_diff2 / 34.0, 0.0, 1.0)
    in_ring2 = np.clip(1.0 - np.abs(dist - (target_radius + 4.0)) / 22.0, 0.0, 1.0)

    alpha2 = np.clip((excess2 - 4.0) / 12.0, 0.0, 1.0) * in_ring2 * west_smooth
    alpha2_f = cv2.GaussianBlur(alpha2, (0, 0), 0.45)[:, :, np.newaxis]

    ruby2_r = np.clip(r2 * 1.95 + excess2 * 1.6, 0, 255)
    ruby2_g = np.clip(g2 * 0.15 + excess2 * 0.05, 0, 255)
    ruby2_b = np.clip(b2 * 0.35 + excess2 * 0.65, 0, 255)
    prom2_bgr = np.stack([ruby2_b, ruby2_g, ruby2_r], axis=2)

    # -------------------------------------------------------------------------
    # 3. Eastern Chromospheric Spicules (C3 @ 98.0s)
    # -------------------------------------------------------------------------
    b3, g3, r3 = f_c3[:, :, 0].astype(np.float32), f_c3[:, :, 1].astype(np.float32), f_c3[:, :, 2].astype(np.float32)
    excess3 = np.maximum(0.0, r3 - 0.94 * np.maximum(g3, b3))

    east_smooth = np.clip(1.0 - np.abs(angles_deg) / 36.0, 0.0, 1.0)
    top_smooth  = np.clip(1.0 - np.abs(angles_deg + 90.0) / 16.0, 0.0, 1.0)
    east_top_smooth = np.maximum(east_smooth, top_smooth)
    in_ring3 = np.clip(1.0 - np.abs(dist - (target_radius + 3.0)) / 16.0, 0.0, 1.0)

    alpha3 = np.clip((excess3 - 5.0) / 14.0, 0.0, 1.0) * in_ring3 * east_top_smooth
    alpha3_f = cv2.GaussianBlur(alpha3, (0, 0), 0.45)[:, :, np.newaxis]

    ruby3_r = np.clip(r3 * 1.95 + excess3 * 1.6, 0, 255)
    ruby3_g = np.clip(g3 * 0.15 + excess3 * 0.05, 0, 255)
    ruby3_b = np.clip(b3 * 0.35 + excess3 * 0.65, 0, 255)
    prom3_bgr = np.stack([ruby3_b, ruby3_g, ruby3_r], axis=2)

    p_tot_alpha = np.clip(alpha2_f + alpha3_f, 0.0, 1.0)
    p_tot_bgr = np.maximum(prom2_bgr, prom3_bgr)

    comp = corona_bgr * (1.0 - 0.90 * p_tot_alpha) + p_tot_bgr * p_tot_alpha * 1.85

    # -------------------------------------------------------------------------
    # 4. Brilliant Sparkling Baily's Beads (Diamond Pearl Necklace with PSF)
    # -------------------------------------------------------------------------
    b_eg_flt = f_eg.astype(np.float32)
    lum_eg = 0.299 * b_eg_flt[:, :, 2] + 0.587 * b_eg_flt[:, :, 1] + 0.114 * b_eg_flt[:, :, 0]
    ang_eg_mask = ((angles_deg >= 32.0) & (angles_deg <= 82.0)).astype(float)
    dist_eg_mask = np.clip(1.0 - np.abs(dist - (target_radius + 0.5)) / 6.0, 0.0, 1.0)
    alpha_bead_eg = np.clip((lum_eg - 55.0) / 50.0, 0.0, 1.0) * ang_eg_mask * dist_eg_mask

    b_in_flt = f_in.astype(np.float32)
    lum_in = 0.299 * b_in_flt[:, :, 2] + 0.587 * b_in_flt[:, :, 1] + 0.114 * b_in_flt[:, :, 0]
    ang_in_mask = (np.abs(np.abs(angles_deg) - 170.0) <= 22.0).astype(float)
    dist_in_mask = np.clip(1.0 - np.abs(dist - (target_radius + 0.5)) / 6.0, 0.0, 1.0)
    alpha_bead_in = np.clip((lum_in - 55.0) / 50.0, 0.0, 1.0) * ang_in_mask * dist_in_mask

    tot_bead_alpha = np.clip(alpha_bead_in + alpha_bead_eg, 0.0, 1.0)
    core_alpha = cv2.GaussianBlur(tot_bead_alpha, (0, 0), 0.5)[:, :, np.newaxis]
    bloom_alpha = cv2.GaussianBlur(tot_bead_alpha, (0, 0), 2.2)[:, :, np.newaxis]

    comp = comp * (1.0 - 0.50 * bloom_alpha) + np.array([230.0, 245.0, 255.0]) * bloom_alpha * 1.20
    comp = comp * (1.0 - 0.85 * core_alpha) + np.array([255.0, 255.0, 255.0]) * core_alpha * 2.10

    # Discrete diamond sparkles overlay
    N_samples = 1440
    thetas = np.linspace(-np.pi, np.pi, N_samples, endpoint=False)
    degs = np.rad2deg(thetas)
    xs = target_center[0] + (target_radius + 0.5) * np.cos(thetas)
    ys = target_center[1] + (target_radius + 0.5) * np.sin(thetas)

    prof_eg = cv2.remap(lum_eg, xs.astype(np.float32).reshape(1, -1), ys.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).flatten()
    mask_eg = np.where((degs >= 32.0) & (degs <= 82.0), prof_eg, 0.0)
    peaks_eg, _ = find_peaks(mask_eg, height=60.0, distance=16)

    prof_in = cv2.remap(lum_in, xs.astype(np.float32).reshape(1, -1), ys.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).flatten()
    mask_in = np.where((degs >= 150.0) | (degs <= -160.0), prof_in, 0.0)
    peaks_in, _ = find_peaks(mask_in, height=60.0, distance=20)

    bead_sparks = np.zeros((h, w, 3), dtype=np.float32)
    for p in list(peaks_eg) + list(peaks_in):
        px, py = xs[p], ys[p]
        d_sq = (X - px)**2 + (Y - py)**2
        core = np.exp(-d_sq / (2.0 * 1.2**2)) * 1.5
        cross = (
            np.exp(-np.abs(X - px) / 0.5) * np.exp(-np.abs(Y - py) / 4.5)
            + np.exp(-np.abs(Y - py) / 0.5) * np.exp(-np.abs(X - px) / 4.5)
        ) * 0.45
        spark_rgb = (core + cross)[:, :, np.newaxis] * np.array([255.0, 255.0, 255.0])
        bead_sparks += spark_rgb

    comp = np.maximum(comp, np.clip(bead_sparks, 0, 255))

    # -------------------------------------------------------------------------
    # 5. Anti-Aliased Pure Void Black Moon Mask
    # -------------------------------------------------------------------------
    moon_mask = np.clip((dist - (target_radius - 1.2)) / 1.5, 0.0, 1.0)[:, :, np.newaxis]
    comp = comp * moon_mask

    return np.clip(comp, 0, 255).astype(np.uint8)


def run_pipeline(
    video_path: str = "010_in/03_video_realtime.mp4",
    out_dir: str = "040_out",
    samples_dir: str = "040_out/hdr_samples"
):
    """Executes the extraction, local synthesis, feature detection, and prompt generation."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Auto-detect stabilized output video if available for optimal sub-pixel accuracy
    alt_out = os.path.join(repo_root, "040_out", "03_video_realtime.mp4")
    if os.path.exists(alt_out):
        video_path = alt_out
    elif not os.path.exists(video_path):
        candidate = os.path.join(repo_root, video_path)
        if os.path.exists(candidate):
            video_path = candidate

    print("=================================================================")
    print("TOTALITY HDR COMPOSITE & AI PROMPT ADAPTER ENGINE")
    print("=================================================================")
    print(f"  Video Source       : {video_path}")
    print(f"  Samples Directory  : {samples_dir}")
    print(f"  Output Directory   : {out_dir}")
    print("=================================================================")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 5 Key Phases (Optimized for sparkling beads & prominence peak)
    timestamps = {
        '1_baily_in': 0.8,
        '2_c2_prom': 8.0,
        '3_mid_corona': 51.7,
        '4_c3_prom': 98.0,
        '5_baily_eg': 102.2
    }

    raw_paths = {}
    aligned_frames = {}

    print("Extracting and saving the 5 key temporal frames...")
    for name, sec in timestamps.items():
        raw_f, aligned_f = extract_aligned_frame_at(cap, sec, fps)
        out_path = os.path.join(samples_dir, f"{name}.jpg")
        cv2.imwrite(out_path, raw_f, [cv2.IMWRITE_JPEG_QUALITY, 98])
        raw_paths[name] = out_path
        aligned_frames[name] = aligned_f
        print(f"  • [{name:14s}] (t = {sec:5.1f}s) -> {out_path}")

    cap.release()

    # 1. Generate Local Mathematical HDR Composite
    print("\nGenerating local mathematical OpenCV HDR composite...")
    local_hdr = generate_local_opencv_composite(aligned_frames)
    local_path = os.path.join(out_dir, "totality_hdr_local.jpg")
    cv2.imwrite(local_path, local_hdr, [cv2.IMWRITE_JPEG_QUALITY, 98])
    print(f"  ✓ Local HDR Composite saved to: {local_path}")

    # 2. Detect Features & Programmatically Adapt Prompt
    print("\nAnalyzing optical features in frames for dynamic prompt adaptation...")
    c2_pos = detect_prominence_features(aligned_frames['2_c2_prom'], is_east=False)
    c3_pos = detect_prominence_features(aligned_frames['4_c3_prom'], is_east=True)

    prompt = build_dynamic_ai_prompt(c2_pos, c3_pos)

    # Save prompt to file
    prompt_file = os.path.join(samples_dir, "ai_prompt.txt")
    with open(prompt_file, "w", encoding="utf-8") as pf:
        pf.write(prompt + "\n")
    print(f"  ✓ Adapted AI prompt saved to: {prompt_file}")

    # 3. Print User Guide
    print("\n" + "=" * 70)
    print("📋 GUÍA RÁPIDA PARA GENERACIÓN IA (WEB / NANO BANANA / CHATGPT)")
    print("=" * 70)
    print("1. Sube a la interfaz web las 5 imágenes extraídas:")
    for p in raw_paths.values():
        print(f"   📷 {p}")
    print("\n2. Pega el siguiente prompt adaptado automáticamente a tus fotos:\n")
    print("-" * 70)
    print(prompt)
    print("-" * 70)
    print(f"\n3. Guarda la imagen resultante como: 010_in/05_totality_6.jpg")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Totality HDR Multi-Exposure Composite Engine & AI Prompt Adapter"
    )
    parser.add_argument("-v", "--video", default="010_in/03_video_realtime.mp4", help="Path to real-time totality video")
    parser.add_argument("-o", "--out-dir", default="040_out", help="Output directory")
    parser.add_argument("-s", "--samples-dir", default="040_out/hdr_samples", help="Samples export directory")

    args = parser.parse_args()
    run_pipeline(video_path=args.video, out_dir=args.out_dir, samples_dir=args.samples_dir)


if __name__ == "__main__":
    main()
