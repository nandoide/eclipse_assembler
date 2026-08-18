#!/usr/bin/env python3
"""
================================================================================
ASTRONOMICAL ECLIPSE RESTORATION & PHOTOSPHERE SYNTHESIS ENGINE
================================================================================
Restores the complete, pristine solar photosphere across any eclipse timelapse
by applying the 100% reconstructed Master Solar Disk texture to the geometric
kinematic orbital model:
  - Preserves all real sunspots, limb darkening, and photospheric granulation.
  - Generates subpixel occultation bites matching the real astronomical orbit.
  - Completely eliminates foreground tree branches, foliage, and cloud dropouts.
  - Renders restored video and side-by-side comparison video with raw footage.

Authors: Fernando (nandoide) & Antigravity (Google Deepmind)
Workspace: eclipse_assembler
================================================================================
"""

import os
import sys
import argparse
import subprocess
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_eclipse_geometry import (
    auto_calibrate_video,
    compute_c1_extrapolation_frames,
    compute_c2_extrapolation_frames,
    compute_c3_extrapolation_frames,
    compute_c4_extrapolation_frames,
    get_extrapolated_output_filename
)


def render_restored_eclipse_video(
    input_path: str,
    master_disk_path: str = "040_out/master_solar_disk.png",
    output_video_path: str = None,
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
    Renders the clean eclipse video by mapping the Master Solar Disk onto calibrated
    photosphere geometry and occulting it with the moving Lunar disk.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(input_path) and not os.path.exists(input_path):
        resolved = os.path.join(repo_root, input_path)
        if os.path.exists(resolved):
            input_path = resolved

    if not os.path.isabs(master_disk_path) and not os.path.exists(master_disk_path):
        resolved_m = os.path.join(repo_root, master_disk_path)
        if os.path.exists(resolved_m):
            master_disk_path = resolved_m

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not os.path.exists(master_disk_path):
        raise FileNotFoundError(f"Master solar disk not found: {master_disk_path}")

    # 1. Load Master Solar Disk
    master_rgba = cv2.imread(master_disk_path, cv2.IMREAD_UNCHANGED)
    if master_rgba is None:
        raise ValueError(f"Could not load master solar disk from: {master_disk_path}")

    tc = master_rgba.shape[0] / 2.0
    target_R = 236.5

    # 2. Autocalibrate orbital kinematics of input video
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

    # Set default output paths
    if output_video_path is None:
        output_video_path = os.path.join(repo_root, f"040_out/restored_eclipse_{phase}.mp4")
    elif not os.path.isabs(output_video_path):
        output_video_path = os.path.join(repo_root, output_video_path)

    if comp_video_path is None:
        comp_video_path = os.path.join(repo_root, f"040_out/restored_comparison_{phase}.mp4")
    elif not os.path.isabs(comp_video_path):
        comp_video_path = os.path.join(repo_root, comp_video_path)

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret0, first_frame = cap.read()

    last_frame = None
    if total_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret_last, last_frame = cap.read()
    cap.release()

    total_rendered_frames = n_extra_before + total_frames + n_extra_after

    print("=" * 68)
    print(f"RESTORING ECLIPSE TIMELAPSE PHOTOSPHERE [{phase.upper()}]")
    print("=" * 68)
    print(f"  Input Raw Video    : {input_path}")
    print(f"  Master Solar Disk  : {master_disk_path}")
    print(f"  Restored Output    : {output_video_path}")
    print(f"  Comparison Output  : {comp_video_path}")
    print(f"  Total Frames       : {total_rendered_frames} ({total_frames} raw + {n_extra_before} before + {n_extra_after} after)")
    print(f"  Solar Radius (Rs)  : {Rs:.2f} px")
    print(f"  Lunar Radius (Rm)  : {Rm:.2f} px")
    print("=" * 68)

    # Setup scratch directory
    scratch_dir = os.path.join(repo_root, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    temp_restored = os.path.join(scratch_dir, f"temp_{phase}_restored_{os.getpid()}.mp4")
    temp_comp = os.path.join(scratch_dir, f"temp_{phase}_comp_{os.getpid()}.mp4") if comp_video_path else None

    # FFmpeg Writers
    proc_restored = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-crf", "14", "-preset", "slow",
        "-pix_fmt", "yuv420p",
        temp_restored
    ], stdin=subprocess.PIPE)

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
    Y_grid, X_grid = np.ogrid[:h, :w]
    scale = Rs / target_R

    try:
        pbar = tqdm(total=total_rendered_frames, desc=f"Rendering Restored {phase.capitalize()} Video")
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

            # Warp Master Solar Disk to current frame solar center
            M = np.array([
                [scale, 0, scx - tc * scale],
                [0, scale, scy - tc * scale]
            ], dtype=np.float32)
            warped_rgba = cv2.warpAffine(master_rgba, M, (w, h), flags=cv2.INTER_LANCZOS4)
            warped_bgr = warped_rgba[:, :, :3]
            warped_alpha = warped_rgba[:, :, 3] / 255.0

            # Lunar Occultation Bite
            dist_moon = np.sqrt((X_grid - mcx)**2 + (Y_grid - mcy)**2)
            moon_alpha = np.clip((dist_moon - Rm) / 1.0, 0.0, 1.0)

            # Composite: Restored visible photosphere
            final_alpha = warped_alpha * moon_alpha
            restored_frame = (warped_bgr * final_alpha[:, :, None]).astype(np.uint8)

            proc_restored.stdin.write(restored_frame.tobytes())

            if proc_comp:
                sbs = np.hstack([frame, restored_frame])
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

                cv2.putText(sbs, f"RESTORED PHOTOSPHERE", (w + 30, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                proc_comp.stdin.write(sbs.tobytes())

                if save_comparison_images and f_idx in comparison_frames:
                    prefix = f"restored_{phase}_comparison"
                    comp_dir = os.path.dirname(comp_video_path) if comp_video_path else os.path.dirname(output_video_path)
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

        proc_restored.stdin.close()
        proc_restored.wait()

        if proc_comp:
            proc_comp.stdin.close()
            proc_comp.wait()

        if os.path.exists(temp_restored):
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            os.rename(temp_restored, output_video_path)

        if comp_video_path and temp_comp and os.path.exists(temp_comp):
            if os.path.exists(comp_video_path):
                os.remove(comp_video_path)
            os.rename(temp_comp, comp_video_path)
    finally:
        # Cleanup temporary files in scratch
        if os.path.exists(temp_restored):
            try:
                os.remove(temp_restored)
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
    print(f"SUCCESS: Restored eclipse video saved at   : {output_video_path}")
    if comp_video_path:
        print(f"SUCCESS: Side-by-side comparison video at : {comp_video_path}")
    if saved_images:
        print(f"Side-by-side comparison images generated in: {os.path.dirname(saved_images[0])}:")
        for cp in saved_images:
            print(f"  • {os.path.basename(cp)}")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(
        description="Astronomical Eclipse Photosphere Restoration Engine"
    )
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to raw eclipse video (Ingress or Egress)")
    parser.add_argument("--master-disk", "-m", type=str,
                        default="040_out/master_solar_disk.png",
                        help="Path to reconstructed Master Solar Disk PNG")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Optional custom output path for restored video")
    parser.add_argument("--comp-video", "-c", type=str, default=None,
                        help="Optional custom output path for side-by-side comparison video")
    parser.add_argument("--no-images", action="store_true",
                        help="Disable comparison images")
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

    render_restored_eclipse_video(
        input_path=args.input,
        master_disk_path=args.master_disk,
        output_video_path=args.output,
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
