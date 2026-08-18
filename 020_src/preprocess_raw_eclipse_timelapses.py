#!/usr/bin/env python3
"""
================================================================================
MASTER RAW ECLIPSE TIMELAPSE PREPROCESSING PIPELINE
================================================================================
Orchestrates the complete astronomical restoration pipeline for raw eclipse
timelapses (Ingress and Egress) and exports the preprocessed videos to
the '005_raw_preprocessed/' directory with the exact original filenames:

Pipeline Steps:
  1. Input Discovery: Locates raw ingress and egress timelapse files in '000_raw/'.
  2. Master Photosphere Reconstruction:
     - Blends complementary clean solar photosphere regions from ingress & egress.
     - Preserves 100% authentic sunspots, limb darkening, and micro-granulation.
     - Saves '040_out/master_solar_disk.png'.
  3. Ingress Video Restoration:
     - Applies the master solar disk to the orbital ingress kinematics.
     - Saves '005_raw_preprocessed/DWARF_mini_TELE_TL_2026-08-12-19-35-13-093.mp4'.
  4. Egress Video Restoration:
     - Applies the master solar disk to the orbital egress kinematics.
     - Completely removes tree branches, foliage, and cloud dropouts.
     - Saves '005_raw_preprocessed/DWARF_mini_TELE_TL_2026-08-12-20-32-53-127.mp4'.
  5. Verification Outputs:
     - Generates side-by-side comparison videos and QA images in '040_out/'.

Authors: Fernando (nandoide) & Antigravity (Google Deepmind)
Workspace: eclipse_assembler
================================================================================
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_master_solar_disk import build_master_solar_disk
from render_restored_eclipse_video import render_restored_eclipse_video
from extract_eclipse_geometry import (
    auto_calibrate_video,
    compute_c1_extrapolation_frames,
    compute_c2_extrapolation_frames,
    compute_c3_extrapolation_frames,
    compute_c4_extrapolation_frames,
    get_extrapolated_output_filename
)


def run_master_preprocessing_pipeline(
    raw_dir: str = "000_raw",
    preprocessed_dir: str = "005_raw_preprocessed",
    out_dir: str = "040_out",
    ingress_filename: str = "DWARF_mini_TELE_TL_2026-08-12-19-35-13-093.mp4",
    egress_filename: str = "DWARF_mini_TELE_TL_2026-08-12-20-32-53-127.mp4",
    extrapolate_c1: bool = True,
    c1_padding_frames: int = 4,
    extrapolate_c2: bool = True,
    c2_padding_frames: int = 4,
    extrapolate_c3: bool = True,
    c3_padding_frames: int = 4,
    extrapolate_c4: bool = True,
    c4_padding_frames: int = 4
):
    """
    Executes the entire raw timelapse preprocessing pipeline end-to-end:
      1. Generates 100% complete Master Solar Disk from clean solar angles.
      2. Renders clean, branch-free Ingress timelapse with geometric photosphere mapping.
      3. Renders clean, branch-free Egress timelapse with geometric photosphere mapping.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isabs(raw_dir):
        raw_dir = os.path.join(repo_root, raw_dir)
    if not os.path.isabs(preprocessed_dir):
        preprocessed_dir = os.path.join(repo_root, preprocessed_dir)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(repo_root, out_dir)

    ingress_raw_path = os.path.join(raw_dir, ingress_filename)
    egress_raw_path = os.path.join(raw_dir, egress_filename)

    if not os.path.exists(ingress_raw_path):
        raise FileNotFoundError(f"Ingress video not found: {ingress_raw_path}")
    if not os.path.exists(egress_raw_path):
        raise FileNotFoundError(f"Egress video not found: {egress_raw_path}")

    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    master_disk_png = os.path.join(out_dir, "master_solar_disk.png")
    master_disk_comp = os.path.join(out_dir, "master_solar_disk_comparison.jpg")

    # Determine dynamic Ingress output filename based on start timestamp
    n_extra_c1 = 0
    if extrapolate_c1:
        scx0, scy0, svx, svy, mcx0, mcy0, mvx, mvy, Rs, Rm, phase, _ = auto_calibrate_video(ingress_raw_path)
        n_extra_c1, _ = compute_c1_extrapolation_frames(
            scx0, scy0, svx, svy, mcx0, mcy0, mvx, mvy, Rs, Rm, padding_frames=c1_padding_frames
        )

    if n_extra_c1 > 0:
        actual_ingress_filename = get_extrapolated_output_filename(ingress_filename, n_extra_c1, interval_s=10.0)
    else:
        actual_ingress_filename = ingress_filename

    # Determine dynamic Egress output filename based on start timestamp
    n_extra_c3 = 0
    if extrapolate_c3:
        scx0_e, scy0_e, svx_e, svy_e, mcx0_e, mcy0_e, mvx_e, mvy_e, Rs_e, Rm_e, _, _ = auto_calibrate_video(egress_raw_path)
        n_extra_c3, _ = compute_c3_extrapolation_frames(
            scx0_e, scy0_e, svx_e, svy_e, mcx0_e, mcy0_e, mvx_e, mvy_e, Rs_e, Rm_e, padding_frames=c3_padding_frames
        )

    if n_extra_c3 > 0:
        actual_egress_filename = get_extrapolated_output_filename(egress_filename, n_extra_c3, interval_s=10.0)
    else:
        actual_egress_filename = egress_filename

    ingress_out_path = os.path.join(preprocessed_dir, actual_ingress_filename)
    egress_out_path = os.path.join(preprocessed_dir, actual_egress_filename)

    # Clean up legacy preprocessed files if names changed
    if actual_ingress_filename != ingress_filename:
        old_ingress_path = os.path.join(preprocessed_dir, ingress_filename)
        if os.path.exists(old_ingress_path):
            try:
                os.remove(old_ingress_path)
                print(f"  • Removed outdated preprocessed ingress file: {ingress_filename}")
            except OSError:
                pass

    if actual_egress_filename != egress_filename:
        old_egress_path = os.path.join(preprocessed_dir, egress_filename)
        if os.path.exists(old_egress_path):
            try:
                os.remove(old_egress_path)
                print(f"  • Removed outdated preprocessed egress file: {egress_filename}")
            except OSError:
                pass

    ingress_comp_path = os.path.join(out_dir, "restored_comparison_ingress.mp4")
    egress_comp_path = os.path.join(out_dir, "restored_comparison_egress.mp4")

    t_start = time.time()

    print("=" * 72)
    print("MASTER RAW ECLIPSE PREPROCESSING PIPELINE")
    print("=" * 72)
    print(f"  Raw Source Directory         : {raw_dir}")
    print(f"  Preprocessed Output Directory: {preprocessed_dir}")
    print(f"  Diagnostics Output Directory : {out_dir}")
    print(f"  Ingress File (Raw)           : {ingress_filename}")
    print(f"  Ingress File (Preprocessed)  : {actual_ingress_filename}")
    print(f"  Egress File (Raw)            : {egress_filename}")
    print(f"  Egress File (Preprocessed)   : {actual_egress_filename}")
    print(f"  Extrapolate C1 First Contact : {extrapolate_c1}")
    print(f"  Extrapolate C2 Totality      : {extrapolate_c2}")
    print(f"  Extrapolate C3 Totality Exit : {extrapolate_c3}")
    print(f"  Extrapolate C4 Fourth Contact: {extrapolate_c4}")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # STEP 1: Reconstruct Master Solar Disk Texture
    # -------------------------------------------------------------------------
    print("\n[STEP 1/3] Reconstructing 100% Complete Master Solar Disk...")
    build_master_solar_disk(
        ingress_video=ingress_raw_path,
        egress_video=egress_raw_path,
        output_png=master_disk_png,
        output_comp=master_disk_comp
    )

    # -------------------------------------------------------------------------
    # STEP 2: Render Restored Ingress Timelapse
    # -------------------------------------------------------------------------
    print("\n[STEP 2/3] Rendering Preprocessed Ingress Timelapse...")
    render_restored_eclipse_video(
        input_path=ingress_raw_path,
        master_disk_path=master_disk_png,
        output_video_path=ingress_out_path,
        comp_video_path=ingress_comp_path,
        save_comparison_images=True,
        extrapolate_c1=extrapolate_c1,
        c1_padding_frames=c1_padding_frames,
        extrapolate_c2=extrapolate_c2,
        c2_padding_frames=c2_padding_frames
    )

    # -------------------------------------------------------------------------
    # STEP 3: Render Restored Egress Timelapse
    # -------------------------------------------------------------------------
    print("\n[STEP 3/3] Rendering Preprocessed Egress Timelapse...")
    render_restored_eclipse_video(
        input_path=egress_raw_path,
        master_disk_path=master_disk_png,
        output_video_path=egress_out_path,
        comp_video_path=egress_comp_path,
        save_comparison_images=True,
        extrapolate_c3=extrapolate_c3,
        c3_padding_frames=c3_padding_frames,
        extrapolate_c4=extrapolate_c4,
        c4_padding_frames=c4_padding_frames
    )

    elapsed = time.time() - t_start

    print("\n" + "=" * 72)
    print(f"PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY in {elapsed:.2f}s")
    print("=" * 72)
    print(f"  • Ingress Preprocessed : {ingress_out_path}")
    print(f"  • Egress Preprocessed  : {egress_out_path}")
    print(f"  • Master Solar Disk    : {master_disk_png}")
    print(f"  • Diagnostics & QA     : {out_dir}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Master Raw Eclipse Timelapse Preprocessing Pipeline"
    )
    parser.add_argument("--raw-dir", "-r", type=str, default="000_raw",
                        help="Directory containing raw eclipse videos")
    parser.add_argument("--preprocessed-dir", "-p", type=str, default="005_raw_preprocessed",
                        help="Directory where restored preprocessed videos will be written")
    parser.add_argument("--out-dir", "-o", type=str, default="040_out",
                        help="Directory for diagnostic artifacts and side-by-side QA videos")
    parser.add_argument("--ingress-name", type=str, default="DWARF_mini_TELE_TL_2026-08-12-19-35-13-093.mp4",
                        help="Filename of raw ingress timelapse")
    parser.add_argument("--egress-name", type=str, default="DWARF_mini_TELE_TL_2026-08-12-20-32-53-127.mp4",
                        help="Filename of raw egress timelapse")
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

    run_master_preprocessing_pipeline(
        raw_dir=args.raw_dir,
        preprocessed_dir=args.preprocessed_dir,
        out_dir=args.out_dir,
        ingress_filename=args.ingress_name,
        egress_filename=args.egress_name,
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
