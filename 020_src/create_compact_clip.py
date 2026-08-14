#!/usr/bin/env python3
"""
Accelerated Compact Solar Eclipse Video Generator
===================================================
Creates an accelerated, lightweight 30-second version of the master film
(full_eclipse.mp4 -> full_eclipse_30s.mp4) optimized for instant sharing,
social media, messaging apps, and web distribution.

Features:
  - Uniform high-quality temporal resampling to target duration (default: 30.0s = 900 frames @ 30 fps).
  - Preserves 720p HD resolution (1280x720).
  - Highly optimized H.264/H.265 compression for compact file size (~2-4 MB) and universal playback compatibility.
  - Fast streaming execution via OpenCV & FFmpeg.

Author: Antigravity
Workspace: eclipse_assembler
"""

import os
import sys
import argparse
import subprocess
import cv2
import numpy as np
from tqdm import tqdm


def generate_compact_eclipse(
    input_path: str = "040_out/full_eclipse.mp4",
    output_path: str = "040_out/full_eclipse_30s.mp4",
    target_duration: float = 30.0,
    target_fps: float = 30.0,
    codec: str = "libx264",
    crf: int = 22,
    preset: str = "slow"
):
    if not os.path.exists(input_path):
        print(f"Error: Input video '{input_path}' not found.", file=sys.stderr)
        return

    cap = cv2.VideoCapture(input_path)
    total_in_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_in_frames <= 0 or w <= 0 or h <= 0:
        print(f"Error: Unable to read valid video metadata from '{input_path}'.", file=sys.stderr)
        return

    in_duration = total_in_frames / in_fps if in_fps > 0 else total_in_frames / 30.0
    target_frames = int(round(target_duration * target_fps))
    speed_factor = total_in_frames / target_frames

    print("=" * 65)
    print(f"Generating Compact Accelerated Video: {input_path} -> {output_path}")
    print(f"  Source: {total_in_frames} frames ({in_duration:.2f}s @ {in_fps:.1f} fps, {w}x{h})")
    print(f"  Target: {target_frames} frames ({target_duration:.1f}s @ {target_fps:.1f} fps)")
    print(f"  Speed multiplier: {speed_factor:.2f}x speedup")
    print(f"  Codec: {codec} (CRF {crf}, Preset {preset})")
    print("=" * 65)

    # Compute sampling indices uniformly across entire master sequence
    sample_indices = np.linspace(0, total_in_frames - 1, target_frames)
    sample_indices = [int(round(idx)) for idx in sample_indices]
    sample_set = set(sample_indices)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'bgr24',
        '-r', str(target_fps),
        '-i', '-',
        '-c:v', codec,
        '-crf', str(crf),
        '-preset', preset,
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-color_range', '1',
        '-colorspace', 'bt709',
        '-color_trc', 'bt709',
        '-color_primaries', 'bt709',
    ]

    if codec == "libx265":
        ffmpeg_cmd.extend(['-tag:v', 'hvc1'])

    ffmpeg_cmd.append(output_path)

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cap = cv2.VideoCapture(input_path)
    pbar = tqdm(total=target_frames, desc="Resampling & encoding compact video")

    curr_frame_idx = 0
    sampled_count = 0
    target_idx_iter = iter(sample_indices)
    next_target = next(target_idx_iter)

    while curr_frame_idx <= sample_indices[-1]:
        if curr_frame_idx == next_target:
            ret, frame = cap.read()
            if not ret:
                break
            proc.stdin.write(frame.tobytes())
            sampled_count += 1
            pbar.update(1)
            try:
                next_target = next(target_idx_iter)
            except StopIteration:
                break
        else:
            # Fast forward through frames not in sample set
            ret = cap.grab()
            if not ret:
                break
        curr_frame_idx += 1

    cap.release()
    pbar.close()

    proc.stdin.close()
    proc.wait()

    if not os.path.exists(output_path):
        print(f"Error: Failed to create output video '{output_path}'.", file=sys.stderr)
        return

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("\n" + "=" * 65)
    print(f"SUCCESS: Compact video generated at: {output_path}")
    print(f"  Resolution: {w}x{h}")
    print(f"  FPS: {target_fps:.2f}")
    print(f"  Total frames: {sampled_count}")
    print(f"  Duration: {sampled_count / target_fps:.2f}s")
    print(f"  File size: {size_mb:.2f} MB (Optimized for quick sharing)")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a 30s accelerated, lightweight compact version of the full eclipse master film")
    parser.add_argument("--input", "-i", type=str, default="040_out/full_eclipse.mp4",
                        help="Input master video path (default: 040_out/full_eclipse.mp4)")
    parser.add_argument("--output", "-o", type=str, default="040_out/full_eclipse_30s.mp4",
                        help="Output compact video path (default: 040_out/full_eclipse_30s.mp4)")
    parser.add_argument("--duration", "-d", type=float, default=30.0,
                        help="Target duration in seconds (default: 30.0s)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Output frame rate (default: 30.0 fps)")
    parser.add_argument("--codec", type=str, choices=["libx264", "libx265"], default="libx264",
                        help="Video codec: 'libx264' (universal compatibility, default) or 'libx265' (HEVC). Default: libx264")
    parser.add_argument("--crf", type=int, default=22,
                        help="Constant Rate Factor / quality level (default: 22)")
    parser.add_argument("--preset", type=str, default="slow",
                        help="Encoder preset for compression efficiency (default: slow)")
    args = parser.parse_args()

    generate_compact_eclipse(
        input_path=args.input,
        output_path=args.output,
        target_duration=args.duration,
        target_fps=args.fps,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset
    )
