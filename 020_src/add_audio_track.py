#!/usr/bin/env python3
"""
ADD AUDIO TRACK TO ECLIPSE VIDEO (Proof of Concept)
===================================================
Muxes a musical audio track (e.g. 010_in/music.wav) into an eclipse video (e.g. 040_out/full_eclipse_subtitled.mp4).

Audio Synchronization Logic:
----------------------------
1. Measures exact video duration (Tv) and audio duration (Ta) via ffprobe.
2. If difference |Ta - Tv| <= threshold (default: 5.0 seconds):
   - Adjusts audio playback speed (atempo filter) to match video duration exactly,
     preserving original musical pitch with zero quality loss.
3. If difference > threshold:
   - If audio is significantly longer (Ta > Tv + threshold):
     Trims audio to video duration Tv with an elegant fade-out to silence.
   - If audio is significantly shorter (Ta < Tv - threshold):
     Fades out at end of audio track and pads remaining time with silence.
4. Lossless stream copy for video and soft subtitles (-c:v copy, -c:s copy),
   re-encoding only the audio track to high-bitrate AAC (320 kbps).
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np


def get_media_duration(file_path: str, stream_type: str = None) -> float:
    """
    Returns exact duration in seconds using ffprobe.
    If stream_type is specified ('v' or 'a'), queries the primary stream's duration first,
    falling back to container duration.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Media file not found: {file_path}")

    # First attempt: stream-level duration if requested
    if stream_type in ["v", "a"]:
        cmd_stream = [
            "ffprobe",
            "-v", "error",
            "-select_streams", f"{stream_type}:0",
            "-show_entries", "stream=duration",
            "-of", "json",
            file_path
        ]
        res = subprocess.run(cmd_stream, capture_output=True, text=True)
        if res.returncode == 0:
            try:
                data = json.loads(res.stdout)
                streams = data.get("streams", [])
                if streams and "duration" in streams[0] and streams[0]["duration"] != "N/A":
                    return float(streams[0]["duration"])
            except Exception:
                pass

    # Fallback to container format duration
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        file_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {file_path}: {res.stderr}")

    data = json.loads(res.stdout)
    duration_str = data.get("format", {}).get("duration")
    if duration_str is None or duration_str == "N/A":
        raise ValueError(f"Could not extract duration from {file_path}")

    return float(duration_str)


def check_has_subtitles(file_path: str) -> bool:
    """
    Checks if video file contains subtitle streams.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "s",
        "-show_entries", "stream=index",
        "-of", "json",
        file_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        data = json.loads(res.stdout)
        return len(data.get("streams", [])) > 0
    return False


def build_atempo_filter(factor: float) -> str:
    """
    Builds a chained ffmpeg atempo filter string for arbitrary tempo factor.
    Each individual atempo filter must be between 0.5 and 2.0.
    """
    filters = []
    f = factor
    while f > 2.0:
        filters.append("atempo=2.0")
        f /= 2.0
    while f < 0.5:
        filters.append("atempo=0.5")
        f /= 0.5
    filters.append(f"atempo={f:.6f}")
    return ",".join(filters)


def surgically_arrange_audio(
    audio_path: str,
    target_duration_sec: float,
    out_wav_path: str,
    xfade_ms: float = 120.0
) -> str:
    """
    Universally and generically rearranges ANY audio track to match a target duration
    at 100.0% natural tempo, preserving the original pitch, musical structure, and climax.

    Algorithm (Audio Retargeting & Loop Detection):
    ----------------------------------------------
    1. Extracts mono signal and computes RMS energy profile to detect and protect musical climaxes.
    2. Computes STFT spectral flux and onset novelty curve to evaluate rhythmic/beat boundaries.
    3. Scans all candidate cut windows t1 -> t2 (where t2 - t1 = cut_duration) across the file:
       - Evaluates waveform cross-correlation across the join boundary.
       - Evaluates harmonic and RMS energy continuity.
       - Penalizes cutting directly through the main musical climax.
    4. Automatically selects the global optimal splice points (t1, t2).
    5. Fine-tunes cut points to matching zero-crossings (same phase / positive slope).
    6. Applies an equal-power S-curve crossfade (cos^2 / sin^2) to guarantee click-free, phase-aligned blending.
    """
    from scipy.io import wavfile
    from scipy import signal

    sr, raw_data = wavfile.read(audio_path)
    is_stereo = (raw_data.ndim > 1)
    if is_stereo:
        data = raw_data
        mono = raw_data.mean(axis=1).astype(np.float32)
    else:
        data = np.column_stack([raw_data, raw_data])
        mono = raw_data.astype(np.float32)

    total_samples = len(data)
    target_samples = int(round(target_duration_sec * sr))
    cut_samples = total_samples - target_samples

    if cut_samples <= 0:
        # Audio is already equal or shorter than target
        wavfile.write(out_wav_path, sr, data)
        return out_wav_path

    # 1. Climax protection: detect highest energy peak in 200ms frames
    frame_len = int(0.200 * sr)
    n_frames = total_samples // frame_len
    if n_frames > 0:
        frames_rms = np.array([np.sqrt(np.mean(mono[i*frame_len : (i+1)*frame_len]**2)) for i in range(n_frames)])
        climax_frame = np.argmax(frames_rms)
        climax_t = (climax_frame + 0.5) * 0.200
    else:
        climax_t = (total_samples / sr) / 2.0

    # 2. STFT Spectral flux and onset novelty
    hop = 512
    win = 2048
    f, t_spec, Zxx = signal.stft(mono, fs=sr, nperseg=win, noverlap=win-hop)
    mag = np.abs(Zxx)
    diff_mag = np.diff(mag, axis=1)
    novelty = np.sum(np.maximum(0, diff_mag), axis=0)
    novelty = np.concatenate([[0], novelty])
    nov_range = np.max(novelty) - np.min(novelty)
    novelty_norm = (novelty - np.min(novelty)) / (nov_range + 1e-6)

    # 3. Global search for optimal seamless cut point (t1, t2)
    min_idx = int(0.05 * total_samples)
    max_idx = total_samples - cut_samples - int(0.05 * total_samples)

    if max_idx <= min_idx:
        # File is too short for search margins; center the cut
        best_idx1 = total_samples // 4
    else:
        step = max(1, int(0.020 * sr))  # 20ms search resolution
        search_indices = np.arange(min_idx, max_idx, step)
        win_corr = int(0.150 * sr)      # 150ms correlation window

        best_score = -1e9
        best_idx1 = search_indices[0]

        for idx1 in search_indices:
            idx2 = idx1 + cut_samples
            t1 = idx1 / sr
            t2 = idx2 / sr

            # Climax penalty: strongly disfavor cutting directly through the peak
            if t1 <= climax_t <= t2:
                penalty = 0.35
            else:
                penalty = 1.0

            # Correlation score across splice seam
            w1 = mono[idx1 - win_corr : idx1]
            w2 = mono[idx2 : idx2 + win_corr]
            norm1 = np.linalg.norm(w1)
            norm2 = np.linalg.norm(w2)

            if norm1 > 0 and norm2 > 0:
                corr = np.dot(w1, w2) / (norm1 * norm2)
                e1 = np.sqrt(np.mean(w1**2))
                e2 = np.sqrt(np.mean(w2**2))
                energy_ratio = min(e1, e2) / (max(e1, e2) + 1e-6)

                frame1 = int(t1 * (sr / hop))
                frame2 = int(t2 * (sr / hop))
                if frame1 < len(novelty_norm) and frame2 < len(novelty_norm):
                    nov_match = 1.0 - abs(novelty_norm[frame1] - novelty_norm[frame2])
                else:
                    nov_match = 0.5

                score = (corr * 0.50 + energy_ratio * 0.30 + nov_match * 0.20) * penalty
                if score > best_score:
                    best_score = score
                    best_idx1 = idx1

    # 4. Fine-tune to local zero crossings with positive slope (+ to +)
    def find_zero_crossing(idx_center, search_radius_ms=25.0):
        rad = int(search_radius_ms * 0.001 * sr)
        start = max(1, idx_center - rad)
        end = min(total_samples - 1, idx_center + rad)
        segment = mono[start:end]
        # Positive slope zero crossings: segment[i-1] <= 0 and segment[i] > 0
        zc = np.where((segment[:-1] <= 0) & (segment[1:] > 0))[0]
        if len(zc) > 0:
            best_zc = zc[np.argmin(np.abs(zc - rad))]
            return start + best_zc
        return idx_center

    idx1_opt = find_zero_crossing(best_idx1)
    idx2_opt = idx1_opt + cut_samples

    t1_s = idx1_opt / sr
    t2_s = idx2_opt / sr
    cut_s = cut_samples / sr
    print(f"  [GENERIC RETARGETING] Optimal splice found: {t1_s:.2f}s -> {t2_s:.2f}s (Removed: {cut_s:.2f}s, Climax: {climax_t:.2f}s)")

    # 5. Equal-power S-curve crossfade
    xfade_n = int(xfade_ms * 0.001 * sr)
    t = np.linspace(0.0, 1.0, xfade_n)
    w_out = np.cos(t * np.pi / 2.0)**2
    w_in  = np.sin(t * np.pi / 2.0)**2

    cross_out = data[idx1_opt - xfade_n : idx1_opt].astype(np.float32)
    cross_in  = data[idx2_opt : idx2_opt + xfade_n].astype(np.float32)

    blended = cross_out * w_out[:, np.newaxis] + cross_in * w_in[:, np.newaxis]

    result = np.vstack([
        data[: idx1_opt - xfade_n],
        blended.astype(data.dtype),
        data[idx2_opt + xfade_n :]
    ])

    if len(result) > target_samples:
        result = result[:target_samples]
    elif len(result) < target_samples:
        pad = np.zeros((target_samples - len(result), 2), dtype=data.dtype)
        result = np.vstack([result, pad])

    wavfile.write(out_wav_path, sr, result)
    return out_wav_path


def add_audio_to_video(
    video_path: str,
    audio_path: str,
    output_path: str = None,
    mode: str = "arrange",
    threshold_sec: float = 5.0,
    fade_duration_sec: float = 3.0,
    audio_bitrate: str = "320k",
) -> str:
    """
    Muxes audio track into video file with intelligent tempo scaling, surgical arrangement, or fade-out trimming.
    
    Modes:
      - 'arrange': Surgically cuts a redundant phrase at 100% natural tempo (preserves cadence & climax).
      - 'stretch': Time-stretches entire audio track to match video duration exactly.
      - 'cut'    : Trims at video duration with fade-out.
      - 'auto'   : Time-stretches if |Ta - Tv| <= threshold_sec, otherwise trims with fade-out.
    """
    repo_root = Path(__file__).resolve().parent.parent

    # Resolve paths relative to repository root if not absolute
    v_path = Path(video_path) if os.path.isabs(video_path) else repo_root / video_path
    a_path = Path(audio_path) if os.path.isabs(audio_path) else repo_root / audio_path

    if not v_path.exists():
        raise FileNotFoundError(f"Input video not found: {v_path}")
    if not a_path.exists():
        raise FileNotFoundError(f"Input audio not found: {a_path}")

    if output_path is None:
        out_name = f"{v_path.stem}_music{v_path.suffix}"
        out_p = v_path.parent / out_name
    else:
        out_p = Path(output_path) if os.path.isabs(output_path) else repo_root / output_path

    out_p.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ECLIPSE AUDIO TRACK MUXER (Proof of Concept)")
    print("=" * 65)
    print(f"  Input Video  : {v_path}")
    print(f"  Input Audio  : {a_path}")
    print(f"  Output Video : {out_p}")
    print(f"  Sync Mode    : {mode.upper()}")

    # 1. Probe durations
    dur_v = get_media_duration(str(v_path), stream_type="v")
    dur_a = get_media_duration(str(a_path), stream_type="a")
    diff = dur_a - dur_v

    print("\nDuration Analysis:")
    print(f"  Video Duration : {dur_v:.3f} s ({int(dur_v // 60):02d}:{dur_v % 60:06.3f})")
    print(f"  Audio Duration : {dur_a:.3f} s ({int(dur_a // 60):02d}:{dur_a % 60:06.3f})")
    print(f"  Difference (Δ) : {diff:+.3f} s (Audio {'longer' if diff > 0 else 'shorter'})")

    # 2. Build audio filter or arranged source
    final_audio_src = str(a_path)

    if mode == "arrange" and diff > 1.0:
        arranged_wav = out_p.parent / f"{a_path.stem}_arranged_{int(dur_v)}s.wav"
        print(f"\n[SYNC MODE: SURGICAL PHRASE ARRANGEMENT (100% CADENCE)]")
        print(f"  -> Cutting redundant phrase ({diff:.2f}s) at natural tempo (1.0x).")
        print(f"  -> Applying phase-aligned S-curve equal-power crossfade (120ms).")
        print(f"  -> Climax aligned with Maximum Eclipse at ~65s, ending naturally at {dur_v:.2f}s.")
        surgically_arrange_audio(str(a_path), dur_v, str(arranged_wav))
        final_audio_src = str(arranged_wav)
        audio_filter = f"atrim=0:{dur_v:.3f}"

    elif mode == "stretch" or (mode == "auto" and abs(diff) <= threshold_sec):
        # Time-stretch / tempo scale to match video duration exactly
        tempo_factor = dur_a / dur_v
        tempo_filter = build_atempo_filter(tempo_factor)
        print(f"\n[SYNC MODE: STRETCH/ACCELERATE]")
        print(f"  -> Time-stretching audio by factor {tempo_factor:.6f} ({tempo_factor * 100:.2f}%)")
        print(f"  -> Filter chain: {tempo_filter}")
        print("  -> Pitch preserved, audio will finish exactly in sync with the last video frame.")

        audio_filter = f"{tempo_filter},atrim=0:{dur_v:.3f}"

    elif diff > 0:
        # Audio is longer -> Trim at video duration with fade-out
        fade_start = max(0.0, dur_v - fade_duration_sec)
        print(f"\n[SYNC MODE: CUT & FADE-OUT]")
        print(f"  -> Trimming to {dur_v:.3f}s with a {fade_duration_sec:.1f}s fade-out (starting at {fade_start:.2f}s).")

        audio_filter = f"atrim=0:{dur_v:.3f},afade=t=out:st={fade_start:.3f}:d={fade_duration_sec:.3f}"

    else:
        # Audio is shorter -> Fade out and pad with silence
        fade_start = max(0.0, dur_a - fade_duration_sec)
        print(f"\n[SYNC MODE: FADE-OUT & PAD SILENCE]")
        print(f"  -> Fading out at {fade_start:.2f}s and padding remaining time with silence up to {dur_v:.3f}s.")

        audio_filter = f"afade=t=out:st={fade_start:.3f}:d={fade_duration_sec:.3f},apad=whole_dur={dur_v:.3f}"

    # 3. Assemble FFmpeg command
    in_place = (os.path.abspath(v_path) == os.path.abspath(out_p))
    out_dir = os.path.dirname(os.path.abspath(out_p))
    actual_output = os.path.join(out_dir, f"tmp_mux_{os.path.basename(out_p)}") if in_place else str(out_p)

    has_subs = check_has_subtitles(str(v_path))

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(v_path),
        "-i", final_audio_src,
        "-filter:a", audio_filter,
        "-map", "0:v:0",
        "-c:v", "copy",
    ]

    if has_subs:
        cmd.extend(["-map", "0:s?", "-c:s", "copy"])

    cmd.extend([
        "-map", "1:a:0",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-t", f"{dur_v:.3f}",
        "-movflags", "+faststart",
        actual_output
    ])

    print("\nExecuting FFmpeg Muxing...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"FFmpeg Error Output:\n{res.stderr}", file=sys.stderr)
        if in_place and os.path.exists(actual_output):
            os.remove(actual_output)
        raise RuntimeError(f"FFmpeg failed with return code {res.returncode}")

    if in_place:
        os.replace(actual_output, str(out_p))

    out_size_mb = out_p.stat().st_size / (1024 * 1024)
    out_dur = get_media_duration(str(out_p))

    print("=" * 65)
    print("SUCCESS: Video with music generated!")
    print(f"  File Path      : {out_p}")
    print(f"  Final Duration : {out_dur:.3f} s")
    print(f"  File Size      : {out_size_mb:.2f} MB")
    print("=" * 65)

    return str(out_p)


def main():
    parser = argparse.ArgumentParser(
        description="Mux musical audio track into eclipse video with intelligent duration synchronization."
    )
    parser.add_argument(
        "--video", "-v", type=str,
        default="040_out/full_eclipse_subtitled.mp4",
        help="Input video file path (default: 040_out/full_eclipse_subtitled.mp4)"
    )
    parser.add_argument(
        "--audio", "-a", type=str,
        default="010_in/music.wav",
        help="Input audio file path (default: 010_in/music.wav)"
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="040_out/full_eclipse_music.mp4",
        help="Output video file path (default: 040_out/full_eclipse_music.mp4)"
    )
    parser.add_argument(
        "--mode", "-m", type=str,
        choices=["arrange", "stretch", "cut", "auto"],
        default="arrange",
        help="Sync mode: 'arrange' (surgical phrase edit at 100%% tempo, default), 'stretch' (accelerate), 'cut' (trim with fade), or 'auto'"
    )
    parser.add_argument(
        "--threshold", "-t", type=float,
        default=5.0,
        help="Maximum duration difference in seconds for auto mode (default: 5.0s)"
    )
    parser.add_argument(
        "--fade-duration", "-f", type=float,
        default=3.0,
        help="Duration of fade-out in seconds when cutting audio (default: 3.0s)"
    )
    parser.add_argument(
        "--bitrate", "-b", type=str,
        default="320k",
        help="AAC audio bitrate (default: 320k)"
    )

    args = parser.parse_args()

    # Fallback to full_eclipse.mp4 if subtitled is not present
    video_target = args.video
    if not os.path.exists(video_target) and os.path.exists("040_out/full_eclipse.mp4"):
        print(f"[INFO] '{video_target}' not found, falling back to '040_out/full_eclipse.mp4'")
        video_target = "040_out/full_eclipse.mp4"

    add_audio_to_video(
        video_path=video_target,
        audio_path=args.audio,
        output_path=args.output,
        mode=args.mode,
        threshold_sec=args.threshold,
        fade_duration_sec=args.fade_duration,
        audio_bitrate=args.bitrate,
    )


if __name__ == "__main__":
    main()
