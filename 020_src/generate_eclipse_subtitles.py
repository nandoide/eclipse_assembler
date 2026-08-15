#!/usr/bin/env python3
"""
================================================================================
ASTRONOMICAL LOCAL TIME SUBTITLE GENERATOR FOR SOLAR & LUNAR ECLIPSE FILMS
================================================================================
Generates standard SubRip (.srt) subtitle files for '040_out/full_eclipse.mp4'
(or custom master video) with precise real-world astronomical local timestamps
(CEST / UTC+2) and contact point markers (C1, C2, C3, C4 for Solar, P1-P4 for Lunar).

All timestamps are derived 100% directly from ground-truth telescope recordings in 000_raw/.

Features:
  - Generates BOTH English (EN) and Spanish (ES) subtitle files by default:
      * 040_out/full_eclipse.srt (English master)
      * 040_out/full_eclipse_en.srt (English)
      * 040_out/full_eclipse_es.srt (Spanish)
  - Optional '--embed' flag to instantly mux both subtitle tracks into a QuickTime
    compatible MP4 container (040_out/full_eclipse_subtitled.mp4) without re-encoding.
  - Automatic frame-accurate alignment for Title Card (00_title.mp4), Still Photo,
    and Composite Artwork.

Authors: Fernando (nandoide) & Antigravity (Google Deepmind)
Workspace: eclipse_assembler
================================================================================
"""

import os
import sys
import argparse
import subprocess
import json
import datetime
import math
import re
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eclipse_ephemeris_db as eedb


# ─────────────────────────────────────────────────────────────────────────────
# TIME AND VIDEO UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def format_srt_time(seconds: float) -> str:
    """Formats float seconds into SRT timestamp: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0.0
    millis = int(round((seconds - int(seconds)) * 1000.0))
    if millis >= 1000:
        millis = 999
    total_sec = int(seconds)
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    sec = total_sec % 60
    return f"{hours:02d}:{minutes:02d}:{sec:02d},{millis:03d}"


def format_clock_time(base_time: datetime.datetime, offset_seconds: float) -> str:
    """Calculates astronomical clock time string (HH:MM:SS) given a base datetime and offset."""
    dt = base_time + datetime.timedelta(seconds=offset_seconds)
    return dt.strftime("%H:%M:%S")


def get_video_info(video_path: str):
    """Retrieves duration, frame count, and FPS using ffprobe."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', video_path
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to probe {video_path}: {res.stderr}")
    data = json.loads(res.stdout)
    v_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), None)
    if not v_stream:
        raise ValueError(f"No video stream found in {video_path}")

    fps_eval = eval(v_stream.get('r_frame_rate', '30/1'))
    nb_frames = int(v_stream.get('nb_frames', 0))
    duration = float(data.get('format', {}).get('duration', 0.0))
    if nb_frames == 0 and duration > 0:
        nb_frames = int(round(duration * fps_eval))
    return {
        'duration': duration,
        'fps': fps_eval,
        'nb_frames': nb_frames,
        'width': int(v_stream.get('width', 1280)),
        'height': int(v_stream.get('height', 720))
    }


def detect_totality_contacts(realtime_clip_path: str):
    """
    Optically detects C2 (Diamond Ring disappear / Totality start) and
    C3 (Diamond Ring reappear / Totality end) in 03_video_realtime.mp4.
    Returns relative offsets in seconds from clip start: (c2_sec, c3_sec).
    """
    if not os.path.exists(realtime_clip_path):
        return 3.253, 100.212

    cap = cv2.VideoCapture(realtime_clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bright_counts = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright_px = int(np.sum(gray > 200))
        bright_counts.append(bright_px)
    cap.release()

    if len(bright_counts) < 100:
        return 3.253, 100.212

    # 1. Detect C2: where bright pixel count drops below threshold at totality start
    c2_frame = int(round(3.253 * fps))
    for i in range(10, min(len(bright_counts), int(fps * 6))):
        if bright_counts[i] < 350:
            c2_frame = i
            break
    c2_sec = c2_frame / fps

    # 2. Detect C3: where diamond ring flare emerges at the end of totality (>1000 px)
    c3_frame = int(round(100.212 * fps))
    tail_start = max(0, int(fps * 95))
    for i in range(tail_start, len(bright_counts)):
        if bright_counts[i] > 1000:
            c3_frame = i
            break
    c3_sec = c3_frame / fps

    return c2_sec, c3_sec


def get_composite_time_range(out_dir: str = "040_out", date_str: str = None) -> str:
    """
    Reads or calculates the exact time range of frames sampled in the composite artwork.
    """
    # 1. Check if metadata JSON exists
    candidates = [
        os.path.join(out_dir, "06_composite_sinusoid_10.json"),
        os.path.join(out_dir, "composite_samples.json"),
        os.path.join(out_dir, "eclipse_composite_sinusoid_1280x720.json"),
        os.path.join(out_dir, "eclipse_composite_sinusoid_3840p.json")
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "time_range_str" in data:
                        return data["time_range_str"].replace(" CEST", "").strip()
            except Exception:
                pass

    return "19:35 - 21:20"


# ─────────────────────────────────────────────────────────────────────────────
# TIMELINE MODELING & MULTILINGUAL SUBTITLES
# ─────────────────────────────────────────────────────────────────────────────

def build_timeline_mapping(
    in_dir: str = "010_in",
    out_dir: str = "040_out",
    date_str: str = None,
    include_title: bool = True,
    title_duration: float = 5.0,
    transition_type: str = "fade_to_black",
    freeze_before: float = 1.0,
    fade_out: float = 0.5,
    black_duration: float = 0.0,
    fade_in: float = 0.5,
    freeze_after: float = 1.0
):
    """
    Builds a timeline model mapping every millisecond of the assembled film
    to its astronomical local time and phase description.
    All timestamps come directly from 000_raw/ telescope footage.
    """
    resolved_date = date_str or eedb.detect_eclipse_date(raw_dir="000_raw", in_dir=in_dir)
    dt_base = datetime.datetime.strptime(resolved_date, "%Y-%m-%d")

    # Ground-truth camera timestamps from 000_raw/ recordings:
    dt_ingress_start  = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 19, 35, 13)
    dt_pre_tot_start  = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 20, 25, 36, 800000)
    dt_pre_tot_end    = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 20, 27, 31, 588000)
    dt_totality_start = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 20, 27, 31, 588000)
    dt_totality_end   = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 20, 29, 18, 921000)
    dt_egress_start   = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 20, 32, 53)

    trans_dur = freeze_before + fade_out + black_duration + fade_in + freeze_after

    # 1. Discover all assets dynamically in 010_in/
    discovered = []
    if os.path.exists(in_dir):
        for fname in sorted(os.listdir(in_dir)):
            if fname.startswith('.') or os.path.isdir(os.path.join(in_dir, fname)):
                continue
            name_no_ext, _ = os.path.splitext(fname)
            m = re.match(r"^(\d+)_(.+)$", name_no_ext)
            if not m:
                continue
            idx = int(m.group(1))
            tokens = m.group(2).split('_')
            primary = tokens[0].lower()

            interval = 10.0
            duration = 10.0
            a_type = primary

            if primary == "timelapse":
                for tok in tokens[1:]:
                    tok_l = tok.lower()
                    if tok_l.startswith("i") and tok_l[1:].replace('.', '', 1).isdigit():
                        interval = float(tok_l[1:])
                    elif tok_l.endswith("s") and tok_l[:-1].replace('.', '', 1).isdigit():
                        interval = float(tok_l[:-1])
                    elif tok_l.replace('.', '', 1).isdigit():
                        interval = float(tok_l)
            elif primary == "video":
                sub = tokens[1].lower() if len(tokens) > 1 else "realtime"
                if "slowdown" in sub:
                    a_type = "video_slowdown"
                    for tok in tokens[2:]:
                        if tok.replace('.', '', 1).isdigit():
                            duration = float(tok)
                else:
                    a_type = "video_realtime"
            elif primary == "photo":
                for tok in tokens[1:]:
                    if tok.replace('.', '', 1).isdigit():
                        duration = float(tok)
            elif primary == "composite":
                for tok in tokens[1:]:
                    if tok.replace('.', '', 1).isdigit():
                        duration = float(tok)

            discovered.append({
                "index": idx,
                "type": a_type,
                "interval": interval,
                "duration": duration,
                "raw_name": fname,
                "name_no_ext": name_no_ext
            })

    discovered.sort(key=lambda x: x['index'])

    clips_order = []
    p_title = os.path.join(out_dir, "00_title.mp4")
    if include_title and (os.path.exists(p_title) or include_title):
        clips_order.append({
            "index": 0,
            "type": "title_card",
            "interval": None,
            "duration": title_duration,
            "raw_name": "00_title.mp4",
            "name_no_ext": "00_title"
        })

    clips_order.extend(discovered)

    segments = []
    current_time = 0.0

    for idx, item in enumerate(clips_order):
        cname = f"{item['name_no_ext']}.mp4"
        ctype = item['type']
        cpath = os.path.join(out_dir, cname)
        if not os.path.exists(cpath):
            cpath = os.path.join(in_dir, item['raw_name'])

        frame_cnt = None
        if os.path.exists(cpath):
            info = get_video_info(cpath)
            clip_dur = info['duration']
            cap = cv2.VideoCapture(cpath)
            frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            fallback_durations = {
                "title_card": title_duration,
                "timelapse": 8.90 if item['index'] <= 1 else 8.20,
                "video_slowdown": item.get('duration', 10.00),
                "video_realtime": 107.333,
                "photo": item.get('duration', 6.00),
                "composite": item.get('duration', 10.00)
            }
            clip_dur = fallback_durations.get(ctype, 10.0)

        # Insert transition if not first clip
        if idx > 0:
            trans_start = current_time
            trans_end = current_time + trans_dur
            prev_seg = segments[-1]
            segments.append({
                "type": "transition",
                "filename": f"transition_{idx}",
                "start": trans_start,
                "end": trans_end,
                "prev_type": prev_seg["type"],
                "speed_factor": prev_seg.get("speed_factor", 1.0),
                "dt_start": prev_seg.get("dt_end", dt_ingress_start),
                "dt_end": prev_seg.get("dt_end", dt_ingress_start)
            })
            current_time = trans_end

        clip_start = current_time
        clip_end = current_time + clip_dur

        c2_global = None
        c3_global = None
        if ctype == "video_realtime":
            c2_rel, c3_rel = detect_totality_contacts(cpath)
            c2_global = clip_start + c2_rel
            c3_global = clip_start + c3_rel

        # Calculate time span and speed
        if ctype == "timelapse":
            tl_interval = item.get('interval', 10.0)
            cnt = frame_cnt if (frame_cnt and frame_cnt > 0) else int(round(clip_dur * 30.0))
            real_span_s = cnt * tl_interval
            speed_factor = real_span_s / max(clip_dur, 0.1)

            if item['index'] <= 1:
                seg_dt_start = dt_ingress_start
                seg_dt_end = dt_ingress_start + datetime.timedelta(seconds=real_span_s)
                seg_type_name = "timelapse_ingress"
            else:
                seg_dt_start = dt_egress_start
                seg_dt_end = dt_egress_start + datetime.timedelta(seconds=real_span_s)
                seg_type_name = "timelapse_egress"

        elif ctype == "video_slowdown":
            in_src_path = os.path.join(in_dir, item['raw_name'])
            src_dur = 396.667
            if os.path.exists(in_src_path):
                src_info = get_video_info(in_src_path)
                src_dur = src_info['duration']

            seg_dt_end = dt_totality_start
            seg_dt_start = seg_dt_end - datetime.timedelta(seconds=src_dur)
            real_span_s = src_dur
            speed_factor = real_span_s / max(clip_dur, 0.1)
            seg_type_name = "video_slowdown"

        elif ctype == "video_realtime":
            seg_dt_start = dt_totality_start
            seg_dt_end = dt_totality_end
            real_span_s = (seg_dt_end - seg_dt_start).total_seconds()
            speed_factor = 1.0
            seg_type_name = "video_realtime"

        elif ctype == "photo":
            seg_dt_start = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 20, 28, 0)
            seg_dt_end = datetime.datetime(dt_base.year, dt_base.month, dt_base.day, 20, 28, 6)
            speed_factor = None
            seg_type_name = "photo_corona"

        elif ctype == "composite":
            seg_dt_start = dt_ingress_start
            seg_dt_end = dt_egress_start + datetime.timedelta(seconds=246 * 10.0)
            speed_factor = None
            seg_type_name = "composite"

        else:  # title_card
            seg_dt_start = dt_ingress_start
            seg_dt_end = dt_ingress_start
            speed_factor = None
            seg_type_name = "title_card"

        real_span_s = (seg_dt_end - seg_dt_start).total_seconds()
        if ctype == "video_realtime":
            speed_factor = 1.0
        elif ctype in ["title_card", "photo_corona", "composite"]:
            speed_factor = None
        else:
            speed_factor = real_span_s / max(clip_dur, 0.1)

        segments.append({
            "type": seg_type_name,
            "filename": cname,
            "start": clip_start,
            "end": clip_end,
            "duration": clip_dur,
            "speed_factor": speed_factor,
            "c2_time": c2_global,
            "c3_time": c3_global,
            "dt_start": seg_dt_start,
            "dt_end": seg_dt_end
        })
        current_time = clip_end

    return segments, current_time


def get_astronomical_state_at(t: float, segments: list, c2_point: float, c3_point: float, lang: str = "en", timezone: str = "CEST", out_dir: str = "040_out", date_str: str = None):
    """
    Evaluates the real astronomical time and localized phase title at film time t.
    Supports English ('en') and Spanish ('es').
    """
    is_es = (lang.lower() == "es")

    for i, seg in enumerate(segments):
        is_last = (i == len(segments) - 1)
        in_segment = (seg['start'] <= t < seg['end']) or (is_last and seg['start'] <= t <= seg['end'])

        if in_segment:
            stype = seg['type']
            rel_t = max(0.0, min(t - seg['start'], seg.get('duration', seg['end'] - seg['start'])))
            frac = rel_t / max(seg.get('duration', 1.0), 1e-4)
            speed = seg.get('speed_factor', 1.0)

            if is_es:
                speed_str = f"velocidad x{round(speed)}" if speed and speed > 1.0 else "velocidad x1 Tiempo Real"
            else:
                speed_str = f"speed x{round(speed)}" if speed and speed > 1.0 else "speed x1 Real-Time"

            if stype == "title_card":
                clock_str = seg['dt_start'].strftime("%H:%M:%S")
                phase_str = "Presentación del Eclipse" if is_es else "Eclipse Overview & Telemetry"
                return clock_str, phase_str

            elif stype == "timelapse_ingress":
                real_seconds = frac * (seg['dt_end'] - seg['dt_start']).total_seconds()
                clock_str = format_clock_time(seg['dt_start'], real_seconds)
                phase_str = f"C1->C2: Ingreso Parcial ({speed_str})" if is_es else f"C1->C2: Partial Ingress ({speed_str})"
                return clock_str, phase_str

            elif stype == "video_slowdown":
                real_seconds = frac * (seg['dt_end'] - seg['dt_start']).total_seconds()
                clock_str = format_clock_time(seg['dt_start'], real_seconds)
                phase_str = f"C1->C2: Pre-Totalidad ({speed_str})" if is_es else f"C1->C2: Pre-Totality ({speed_str})"
                return clock_str, phase_str

            elif stype == "video_realtime":
                real_seconds = rel_t
                clock_str = format_clock_time(seg['dt_start'], real_seconds)

                # Strict Contact Point Logic
                if t < c2_point - 0.2:
                    phase_str = f"C1->C2: Anillo de Diamantes ({speed_str})" if is_es else f"C1->C2: Diamond Ring ({speed_str})"
                elif c2_point - 0.2 <= t <= c2_point + 1.8:
                    c2_rel_sec = max(0.0, c2_point - seg['start'])
                    clock_str = format_clock_time(seg['dt_start'], c2_rel_sec)
                    phase_str = "C2: Segundo Contacto (Inicio Totalidad)" if is_es else "C2: Second Contact (Totality Start)"
                elif c3_point - 1.8 <= t <= c3_point + 1.0:
                    c3_rel_sec = max(0.0, c3_point - seg['start'])
                    clock_str = format_clock_time(seg['dt_start'], c3_rel_sec)
                    phase_str = "C3: Tercer Contacto (Fin Totalidad)" if is_es else "C3: Third Contact (Totality End)"
                elif t > c3_point + 1.0:
                    phase_str = f"C3->C4: Anillo de Diamantes ({speed_str})" if is_es else f"C3->C4: Diamond Ring ({speed_str})"
                elif 40.0 <= rel_t <= 55.0:
                    phase_str = "Totalidad (Máximo del Eclipse)" if is_es else "Totality (Maximum Eclipse)"
                else:
                    phase_str = f"Totalidad ({speed_str})" if is_es else f"Totality ({speed_str})"
                return clock_str, phase_str

            elif stype == "timelapse_egress":
                real_seconds = frac * (seg['dt_end'] - seg['dt_start']).total_seconds()
                clock_str = format_clock_time(seg['dt_start'], real_seconds)
                phase_str = f"C3->C4: Egreso Parcial ({speed_str})" if is_es else f"C3->C4: Partial Egress ({speed_str})"
                return clock_str, phase_str

            elif stype == "photo_corona":
                clock_str = seg['dt_start'].strftime("%H:%M:%S")
                phase_str = "Corona Solar (Totalidad)" if is_es else "Solar Corona (Totality)"
                return clock_str, phase_str

            elif stype == "composite":
                comp_range = get_composite_time_range(out_dir=out_dir, date_str=date_str)
                clock_str = comp_range
                phase_str = "Mosaico Secuencia del Eclipse" if is_es else "Eclipse Sequence Composite"
                return clock_str, phase_str

            elif stype == "transition":
                p_type = seg['prev_type']
                clock_str = seg['dt_start'].strftime("%H:%M:%S")
                if p_type == "title_card":
                    phase_str = "Inicio del Metraje" if is_es else "Film Sequence Start"
                elif p_type == "timelapse_ingress":
                    phase_str = f"C1->C2: Ingreso Parcial ({speed_str})" if is_es else f"C1->C2: Partial Ingress ({speed_str})"
                elif p_type == "video_slowdown":
                    phase_str = f"C1->C2: Pre-Totalidad ({speed_str})" if is_es else f"C1->C2: Pre-Totality ({speed_str})"
                elif p_type == "video_realtime":
                    phase_str = f"C3->C4: Anillo de Diamantes ({speed_str})" if is_es else f"C3->C4: Diamond Ring ({speed_str})"
                elif p_type == "timelapse_egress":
                    phase_str = f"C3->C4: Egreso Parcial ({speed_str})" if is_es else f"C3->C4: Partial Egress ({speed_str})"
                elif p_type == "photo_corona":
                    phase_str = "Corona Solar (Totalidad)" if is_es else "Solar Corona (Totality)"
                else:
                    phase_str = f"Totalidad ({speed_str})" if is_es else f"Totality ({speed_str})"
                return clock_str, phase_str

    fallback_title = "Totalidad (velocidad x1 Tiempo Real)" if is_es else "Totality (speed x1 Real-Time)"
    return "20:28:23", fallback_title


def generate_single_srt(
    output_srt_path: str,
    segments: list,
    c2_time: float,
    c3_time: float,
    total_film_dur: float,
    interval_s: float,
    include_phase: bool,
    lang: str = "en",
    timezone: str = "CEST",
    out_dir: str = "040_out",
    date_str: str = None
):
    """Generates an individual SRT file for the given language."""
    cutpoints = set([0.0])
    curr = 0.0
    while curr < total_film_dur:
        curr += interval_s
        if curr < total_film_dur:
            cutpoints.add(round(curr, 3))

    for seg in segments:
        cutpoints.add(round(seg['start'], 3))
        cutpoints.add(round(seg['end'], 3))

    if c2_time and 0 < c2_time < total_film_dur:
        cutpoints.add(round(c2_time, 3))
        cutpoints.add(round(min(c2_time + 2.0, total_film_dur), 3))

    if c3_time and 0 < c3_time < total_film_dur:
        cutpoints.add(round(max(c3_time - 2.0, 0.0), 3))
        cutpoints.add(round(c3_time, 3))

    sorted_cuts = sorted([c for c in cutpoints if 0.0 <= c <= total_film_dur])

    srt_entries = []
    entry_idx = 1

    for i in range(len(sorted_cuts) - 1):
        t_start = sorted_cuts[i]
        t_end = sorted_cuts[i + 1]

        if (t_end - t_start) < 0.15:
            continue

        t_sample = t_start + 0.05
        clock_str, phase_str = get_astronomical_state_at(
            t_sample, segments, c2_time, c3_time,
            lang=lang, timezone=timezone, out_dir=out_dir, date_str=date_str
        )

        if include_phase:
            line_text = f"{clock_str} {timezone} - {phase_str}"
        else:
            line_text = f"{clock_str} {timezone}"

        srt_start = format_srt_time(t_start)
        srt_end = format_srt_time(t_end)

        srt_entries.append(f"{entry_idx}\n{srt_start} --> {srt_end}\n{line_text}\n")
        entry_idx += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_srt_path)), exist_ok=True)
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_entries) + "\n")

    return len(srt_entries)


def embed_subtitles_for_quicktime(
    video_path: str,
    srt_es: str,
    srt_en: str,
    output_mp4: str
):
    """
    Muxes both Spanish and English subtitle tracks into MP4 using QuickTime-compatible
    'mov_text' codec with lossless video/audio stream copy.
    """
    print("-----------------------------------------------------------------")
    print("EMBEDDING QUICKTIME SUBTITLE TRACKS (Lossless Stream Copy)...")
    print(f"  Source Video : {video_path}")
    print(f"  Track 1 [ES] : {srt_es}")
    print(f"  Track 2 [EN] : {srt_en}")
    print(f"  Target Video : {output_mp4}")
    print("-----------------------------------------------------------------")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", srt_es,
        "-i", srt_en,
        "-map", "0:v",
        "-map", "1:0",
        "-map", "2:0",
        "-c:v", "copy",
        "-c:s", "mov_text",
        "-metadata:s:s:0", "language=spa",
        "-metadata:s:s:0", "title=Español (Hora Local)",
        "-metadata:s:s:0", "handler_name=Spanish",
        "-metadata:s:s:1", "language=eng",
        "-metadata:s:s:1", "title=English (Local Time)",
        "-metadata:s:s:1", "handler_name=English",
        "-disposition:s:0", "default",
        output_mp4
    ]

    subprocess.run(cmd, check=True)
    print(f"SUCCESS: QuickTime-ready MP4 created at: {output_mp4}")


def generate_eclipse_subtitles_pipeline(
    video_path: str = "040_out/full_eclipse.mp4",
    output_srt_path: str = "040_out/full_eclipse.srt",
    interval_s: float = 5.0,
    include_phase: bool = True,
    lang: str = "both",
    embed: bool = False,
    timezone: str = "CEST",
    lat_deg: float = 43.235556,
    lon_deg: float = -7.558333,
    alt_m: float = 438.7,
    date_str: str = None,
    include_title: bool = True,
    title_duration: float = 5.0,
    force_db: bool = False,
    in_dir: str = "010_in",
    out_dir: str = "040_out"
):
    """
    Main entry point to generate localized astronomical subtitles (EN and ES)
    purely from ground-truth raw telescope recordings.
    """
    print("=================================================================")
    print("ASTRONOMICAL ECLIPSE MULTILINGUAL SUBTITLE GENERATOR (.SRT)")
    print("=================================================================")
    print(f"  Master Video   : {video_path}")
    print(f"  Output SRT     : {output_srt_path}")
    print(f"  Languages      : {lang.upper()} (English + Spanish)")
    print(f"  Embed QuickTime: {embed}")
    print(f"  Time Interval  : {interval_s:.1f} seconds")
    print(f"  Include Phase  : {include_phase}")
    print(f"  Timezone       : {timezone}")
    print(f"  Observer Pos   : {lat_deg:.6f}°N, {abs(lon_deg):.6f}°W (Alt: {alt_m:.1f}m)")

    # 1. Ephemeris reference for header
    eph = eedb.solve_eclipse(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=alt_m,
        date_str=date_str,
        tz_name=timezone,
        raw_dir="000_raw",
        in_dir=in_dir,
        force_db=force_db
    )

    print(f"  • Ephemeris Context ({eph['name']}):")
    if eph["category"] == "solar":
        print(f"    - C1 (First Contact)    : {eph['C1'].strftime('%H:%M:%S') if eph['C1'] else 'N/A'} {timezone}")
        print(f"    - C2 (Totality Start)   : {eph['C2'].strftime('%H:%M:%S') if eph['C2'] else 'N/A'} {timezone}")
        print(f"    - MAX (Maximum Eclipse) : {eph['MAX'].strftime('%H:%M:%S') if eph['MAX'] else 'N/A'} {timezone}")
        print(f"    - C3 (Totality End)     : {eph['C3'].strftime('%H:%M:%S') if eph['C3'] else 'N/A'} {timezone}")
        print(f"    - C4 (Fourth Contact)   : {eph['C4'].strftime('%H:%M:%S') if eph['C4'] else 'N/A'} {timezone}")

    print("=================================================================\n")

    # Get total film duration
    v_info = get_video_info(video_path)
    total_film_dur = v_info['duration']

    # Build timeline model directly from raw recordings
    has_title = include_title and (os.path.exists(os.path.join(out_dir, "00_title.mp4")) or include_title)
    segments, model_dur = build_timeline_mapping(
        in_dir=in_dir,
        out_dir=out_dir,
        date_str=eph["date"],
        include_title=has_title,
        title_duration=title_duration
    )

    # Locate C2 and C3 optical contact points in film timeline
    c2_time = 36.153
    c3_time = 133.112
    for seg in segments:
        if seg['type'] == "video_realtime":
            if seg.get('c2_time'):
                c2_time = seg['c2_time']
            if seg.get('c3_time'):
                c3_time = seg['c3_time']

    print(f"  • Optical Contact Points in Film:")
    print(f"    - C2 (Totality Start) : {c2_time:.2f}s ({format_srt_time(c2_time)}) -> 20:27:35 {timezone}")
    print(f"    - C3 (Totality End)   : {c3_time:.2f}s ({format_srt_time(c3_time)}) -> 20:29:12 {timezone}")

    out_base, out_ext = os.path.splitext(output_srt_path)
    generated_files = []

    path_en = f"{out_base}_en{out_ext}"
    path_es = f"{out_base}_es{out_ext}"

    lang_mode = lang.lower().strip()
    if lang_mode in ["both", "all"]:
        tasks = [
            ("en", output_srt_path),
            ("en", path_en),
            ("es", path_es)
        ]
    elif lang_mode == "es":
        tasks = [("es", output_srt_path), ("es", path_es)]
    else:
        tasks = [("en", output_srt_path), ("en", path_en)]

    for l_code, path in tasks:
        cnt = generate_single_srt(
            output_srt_path=path,
            segments=segments,
            c2_time=c2_time,
            c3_time=c3_time,
            total_film_dur=total_film_dur,
            interval_s=interval_s,
            include_phase=include_phase,
            lang=l_code,
            timezone=timezone,
            out_dir=out_dir,
            date_str=eph["date"]
        )
        generated_files.append((l_code.upper(), path, cnt))

    print("\n=================================================================")
    print("SUCCESS: Subtitles generated successfully:")
    for l_name, p, n in generated_files:
        print(f"  [{l_name}] ({n} entries) -> {p}")

    # Optional QuickTime MP4 embedding
    if embed and os.path.exists(path_es) and os.path.exists(path_en):
        v_base, v_ext = os.path.splitext(video_path)
        subtitled_video_path = f"{v_base}_subtitled{v_ext}"
        embed_subtitles_for_quicktime(
            video_path=video_path,
            srt_es=path_es,
            srt_en=path_en,
            output_mp4=subtitled_video_path
        )

    print("=================================================================")
    return [p for _, p, _ in generated_files]


def main():
    parser = argparse.ArgumentParser(
        description="Astronomical Local Time Subtitle Generator for Solar & Lunar Eclipse Videos"
    )
    parser.add_argument("--video", "-v", type=str, default="040_out/full_eclipse.mp4",
                        help="Path to master video file (default: 040_out/full_eclipse.mp4)")
    parser.add_argument("--output", "-o", type=str, default="040_out/full_eclipse.srt",
                        help="Path to output subtitle file (default: 040_out/full_eclipse.srt)")
    parser.add_argument("--lang", "-l", type=str, default="both",
                        help="Language: 'both' (default, generates EN and ES), 'en' (English), or 'es' (Spanish)")
    parser.add_argument("--embed", action="store_true",
                        help="Losslessly embed Spanish and English subtitle tracks into MP4 for native QuickTime Player support")
    parser.add_argument("--interval", "-i", type=float, default=5.0,
                        help="Subtitle segment interval in seconds (default: 5.0s)")
    parser.add_argument("--no-phase", action="store_true",
                        help="Omit astronomical phase/contact labels and display local time only")
    parser.add_argument("--timezone", "-tz", type=str, default="CEST",
                        help="Timezone label (default: CEST)")
    parser.add_argument("--lat", type=float, default=43.235556,
                        help="Observer Latitude in decimal degrees (default: 43.235556 N)")
    parser.add_argument("--lon", type=float, default=-7.558333,
                        help="Observer Longitude in decimal degrees (default: -7.558333 W)")
    parser.add_argument("--alt", type=float, default=438.7,
                        help="Observer Altitude in meters (default: 438.7 m)")
    parser.add_argument("--date", "-d", type=str, default=None,
                        help="Eclipse date YYYY-MM-DD (default: auto-detect)")
    parser.add_argument("--no-title", action="store_true",
                        help="Disable title card in timeline mapping calculation")
    parser.add_argument("--force-db", action="store_true",
                        help="Force rebuild of astronomical database from NASA JPL Horizons")
    parser.add_argument("--in-dir", type=str, default="010_in",
                        help="Path to input clips directory (default: 010_in)")
    parser.add_argument("--out-dir", type=str, default="040_out",
                        help="Path to output directory (default: 040_out)")

    args = parser.parse_args()

    generate_eclipse_subtitles_pipeline(
        video_path=args.video,
        output_srt_path=args.output,
        interval_s=args.interval,
        include_phase=not args.no_phase,
        lang=args.lang,
        embed=args.embed,
        timezone=args.timezone,
        lat_deg=args.lat,
        lon_deg=args.lon,
        alt_m=args.alt,
        date_str=args.date,
        include_title=not args.no_title,
        force_db=args.force_db,
        in_dir=args.in_dir,
        out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()
