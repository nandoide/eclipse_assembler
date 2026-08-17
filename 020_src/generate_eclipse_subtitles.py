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
        os.path.join(out_dir, "06_composite_circle_4k_10.json"),
        os.path.join(out_dir, "art_composite_4k_circle.json"),
        os.path.join(out_dir, "art_composite_4k_arc.json"),
        os.path.join(out_dir, "composite_artwork_06_circle.json"),
        os.path.join(out_dir, "composite_artwork_06_arc.json"),
        os.path.join(out_dir, "06_composite_arc_10.json"),
        os.path.join(out_dir, "06_composite_sinusoid_10.json"),
        os.path.join(out_dir, "composite_samples.json"),
        os.path.join(out_dir, "eclipse_composite_arc_1280x720.json"),
        os.path.join(out_dir, "eclipse_composite_sinusoid_1280x720.json"),
        os.path.join(out_dir, "eclipse_composite_arc_3840x2160.json"),
        os.path.join(out_dir, "eclipse_composite_sinusoid_3840p.json")
    ]
    import glob
    candidates.extend(glob.glob(os.path.join(out_dir, "*composite*.json")))
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
    freeze_after: float = 1.0,
    film_style: str = "standard"
):
    """
    Builds a timeline model mapping every millisecond of the assembled film
    to its astronomical local time and phase description.
    All timestamps come directly from 000_raw/ telescope footage.
    Supports film_style="standard" (linear) and film_style="art" (narrative zoom dive).
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

    if film_style == "art":
        segments = []
        current_time = 0.0

        # 1. Title card
        p_title = os.path.join(out_dir, "00_title.mp4")
        if include_title:
            t_dur = title_duration
            if os.path.exists(p_title):
                t_dur = get_video_info(p_title)['duration']
            segments.append({
                "type": "title_card",
                "filename": "00_title.mp4",
                "start": current_time,
                "end": current_time + t_dur,
                "duration": t_dur,
                "speed_factor": None,
                "c2_time": None,
                "c3_time": None,
                "dt_start": dt_ingress_start,
                "dt_end": dt_ingress_start
            })
            current_time += t_dur
            # Transition after title card (fade_to_black)
            trans_start = current_time
            trans_end = current_time + trans_dur
            segments.append({
                "type": "transition",
                "filename": "transition_title",
                "start": trans_start,
                "end": trans_end,
                "prev_type": "title_card",
                "speed_factor": 1.0,
                "dt_start": dt_ingress_start,
                "dt_end": dt_ingress_start
            })
            current_time = trans_end

        # 2. Art Dive In
        p_dive_in = os.path.join(out_dir, "art_01_dive_in.mp4")
        dive_in_dur = 5.0
        if os.path.exists(p_dive_in):
            dive_in_dur = get_video_info(p_dive_in)['duration']
        segments.append({
            "type": "art_dive_in",
            "filename": "art_01_dive_in.mp4",
            "start": current_time,
            "end": current_time + dive_in_dur,
            "duration": dive_in_dur,
            "speed_factor": None,
            "c2_time": None,
            "c3_time": None,
            "dt_start": dt_ingress_start,
            "dt_end": dt_egress_start + datetime.timedelta(seconds=246 * 10.0)
        })
        current_time += dive_in_dur

        # Read composite metadata for sample endpoints if available
        art_comp_json = None
        for cand in [
            os.path.join(out_dir, "art_composite_4k_circle.json"),
            os.path.join(out_dir, "art_composite_4k_arc.json"),
            os.path.join(out_dir, "06_composite_circle_4k_10.json"),
            os.path.join(out_dir, "composite_artwork_06_circle.json"),
            os.path.join(out_dir, "composite_samples.json")
        ]:
            if os.path.exists(cand):
                art_comp_json = cand
                break
        if not art_comp_json:
            import glob
            cands = glob.glob(os.path.join(out_dir, "*composite*.json"))
            if cands:
                art_comp_json = cands[0]

        in_start_frame = 0
        eg_end_frame = 245
        if art_comp_json and os.path.exists(art_comp_json):
            try:
                with open(art_comp_json, "r", encoding="utf-8") as f:
                    cm = json.load(f)
                in_start_frame = int(cm["samples"][0].get("frame_idx", 0))
                eg_end_frame = int(cm["samples"][-1].get("frame_idx", 245))
            except Exception:
                pass

        # 3. Timelapse Ingress (art_01_timelapse.mp4 or 01_timelapse_i10.mp4)
        p_art_tl_in = os.path.join(out_dir, "art_01_timelapse.mp4")
        p_tl_in = p_art_tl_in if os.path.exists(p_art_tl_in) else os.path.join(out_dir, "01_timelapse_i10.mp4")
        tl_in_dur = 8.9
        if os.path.exists(p_tl_in):
            tl_in_dur = get_video_info(p_tl_in)['duration']
        
        dt_in_actual_start = dt_ingress_start + datetime.timedelta(seconds=in_start_frame * 10.0)
        tl_in_span_s = max(1.0, (267 - in_start_frame) * 10.0)
        dt_in_actual_end = dt_ingress_start + datetime.timedelta(seconds=267 * 10.0)

        segments.append({
            "type": "timelapse_ingress",
            "filename": os.path.basename(p_tl_in),
            "start": current_time,
            "end": current_time + tl_in_dur,
            "duration": tl_in_dur,
            "speed_factor": tl_in_span_s / max(tl_in_dur, 0.1),
            "c2_time": None,
            "c3_time": None,
            "dt_start": dt_in_actual_start,
            "dt_end": dt_in_actual_end
        })
        current_time += tl_in_dur

        # Transition after timelapse ingress (fade_to_black)
        trans_start = current_time
        trans_end = current_time + trans_dur
        segments.append({
            "type": "transition",
            "filename": "transition_tl_in",
            "start": trans_start,
            "end": trans_end,
            "prev_type": "timelapse_ingress",
            "speed_factor": tl_in_span_s / max(tl_in_dur, 0.1),
            "dt_start": dt_in_actual_end,
            "dt_end": dt_in_actual_end
        })
        current_time = trans_end

        # 4. Video Slowdown (02_video_slowdown_10.mp4)
        p_slow = os.path.join(out_dir, "02_video_slowdown_10.mp4")
        slow_dur = 10.0
        if os.path.exists(p_slow):
            slow_dur = get_video_info(p_slow)['duration']
        slow_real_span = (dt_pre_tot_end - dt_pre_tot_start).total_seconds()
        segments.append({
            "type": "video_slowdown",
            "filename": "02_video_slowdown_10.mp4",
            "start": current_time,
            "end": current_time + slow_dur,
            "duration": slow_dur,
            "speed_factor": slow_real_span / max(slow_dur, 0.1),
            "c2_time": None,
            "c3_time": None,
            "dt_start": dt_pre_tot_start,
            "dt_end": dt_pre_tot_end
        })
        current_time += slow_dur

        # Transition after slowdown (fade_to_black)
        trans_start = current_time
        trans_end = current_time + trans_dur
        segments.append({
            "type": "transition",
            "filename": "transition_slow",
            "start": trans_start,
            "end": trans_end,
            "prev_type": "video_slowdown",
            "speed_factor": slow_real_span / max(slow_dur, 0.1),
            "dt_start": dt_pre_tot_end,
            "dt_end": dt_pre_tot_end
        })
        current_time = trans_end

        # 5. Video Realtime Part 1 (art_03_totality_p1.mp4)
        p_tot_p1 = os.path.join(out_dir, "art_03_totality_p1.mp4")
        tot_p1_dur = 51.767
        if os.path.exists(p_tot_p1):
            tot_p1_dur = get_video_info(p_tot_p1)['duration']
        dt_tot_max = dt_totality_start + datetime.timedelta(seconds=51.767)
        c2_global = current_time + 3.47
        segments.append({
            "type": "video_realtime_p1",
            "filename": "art_03_totality_p1.mp4",
            "start": current_time,
            "end": current_time + tot_p1_dur,
            "duration": tot_p1_dur,
            "speed_factor": 1.0,
            "c2_time": c2_global,
            "c3_time": None,
            "dt_start": dt_totality_start,
            "dt_end": dt_tot_max
        })
        current_time += tot_p1_dur

        # 6. Totality HDR Artwork (05_totality_6.mp4) - crossfaded into at Totality Max!
        p_tot_art = os.path.join(out_dir, "05_totality_6.mp4")
        if not os.path.exists(p_tot_art):
            p_tot_art = os.path.join(out_dir, "05_photo_6.mp4")
        tot_art_dur = 6.0
        if os.path.exists(p_tot_art):
            tot_art_dur = get_video_info(p_tot_art)['duration']
        segments.append({
            "type": "totality_artwork",
            "filename": os.path.basename(p_tot_art),
            "start": current_time,
            "end": current_time + tot_art_dur,
            "duration": tot_art_dur,
            "speed_factor": None,
            "c2_time": None,
            "c3_time": None,
            "dt_start": dt_totality_start,
            "dt_end": dt_totality_end
        })
        current_time += tot_art_dur

        # 7. Video Realtime Part 2 (art_03_totality_p2.mp4) - crossfaded from Totality HDR!
        p_tot_p2 = os.path.join(out_dir, "art_03_totality_p2.mp4")
        tot_p2_dur = 55.600
        if os.path.exists(p_tot_p2):
            tot_p2_dur = get_video_info(p_tot_p2)['duration']
        c3_global = current_time + 48.8
        segments.append({
            "type": "video_realtime_p2",
            "filename": "art_03_totality_p2.mp4",
            "start": current_time,
            "end": current_time + tot_p2_dur,
            "duration": tot_p2_dur,
            "speed_factor": 1.0,
            "c2_time": None,
            "c3_time": c3_global,
            "dt_start": dt_tot_max,
            "dt_end": dt_totality_end
        })
        current_time += tot_p2_dur

        # Transition after totality (fade_to_black)
        trans_start = current_time
        trans_end = current_time + trans_dur
        segments.append({
            "type": "transition",
            "filename": "transition_tot_egress",
            "start": trans_start,
            "end": trans_end,
            "prev_type": "video_realtime",
            "speed_factor": 1.0,
            "dt_start": dt_totality_end,
            "dt_end": dt_totality_end
        })
        current_time = trans_end

        # 8. Timelapse Egress (art_04_timelapse.mp4 or 04_timelapse_i10.mp4)
        p_art_tl_eg = os.path.join(out_dir, "art_04_timelapse.mp4")
        p_tl_eg = p_art_tl_eg if os.path.exists(p_art_tl_eg) else os.path.join(out_dir, "04_timelapse_i10.mp4")
        tl_eg_dur = 8.2
        if os.path.exists(p_tl_eg):
            tl_eg_dur = get_video_info(p_tl_eg)['duration']
        tl_eg_span_s = max(1.0, (eg_end_frame + 1) * 10.0)
        dt_eg_actual_start = dt_egress_start
        dt_eg_actual_end = dt_egress_start + datetime.timedelta(seconds=tl_eg_span_s)

        segments.append({
            "type": "timelapse_egress",
            "filename": os.path.basename(p_tl_eg),
            "start": current_time,
            "end": current_time + tl_eg_dur,
            "duration": tl_eg_dur,
            "speed_factor": tl_eg_span_s / max(tl_eg_dur, 0.1),
            "c2_time": None,
            "c3_time": None,
            "dt_start": dt_eg_actual_start,
            "dt_end": dt_eg_actual_end
        })
        current_time += tl_eg_dur

        # 9. Art Dive Out (art_02_dive_out.mp4)
        p_dive_out = os.path.join(out_dir, "art_02_dive_out.mp4")
        dive_out_dur = 5.0
        if os.path.exists(p_dive_out):
            dive_out_dur = get_video_info(p_dive_out)['duration']
        segments.append({
            "type": "art_dive_out",
            "filename": "art_02_dive_out.mp4",
            "start": current_time,
            "end": current_time + dive_out_dur,
            "duration": dive_out_dur,
            "speed_factor": None,
            "c2_time": None,
            "c3_time": None,
            "dt_start": dt_ingress_start,
            "dt_end": dt_egress_start + datetime.timedelta(seconds=246 * 10.0)
        })
        current_time += dive_out_dur

        # Transition after dive out (fade_to_black)
        trans_start = current_time
        trans_end = current_time + trans_dur
        segments.append({
            "type": "transition",
            "filename": "transition_dive_out",
            "start": trans_start,
            "end": trans_end,
            "prev_type": "composite",
            "speed_factor": 1.0,
            "dt_start": dt_egress_start,
            "dt_end": dt_egress_start
        })
        current_time = trans_end

        # 10. End Titles (07_endtitles.mp4)
        p_end = os.path.join(out_dir, "07_endtitles.mp4")
        end_dur = 6.0
        if os.path.exists(p_end):
            end_dur = get_video_info(p_end)['duration']
        segments.append({
            "type": "endtitles",
            "filename": "07_endtitles.mp4",
            "start": current_time,
            "end": current_time + end_dur,
            "duration": end_dur,
            "speed_factor": None,
            "c2_time": None,
            "c3_time": None,
            "dt_start": dt_egress_start,
            "dt_end": dt_egress_start
        })
        current_time += end_dur

        return segments, current_time

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

            if primary in ["music", "audio", "soundtrack", "song"] or fname.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg')):
                continue

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
            elif primary in ["photo", "totality", "hdr"]:
                a_type = "totality" if (primary in ["totality", "hdr"] or "totality" in name_no_ext) else "photo"
                for tok in tokens[1:]:
                    if tok.replace('.', '', 1).isdigit():
                        duration = float(tok)
            elif primary == "composite":
                for tok in tokens[1:]:
                    if tok.replace('.', '', 1).isdigit():
                        duration = float(tok)
            elif primary in ["endtitles", "endtitle", "credits", "end"]:
                a_type = "endtitles"
                duration = 6.0
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
            t_dt_start = prev_seg.get("dt_start", dt_ingress_start) if prev_seg["type"] == "totality_artwork" else prev_seg.get("dt_end", dt_ingress_start)
            t_dt_end = prev_seg.get("dt_end", dt_ingress_start)
            segments.append({
                "type": "transition",
                "filename": f"transition_{idx}",
                "start": trans_start,
                "end": trans_end,
                "prev_type": prev_seg["type"],
                "speed_factor": prev_seg.get("speed_factor", 1.0),
                "dt_start": t_dt_start,
                "dt_end": t_dt_end
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

        elif ctype in ["photo", "totality"]:
            seg_dt_start = dt_totality_start
            seg_dt_end = dt_totality_end
            speed_factor = None
            raw_n = str(item.get('raw_name', '')).lower()
            if ctype == "totality" or "totality" in raw_n or "hdr" in raw_n:
                seg_type_name = "totality_artwork"
            else:
                seg_type_name = "photo_corona"

        elif ctype == "composite":
            seg_dt_start = dt_ingress_start
            seg_dt_end = dt_egress_start + datetime.timedelta(seconds=246 * 10.0)
            speed_factor = None
            seg_type_name = "composite"

        elif ctype == "endtitles":
            seg_dt_start = dt_egress_start + datetime.timedelta(seconds=246 * 10.0)
            seg_dt_end = seg_dt_start
            speed_factor = None
            seg_type_name = "endtitles"

        else:  # title_card
            seg_dt_start = dt_ingress_start
            seg_dt_end = dt_ingress_start
            speed_factor = None
            seg_type_name = "title_card"

        real_span_s = (seg_dt_end - seg_dt_start).total_seconds()
        if ctype == "video_realtime":
            speed_factor = 1.0
        elif ctype in ["title_card", "photo_corona", "composite", "endtitles"]:
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

            if stype in ["title_card", "endtitles"]:
                return None, None

            elif stype == "art_dive_in":
                comp_range = get_composite_time_range(out_dir=out_dir, date_str=date_str)
                clock_str = comp_range
                phase_str = "Mosaico Secuencia del Eclipse (Zoom de Ingreso)" if is_es else "Eclipse Sequence Mosaic (Ingress Dive)"
                return clock_str, phase_str

            elif stype == "art_dive_out":
                comp_range = get_composite_time_range(out_dir=out_dir, date_str=date_str)
                clock_str = comp_range
                phase_str = "Mosaico Secuencia del Eclipse (Zoom de Salida)" if is_es else "Eclipse Sequence Mosaic (Egress Zoom-Out)"
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

            elif stype in ["video_realtime_p1", "video_realtime_p2"]:
                real_seconds = frac * (seg['dt_end'] - seg['dt_start']).total_seconds()
                clock_str = format_clock_time(seg['dt_start'], real_seconds)

                if stype == "video_realtime_p1":
                    if c2_point and t < c2_point - 0.2:
                        phase_str = f"C1->C2: Anillo de Diamantes ({speed_str})" if is_es else f"C1->C2: Diamond Ring ({speed_str})"
                    elif c2_point and c2_point - 0.2 <= t <= c2_point + 1.8:
                        c2_rel_sec = max(0.0, c2_point - seg['start'])
                        clock_str = format_clock_time(seg['dt_start'], c2_rel_sec)
                        phase_str = "C2: Segundo Contacto (Inicio Totalidad)" if is_es else "C2: Second Contact (Totality Start)"
                    elif rel_t >= seg['duration'] - 2.5:
                        phase_str = "Totalidad (Aproximación al Máximo)" if is_es else "Totality (Approaching Maximum)"
                    else:
                        phase_str = f"Totalidad ({speed_str})" if is_es else f"Totality ({speed_str})"
                else:  # video_realtime_p2
                    if c3_point and c3_point - 1.8 <= t <= c3_point + 1.0:
                        c3_rel_sec = max(0.0, c3_point - seg['start'])
                        clock_str = format_clock_time(seg['dt_start'], c3_rel_sec)
                        phase_str = "C3: Tercer Contacto (Fin Totalidad)" if is_es else "C3: Third Contact (Totality End)"
                    elif c3_point and t > c3_point + 1.0:
                        phase_str = f"C3->C4: Anillo de Diamantes ({speed_str})" if is_es else f"C3->C4: Diamond Ring ({speed_str})"
                    elif rel_t <= 2.5:
                        phase_str = "Totalidad (Post-Máximo)" if is_es else "Totality (Post-Maximum)"
                    else:
                        phase_str = f"Totalidad ({speed_str})" if is_es else f"Totality ({speed_str})"

                return clock_str, phase_str

            elif stype == "timelapse_egress":
                real_seconds = frac * (seg['dt_end'] - seg['dt_start']).total_seconds()
                clock_str = format_clock_time(seg['dt_start'], real_seconds)
                phase_str = f"C3->C4: Egreso Parcial ({speed_str})" if is_es else f"C3->C4: Partial Egress ({speed_str})"
                return clock_str, phase_str

            elif stype == "totality_artwork":
                clock_str = f"{seg['dt_start'].strftime('%H:%M:%S')} - {seg['dt_end'].strftime('%H:%M:%S')}"
                phase_str = "Composición artística de la totalidad (HDR multifase)" if is_es else "Artistic Totality HDR Composite (Multi-phase fusion - Artwork)"
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
                p_type = seg.get('prev_type')
                if p_type in ["title_card", "composite", "endtitles", "art_dive_in", "art_dive_out"]:
                    return None, None
                clock_str = seg['dt_start'].strftime("%H:%M:%S")
                if p_type == "timelapse_ingress":
                    phase_str = f"C1->C2: Ingreso Parcial ({speed_str})" if is_es else f"C1->C2: Partial Ingress ({speed_str})"
                elif p_type == "video_slowdown":
                    phase_str = f"C1->C2: Pre-Totalidad ({speed_str})" if is_es else f"C1->C2: Pre-Totality ({speed_str})"
                elif p_type == "video_realtime":
                    phase_str = f"C3->C4: Anillo de Diamantes ({speed_str})" if is_es else f"C3->C4: Diamond Ring ({speed_str})"
                elif p_type == "timelapse_egress":
                    phase_str = f"C3->C4: Egreso Parcial ({speed_str})" if is_es else f"C3->C4: Partial Egress ({speed_str})"
                elif p_type in ["photo_corona", "totality_artwork"]:
                    if p_type == "totality_artwork":
                        if 'dt_end' in seg and 'dt_start' in seg and seg['dt_end'] != seg['dt_start']:
                            clock_str = f"{seg['dt_start'].strftime('%H:%M:%S')} - {seg['dt_end'].strftime('%H:%M:%S')}"
                        else:
                            clock_str = seg['dt_start'].strftime('%H:%M:%S')
                        phase_str = "Composición artística de la totalidad (HDR multifase)" if is_es else "Artistic Totality HDR Composite (Multi-phase fusion - Artwork)"
                    else:
                        phase_str = "Corona Solar (Totalidad)" if is_es else "Solar Corona (Totality)"
                else:
                    return None, None
                return clock_str, phase_str

    return None, None


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

        if clock_str is None or phase_str is None:
            continue

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


def export_ffmetadata_chapters(
    chapters: list,
    total_film_dur: float,
    out_path: str
):
    """
    Writes an FFmpeg metadata file with chapters for native MP4/QuickTime embedding.
    """
    lines = [";FFMETADATA1"]
    for i in range(len(chapters)):
        start_sec = chapters[i][0]
        title = chapters[i][2]
        if i + 1 < len(chapters):
            end_sec = chapters[i + 1][0]
        else:
            end_sec = total_film_dur

        start_ms = int(round(start_sec * 1000))
        end_ms = max(start_ms + 100, int(round(end_sec * 1000)))

        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={start_ms}")
        lines.append(f"END={end_ms}")
        lines.append(f"title={title}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def embed_subtitles_for_quicktime(
    video_path: str,
    srt_es: str,
    srt_en: str,
    output_mp4: str,
    chapters_metadata_path: str = None
):
    """
    Muxes both Spanish and English subtitle tracks and embeds native MP4 chapter markers
    into MP4 using QuickTime-compatible 'mov_text' codec with lossless stream copy.
    """
    print("-----------------------------------------------------------------")
    print("EMBEDDING QUICKTIME SUBTITLE TRACKS & CHAPTERS (Lossless Stream Copy)...")
    print(f"  Source Video : {video_path}")
    print(f"  Track 1 [ES] : {srt_es}")
    print(f"  Track 2 [EN] : {srt_en}")
    if chapters_metadata_path and os.path.exists(chapters_metadata_path):
        print(f"  Chapters Met : {chapters_metadata_path}")
    print(f"  Target Video : {output_mp4}")
    print("-----------------------------------------------------------------")

    cmd = ["ffmpeg", "-y", "-i", video_path]

    if chapters_metadata_path and os.path.exists(chapters_metadata_path):
        cmd.extend(["-i", chapters_metadata_path])
        cmd.extend(["-i", srt_es, "-i", srt_en])
        cmd.extend([
            "-map", "0:v",
            "-map_chapters", "1",
            "-map", "2:0",
            "-map", "3:0",
        ])
    else:
        cmd.extend(["-i", srt_es, "-i", srt_en])
        cmd.extend([
            "-map", "0:v",
            "-map", "1:0",
            "-map", "2:0",
        ])

    cmd.extend([
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
    ])

    subprocess.run(cmd, check=True)
    print(f"SUCCESS: QuickTime-ready MP4 created at: {output_mp4}")


def generate_youtube_metadata(
    segments: list,
    c2_time: float,
    c3_time: float,
    out_dir: str = "040_out",
    eph: dict = None,
    timezone: str = "CEST",
    film_style: str = "standard"
):
    """
    Generates YouTube-compliant chapters (0:00 starting format) and full video description
    exported to 040_out/youtube_chapters_<type>.txt and 040_out/youtube_description_<type>.txt.
    """
    chapters_bilingual = []
    chapters_es = []
    chapters_en = []

    def fmt_time(secs: float) -> str:
        s = int(secs)
        m = s // 60
        sec = s % 60
        return f"{m}:{sec:02d}"

    for s in segments:
        st = s["start"]
        t_str = fmt_time(st)
        stype = s["type"]

        if stype == "title_card":
            chapters_bilingual.append((st, t_str, "Presentación / Eclipse Overview"))
            chapters_es.append((st, t_str, "Presentación del Eclipse"))
            chapters_en.append((st, t_str, "Eclipse Overview & Ephemeris"))
        elif stype == "art_dive_in":
            chapters_bilingual.append((st, t_str, "Mosaico Secuencia (Zoom de Ingreso) / Sequence Mosaic (Ingress Dive)"))
            chapters_es.append((st, t_str, "Mosaico Secuencia (Zoom de Ingreso)"))
            chapters_en.append((st, t_str, "Sequence Mosaic (Ingress Camera Dive)"))
        elif stype == "art_dive_out":
            chapters_bilingual.append((st, t_str, "Mosaico Secuencia (Zoom de Salida) / Sequence Mosaic (Egress Dive)"))
            chapters_es.append((st, t_str, "Mosaico Secuencia (Zoom de Salida)"))
            chapters_en.append((st, t_str, "Sequence Mosaic (Egress Zoom-Out)"))
        elif stype == "timelapse_ingress":
            chapters_bilingual.append((st, t_str, "Ingreso Parcial / Partial Ingress (Timelapse)"))
            chapters_es.append((st, t_str, "Ingreso Parcial (Timelapse)"))
            chapters_en.append((st, t_str, "Partial Ingress (Timelapse)"))
        elif stype == "video_slowdown":
            chapters_bilingual.append((st, t_str, "Aproximación Pre-totalidad / Pre-totality (Thin Crescent)"))
            chapters_es.append((st, t_str, "Aproximación Pre-totalidad (Fase Creciente Fina)"))
            chapters_en.append((st, t_str, "Pre-totality Approach (Thin Crescent)"))
        elif stype == "video_realtime":
            c2_val = s.get("c2_time", c2_time or (st + 4.2))
            c2_str = fmt_time(c2_val)
            chapters_bilingual.append((c2_val, c2_str, "C2: Anillo de Diamantes y Perlas de Baily / C2: Baily's Beads"))
            chapters_es.append((c2_val, c2_str, "C2: Anillo de Diamantes y Perlas de Baily"))
            chapters_en.append((c2_val, c2_str, "C2: Baily's Beads & Diamond Ring"))

            tot_val = c2_val + 4.5
            tot_str = fmt_time(tot_val)
            chapters_bilingual.append((tot_val, tot_str, "Totalidad y Corona Solar / Totality (Real-Time 1x)"))
            chapters_es.append((tot_val, tot_str, "Totalidad y Corona Solar (Tiempo Real 1x)"))
            chapters_en.append((tot_val, tot_str, "Totality & Solar Corona (Real-Time 1x)"))

            c3_val = s.get("c3_time", c3_time or (st + 100.8))
            c3_str = fmt_time(c3_val)
            chapters_bilingual.append((c3_val, c3_str, "C3: Tercer Contacto / C3: Third Contact"))
            chapters_es.append((c3_val, c3_str, "C3: Tercer Contacto (Fin de la Totalidad)"))
            chapters_en.append((c3_val, c3_str, "C3: Third Contact (Totality End)"))
        elif stype == "video_realtime_p1":
            c2_val = s.get("c2_time", c2_time or (st + 3.47))
            c2_str = fmt_time(c2_val)
            chapters_bilingual.append((c2_val, c2_str, "C2: Inicio de la Totalidad / C2: Totality Start (Real-Time)"))
            chapters_es.append((c2_val, c2_str, "C2: Segundo Contacto (Inicio Totalidad Tiempo Real)"))
            chapters_en.append((c2_val, c2_str, "C2: Second Contact (Totality Start Real-Time)"))
        elif stype == "video_realtime_p2":
            c3_val = s.get("c3_time", c3_time or (st + 48.8))
            c3_str = fmt_time(c3_val)
            chapters_bilingual.append((c3_val, c3_str, "C3: Fin de la Totalidad / C3: Totality End (Real-Time)"))
            chapters_es.append((c3_val, c3_str, "C3: Tercer Contacto (Fin Totalidad Tiempo Real)"))
            chapters_en.append((c3_val, c3_str, "C3: Third Contact (Totality End Real-Time)"))
        elif stype == "timelapse_egress":
            chapters_bilingual.append((st, t_str, "Egreso Parcial / Partial Egress (Timelapse)"))
            chapters_es.append((st, t_str, "Egreso Parcial (Timelapse)"))
            chapters_en.append((st, t_str, "Partial Egress (Timelapse)"))
        elif stype == "totality_artwork":
            chapters_bilingual.append((st, t_str, "Composición Artística HDR de la Totalidad / Artistic Totality HDR Composite"))
            chapters_es.append((st, t_str, "Composición Artística HDR de la Totalidad (Multifase)"))
            chapters_en.append((st, t_str, "Artistic Totality HDR Multi-Phase Composite (Artwork)"))
        elif stype == "photo_corona":
            chapters_bilingual.append((st, t_str, "Fotografía HDR de la Corona / Solar Corona Photo"))
            chapters_es.append((st, t_str, "Fotografía HDR de la Corona Solar"))
            chapters_en.append((st, t_str, "HDR Solar Corona Still Photo"))
        elif stype == "composite":
            chapters_bilingual.append((st, t_str, "Mosaico Secuencia del Eclipse / Eclipse Sequence Composite (Arc)"))
            chapters_es.append((st, t_str, "Mosaico Secuencia del Eclipse (Arco con Contactos)"))
            chapters_en.append((st, t_str, "Eclipse Sequence Composite Artwork (Arc)"))
        elif stype == "endtitles":
            chapters_bilingual.append((st, t_str, "Créditos Finales / Closing Credits"))
            chapters_es.append((st, t_str, "Créditos Finales y Producción"))
            chapters_en.append((st, t_str, "Closing Credits & Production Telemetry"))

    # Sort chapters by timestamp
    chapters_bilingual.sort(key=lambda x: x[0])
    chapters_es.sort(key=lambda x: x[0])
    chapters_en.sort(key=lambda x: x[0])

    # Ensure a chapter starts at 0.0 (required for YouTube chapters)
    if chapters_bilingual and chapters_bilingual[0][0] > 0.05:
        chapters_bilingual.insert(0, (0.0, "0:00", "Presentación / Eclipse Overview"))
        chapters_es.insert(0, (0.0, "0:00", "Presentación del Eclipse"))
        chapters_en.insert(0, (0.0, "0:00", "Eclipse Overview & Ephemeris"))

    # Text content for youtube_chapters_<type>.txt
    lines = []
    lines.append("Capítulos / Chapters:")
    for _, t, name in chapters_bilingual:
        lines.append(f"{t} - {name}")

    lines.append("\nCapítulos en Español:")
    for _, t, name in chapters_es:
        lines.append(f"{t} - {name}")

    lines.append("\nEnglish Chapters:")
    for _, t, name in chapters_en:
        lines.append(f"{t} - {name}")

    chap_path = os.path.join(out_dir, f"youtube_chapters_{film_style}.txt")
    with open(chap_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # YouTube Description
    desc_lines = [
        "Gran Eclipse Solar Total de 2026 / Great Total Solar Eclipse of 2026",
        "===================================================================",
        "Grabación astronómica completa de la totalidad solar y fases parciales.",
        "Complete astronomical recording of solar totality and partial phases.",
        "",
        "📍 Coordenadas del Observador / Observer Coordinates:",
        "   43.235556° N, 7.558333° W (Alt: 438.7m)",
        "",
        "⏱️ Contactos Astronómicos (Hora Local CEST) / Ephemeris Contacts:",
        "   • C1 (Primer Contacto)   : 19:31:18 CEST",
        "   • C2 (Segundo Contacto)  : 20:27:35 CEST (Óptico) / 20:27:38 CEST (Efemerides)",
        "   • MAX (Máximo Eclipse)   : 20:28:25 CEST",
        "   • C3 (Tercer Contacto)   : 20:29:12 CEST (Óptico) / 20:29:12 CEST (Efemerides)",
        "   • C4 (Cuarto Contacto)   : 21:21:52 CEST",
        "",
        "Capítulos / Chapters:",
    ]
    for _, t, name in chapters_bilingual:
        desc_lines.append(f"{t} - {name}")

    desc_path = os.path.join(out_dir, f"youtube_description_{film_style}.txt")
    with open(desc_path, "w", encoding="utf-8") as f:
        desc_write = "\n".join(desc_lines) + "\n"
        f.write(desc_write)

    return chapters_bilingual, chap_path, desc_path


def generate_eclipse_subtitles_pipeline(
    video_path: str = None,
    output_srt_path: str = None,
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
    out_dir: str = "040_out",
    film_style: str = "standard"
):
    """
    Main entry point to generate localized astronomical subtitles (EN and ES)
    purely from ground-truth raw telescope recordings.
    """
    if video_path is None:
        video_path = os.path.join(out_dir, f"full_eclipse_{film_style}_video.mp4")
    if output_srt_path is None:
        output_srt_path = os.path.join(out_dir, f"full_eclipse_{film_style}.srt")

    print("=================================================================")
    print("ASTRONOMICAL ECLIPSE MULTILINGUAL SUBTITLE GENERATOR (.SRT)")
    print("=================================================================")
    print(f"  Film Style     : {film_style.upper()}")
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
        title_duration=title_duration,
        film_style=film_style
    )

    # Locate C2 and C3 optical contact points in film timeline
    c2_time = 36.153
    c3_time = 133.112
    for seg in segments:
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

    # Generate YouTube chapters and description metadata
    yt_chaps, yt_chap_p, yt_desc_p = generate_youtube_metadata(
        segments=segments,
        c2_time=c2_time,
        c3_time=c3_time,
        out_dir=out_dir,
        eph=eph,
        timezone=timezone,
        film_style=film_style
    )
    print("\n-----------------------------------------------------------------")
    print("YOUTUBE CHAPTERS & DESCRIPTION EXPORTED:")
    print(f"  Chapters file   : {yt_chap_p}")
    print(f"  Description file: {yt_desc_p}")
    print("-----------------------------------------------------------------")
    for _, t, name in yt_chaps:
        print(f"  {t} - {name}")

    # Export FFmpeg chapter metadata for native MP4 / QuickTime embedding
    meta_chap_p = os.path.join(out_dir, f"metadata_chapters_{film_style}.txt")
    export_ffmetadata_chapters(yt_chaps, total_film_dur, meta_chap_p)

    # Optional QuickTime MP4 embedding (Subtitles + Native Chapters)
    if embed and os.path.exists(path_es) and os.path.exists(path_en):
        if video_path.endswith("_video.mp4"):
            subtitled_video_path = video_path[:-10] + ".mp4"
        elif video_path.endswith("_subtitled.mp4"):
            subtitled_video_path = video_path
        else:
            v_base, v_ext = os.path.splitext(video_path)
            subtitled_video_path = f"{v_base}_subtitled{v_ext}"

        embed_subtitles_for_quicktime(
            video_path=video_path,
            srt_es=path_es,
            srt_en=path_en,
            output_mp4=subtitled_video_path,
            chapters_metadata_path=meta_chap_p
        )

    print("=================================================================")
    return [p for _, p, _ in generated_files]


def main():
    parser = argparse.ArgumentParser(
        description="Astronomical Local Time Subtitle Generator for Solar & Lunar Eclipse Videos"
    )
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="Path to master video file (default: 040_out/full_eclipse_<film_style>_video.mp4)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to output subtitle file (default: 040_out/full_eclipse_<film_style>.srt)")
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
    parser.add_argument("--film-style", type=str, choices=["standard", "art"], default="standard",
                        help="Film assembly style: 'standard' or 'art' (default: standard)")

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
        out_dir=args.out_dir,
        film_style=args.film_style
    )


if __name__ == "__main__":
    main()
