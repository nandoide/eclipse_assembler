#!/usr/bin/env python3
"""
================================================================================
CINEMATIC TITLE CARD GENERATOR FOR SOLAR & LUNAR ECLIPSE FILMS
================================================================================
Generates an elegant, high-precision cinematic opening title card (00_title.png
and 00_title.mp4) using dynamic astronomical ephemeris data from eclipse_ephemeris_db.py.

Outputs:
  - 040_out/00_title.png (High-Res Anti-Aliased Graphic)
  - 040_out/00_title.mp4 (HEVC / H.264 Video Clip of specified duration, default: 5.0s)

Authors: Fernando (nandoide) & Antigravity (Google Deepmind)
================================================================================
"""

import os
import sys
import argparse
import subprocess
import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eclipse_ephemeris_db as eedb


def get_font(size: int, bold: bool = False):
    """Retrieves standard clean high-quality font."""
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Geneva.dfont",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arial.ttf"
    ]
    for p in font_candidates:
        if os.path.exists(p):
            try:
                # index=0 for standard font inside ttc
                return ImageFont.truetype(p, size, index=0)
            except Exception:
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
    return ImageFont.load_default()


def render_title_card_image(
    eph_meta: dict,
    width: int = 1280,
    height: int = 720
) -> Image.Image:
    """
    Renders an elegant, anti-aliased cinematic title card on a solid black canvas (English only).
    """
    # 2x supersampling for ultra-crisp typography
    ss = 2
    sw, sh = width * ss, height * ss
    img = Image.new("RGB", (sw, sh), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    cat = eph_meta.get("category", "solar")
    etype = eph_meta.get("type", "Total")
    date_str = eph_meta.get("date", "2026-08-12")
    tz_name = eph_meta.get("tz_name", "CEST")

    # Format date string nicely (e.g. August 12, 2026)
    try:
        dt_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        formatted_date = dt_obj.strftime("%B %d, %Y")
        year_str = str(dt_obj.year)
    except Exception:
        formatted_date = date_str
        year_str = "2026"

    # 1. Main Title & Subheading (No year in first line)
    if cat == "solar":
        main_title = f"{etype.upper()} SOLAR ECLIPSE"
    else:
        main_title = f"{etype.upper()} LUNAR ECLIPSE"

    date_title = formatted_date

    # 2. Location String
    lat = eph_meta.get("lat_deg", 43.235556)
    lon = eph_meta.get("lon_deg", -7.558333)
    alt = eph_meta.get("alt_m", 438.7)
    lat_card = "N" if lat >= 0 else "S"
    lon_card = "E" if lon >= 0 else "W"
    loc_str = f"Observation Site: {abs(lat):.4f}° {lat_card}, {abs(lon):.4f}° {lon_card}   |   Alt: {int(round(alt))} m"

    # 3. Phase timings strings
    if cat == "solar":
        t_c1 = eph_meta.get("C1")
        t_c2 = eph_meta.get("C2")
        t_c3 = eph_meta.get("C3")
        t_c4 = eph_meta.get("C4")

        c1_s = t_c1.strftime("%H:%M:%S") if t_c1 else "--:--:--"
        c4_s = t_c4.strftime("%H:%M:%S") if t_c4 else "--:--:--"
        partial_span_str = f"{c1_s} - {c4_s} {tz_name}"

        if t_c2 and t_c3:
            c2_s = t_c2.strftime("%H:%M:%S")
            c3_s = t_c3.strftime("%H:%M:%S")
            dur_s = (t_c3 - t_c2).total_seconds()
            m = int(dur_s // 60)
            s = int(round(dur_s % 60))
            dur_label = f"{m}m {s:02d}s" if m > 0 else f"{s}s"
            totality_span_str = f"Totality (C2 -> C3):  {c2_s} - {c3_s} {tz_name}  (Duration: {dur_label})"
        else:
            totality_span_str = f"Maximum Eclipse:  {eph_meta.get('MAX').strftime('%H:%M:%S') if eph_meta.get('MAX') else '--:--:--'} {tz_name}"

    else:
        # Lunar Eclipse
        t_p1 = eph_meta.get("P1")
        t_p4 = eph_meta.get("P4")
        t_u2 = eph_meta.get("U2")
        t_u3 = eph_meta.get("U3")

        p1_s = t_p1.strftime("%H:%M:%S") if t_p1 else "--:--:--"
        p4_s = t_p4.strftime("%H:%M:%S") if t_p4 else "--:--:--"
        partial_span_str = f"{p1_s} - {p4_s} {tz_name}"

        if t_u2 and t_u3:
            u2_s = t_u2.strftime("%H:%M:%S")
            u3_s = t_u3.strftime("%H:%M:%S")
            dur_s = (t_u3 - t_u2).total_seconds()
            h = int(dur_s // 3600)
            m = int((dur_s % 3600) // 60)
            dur_label = f"{h}h {m:02d}m" if h > 0 else f"{m}m"
            totality_span_str = f"Totality (U2 -> U3):  {u2_s} - {u3_s} {tz_name}  (Duration: {dur_label})"
        else:
            totality_span_str = f"Maximum Eclipse:  {eph_meta.get('MAX').strftime('%H:%M:%S') if eph_meta.get('MAX') else '--:--:--'} {tz_name}"

    # Typography & Scaling
    font_main  = get_font(int(46 * ss), bold=True)
    font_date  = get_font(int(28 * ss), bold=False)
    font_loc   = get_font(int(22 * ss), bold=False)
    font_meta  = get_font(int(21 * ss), bold=False)
    font_tot   = get_font(int(21 * ss), bold=True)

    # Color palette
    color_gold  = (255, 204, 102)   # Warm Solar Gold
    color_white = (245, 245, 245)   # Off-White
    color_muted = (170, 175, 185)   # Cool Slate Gray
    color_amber = (255, 220, 150)   # Pale Amber

    # Center placement
    cx = sw / 2.0
    cy = sh / 2.0

    # Layout structure:
    # 1. Main Title
    # 2. Date
    # [Separator Line]
    # 3. Location
    # 4. Partial Span
    # 5. Totality Span (Gold Highlight)

    # Calculate vertical heights
    h_main = 55 * ss
    h_date = 36 * ss
    h_gap1 = 28 * ss
    h_line = 2 * ss
    h_gap2 = 32 * ss
    h_loc  = 30 * ss
    h_part = 30 * ss
    h_tot  = 30 * ss

    total_block_h = h_main + h_date + h_gap1 + h_line + h_gap2 + h_loc + h_part + h_tot
    top_y = cy - total_block_h / 2.0

    curr_y = top_y

    # Draw Main Title
    bbox = draw.textbbox((0, 0), main_title, font=font_main)
    w_txt = bbox[2] - bbox[0]
    draw.text((cx - w_txt / 2.0, curr_y), main_title, font=font_main, fill=color_white)
    curr_y += h_main

    # Draw Date
    bbox = draw.textbbox((0, 0), date_title, font=font_date)
    w_txt = bbox[2] - bbox[0]
    draw.text((cx - w_txt / 2.0, curr_y), date_title, font=font_date, fill=color_gold)
    curr_y += h_date + h_gap1

    # Draw Subtle Divider Line
    line_w = 420 * ss
    draw.line([(cx - line_w / 2.0, curr_y), (cx + line_w / 2.0, curr_y)], fill=(70, 75, 85), width=max(1, int(1.5 * ss)))
    curr_y += h_line + h_gap2

    # Draw Location
    bbox = draw.textbbox((0, 0), loc_str, font=font_loc)
    w_txt = bbox[2] - bbox[0]
    draw.text((cx - w_txt / 2.0, curr_y), loc_str, font=font_loc, fill=color_muted)
    curr_y += h_loc + int(10 * ss)

    # Draw Partial Phase
    bbox = draw.textbbox((0, 0), partial_span_str, font=font_meta)
    w_txt = bbox[2] - bbox[0]
    draw.text((cx - w_txt / 2.0, curr_y), partial_span_str, font=font_meta, fill=color_muted)
    curr_y += h_part

    # Draw Totality Phase
    bbox = draw.textbbox((0, 0), totality_span_str, font=font_tot)
    w_txt = bbox[2] - bbox[0]
    draw.text((cx - w_txt / 2.0, curr_y), totality_span_str, font=font_tot, fill=color_amber)

    # Downsample with Lanczos for ultra-smooth anti-aliased rendering
    final_img = img.resize((width, height), Image.Resampling.LANCZOS)
    return final_img


def generate_title_card(
    date_str: str = None,
    lat_deg: float = 43.235556,
    lon_deg: float = -7.558333,
    alt_m: float = 438.7,
    tz_name: str = "CEST",
    tz_offset_hours: float = 2.0,
    duration_s: float = 5.0,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    crf: int = 16,
    out_dir: str = "040_out",
    output_mp4: str = None,
    force_db: bool = False
) -> tuple:
    """
    Solves ephemeris, generates title card image and renders output video clip.
    Returns: (png_path, mp4_path).
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Solve ephemeris for the eclipse date and coordinates
    eph = eedb.solve_eclipse(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=alt_m,
        date_str=date_str,
        tz_name=tz_name,
        tz_offset_hours=tz_offset_hours,
        force_db=force_db
    )

    # 2. Render image
    title_img = render_title_card_image(eph, width=width, height=height)

    png_path = os.path.join(out_dir, "00_title.png")
    jpg_path = os.path.join(out_dir, "00_title.jpg")
    title_img.save(png_path, format="PNG")
    title_img.save(jpg_path, format="JPEG", quality=98)

    # 3. Render video clip of duration_s
    mp4_target = output_mp4 or os.path.join(out_dir, "00_title.mp4")
    total_frames = int(round(duration_s * fps))

    # Convert PIL Image to BGR for OpenCV / FFmpeg streaming
    cv_frame = cv2.cvtColor(np.array(title_img), cv2.COLOR_RGB2BGR)
    frame_bytes = cv_frame.tobytes()

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx265',
        '-crf', str(crf),
        '-preset', 'fast',
        '-tag:v', 'hvc1',
        '-movflags', '+faststart',
        '-pix_fmt', 'yuv420p',
        '-color_range', '1',
        '-colorspace', 'bt709',
        '-color_trc', 'bt709',
        '-color_primaries', 'bt709',
        mp4_target
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(total_frames):
        proc.stdin.write(frame_bytes)
    proc.stdin.close()
    proc.wait()

    return png_path, mp4_target


def main():
    parser = argparse.ArgumentParser(description="Cinematic Eclipse Title Card Generator (00_title.mp4)")
    parser.add_argument("--date", "-d", type=str, default=None, help="Eclipse date YYYY-MM-DD (default: auto-detect)")
    parser.add_argument("--lat", type=float, default=43.235556, help="Observer Latitude in degrees")
    parser.add_argument("--lon", type=float, default=-7.558333, help="Observer Longitude in degrees")
    parser.add_argument("--alt", type=float, default=438.7, help="Observer Altitude in meters")
    parser.add_argument("--timezone", "-tz", type=str, default="CEST", help="Timezone label (default: CEST)")
    parser.add_argument("--tz-offset", type=float, default=2.0, help="UTC offset in hours (default: 2.0)")
    parser.add_argument("--duration", type=float, default=5.0, help="Clip duration in seconds (default: 5.0s)")
    parser.add_argument("--width", "-W", type=int, default=1280, help="Canvas width (default: 1280)")
    parser.add_argument("--height", "-H", type=int, default=720, help="Canvas height (default: 720)")
    parser.add_argument("--out-dir", type=str, default="040_out", help="Output directory (default: 040_out)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Custom output MP4 path")
    parser.add_argument("--force-db", action="store_true", help="Force refresh eclipse database")
    args = parser.parse_args()

    png_out, mp4_out = generate_title_card(
        date_str=args.date,
        lat_deg=args.lat,
        lon_deg=args.lon,
        alt_m=args.alt,
        tz_name=args.timezone,
        tz_offset_hours=args.tz_offset,
        duration_s=args.duration,
        width=args.width,
        height=args.height,
        out_dir=args.out_dir,
        output_mp4=args.output,
        force_db=args.force_db
    )

    print("=================================================================")
    print("SUCCESS: Title card generated!")
    print(f"  Image (Lossless) : {png_out}")
    print(f"  Video Clip ({args.duration}s) : {mp4_out}")
    print("=================================================================")


if __name__ == "__main__":
    main()
