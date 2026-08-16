#!/usr/bin/env python3
"""
================================================================================
CINEMATIC END TITLES & CREDITS GENERATOR FOR ECLIPSE FILMS
================================================================================
Generates an elegant, high-precision cinematic closing credits card
(07_endtitles.png and 07_endtitles.mp4) parsed from a markdown descriptor
(e.g., 010_in/07_endtitles.md).

Includes:
  - Custom sections provided by the user (# Author, # Telescope, # Software, etc.)
  - Software processing credits (integrating eclipse_assembler and user tools)
  - Creation & Generation metadata and timestamp
  - 2x supersampled Lanczos anti-aliasing on solid black canvas

Outputs:
  - 040_out/07_endtitles.png (High-Res Anti-Aliased Graphic)
  - 040_out/07_endtitles.mp4 (HEVC Video Clip of specified duration, default: 6.0s)

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
                return ImageFont.truetype(p, size, index=0)
            except Exception:
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
    return ImageFont.load_default()


def parse_endtitles_markdown(md_path: str) -> list:
    """
    Parses a markdown file structured with `# Header` followed by content lines.
    Returns an ordered list of (header_str, [line_str, ...]).
    """
    if not os.path.exists(md_path):
        return []

    sections = []
    current_header = None
    current_lines = []

    with open(md_path, 'r', encoding='utf-8') as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            if raw.startswith('#'):
                if current_header is not None and (current_lines or current_header):
                    sections.append((current_header, current_lines))
                current_header = raw.lstrip('#').strip()
                current_lines = []
            else:
                if current_header is None:
                    current_header = "Information"
                current_lines.append(raw)

    if current_header is not None:
        sections.append((current_header, current_lines))

    return sections


def render_end_titles_image(
    sections: list,
    width: int = 1280,
    height: int = 720,
    generation_dt: datetime.datetime = None,
    eclipse_date_str: str = None
) -> Image.Image:
    """
    Renders an elegant, anti-aliased cinematic end credits card on solid black canvas.
    """
    if generation_dt is None:
        generation_dt = datetime.datetime.now()

    # 2x supersampling for ultra-crisp typography
    ss = 2
    sw, sh = width * ss, height * ss
    img = Image.new("RGB", (sw, sh), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Color palette matching title card
    color_gold  = (255, 204, 102)   # Warm Solar Gold
    color_white = (245, 245, 245)   # Off-White
    color_muted = (165, 170, 180)   # Cool Slate Gray
    color_line  = (65, 70, 80)      # Muted divider line

    # Typography & Scaling
    font_sec_hdr  = get_font(int(17 * ss), bold=True)
    font_sec_body = get_font(int(23 * ss), bold=False)
    font_footer   = get_font(int(17 * ss), bold=False)

    cx = sw / 2.0
    cy = sh / 2.0

    # Ensure Software section exists and includes eclipse_assembler + user software
    processed_sections = []
    has_software = False

    for hdr, lines in sections:
        hdr_lower = hdr.lower()
        if "software" in hdr_lower:
            has_software = True
            sw_lines = ["Processing: eclipse-assembler"]
            for l in lines:
                if "eclipse_assembler" not in l.lower() and "eclipse-assembler" not in l.lower():
                    sw_lines.append(l)
            processed_sections.append(("SOFTWARE", sw_lines))
        elif "author" in hdr_lower or "autor" in hdr_lower:
            processed_sections.append(("AUTHOR", lines))
        elif "telescope" in hdr_lower or "telescopio" in hdr_lower or "optics" in hdr_lower:
            processed_sections.append(("TELESCOPE", lines))
        elif "location" in hdr_lower or "lugar" in hdr_lower or "site" in hdr_lower:
            processed_sections.append(("OBSERVATION SITE", lines))
        else:
            processed_sections.append((hdr.upper(), lines))

    if not has_software:
        processed_sections.append(("SOFTWARE", ["Processing: eclipse-assembler"]))

    # Date formatting (Formatted e.g. August 16, 2026)
    if eclipse_date_str:
        try:
            dt_obj = datetime.datetime.strptime(eclipse_date_str, "%Y-%m-%d")
            date_display = dt_obj.strftime("%B %d, %Y")
        except Exception:
            date_display = eclipse_date_str
    else:
        date_display = generation_dt.strftime("%B %d, %Y")

    # Ensure DATE section is present in same format as other sections
    has_date = any(h.lower() in ["date", "fecha"] for h, _ in processed_sections)
    if not has_date:
        processed_sections.append(("DATE", [date_display]))

    # Calculate height of sections for dynamic vertical centering
    sec_heights = []
    for hdr, lines in processed_sections:
        h_sec = int(22 * ss)               # header text
        h_sec += int(len(lines) * 32 * ss)  # body lines
        h_sec += int(22 * ss)               # gap after section
        sec_heights.append(h_sec)

    total_content_h = sum(sec_heights)
    curr_y = max(int(40 * ss), int(cy - total_content_h / 2.0))

    # Render Sections
    for hdr, lines in processed_sections:
        # Section Header (Muted uppercase label)
        bbox = draw.textbbox((0, 0), hdr, font=font_sec_hdr)
        w_txt = bbox[2] - bbox[0]
        draw.text((cx - w_txt / 2.0, curr_y), hdr, font=font_sec_hdr, fill=color_muted)
        curr_y += int(26 * ss)

        # Section Body Lines (Exact same format/color/font for all lines)
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font_sec_body)
            w_txt = bbox[2] - bbox[0]
            draw.text((cx - w_txt / 2.0, curr_y), line, font=font_sec_body, fill=color_gold)
            curr_y += int(32 * ss)

        curr_y += int(18 * ss)

    # Downsample with Lanczos for ultra-smooth anti-aliasing
    final_img = img.resize((width, height), Image.Resampling.LANCZOS)
    return final_img


def generate_end_titles(
    md_path: str = "010_in/07_endtitles.md",
    duration_s: float = 6.0,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    crf: int = 16,
    out_dir: str = "040_out",
    output_mp4: str = None,
    generation_dt: datetime.datetime = None
) -> tuple:
    """
    Parses markdown descriptor, renders closing credits graphic and encodes video clip.
    Returns: (png_path, mp4_path).
    """
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(md_path):
        # Fallback default content if markdown does not exist
        sections = [
            ("SOFTWARE & PROCESSING", ["eclipse_assembler (Astro-CV & Stabilization Engine)"])
        ]
    else:
        sections = parse_endtitles_markdown(md_path)

    title_img = render_end_titles_image(
        sections=sections,
        width=width,
        height=height,
        generation_dt=generation_dt
    )

    base_name = os.path.splitext(os.path.basename(md_path))[0] if md_path else "07_endtitles"
    png_path = os.path.join(out_dir, f"{base_name}.png")
    jpg_path = os.path.join(out_dir, f"{base_name}.jpg")
    title_img.save(png_path, format="PNG")
    title_img.save(jpg_path, format="JPEG", quality=98)

    mp4_target = output_mp4 or os.path.join(out_dir, f"{base_name}.mp4")
    total_frames = int(round(duration_s * fps))

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
    parser = argparse.ArgumentParser(description="Cinematic Eclipse End Titles & Credits Generator")
    parser.add_argument("--input", "-i", type=str, default="010_in/07_endtitles.md", help="Path to endtitles markdown descriptor")
    parser.add_argument("--duration", type=float, default=6.0, help="Clip duration in seconds (default: 6.0s)")
    parser.add_argument("--width", "-W", type=int, default=1280, help="Canvas width (default: 1280)")
    parser.add_argument("--height", "-H", type=int, default=720, help="Canvas height (default: 720)")
    parser.add_argument("--out-dir", type=str, default="040_out", help="Output directory (default: 040_out)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Custom output MP4 path")
    args = parser.parse_args()

    png_out, mp4_out = generate_end_titles(
        md_path=args.input,
        duration_s=args.duration,
        width=args.width,
        height=args.height,
        out_dir=args.out_dir,
        output_mp4=args.output
    )

    print("=================================================================")
    print("SUCCESS: End titles clip generated!")
    print(f"  Image (Lossless) : {png_out}")
    print(f"  Video Clip ({args.duration}s) : {mp4_out}")
    print("=================================================================")


if __name__ == "__main__":
    main()
