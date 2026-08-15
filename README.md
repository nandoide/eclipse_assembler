# Eclipse assembler: Cinematic Processing, Assembly & Astronomical Subtitle Pipeline

An automated, high-precision computer vision and astronomical pipeline designed to stabilize, color-equalize, align, and assemble multi-phase solar and lunar eclipse footage, high-resolution still photographs, on-the-fly composite artwork, and dynamic title cards into a single, perfectly centered cinematic master film (`full_eclipse.mp4`), accompanied by synchronized multilingual astronomical subtitles and QuickTime-ready embedded tracks.

---

## 🔭 Source Telemetry & Raw Footage (`000_raw/`)

Smart telescopes (such as **DWARFLAB DWARF mini / DWARF II / DWARF 3**, **ZWO Seestar S50**, **Vaonis Vespera**, or intervalometer-controlled DSLRs) embed the exact real-world capture timestamps directly into the raw video filenames and file metadata headers.

Users should place their **original, unedited raw telescope captures** inside the **`000_raw/`** directory. This directory serves as the **ground-truth telemetry source** for all astronomical timestamps, contact point calculations, speed multiplier derivations, and automatic eclipse date detection across the pipeline.

### Example Files in `000_raw/` (DWARFLAB DWARF mini):

| Raw File in `000_raw/` | Capture Mode | Embedded Timestamp | Purpose & Telemetry in Pipeline |
| :--- | :--- | :--- | :--- |
| **`DWARF_mini_TELE_TL_2026-08-12-19-35-13-093.mp4`** | Ingress Timelapse | `19:35:13 CEST` | Baseline start timestamp ($t_0$) for partial ingress progression (1 frame every 10s). |
| **`DWARF_mini_TELE_2026-08-12-20-20-36-811.mp4`** | Real-Time Video Burst | `20:20:36 CEST` | High-speed video encompassing pre-totality thin crescent, C2 diamond ring, totality corona, and C3 egress diamond ring. |
| **`DWARF_mini_TELE_TL_2026-08-12-20-32-53-127.mp4`** | Egress Timelapse | `20:32:53 CEST` | Baseline timestamps for partial egress progression until sunset/fourth contact (1 frame every 10s). |
| **`gps.jpg`** | Observer GPS Reference | `43.235556°N, 7.558333°W` (Alt: `438.7m`) | Any photograph taken with the telescope (or smartphone) from the exact observing site containing embedded EXIF GPS tags (Latitude, Longitude, Altitude), used as the location ground truth to calculate topocentric NASA Besselian ephemeris. |

> [!IMPORTANT]
> **Why keep `000_raw/` separate from `010_in/`?**
> Video editing or trimming software often strips original capture creation timestamps and EXIF GPS tags. By keeping original files in `000_raw/` (including any reference photo like `gps.jpg`), the subtitle generator (`020_src/generate_eclipse_subtitles.py`) and composite generator (`020_src/create_eclipse_composite.py`) reference authentic ground-truth timestamps and exact geographic coordinates, computing frame-accurate astronomical local times directly from the lens.

---

## 🌌 Universal Eclipse Database (`030_db/eclipses_db.json`) & Ephemeris Engine

Instead of hardcoding astronomical coefficients in source files, the pipeline utilizes a dedicated, extensible **Universal Eclipse Database (`030_db/eclipses_db.json`)** dynamically queried from the **NASA JPL Horizons REST API** and managed by **`020_src/eclipse_ephemeris_db.py`**.

```
000_raw/ (Files with date: e.g. 2026-08-12)
   │
   ▼
[Date Detection Engine] (Auto-detects YYYY-MM-DD from RAW / CoC / CLI)
   │
   ▼
[020_src/eclipse_ephemeris_db.py]
   ├── Checks 030_db/eclipses_db.json (Auto-builds / refreshes with --force-db)
   └── Loads exact Besselian polynomials (Solar) or Umbral/Penumbral contacts (Lunar)
   │
   ▼
[General Astronomical Solver]
   ├── Calculates topocentric C1, C2, MAX, C3, C4 (Solar) or P1-P4 (Lunar)
   ├── Generates Cinematic Title Card (00_title.mp4 via 020_src/create_title_card.py)
   ├── Feeds compare_contacts_ephemeris.py (Observational vs. Ephemeris Comparison)
   └── Feeds create_eclipse_composite.py & build_full_eclipse.py (Sample timestamp tagging)
```

### 1. Dual Support: Solar and Lunar Eclipses (2026–2036)

- **Solar Eclipses (`solar_eclipses`)**: Solves topocentric **NASA Besselian polynomials** ($x, y, d, l_1, l_2, \mu, \tan f_1, \tan f_2$) in Terrestrial Time (TT) for the observer's exact geographic coordinates and elevation ($C_1, C_2, \text{MAX}, C_3, C_4$).
- **Lunar Eclipses (`lunar_eclipses`)**: Calculates geocentric and topocentric contact times for penumbral immersion ($P_1, P_4$) and umbral totality ($U_1, U_2, \text{MAX}, U_3, U_4$) with atmospheric enlargement.

### 2. Observational Telemetry vs. NASA Ephemeris Comparison

Modern smart telescopes synchronize their internal system clock via NTP/GPS to within seconds of UTC. The pipeline uses **100% direct raw camera timestamps** for all video subtitles and metadata, without applying artificial compensations.

To inspect how optical measurements compare against theoretical smooth-sphere ephemerides, run:
```bash
python 020_src/compare_contacts_ephemeris.py
```

| Astronomical Event | Physical Criterion | NASA Ephemeris | Optical / Camera (RAW) | Residual (Obs − Eph) |
| :--- | :--- | :--- | :--- | :--- |
| **$C_1$ (First Contact)** | Moon enters solar photosphere | `19:31:28.9 CEST` | `19:35:13 CEST` *(TL start)* | *Documented in 01_timelapse_i10* |
| **$C_2$ (Second Contact)** | Baily's Beads extinguish $\to$ **Totality Begins** | `20:27:43.6 CEST` | **`20:27:34.8 CEST`** | **`-8.85 s`** |
| **$\text{MAX}$ (Maximum)** | Point of deepest eclipse / Midpoint | `20:28:36.9 CEST` | **`20:28:23.3 CEST`** | **`-13.65 s`** |
| **$C_3$ (Third Contact)** | Baily's Beads emerge $\to$ **Totality Ends** | `20:29:30.2 CEST` | **`20:29:11.8 CEST`** | **`-18.45 s`** |
| **$C_4$ (Fourth Contact)** | Moon completely leaves solar disk | `21:22:05.6 CEST` | `21:13:53 CEST` *(TL end)* | *Documented in 04_timelapse_i10* |
| **Totality Duration** | Full 100% eclipse window ($C_3 - C_2$) | `1m 46.56s` ($106.6\text{ s}$) | **`1m 36.96s`** ($97.0\text{ s}$) | **`-9.60 s`** |

> [!NOTE]
> The small residuals ($\sim 8-18\text{ s}$) are primarily driven by the **lunar limb topography** (valleys and mountain peaks in the *Kaguya / LRO* profile), sensor exposure integration times, and local horizon atmospheric refraction.

---

## 🎬 Cinematic Opening Title Card (`00_title.mp4`)

Before the first timelapse clip, `build_full_eclipse.py` automatically generates and prepends an opening title card (**`00_title.mp4`**, default 5.0s) rendered by **`020_src/create_title_card.py`**:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                         TOTAL SOLAR ECLIPSE                              │
│                           August 12, 2026                                │
│                                                                          │
│          Observation Site: 43.2356° N, 7.5583° W   |   Alt: 439 m        │
│                                                                          │
│                       19:31:28 - 21:22:05 CEST                           │
│          Totality (C2 -> C3):  20:27:43 - 20:29:30 CEST (1m 47s)        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

- **English Typography**: Clean, anti-aliased typography supersampled at 2x and downscaled via Lanczos filtering on solid black.
- **Dynamic Content**: Displays the astronomical name, date, observer coordinates, partial phase window, and totality window with exact duration.
- **Seamless Transition**: Fades smoothly into the initial ingress timelapse (`01_timelapse_i10.mp4`).

---

## 🌒 Overview & CoC Architecture (`010_in/`)

This pipeline follows **Convention over Configuration (CoC)**. Curated assets in **`010_in/`** define sequence ordering, asset types, timelapse shoot intervals, custom durations, and processing pipelines without needing external configuration files:

`[INDEX]_[TYPE]_[PARAMS/INTERVAL/LAYOUT]_[DURATION].[ext]`

### Supported CoC Asset Types in `010_in/`

| Filename Example | Type | Behavior & Parameters | Speed / Duration |
| :--- | :--- | :--- | :--- |
| **`01_timelapse_i10.mp4`** | `timelapse` | Ingress timelapse with `_i[INTERVAL]` parameter (e.g. `_i10` = 1 frame every 10s). Subpixel solar limb convex-hull stabilization and white balance equalization. | Native clip duration (speed: $10\text{s} \times 30\text{fps} = \mathbf{300\times}$) |
| **`02_video_slowdown_10.mp4`** | `video_slowdown` | High-speed burst video. Automatically filters black/corrupt frames, resamples $396.7\text{s}$ of pre-totality footage to `duration` (e.g. 10s), smooths camera exposure jumps, and stabilizes solar limb. | 10.0s (speed: $\frac{396.7\text{s}}{10\text{s}} \approx \mathbf{40\times}$) |
| **`03_video_realtime.mp4`** | `video_realtime` | Continuous 1x real-time video (30 fps) with lunar silhouette, corona, and Baily's beads tracking. Preserves exact 1:1 real-time duration. | Native real duration ($\mathbf{1\times}$ Real-Time) |
| **`04_timelapse_i10.mp4`** | `timelapse` | Egress timelapse with `_i[INTERVAL]` parameter (`_i10` = 1 frame every 10s). Subpixel solar limb stabilization. | Native clip duration (speed: $10\text{s} \times 30\text{fps} = \mathbf{300\times}$) |
| **`05_photo_6.jpg`** | `photo` | Visual still image, scaled preserving aspect ratio with clean black letterbox padding for specified duration (e.g. 6s). | 10.0s (or `_6` for 6s) |
| **`06_composite_sinusoid_10`** | `composite` | Generates high-resolution composite artwork mosaic on-the-fly, scaled to project resolution (16:9 1280x720), held for specified duration (e.g. 10s). | 10.0s |

---

## 🔬 Algorithmic Pipeline & Flow

```
000_raw/ (Original Telescope Telemetry: DWARF mini, Seestar, etc.)
  │  • DWARF_mini_TELE_TL_2026-08-12-19-35-13-093.mp4
  │  • DWARF_mini_TELE_2026-08-12-20-20-36-811.mp4
  │  • DWARF_mini_TELE_TL_2026-08-12-20-32-53-127.mp4
  │  • gps.jpg (Coordinates: 43.235556°N, 7.558333°W, Alt: 438.7m)
  ▼
030_db/eclipses_db.json (NASA Besselian Elements & Ephemeris DB)
  ▼
010_in/ (Curated CoC Assets)                                              040_out/
┌────────────────────────┐                                        ┌────────────────────────┐
│ [Dynamic Title Card]   │ ───► [Render Typography Card] ───────► │ 00_title.mp4           │
│ (020_src/create_title) │      (2x Lanczos + Ephemeris Data)     │ (5.00s @ 30 fps)       │
├────────────────────────┤                                        ├────────────────────────┤
│ 01_timelapse_i10.mp4   │ ───► [Solar Limb Stabilization] ─────► │ 01_timelapse_i10.mp4   │
│ (Ingress Timelapse)    │      (Subpixel Convex Hull)            │ (8.90s @ 30 fps, x300) │
├────────────────────────┤                                        ├────────────────────────┤
│ 02_video_slowdown_10   │ ───► [Black Frame Filter +          ─► │ 02_video_slowdown_10   │
│ (C2 Approach Burst)    │      10s Resample + Limb Track]        │ (10.00s @ 30 fps, x40) │
├────────────────────────┤                                        ├────────────────────────┤
│ 03_video_realtime.mp4  │ ───► [Totality Silhouette Track     ─► │ 03_video_realtime.mp4  │
│ (Totality Video)       │      + Corona / Beads Tracking]        │ (107.33s @ 30 fps, 1x) │
├────────────────────────┤                                        ├────────────────────────┤
│ 04_timelapse_i10.mp4   │ ───► [Adaptive Solar Limb Track     ─► │ 04_timelapse_i10.mp4   │
│ (Egress Timelapse)     │      + Color Balance Eq]               │ (8.20s @ 30 fps, x300) │
├────────────────────────┤                                        ├────────────────────────┤
│ 05_photo_6.jpg         │ ───► [Scale & Pad to Canvas]        ─► │ 05_photo_6.mp4         │
│ (Still Photograph)     │      (Preserve AR + Black Border)      │ (6.00s @ 30 fps)       │
├────────────────────────┤                                        ├────────────────────────┤
│ 06_composite_sinusoid  │ ───► [On-The-Fly Artwork Render]    ─► │ 06_composite_sinusoid  │
│ (Mosaic Directive)     │      (Native 16:9 Mosaic + JSON Meta)  │ (10.00s @ 30 fps)      │
└────────────────────────┘                                        └───────────┬────────────┘
                                                                              │
                                                                    [Master Assembly]
                                                             (Cinematic Transitions: Freeze,
                                                             Dip to Black / Hard Cut / Crossfade)
                                                                              │
                                                                              ▼
                                                                  ┌────────────────────────┐
                                                                  │ full_eclipse.mp4       │
                                                                  │ (172.60s @ 30 fps)     │
                                                                  └───────────┬────────────┘
                                                                              │
                                                                    [Astronomical Engine]
                                                                (100% Direct RAW Timestamps +
                                                                 Multilingual Subtitles +
                                                                 QuickTime mov_text Muxing)
                                                                              │
                                                                              ▼
                                          ┌───────────────────────────────────┴───────────────────────────────────┐
                                          ▼                                                                       ▼
                              ┌────────────────────────┐                                              ┌────────────────────────┐
                              │ full_eclipse_es.srt    │                                              │ full_eclipse_subtitled │
                              │ full_eclipse_en.srt    │                                              │ .mp4 (QuickTime Dual)  │
                              │ (YouTube / VLC)        │                                              │ (Native mov_text)      │
                              └────────────────────────┘                                              └────────────────────────┘
```

---

## 📦 Installation & Requirements

### Prerequisites
- **Python 3.10+**
- **FFmpeg** with `libx264`, `libx265` HEVC, and `mov_text` subtitle encoder support.

### Setup Virtual Environment
```bash
# Clone and navigate to the repository
git clone https://github.com/nandoide/eclipse_assembler.git
cd eclipse_assembler

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install numpy opencv-python pillow scipy tqdm
```

---

## 🚀 Running the Pipeline

### 1. Default End-to-End Build
Discovers all assets in `010_in/`, renders the opening title card `00_title.mp4`, stabilizes clips, renders the 16:9 composite artwork, assembles `040_out/full_eclipse.mp4`, and automatically generates multilingual subtitles and the QuickTime-ready master film:
```bash
python 020_src/build_full_eclipse.py
```

### 2. Standalone Subtitle Generation & QuickTime Embedding
To generate or update subtitles independently without re-rendering videos:
```bash
# Generate English + Spanish SRTs and embed in QuickTime MP4:
python 020_src/generate_eclipse_subtitles.py --embed

# Custom interval (e.g. every 10 seconds):
python 020_src/generate_eclipse_subtitles.py --interval 10.0 --embed
```

### 3. Observational vs. Ephemeris Comparison Report
```bash
# Generate comparison table and markdown/JSON reports:
python 020_src/compare_contacts_ephemeris.py
```

### 4. Standalone Title Card Generation (`00_title.mp4`)
```bash
# Generate 5.0s 1080p / 720p title card:
python 020_src/create_title_card.py --duration 5.0

# Generate title card for a specific eclipse date:
python 020_src/create_title_card.py --date 2026-08-12 --duration 5.0
```

### 5. Standalone Composite Artwork Mosaic Generation
To create standalone high-resolution photographic composite artwork with JSON metadata:
```bash
# 1. Sinusoidal S-Curve (default: 1280x720 16:9 matching project resolution):
python 020_src/create_eclipse_composite.py --layout sinusoid

# 2. UHD 4K Squared (3840x3840 px):
python 020_src/create_eclipse_composite.py --layout sinusoid --size 3840

# 3. 4K Desktop Widescreen (3840x2160 px):
python 020_src/create_eclipse_composite.py --layout sinusoid --width 3840 --height 2160

# 4. Vertical Mobile Wallpaper 9:16 (2160x3840 px):
python 020_src/create_eclipse_composite.py --layout vertical
python 020_src/create_eclipse_composite.py --layout vertical-s

# 5. Generate all layouts at once (sinusoid, vertical, vertical-s, circle, diagonal, horizontal, arc):
python 020_src/create_eclipse_composite.py --layout all
```

### 6. Ephemeris Database CLI & Refresh
```bash
# Query ephemeris contacts for auto-detected date:
python 020_src/eclipse_ephemeris_db.py

# Query for a specific solar or lunar eclipse date:
python 020_src/eclipse_ephemeris_db.py --date 2026-08-12

# Force refresh/rebuild database:
python 020_src/eclipse_ephemeris_db.py --force-db
```

### 7. Lightweight 30-Second Compact Video (for Easy Sharing)
Generates an accelerated, universally compatible 30-second version in H.264 (~1.19 MB in 720p HD) for quick distribution via messaging apps (WhatsApp, Telegram) or social media:
```bash
python 020_src/create_compact_clip.py
```

### 8. Force Re-processing from Scratch
```bash
python 020_src/build_full_eclipse.py --force-all
```

---

## 📂 Project Directory Structure

```
eclipse_assembler/
├── 000_raw/                          # ★ Ground-truth raw telescope footage & telemetry
│   ├── DWARF_mini_TELE_TL_2026-08-12-19-35-13-093.mp4
│   ├── DWARF_mini_TELE_2026-08-12-20-20-36-811.mp4
│   ├── DWARF_mini_TELE_TL_2026-08-12-20-32-53-127.mp4
│   └── gps.jpg                       # Observer coordinates & elevation metadata photo
│
├── 010_in/                           # Curated source input assets (CoC naming)
│   ├── 01_timelapse_i10.mp4          # Ingress time-lapse (1 frame every 10s)
│   ├── 02_video_slowdown_10.mp4      # Pre-totality burst video (10s resample)
│   ├── 03_video_realtime.mp4         # Totality real-time continuous video (1x)
│   ├── 04_timelapse_i10.mp4          # Egress time-lapse (1 frame every 10s)
│   ├── 05_photo_6.jpg                # Still photograph (6s hold)
│   └── 06_composite_sinusoid_10      # On-the-fly composite mosaic (10s hold)
│
├── 020_src/                          # Python source code
│   ├── build_full_eclipse.py         # ★ CoC Master pipeline and assembly script
│   ├── eclipse_ephemeris_db.py       # ★ Universal Solar/Lunar Besselian database & solver
│   ├── fetch_eclipses_horizons.py    # ★ NASA JPL Horizons API fetcher & DB builder (2026-2036)
│   ├── compare_contacts_ephemeris.py # ★ Observational telemetry vs. NASA ephemeris comparison
│   ├── create_title_card.py          # ★ Cinematic title card generator (00_title.mp4)
│   ├── generate_eclipse_subtitles.py # ★ 100% RAW-based multilingual subtitle generator
│   ├── create_eclipse_composite.py   # ★ High-resolution composite mosaic generator & JSON metadata
│   ├── create_compact_clip.py        # 30s accelerated lightweight generator
│   ├── stabilize_eclipse.py          # Core solar limb subpixel stabilizer
│   └── stabilize_full_film.py        # Global 2-pass master film stabilizer
│
├── 030_db/                           # ★ Universal astronomical database
│   └── eclipses_db.json              # Solar & Lunar eclipse NASA Besselian elements catalog (2026-2036)
│
├── 040_out/                          # Generated stabilized output videos, artwork & subtitles
│   ├── 00_title.mp4                  # Cinematic opening title card clip (5.00s)
│   ├── 00_title.png                  # Lossless anti-aliased title graphic
│   ├── 01_timelapse_i10.mp4          # Stabilized partial ingress (8.90s, speed x300)
│   ├── 02_video_slowdown_10.mp4      # Filtered & stabilized pre-totality (10.00s)
│   ├── 03_video_realtime.mp4         # Stabilized totality (107.33s, speed 1x)
│   ├── 04_timelapse_i10.mp4          # Stabilized partial egress (8.20s, speed x300)
│   ├── 05_photo_6.mp4                # Letterboxed photo video (6.00s)
│   ├── 06_composite_sinusoid_10.mp4  # 16:9 composite video (10.00s)
│   ├── 06_composite_sinusoid_10.json # 📋 Frame timestamps metadata for composite
│   ├── full_eclipse.mp4              # ★ Complete master film (172.60s)
│   ├── full_eclipse_subtitled.mp4    # 🍎 Master film with embedded QuickTime subtitles (ES + EN)
│   ├── full_eclipse_es.srt           # 🇪🇸 Spanish subtitles (YouTube / VLC)
│   ├── full_eclipse_en.srt           # 🇬🇧 English subtitles (YouTube / VLC)
│   ├── full_eclipse.srt              # Master English subtitle file
│   ├── full_eclipse_30s.mp4          # ⚡ Compact 30s accelerated film
│   ├── eclipse_contacts_comparison.md# 📊 Markdown telemetry vs ephemeris comparison report
│   ├── eclipse_contacts_comparison.json# 📋 JSON dataset of observational residuals
│   └── eclipse_composite_*.png       # 🖼️ High-resolution composite artwork
│
└── README.md                         # Project documentation
```

---

## 📊 Summary of Master Output Assets

| Output Asset | Resolution | Frame Rate / Type | Duration / Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`00_title.mp4`** | 1280x720 | 30.0 fps | 5.00s (0.02 MB) | Cinematic title card with ephemeris metadata |
| **`01_timelapse_i10.mp4`** | 1280x720 | 30.0 fps | 8.90s (1.31 MB) | Stabilized partial ingress ($R = 238.5\text{ px}$, speed x300) |
| **`02_video_slowdown_10.mp4`** | 1280x720 | 30.0 fps | 10.00s (0.69 MB) | Filtered & stabilized pre-totality ($396.7\text{s} \to 10\text{s}$, speed x40) |
| **`03_video_realtime.mp4`** | 1280x720 | 30.0 fps | 107.33s (13.71 MB) | Totality & corona in exact 1x Real-Time |
| **`04_timelapse_i10.mp4`** | 1280x720 | 30.0 fps | 8.20s (2.66 MB) | Stabilized partial egress ($R = 237.5\text{ px}$, speed x300) |
| **`05_photo_6.mp4`** | 1280x720 | 30.0 fps | 6.00s (0.12 MB) | Still photo scaled with black letterboxing |
| **`06_composite_sinusoid_10.mp4`** | 1280x720 | 30.0 fps | 10.00s (0.05 MB) | Native 16:9 sinusoidal mosaic clip |
| **`full_eclipse.mp4`** | **1280x720** | **30.0 fps** | **172.60s (12.46 MB)** | 🎬 **Master film with title card & cinematic transitions** |
| **`full_eclipse_subtitled.mp4`** | **1280x720** | **30.0 fps** | **172.60s (12.48 MB)** | 🍎 **Master film with dual embedded QuickTime tracks (`mov_text`)** |
| **`full_eclipse_es.srt`** | SubRip | UTF-8 | 48 entries | 🇪🇸 **Astronomical subtitles in Spanish with speed multipliers** |
| **`full_eclipse_en.srt`** | SubRip | UTF-8 | 48 entries | 🇬🇧 **Astronomical subtitles in English with speed multipliers** |

---

## 📄 License & Attribution
Developed for high-precision astronomical astrophotography and eclipse video processing.

- **Project Lead & Domain Guidance**: Fernando ([@nandoide](https://github.com/nandoide))
- **AI Coding Agent & Computer Vision Architecture**: Antigravity (Google Deepmind)

Code released under the MIT License.
