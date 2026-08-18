# Eclipse assembler: Cinematic Processing, Assembly & Astronomical Subtitle Pipeline

An automated, high-precision computer vision and astronomical pipeline designed to stabilize, color-equalize, align, and assemble multi-phase solar and lunar eclipse footage, high-resolution still photographs, on-the-fly composite artwork, and dynamic title cards into a single, perfectly centered cinematic master film (`full_eclipse.mp4`), accompanied by synchronized multilingual astronomical subtitles and QuickTime-ready embedded tracks.

---

## 🔭 Source Telemetry & Raw Footage (`000_raw/<observation>/`)

Smart telescopes (such as **DWARFLAB DWARF mini / DWARF II / DWARF 3**, **ZWO Seestar S50**, **Vaonis Vespera**, or intervalometer-controlled DSLRs) embed the exact real-world capture timestamps directly into the raw video filenames and file metadata headers.

The pipeline organizes raw data into **observation folders** inside `000_raw/<observation>/` (e.g. `000_raw/solar26/`). Each observation directory represents a distinct observation campaign or observing location, serving as the **ground-truth telemetry source** for all astronomical timestamps, contact point calculations, speed multiplier derivations, and automatic eclipse date detection across the pipeline.

### Example Files in `000_raw/solar26/` (DWARFLAB DWARF mini):

| Raw File in `000_raw/solar26/`                       | Capture Mode           | Embedded Timestamp                        | Purpose & Telemetry in Pipeline                                                                                                                                                                                                               |
| :--------------------------------------------------- | :--------------------- | :---------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`DWARF_mini_TELE_TL_2026-08-12-19-35-13-093.mp4`** | Ingress Timelapse      | `19:35:13 CEST`                           | Baseline start timestamp ($t_0$) for partial ingress progression (1 frame every 10s).                                                                                                                                                         |
| **`DWARF_mini_TELE_2026-08-12-20-20-36-811.mp4`**    | Real-Time Video Burst  | `20:20:36 CEST`                           | High-speed video encompassing pre-totality thin crescent, C2 diamond ring, totality corona, and C3 egress diamond ring.                                                                                                                       |
| **`DWARF_mini_TELE_TL_2026-08-12-20-32-53-127.mp4`** | Egress Timelapse       | `20:32:53 CEST`                           | Baseline timestamps for partial egress progression until sunset/fourth contact (1 frame every 10s).                                                                                                                                           |
| **`gps.jpg`**                                        | Observer GPS Reference | `43.235556°N, 7.558333°W` (Alt: `438.7m`) | Any photograph taken with the telescope (or smartphone) from the exact observing site containing embedded EXIF GPS tags (Latitude, Longitude, Altitude), used as the location ground truth to calculate topocentric NASA Besselian ephemeris. |

> [!IMPORTANT]
> **Why keep `000_raw/<observation>/` separate from `010_in/`?**
> Video editing or trimming software often strips original capture creation timestamps and EXIF GPS tags. By keeping original files in `000_raw/<observation>/` (including any reference photo like `gps.jpg`), the subtitle generator (`020_src/generate_eclipse_subtitles.py`) and composite generator (`020_src/create_eclipse_composite.py`) reference authentic ground-truth timestamps and exact geographic coordinates, computing frame-accurate astronomical local times directly from the lens.

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

| Astronomical Event         | Physical Criterion                                 | NASA Ephemeris                | Optical / Camera (RAW)            | Residual (Obs − Eph)             |
| :------------------------- | :------------------------------------------------- | :---------------------------- | :-------------------------------- | :------------------------------- |
| **$C_1$ (First Contact)**  | Moon enters solar photosphere                      | `19:31:18.5 CEST`             | `19:35:13 CEST` *(TL start)*      | *Documented in 01_timelapse_i10* |
| **$C_2$ (Second Contact)** | Baily's Beads extinguish $\to$ **Totality Begins** | `20:27:38.2 CEST`             | **`20:27:34.8 CEST`**             | **`-3.41 s`**                    |
| **$\text{MAX}$ (Maximum)** | Point of deepest eclipse / Midpoint                | `20:28:25.0 CEST`             | **`20:28:23.3 CEST`**             | **`-1.73 s`**                    |
| **$C_3$ (Third Contact)**  | Baily's Beads emerge $\to$ **Totality Ends**       | `20:29:12.2 CEST`             | **`20:29:11.8 CEST`**             | **`-0.41 s`**                    |
| **$C_4$ (Fourth Contact)** | Moon completely leaves solar disk                  | `21:21:52.6 CEST`             | `21:13:53 CEST` *(TL end)*        | *Documented in 04_timelapse_i10* |
| **Totality Duration**      | Full 100% eclipse window ($C_3 - C_2$)             | `1m 33.96s` ($94.0\text{ s}$) | **`1m 36.96s`** ($97.0\text{ s}$) | **`+3.00 s`**                    |

> [!NOTE]
> The remarkable sub-second to few-second precision ($\sim 0.4 - 3.4\text{ s}$) demonstrates high telescope NTP clock accuracy ($\sim 1.8\text{ s}$ of UTC). The remaining residuals are physically accounted for by **lunar limb topography** (valleys and mountain peaks in the *Kaguya / LRO* elevation profile that advance or delay the bead extinguish/emergence points) and sensor auto-exposure response regimes.

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
│                       19:31:18 - 21:21:53 CEST                           │
│          Totality (C2 -> C3):  20:27:38 - 20:29:12 CEST (1m 34s)        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

- **English Typography**: Clean, anti-aliased typography supersampled at 2x and downscaled via Lanczos filtering on solid black.
- **Dynamic Content**: Displays the astronomical name, date, observer coordinates, partial phase window, and totality window with exact duration.
- **Seamless Transition**: Fades smoothly into the initial ingress timelapse (`01_timelapse_i10.mp4`).

---

## 🌒 Multi-Project Architecture & CoC (`010_in/`)

The pipeline supports a robust **Multi-Project Architecture** based on **Convention over Configuration (CoC)**. Multiple projects can coexist inside `010_in/` as independent subdirectories, mapping seamlessly to their corresponding observation raw data in `000_raw/<observation>/` and outputting cleanly into `040_out/<project_folder>/`.

### 1. Project Folder Naming Nomenclature

Each project directory in `010_in/` follows the convention:

```
010_in/<observation>_<project_name>_<layout>[_<preproc>]/
```

- **`<observation>`**: Identifies the observation campaign / telescope telemetry dataset inside `000_raw/<observation>/` (e.g. `solar26`).
- **`<project_name>`**: Sub-identifier or variant (e.g. `main`, `v1`, `v2`).
- **`project_id`**: The canonical base project ID derived as `<observation>_<project_name>` (e.g. `solar26_main`, `solar26_v1`, `solar26_v2`). This ID names all primary output files (`.mp4`, `_chapters.txt`, `_en.srt`, `_es.srt`).
- **`<layout>`**: Film assembly style:
  - **`standard`**: Linear chronological sequence with smooth fades to black.
  - **`art`**: Cinematic narrative assembly featuring 4K continuous sub-pixel camera dives, crossfades at Totality Max, and dynamic mosaics.
- **`[_preproc]`**: Optional flag indicating that the project consumes obstacle-free, centered, and C1/C4-extrapolated timelapses from `005_raw_preprocessed/<observation>/` (automatically generated on first run if missing).

#### Real-World Project Examples:

| Project Folder in `010_in/`             | Base ID (`project_id`) | Observation Source            | Film Style | Preprocessing | Description                                                                                    |
| :-------------------------------------- | :--------------------- | :---------------------------- | :--------- | :------------ | :--------------------------------------------------------------------------------------------- |
| **`solar26_main_standard_preproc`**     | `solar26_main`         | `000_raw/solar26/` (via 005)  | `standard` | ✅ Enabled     | Main official solar eclipse film in standard linear format using preprocessed restored footage. |
| **`solar26_v1_standard`**               | `solar26_v1`           | `000_raw/solar26/` (original) | `standard` | ❌ Raw assets | Linear sequence using raw telescope recordings and high-speed pre-totality slowdown burst.      |
| **`solar26_v2_art_preproc`**            | `solar26_v2`           | `000_raw/solar26/` (via 005)  | `art`      | ✅ Enabled     | Cinematic Art film with 4K camera dives, Totality HDR insert, and preprocessed timelapses.      |

---

### 2. Supported CoC Asset Types inside `010_in/<project>/`

`[INDEX]_[TYPE]_[PARAMS/INTERVAL/LAYOUT]_[DURATION].[ext]`

| Filename Example                      | Type             | Behavior & Parameters                                                                                                                                                                                                                                                                                                  | Speed / Duration                                                                    |
| :------------------------------------ | :--------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------- |
| **`01_timelapse_prep_i10`**           | `timelapse_prep` | **Preprocessed & Photosphere-Restored Ingress Timelapse**: Automatically consumes restored, obstacle-free, centered footage from `005_raw_preprocessed/<observation>/` (auto-triggered from `000_raw/` if missing). Stabilization is skipped as the clip is already 100% jitter-free. Interval is set to 10s (`_i10`). | Native clip duration (speed: $10\text{s} \times 30\text{fps} = \mathbf{300\times}$) |
| **`01_timelapse_i10.mp4`**            | `timelapse`      | Standard Ingress timelapse with `_i[INTERVAL]` parameter (e.g. `_i10` = 1 frame every 10s). Subpixel solar limb convex-hull stabilization and white balance equalization.                                                                                                                                              | Native clip duration (speed: $10\text{s} \times 30\text{fps} = \mathbf{300\times}$) |
| **`02_video_slowdown_10.mp4`**        | `video_slowdown` | High-speed burst video. Automatically filters black/corrupt frames, resamples $396.7\text{s}$ of pre-totality footage to `duration` (e.g. 10s), smooths camera exposure jumps, and stabilizes solar limb.                                                                                                              | 10.0s (speed: $\frac{396.7\text{s}}{10\text{s}} \approx \mathbf{40\times}$)         |
| **`03_video_realtime.mp4`**           | `video_realtime` | Continuous 1x real-time video (30 fps) with lunar silhouette, corona, and Baily's beads tracking. Preserves exact 1:1 real-time duration.                                                                                                                                                                              | Native real duration ($\mathbf{1\times}$ Real-Time)                                 |
| **`04_timelapse_prep_i10`**           | `timelapse_prep` | **Preprocessed & Photosphere-Restored Egress Timelapse**: Automatically consumes restored, foliage/branch-free footage from `005_raw_preprocessed/<observation>/`. Interval is set to 10s (`_i10`).                                                                                                                   | Native clip duration (speed: $10\text{s} \times 30\text{fps} = \mathbf{300\times}$) |
| **`04_timelapse_i10.mp4`**            | `timelapse`      | Standard Egress timelapse with `_i[INTERVAL]` parameter (`_i10` = 1 frame every 10s). Subpixel solar limb stabilization.                                                                                                                                                                                               | Native clip duration (speed: $10\text{s} \times 30\text{fps} = \mathbf{300\times}$) |
| **`05_totality_6.jpg`**               | `totality`       | Multi-exposure Totality HDR composite artwork (synthesizing corona, ruby $H\alpha$ prominences, and Baily's beads across totality). Scaled with black letterbox padding.                                                                                                                                            | 10.0s (or `_6` for 6s)                                                              |
| **`05_photo_6.jpg`**                  | `photo`          | Single-exposure still image, scaled preserving aspect ratio with clean black letterbox padding for specified duration (e.g. 6s).                                                                                                                                                                                       | 10.0s (or `_6` for 6s)                                                              |
| **`06_composite_circle_4k_10`**       | `composite`      | Generates high-resolution composite artwork mosaic on-the-fly (`arc`, `circle`, `ellipse`, `sinusoid`, `spiral`, etc.) with prominent totality contacts (`C2`, `MAX`, `C3`), scaled to project canvas, held for specified duration (e.g. 10s).                                                                      | 10.0s                                                                               |
| **`07_endtitles.md`**                 | `endtitles`      | Markdown descriptor for closing credits and telemetry card (`# Telescope`, `# Cameras`, `# Software`, `# Author`, `# Music`). Renders 2x supersampled anti-aliased card with software pipeline credits, dynamic music credits, and date.                                                                               | 6.0s (or custom `_8` for 8s)                                                        |
| **`01_music_corrubedo_nandoide.wav`** | `music`          | Soundtrack audio track (`[INDEX]_music_[TITLE]_[AUTHOR].[ext]`). Generically retargets audio at 100% natural tempo with phase-aligned crossfades, synchronizing climax with Totality Max, muxing high-fidelity AAC 320 kbps into master video, and dynamically crediting in end titles.                                | Full film duration                                                                  |

---

### 3. Clean Deliverables & Temporary Files Architecture (`040_out/`)

To keep production outputs uncluttered, the pipeline maintains a strict separation between **final deliverables** (at the root of `040_out/<project_folder>/`) and **intermediate assets** (inside `040_out/<project_folder>/temp/`):

```
040_out/<project_folder>/
├── <project_id>.mp4              # 🍎 Final Master Film (Embedded QuickTime ES/EN subs + Chapters + Music)
├── <project_id>_chapters.txt     # 📺 YouTube-compliant chapters with 0:00 timestamp format
├── <project_id>_en.srt           # 🇬🇧 English astronomical subtitles (Real-time telemetry)
├── <project_id>_es.srt           # 🇪🇸 Spanish astronomical subtitles (Real-time telemetry)
├── eclipse_composite_*.png/.jpg  # 🖼️ High-resolution standalone composite posters (when run directly)
└── temp/                         # 📁 All intermediate clips, 4K canvases, waveforms, descriptions & JSONs
    ├── 00_title.mp4
    ├── 01_timelapse_*.mp4
    ├── 03_video_realtime.mp4
    ├── 07_endtitles.mp4
    ├── full_eclipse_*_video.mp4
    ├── art_01_dive_in.mp4
    ├── art_02_dive_out.mp4
    ├── art_composite_4k_*.png / .json
    ├── composite_samples.json
    ├── metadata_chapters_*.txt
    └── youtube_description_*.txt
```

---

## 🔬 Algorithmic Pipeline & Flow

```
000_raw/<observation>/ (Original Telescope Telemetry: DWARF mini, Seestar, etc.)
  │  • DWARF_mini_TELE_TL_2026-08-12-19-35-13-093.mp4
  │  • DWARF_mini_TELE_2026-08-12-20-20-36-811.mp4
  │  • DWARF_mini_TELE_TL_2026-08-12-20-32-53-127.mp4
  │  • gps.jpg (Coordinates: 43.235556°N, 7.558333°W, Alt: 438.7m)
  ▼
005_raw_preprocessed/<observation>/ (Obstacle-free, Photosphere-Restored & C1-Extrapolated)
  ▼
030_db/eclipses_db.json (NASA Besselian Elements & Ephemeris DB)
  ▼
010_in/<project_folder>/ (Curated CoC Project)                   040_out/<project_folder>/temp/
┌────────────────────────┐                                        ┌────────────────────────┐
│ [Dynamic Title Card]   │ ───► [Render Typography Card] ───────► │ 00_title.mp4           │
│ (020_src/create_title) │      (2x Lanczos + Ephemeris Data)     │ (5.00s @ 30 fps)       │
├────────────────────────┤                                        ├────────────────────────┤
│ 01_timelapse_prep_i10  │ ───► [Resolve Restored Footage] ─────► │ 01_timelapse_i10.mp4   │
│ (Ingress Timelapse)    │      (Centered Photosphere)            │ (8.90s @ 30 fps, x300) │
├────────────────────────┤                                        ├────────────────────────┤
│ 03_video_realtime.mp4  │ ───► [Totality Silhouette Track     ─► │ 03_video_realtime.mp4  │
│ (Totality Video)       │      + Corona / Beads Tracking]        │ (107.33s @ 30 fps, 1x) │
├────────────────────────┤                                        ├────────────────────────┤
│ 04_timelapse_prep_i10  │ ───► [Resolve Restored Footage] ─────► │ 04_timelapse_i10.mp4   │
│ (Egress Timelapse)     │      (Foliage-Free Photosphere)        │ (8.20s @ 30 fps, x300) │
├────────────────────────┤                                        ├────────────────────────┤
│ 05_totality_6.jpg      │ ───► [Scale & Pad to Canvas]        ─► │ 05_totality_6.mp4      │
│ (Totality HDR Artwork) │      (Preserve AR + Black Border)      │ (6.00s @ 30 fps)       │
├────────────────────────┤                                        ├────────────────────────┤
│ 06_composite_circle_10 │ ───► [On-The-Fly Artwork Render]    ─► │ 06_composite_circle.mp4│
│ (Mosaic Directive)     │      (No Metadata Block in Build)      │ (10.00s @ 30 fps)      │
├────────────────────────┤                                        ├────────────────────────┤
│ 07_endtitles.md        │ ───► [Dynamic End Credits Render]   ─► │ 07_endtitles.mp4       │
│ (Closing Credits Card) │      (2x Lanczos + Dynamic Music Info) │ (6.00s @ 30 fps)       │
└────────────────────────┘                                        └───────────┬────────────┘
                                                                              │
                                                                    [Master Assembly]
                                                             (Standard Linear / Cinematic Art Dives)
                                                                              │
                                                                              ▼
                                                                  ┌────────────────────────┐
                                                                  │ temp/full_eclipse_     │
                                                                  │ <style>_video.mp4      │
                                                                  └───────────┬────────────┘
                                                                              │
                                                                    [Astronomical Engine]
                                                                (100% Direct RAW Timestamps +
                                                                 Multilingual Subtitles +
                                                                 QuickTime mov_text Muxing +
                                                                 Intelligent Music Sync)
                                                                              │
                                                                              ▼
                                          ┌───────────────────────────────────┼───────────────────────────────────┐
                                          ▼                                   ▼                                   ▼
                              ┌────────────────────────┐          ┌────────────────────────┐          ┌────────────────────────┐
                              │ <project_id>_es.srt    │          │ <project_id>.mp4       │          │ <project_id>           │
                              │ <project_id>_en.srt    │          │ (QuickTime Dual Subs + │          │ _chapters.txt          │
                              │ (YouTube / VLC)        │          │ Native Chapters + AAC) │          │ (YouTube Ready)        │
                              └────────────────────────┘          └────────────────────────┘          └────────────────────────┘
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

# Switch to the multi-phase processing branch
git checkout multi

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies from requirements.txt
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

### 1. Default End-to-End Build
Discovers all assets in `010_in/`, renders the opening title card `00_title.mp4`, stabilizes clips, renders the 16:9 composite artwork, assembles `040_out/full_eclipse.mp4`, and automatically generates multilingual subtitles and the QuickTime-ready master film:
```bash
python 020_src/build_full_eclipse.py
```

### 2. Standalone Subtitle Generation, QuickTime Embedding & YouTube Chapters
To generate or update subtitles and YouTube chapters independently without re-rendering videos:
```bash
# Generate English + Spanish SRTs, YouTube chapters, and embed in QuickTime MP4:
python 020_src/generate_eclipse_subtitles.py --embed

# Custom interval (e.g. every 10 seconds):
python 020_src/generate_eclipse_subtitles.py --interval 10.0 --embed
```

This generates:
- **`040_out/full_eclipse_es.srt` & `full_eclipse_en.srt`**: Multilingual subtitles synchronized to raw camera telemetry.
- **`040_out/youtube_chapters.txt`**: YouTube-compliant chapter markers (starting at `0:00`) in Bilingual, Spanish, and English formats.
- **`040_out/youtube_description.txt`**: Complete ready-to-paste YouTube video description with observer coordinates, ephemeris contact points, and timestamps.

```text
Capítulos / Chapters:
0:00 - Presentación / Eclipse Overview
0:08 - Ingreso Parcial / Partial Ingress (Timelapse)
0:19 - Aproximación Pre-totalidad / Pre-totality (Thin Crescent)
0:37 - C2: Anillo de Diamantes y Perlas de Baily / C2: Baily's Beads
0:41 - Totalidad y Corona Solar / Totality (Real-Time 1x)
2:13 - C3: Tercer Contacto / C3: Third Contact
2:22 - Egreso Parcial / Partial Egress (Timelapse)
2:33 - Fotografía HDR de la Corona / Solar Corona Photo
2:42 - Mosaico Secuencia del Eclipse / Eclipse Sequence Composite (Arc)
2:55 - Créditos Finales / Closing Credits
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

### 5. High-Resolution Composite Mosaic Artwork Generator (`create_eclipse_composite.py`)
Creates standalone high-resolution astronomical composite artwork across diverse mathematical progressions and aspect ratios, accompanied by comprehensive JSON metadata:

#### Key Layouts & Geometries:
- **`spiral` (Expanding Spiral)**: Progression starting directly at the center (Frame 0) and spiraling outward to the canvas perimeter with equal arc-length spacing. Tailored exclusively for square formats (e.g. 3840p) to maximize canvas fill and visual balance without central contacts.
- **`arc` (Parabolic Arch)**: Progression following a celestial solar arc. In portrait orientation ($9:16$), it expands vertically to maximize negative space.
- **`circle` / `ellipse`**: Circular progression with totality at 12 o'clock (top). Automatically transforms into an adaptive ellipse on non-square canvases ($16:9$, $9:16$).
- **`sinusoid` (S-Curve)**: Continuous sinusoidal wave parameterization with equal arc-length spacing.
- **`vertical` / `vertical-s`**: Linear and serpentine vertical columns tailored for mobile wallpapers ($9:16$).
- **`horizontal` / `diagonal`**: Linear horizontal and bottom-left to top-right diagonal trajectories.

#### Featured Totality Contacts (`--contacts`):
Inside the cavity of `arc`, `circle`, `ellipse`, and `sinusoid` layouts, the generator can render prominent enlarged keyframes ($1.22\times$ scale) of the totality sequence: **`C2`** *(diamond ring / Baily's beads at 20:27:35 CEST)*, **`TOTAL`** *(maximum grand corona at 20:28:23 CEST)*, and **`C3`** *(third contact diamond ring at 20:29:12 CEST)*. (Note: `spiral` layout does not use contacts as the sequence itself naturally fills the central core).

```bash
# 1. 8K Master Inward Spiral (7680x7680 px - 20 Frames with Timestamps):
python 020_src/create_eclipse_composite.py --layout spiral --size 7680 --frames 20 --show-labels

# 2. 4K Squared Inward Spiral (3840x3840 px - Perimeter-to-Center):
python 020_src/create_eclipse_composite.py --layout spiral --size 3840 --show-labels

# 3. Parabolic Celestial Arc with Contacts (Default for Video Assembly - 1280x720 16:9):
python 020_src/create_eclipse_composite.py --layout arc -W 1280 -H 720 --contacts

# 3. 4K UHD Desktop Arc with Timestamps (3840x2160 px):
python 020_src/create_eclipse_composite.py --layout arc -W 3840 -H 2160 --contacts --show-labels

# 4. 4K Squared Circle/Ring with Contacts & Centered Labels (3840x3840 px):
python 020_src/create_eclipse_composite.py --layout circle --size 3840 --contacts --show-labels

# 5. Vertical Mobile Wallpaper 9:16 with Vertical Contacts (2160x3840 px):
python 020_src/create_eclipse_composite.py --layout arc -W 2160 -H 3840 --contacts vertical --show-labels
python 020_src/create_eclipse_composite.py --layout circle -W 2160 -H 3840 --contacts vertical --show-labels

# 6. Sinusoidal S-Curve 8K Master (7680x7680 px):
python 020_src/create_eclipse_composite.py --layout sinusoid --size 7680 --contacts --show-labels

# 7. Clean Orbital Curve without Central Contacts:
python 020_src/create_eclipse_composite.py --layout circle --size 3840 --no-contacts --show-labels

# 8. Generate all layouts simultaneously:
python 020_src/create_eclipse_composite.py --layout all --contacts --show-labels
```

### 6. Standalone Closing Credits & End Titles Generator (`create_end_titles.py`)
Generates an anti-aliased, 2x supersampled closing credits card and video clip from a markdown file (e.g. `010_in/07_endtitles.md`):

```bash
# Generate 6.0s closing credits clip from markdown descriptor:
python 020_src/create_end_titles.py -i 010_in/07_endtitles.md --duration 6.0

# Generate for custom 4K resolution:
python 020_src/create_end_titles.py -i 010_in/07_endtitles.md -W 3840 -H 2160 -o 040_out/07_endtitles_4k.mp4
```

Example markdown descriptor (`010_in/07_endtitles.md`):
```markdown
# Telescope
DWARF mini
# Cameras
Lumix GH6
# Software
AI Scaling: Topaz Video AI Rhea
# Author
Nandoide
```

The engine automatically:
- Parses markdown headers into elegant uppercase section labels (`TELESCOPE`, `CAMERAS`, `SOFTWARE`, `AUTHOR`).
- Harmonizes pipeline software attribution (`Processing: eclipse-assembler`) in identical style and typography as user tools (`AI Upscaling: Topaz Video 1.7.0 (Rhea)`).
- Appends the generation/observation date (`DATE`) formatted uniformly.
- Renders at 2x resolution with Lanczos downsampling on solid black canvas.

### 7. Totality HDR Composite Photo & AI Prompt Generator (`create_totality_hdr.py`)
A standalone assistant tool to extract the 5 key temporal phases from raw totality footage, generate an immediate local OpenCV mathematical HDR composite, and dynamically format a tailored prompt adapted to the detected prominence and Baily bead positions for web-based multi-modal AI generation (Nano Banana / Gemini / ChatGPT):

```bash
# Run the automated extraction, local synthesis, and prompt generation:
python 020_src/create_totality_hdr.py

# Custom video input or custom output directory:
python 020_src/create_totality_hdr.py -v 010_in/03_video_realtime.mp4 -o 040_out
```

#### What `create_totality_hdr.py` produces:
1. **5 Clean Numbered Temporal Frames** (`040_out/hdr_samples/`):
   - `1_baily_in.jpg` ($t = 2.0\text{s}$): Ingress Baily's beads & western limb.
   - `2_c2_prom.jpg` ($t = 8.0\text{s}$): Western ruby-red $H\alpha$ prominence loops.
   - `3_mid_corona.jpg` ($t = 51.7\text{s}$): Soft, natural mid-totality solar corona.
   - `4_c3_prom.jpg` ($t = 98.0\text{s}$): Eastern carmine chromospheric spikes.
   - `5_baily_eg.jpg` ($t = 101.2\text{s}$): Egress diamond sparks along southeast limb.
2. **Local Mathematical OpenCV HDR Composite** (`040_out/totality_hdr_local.jpg`):
   - Instantaneous, 100% offline composite with sub-pixel ray-tracing alignment and continuous $H\alpha$ spectral feathering.
3. **Tailored AI Generation Prompt** (`040_out/hdr_samples/ai_prompt.txt`):
   - Programmatically detects angular positions of prominences and beads on the limb and formats a ready-to-paste prompt for web AI generation.

> [!TIP]
> **Workflow**: Save the resulting composite image (from web AI or the local OpenCV engine) as **`010_in/05_totality_6.jpg`**. The master build pipeline (`build_full_eclipse.py`) will automatically discover it, apply black letterbox padding, and tag it in subtitles and chapters.

### 8. Multi-Project Assembly CLI (`-p` / `--project`)

The master film assembler (`build_full_eclipse.py`) and standalone composite generator (`create_eclipse_composite.py`) accept a target project name:

```bash
# 1. Assemble specific project (e.g. Standard Linear with Preprocessing):
python 020_src/build_full_eclipse.py -p solar26_main_standard_preproc

# 2. Assemble Cinematic 'Art' Project with 4K Camera Dives:
python 020_src/build_full_eclipse.py -p solar26_v2_art_preproc

# 3. Assemble Raw Linear Footage with pre-totality slowdown burst:
python 020_src/build_full_eclipse.py -p solar26_v1_standard

# 4. Auto-detect project (picks the most recently modified project in 010_in/):
python 020_src/build_full_eclipse.py

# 5. Standalone 6000x6000 Circular Poster for specific project:
python 020_src/create_eclipse_composite.py -p solar26_main_standard_preproc -l circle --size 6000 --contacts auto --show-labels
```

| Montage Style              | Structure & Narrative Flow                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Key Visual Transitions                                                                       |
| :------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------- |
| **`standard`** *(default)* | Linear chronological progression through all CoC numbered assets in `010_in/<project>/`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Fades to black with customizable freeze holds between assets.                                |
| **`art`**                  | 1. **Title Card** (`00_title.mp4`).<br>2. **Composite Overview & Continuous Ingress Dive** (`art_01_dive_in.mp4`): Holds full composite canvas $\to$ dives into first sample disk $\to$ dissolves seamlessly into telescope footage.<br>3. **Partial Ingress Timelapse** (`01_timelapse_prep_i10` or `01_timelapse_i10.mp4`).<br>4. **Real-Time Totality Part 1** up to Totality Max.<br>5. **Totality Multi-Exposure HDR Artwork Insert** (`05_totality_6.jpg`): Crossfades smoothly at Max Eclipse ($t = 20:28:23\text{ CEST}$), holds, and crossfades back into footage.<br>6. **Real-Time Totality Part 2** from Totality Max through C3.<br>7. **Partial Egress Timelapse** (`04_timelapse_prep_i10` or `04_timelapse_i10.mp4`).<br>8. **Continuous Egress Zoom-Out Dive** (`art_02_dive_out.mp4`): Zooms out from last sample disk back to full composite overview.<br>9. **Closing Credits & End Titles** (`07_endtitles.md`).               | Continuous sub-pixel Lanczos-4 camera zoom dives + 0.8s crossfade dissolves at Totality Max. |

### 9. Continuous Camera Dive Engine (`create_camera_dive.py`)
Generates high-precision, sub-pixel Lanczos-4 affine camera dives between high-resolution composite mosaics (4K/8K) and telescope video footage:
- **Quintic Smooth Easing**: Perceptually uniform logarithmic scale acceleration ($S(t) = S_0 \cdot (S_1 / S_0)^{\sigma(t)}$).
- **Sub-Pixel Warping**: High-quality sinc interpolation prevents edge shimmering and moiré artifacts during magnification.
- **Reference Frame Dissolving**: Seamlessly dissolves into/out of the telescope video frames at the dive endpoints.

```bash
# Standalone Ingress Zoom-In Dive:
python 020_src/create_camera_dive.py --composite 040_out/solar26_v2_art_preproc/temp/art_composite_4k_circle.png --meta 040_out/solar26_v2_art_preproc/temp/art_composite_4k_circle.json --sample 0 --direction zoom_in -o 040_out/solar26_v2_art_preproc/temp/art_01_dive_in.mp4

# Standalone Egress Zoom-Out Dive:
python 020_src/create_camera_dive.py --composite 040_out/solar26_v2_art_preproc/temp/art_composite_4k_circle.png --meta 040_out/solar26_v2_art_preproc/temp/art_composite_4k_circle.json --sample -1 --direction zoom_out -o 040_out/solar26_v2_art_preproc/temp/art_02_dive_out.mp4
```

### 10. Ephemeris Database CLI & Refresh
```bash
# Query ephemeris contacts for auto-detected date:
python 020_src/eclipse_ephemeris_db.py

# Query for a specific solar or lunar eclipse date:
python 020_src/eclipse_ephemeris_db.py --date 2026-08-12

# Force refresh/rebuild database:
python 020_src/eclipse_ephemeris_db.py --force-db
```

### 11. Lightweight 30-Second Compact Video (for Easy Sharing)
Generates an accelerated, universally compatible 30-second version in H.264 (~1.19 MB in 720p HD) for quick distribution via messaging apps (WhatsApp, Telegram) or social media:
```bash
python 020_src/create_compact_clip.py
```

### 12. Force Re-processing from Scratch
```bash
python 020_src/build_full_eclipse.py -p solar26_main_standard_preproc --force-all
```

---

## 📂 Project Directory Structure

```
eclipse_assembler/
├── 000_raw/                                  # Untouched raw telescope footage & observer GPS photo by observation
│   └── solar26/                              # Observation campaign: Total Solar Eclipse 2026
│       ├── DWARF_mini_TELE_TL_2026-08-12-19-35-13-093.mp4
│       ├── DWARF_mini_TELE_2026-08-12-20-20-36-811.mp4
│       ├── DWARF_mini_TELE_TL_2026-08-12-20-32-53-127.mp4
│       └── gps.jpg                           # Observer coordinates & elevation metadata photo
│
├── 005_raw_preprocessed/                     # ★ Autonomous Photosphere-Restored & C1-Extrapolated Raw Timelapses
│   └── solar26/                              # Restored timelapses mapped to observation campaign
│       ├── DWARF_mini_TELE_TL_2026-08-12-19-35-13-093.mp4 # Ingress (288 frames: pre-C1, C1 contact & full sequence)
│       └── DWARF_mini_TELE_TL_2026-08-12-20-32-53-127.mp4 # Egress (293 frames: foliage/branch-free restored photosphere)
│
├── 010_in/                                   # Curated projects (Convention-over-Configuration)
│   ├── solar26_main_standard_preproc/        # Project: Official main film (Standard linear + Restored timelapses)
│   │   ├── 01_music_corrubedo_nandoide.wav   # Soundtrack audio track
│   │   ├── 01_timelapse_prep_i10             # Ingress directive (points to 005_raw_preprocessed/solar26/)
│   │   ├── 03_video_realtime.mp4             # Totality real-time continuous video (1x)
│   │   ├── 04_timelapse_prep_i10             # Egress directive (points to 005_raw_preprocessed/solar26/)
│   │   ├── 05_totality_6.jpg                 # Totality HDR multi-phase artwork (6s hold)
│   │   ├── 06_composite_circle_4k_10         # Circular 4K composite mosaic directive (10s hold)
│   │   └── 07_endtitles.md                   # Closing credits descriptor (6s hold)
│   │
│   ├── solar26_v1_standard/                  # Project: Version 1 (Standard linear + Raw footage + Slowdown burst)
│   │   ├── 01_music_corrubedo_nandoide.wav
│   │   ├── 01_timelapse_i10.mp4
│   │   ├── 02_video_slowdown_10.mp4
│   │   ├── 03_video_realtime.mp4
│   │   ├── 04_timelapse_i10.mp4
│   │   ├── 05_totality_6.jpg
│   │   ├── 06_composite_arc_10
│   │   └── 07_endtitles.md
│   │
│   └── solar26_v2_art_preproc/               # Project: Version 2 (Cinematic Art dives + Restored timelapses)
│       ├── 01_music_corrubedo_nandoide.wav
│       ├── 01_timelapse_prep_i10
│       ├── 03_video_realtime.mp4
│       ├── 04_timelapse_prep_i10
│       ├── 05_totality_6.jpg
│       ├── 06_composite_circle_4k_10
│       └── 07_endtitles.md
│
├── 020_src/                                  # Python source code & modular pipeline engines
│   ├── project_manager.py                    # ★ Multi-project & multi-observation resolution manager
│   ├── extract_eclipse_geometry.py           # ★ Orbital kinematics solver & C1 extrapolation engine
│   ├── build_master_solar_disk.py            # ★ 100% complete solar photosphere fusion engine
│   ├── render_restored_eclipse_video.py      # ★ Photosphere applicator & occultation bite renderer
│   ├── preprocess_raw_eclipse_timelapses.py  # ★ Master raw timelapse preprocessing orchestrator
│   ├── build_full_eclipse.py                 # ★ CoC Master pipeline and assembly script (--project, --film-style)
│   ├── create_camera_dive.py                 # ★ Continuous sub-pixel Lanczos-4 camera dive zoom generator
│   ├── add_audio_track.py                    # ★ Intelligent musical phrase retargeting and audio muxing
│   ├── create_totality_hdr.py                # ★ Totality HDR composite engine & dynamic AI prompt adapter
│   ├── eclipse_ephemeris_db.py               # ★ Universal Solar/Lunar Besselian database & solver
│   ├── fetch_eclipses_horizons.py            # ★ NASA JPL Horizons API fetcher & DB builder (2026-2036)
│   ├── compare_contacts_ephemeris.py         # ★ Observational telemetry vs. NASA ephemeris comparison
│   ├── create_title_card.py                  # ★ Cinematic opening title card generator (00_title.mp4)
│   ├── create_end_titles.py                  # ★ Cinematic closing credits generator (07_endtitles.mp4)
│   ├── generate_eclipse_subtitles.py         # ★ 100% RAW-based multilingual subtitle & YouTube chapters generator
│   ├── create_eclipse_composite.py           # ★ High-resolution composite mosaic generator & JSON metadata
│   ├── create_compact_clip.py                # 30s accelerated lightweight generator
│   ├── stabilize_eclipse.py                  # Core solar limb subpixel stabilizer
│   └── stabilize_full_film.py                # Global 2-pass master film stabilizer
│
├── 030_db/                                   # ★ Universal astronomical database
│   └── eclipses_db.json                      # Solar & Lunar eclipse NASA Besselian elements catalog (2026-2036)
│
├── 040_out/                                  # Generated stabilized master deliverables & project exports
│   ├── solar26_main_standard_preproc/        # Output for project solar26_main
│   │   ├── solar26_main.mp4                  # 🍎 Final master film with QuickTime dual subs & AAC music
│   │   ├── solar26_main_chapters.txt         # 📺 YouTube-formatted chapters
│   │   ├── solar26_main_en.srt               # 🇬🇧 English astronomical subtitles
│   │   ├── solar26_main_es.srt               # 🇪🇸 Spanish astronomical subtitles
│   │   ├── eclipse_composite_circle_6000p.png# 🖼️ High-resolution standalone poster (when run directly)
│   │   ├── eclipse_composite_circle_6000p.jpg
│   │   └── temp/                             # 📁 Intermediate clips, title cards, waveforms & JSONs
│   │
│   ├── solar26_v1_standard/                  # Output for project solar26_v1
│   │   ├── solar26_v1.mp4                    # 🍎 Final master film
│   │   ├── solar26_v1_chapters.txt
│   │   ├── solar26_v1_en.srt
│   │   ├── solar26_v1_es.srt
│   │   └── temp/
│   │
│   └── solar26_v2_art_preproc/               # Output for project solar26_v2
│       ├── solar26_v2.mp4                    # 🍎 Final cinematic Art master film
│       ├── solar26_v2_chapters.txt
│       ├── solar26_v2_en.srt
│       ├── solar26_v2_es.srt
│       └── temp/
│
├── requirements.txt                          # Python package dependencies
└── README.md                                 # Project documentation
```

---

## 📊 Summary of Master Output Assets

| Output Asset (in `040_out/<project>/`) | Resolution / Format | Frame Rate / Type | Duration / Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`<project_id>.mp4`** | **1280x720** (or target) | **30.0 fps** | **~168–181s** | 🍎 **Master Film: Dual QuickTime embedded subtitles (ES+EN), chapters, synchronized musical score** |
| **`<project_id>_en.srt`** | SubRip (.srt) | UTF-8 | Subtitles | 🇬🇧 **Astronomical subtitles in English with real-time telemetry from camera NTP/GPS** |
| **`<project_id>_es.srt`** | SubRip (.srt) | UTF-8 | Subtitles | 🇪🇸 **Astronomical subtitles in Spanish with real-time telemetry from camera NTP/GPS** |
| **`<project_id>_chapters.txt`** | Plain Text | UTF-8 | Chapters | 📺 **YouTube chapters formatted with exact timestamps** |
| **`eclipse_composite_*.png/.jpg`** | Up to 16K UHD | Lossless / JPEG | Static Artwork | 🖼️ **High-resolution standalone eclipse progression poster** |
| **`temp/`** | Directory | — | — | 📁 **Intermediate rendering directory containing raw stitched video, audio waveforms, and JSON metadata** |

---

## 📄 License & Attribution
Developed for high-precision astronomical astrophotography and eclipse video processing.

- **Project Lead & Domain Guidance**: Fernando ([@nandoide](https://github.com/nandoide))
- **AI Coding Agent & Computer Vision Architecture**: Antigravity (Google Deepmind)

Code released under the MIT License.
