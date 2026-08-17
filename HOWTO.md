# Comprehensive Technical Manual & How-To Guide (`HOWTO.md`)

> **Eclipse Assembler Suite**: An end-to-end, high-precision computer vision, astronomical telemetry, and cinematic assembly pipeline for total & partial solar eclipses.

---

## 📑 Table of Contents

1. [Architectural Overview & Design Philosophy](#1-architectural-overview--design-philosophy)
2. [Directory Structure & Convention-Over-Configuration (CoC)](#2-directory-structure--convention-over-configuration-coc)
3. [Computer Vision & Sub-Pixel Stabilization Engines](#3-computer-vision--sub-pixel-stabilization-engines)
   - [3.1 Partial Ingress & Egress Timelapses (The Oblate Atmospheric Solver)](#31-partial-ingress--egress-timelapses-the-oblate-atmospheric-solver)
   - [3.2 Pre-Totality Crescent Approach Slowdown](#32-pre-totality-crescent-approach-slowdown)
   - [3.3 Totality Real-Time 1x Engine (Lunar Silhouette & Corona Tracking)](#33-totality-real-time-1x-engine-lunar-silhouette--corona-tracking)
   - [3.4 Sub-Pixel Affine Warping & Centering](#34-sub-pixel-affine-warping--centering)
4. [NASA JPL Ephemeris & Topocentric Astronomical Geometry](#4-nasa-jpl-ephemeris--topocentric-astronomical-geometry)
   - [4.1 Besselian Polynomials & Topocentric Contact Solver](#41-besselian-polynomials--topocentric-contact-solver)
   - [4.2 Optical vs. Theoretical Ephemeris Benchmarking](#42-optical-vs-theoretical-ephemeris-benchmarking)
5. [Astronomical Multilingual Subtitle Engine](#5-astronomical-multilingual-subtitle-engine)
   - [5.1 Real-Time Telemetry Calculation](#51-real-time-telemetry-calculation)
   - [5.2 Dual QuickTime / Apple TV Subtitle Embedding](#52-dual-quicktime--apple-tv-subtitle-embedding)
   - [5.3 YouTube Chapter & Description Export](#53-youtube-chapter--description-export)
6. [Multi-Layout Eclipse Composite Artwork Generator](#6-multi-layout-eclipse-composite-artwork-generator)
   - [6.1 Layout Geometries (Spiral, Arc, Sinusoid, Circle, Linear)](#61-layout-geometries-spiral-arc-sinusoid-circle-linear)
   - [6.2 Featured Contact Alignment (C2, MAX, C3)](#62-featured-contact-alignment-c2-max-c3)
   - [6.3 Consistent Disk Scaling & 8K/16K Ultra-HD Export](#63-consistent-disk-scaling--8k16k-ultra-hd-export)
7. [Cinematic Title Card Engine](#7-cinematic-title-card-engine)
8. [Master Film Concatenation & H.265 Broadcast Encoding](#8-master-film-concatenation--h265-broadcast-encoding)
9. [Step-by-Step Recipes & CLI Reference](#9-step-by-step-recipes--cli-reference)
10. [Troubleshooting & Best Practices](#10-troubleshooting--best-practices)

---

## 1. Architectural Overview & Design Philosophy

The **Eclipse Assembler** is engineered to transform heterogeneous raw captures from smart telescopes (e.g., DWARFLAB DWARF mini / II / 3, ZWO Seestar S50, Vaonis Vespera) or intervalometer-equipped mirrorless cameras into a broadcast-quality, perfectly stabilized cinematic master film (`040_out/full_eclipse.mp4`).

```
                    ┌──────────────────────────────┐
                    │   000_raw/ (Ground Truth)    │
                    │  Raw Video + EXIF GPS Image  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │ 020_src/eclipse_ephemeris_db.py                      │
        │ NASA JPL Horizons API + Topocentric Besselian Solver │
        └──────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │ 010_in/ (Curated Convention-Over-Configuration)      │
        │  • 01_timelapse_i10.mp4                              │
        │  • 02_video_slowdown_10.mp4                          │
        │  • 03_video_realtime.mp4                             │
        │  • 04_timelapse_i10.mp4                              │
        │  • 05_totality_6.jpg (or 05_photo_6.jpg)             │
        │  • 06_composite_arc_10                               │
        │  • 07_endtitles.md                                   │
        └──────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │ Computer Vision & Sub-Pixel Stabilization Pipeline   │
        │  • Oblate Atmospheric Refraction Solver              │
        │  • 3-Pass Trimmed Inlier Huber Optimizer             │
        │  • Lunar Silhouette & Corona Annular Edge Tracker    │
        │  • Sub-pixel Affine Warper (cv2.INTER_LANCZOS4)      │
        └──────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │ Assembly & Multilingual Subtitle Pipeline            │
        │  • H.265 / HEVC (Apple hvc1) Concat Streaming        │
        │  • Dual-Language Subtitle Tracks (EN / ES)           │
        │  • YouTube Chapters & Composite Artwork              │
        └──────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ 040_out/full_eclipse.mp4     │
                    │ 040_out/full_eclipse_sub.mp4 │
                    └──────────────────────────────┘
```

### Core Design Tenets:
1. **Zero Artificial Temporal Fudging**: Raw telescope timestamps ($t_0$) are treated as inviolable physical ground truth. No manual clock offsets are applied to mask reality; optical contacts are mathematically compared to NASA ephemeris predictions.
2. **Sub-Pixel Spatial Anchoring**: Every solar and lunar phase is mathematically solved and warped onto the exact optical center $(W/2, H/2)$ of the master video canvas (for example, $(640.0, 360.0)$ for the 720p master demonstrated in this repository, $(960.0, 540.0)$ for 1080p, or $(1920.0, 1080.0)$ for 4K UHD—scaling dynamically with the target project resolution), eliminating telescope mount drift, wind vibrations, and tracking backlash.
3. **Atmospheric Physical Modeling**: Near-horizon low-altitude solar flattening (atmospheric refraction) is explicitly handled via oblate elliptical models rather than naive circular approximations.
4. **Convention Over Configuration (CoC)**: All pipeline parameters (speed multipliers, target durations, layouts) are derived from the filenames in `010_in/`.

---

## 2. Directory Structure & Convention-Over-Configuration (CoC)

> [!NOTE]
> **Illustrative Concrete Case Study**: The directory structure and file names shown below illustrate a specific real-world execution for the **August 12, 2026 Total Solar Eclipse** recorded with a DWARFLAB DWARF mini smart telescope in 720p. The architecture is fully generic and automatically scales to any eclipse date, observer coordinates, telescope or camera brand (e.g. ZWO Seestar, Vaonis, Sony/Canon DSLRs), and arbitrary output resolutions (1080p, 4K, 8K).

```
eclipse26/
├── 000_raw/                 # Untouched telescope files (timestamps & GPS EXIF source)
│   ├── DWARF_mini_TELE_TL_2026-08-12-19-35-13-093.mp4
│   ├── DWARF_mini_TELE_2026-08-12-20-20-36-811.mp4
│   ├── DWARF_mini_TELE_TL_2026-08-12-20-32-53-127.mp4
│   └── gps.jpg              # Any photo with EXIF GPS coordinates of observation site
├── 010_in/                  # Curated CoC source clips and instructions
│   ├── 01_music_corrubedo_nandoide.wav # Soundtrack ([INDEX]_music_[TITLE]_[AUTHOR].[ext])
│   ├── 01_timelapse_i10.mp4
│   ├── 02_video_slowdown_10.mp4
│   ├── 03_video_realtime.mp4
│   ├── 04_timelapse_i10.mp4
│   ├── 05_totality_6.jpg (or 05_photo_6.jpg)
│   ├── 06_composite_arc_10
│   └── 07_endtitles.md
├── 020_src/                 # Core Python engine modules
│   ├── add_audio_track.py
│   ├── build_full_eclipse.py
│   ├── create_totality_hdr.py
│   ├── create_eclipse_composite.py
│   ├── create_title_card.py
│   ├── create_end_titles.py
│   ├── eclipse_ephemeris_db.py
│   ├── fetch_eclipses_horizons.py
│   ├── generate_eclipse_subtitles.py
│   ├── stabilize_eclipse.py
│   └── compare_contacts_ephemeris.py
├── 030_db/                  # Persistent SQLite & JSON ephemeris cache
│   └── eclipses_db.json
├── 040_out/                 # Rendered assets, composite art, subtitles, and master film
└── requirements.txt
```

### CoC Asset Naming Grammar:
`[INDEX]_[TYPE]_[PARAMS/INTERVAL/LAYOUT/TITLE]_[DURATION/AUTHOR].[ext]`

- `_i[N]` specifies intervalometer step in seconds (e.g., `_i10` = 1 frame every 10s $\to$ $300\times$ speedup at 30 fps).
- `_slowdown_[N]` specifies target duration in seconds for burst pre-totality footage (e.g., `_slowdown_10` = 10s).
- `_composite_[LAYOUT]_[N]` specifies on-the-fly artwork generation (`arc`, `spiral`, `sinusoid`, `circle`) held for $N$ seconds.
- `_photo_[N]` specifies static photo held for $N$ seconds.
- `_music_[TITLE]_[AUTHOR].[ext]` specifies a soundtrack audio track (e.g., `01_music_corrubedo_nandoide.wav` $\to$ Title: *"Corrubedo"*, Composer: *Nandoide*). Generically analyzes spectral novelty, onset boundaries, and RMS energy to seamlessly retarget the music at 100% natural tempo with phase-aligned crossfades, synchronizing the primary climax with Totality Max and automatically appending a `MUSIC` credits card to `07_endtitles.mp4`.

---

## 3. Computer Vision & Sub-Pixel Stabilization Engines

Stabilizing an eclipse sequence presents unique optical hurdles:
- The crescent changes shape and area continuously.
- Lunar cusps deceive centroid detectors.
- Passing clouds alter contrast gradients.
- Atmospheric refraction at low solar elevations vertically compresses the Sun into an oblate ellipse.
- Totality exhibits extremely faint coronal streamers against an ultra-dark sky.

The pipeline implements dedicated computer-vision solvers for every phase:

---

### 3.1 Partial Ingress & Egress Timelapses (The Oblate Atmospheric Solver)

#### The Problem with Circular Solvers at Sunset/Sunrise:
At low solar elevations (e.g., $< 10^\circ$ above the horizon during the August 12, 2026 egress), atmospheric refraction compresses the vertical axis by ~4%. The Sun's horizontal semi-axis remains $R_x \approx 237.5\text{ px}$, but the vertical semi-axis shrinks to $R_y \approx 228.0\text{ px}$. Fitting a rigid circle ($R = 237.5$) creates an ill-conditioned optimization valley where passing clouds or moon transit cause the solver to oscillate vertically by $4-14\text{ px}$.

#### The Oblate Physical Formulation:
The boundary of the flattened solar disk is modeled as a time-varying oblate ellipse:

$$\left(\frac{x - c_x}{R_x}\right)^2 + \left(\frac{y - c_y}{R_y(t)}\right)^2 = 1$$

where:
$$R_x = 238.0 \cdot \left(\frac{H_{\text{master}}}{720}\right)$$
$$R_y(t) = R_x \cdot \left[1.0 - 0.042 \cdot \left(\frac{t - t_{\text{start}}}{t_{\text{end}} - t_{\text{start}}}\right)\right]$$

#### Multi-Threshold Ensemble & Convex-Hull Extraction:
1. Three percentile thresholds ($p \in [0.20, 0.35, 0.50]$, corresponding to 20%, 35%, and 50%) are computed on the green channel to extract candidate limb contours across differing cloud-attenuation regimes.
2. Contours smaller than 15% of expected solar area are pruned.
3. The convex hull of the exterior contour is computed to exclude the inner lunar intrusion and cloud cutouts.
4. Edge normals are filtered so only outer boundary points facing outward contribute to solar tracking.

#### 3-Pass Trimmed Inlier Optimizer:
To guarantee absolute immunity against dense passing clouds:
- **Pass 1**: Initial Nelder-Mead optimization using Huber loss ($\delta = 3.0\text{ px}$):
  $$\mathcal{L}_{\text{Huber}}(r) = \begin{cases} \frac{1}{2}r^2 & \text{for } |r| \le \delta \\ \delta(|r| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$
- **Pass 2**: Compute radial residuals $r_i = \left|\sqrt{\frac{(x_i-c_x)^2}{R_x^2} + \frac{(y_i-c_y)^2}{R_y^2}} - 1.0\right| \cdot R_x$. Discard all outlier points with $r_i > 2.0\text{ px}$.
- **Pass 3**: Re-optimize $(c_x, c_y)$ strictly on the surviving > 95% confidence inlier set.

Result: Center variance across egress clouds is reduced from $\sigma_y = 4.2\text{ px}$ to $\sigma_y \le 0.3\text{ px}$.

---

### 3.2 Pre-Totality Crescent Approach Slowdown (`02_video_slowdown_10`)

The pre-totality phase captures the rapid collapse of the photospheric crescent over $\sim 400\text{ seconds}$ before C2:

1. **Corrupt Frame Filter**: Intervalometer video streams often record 1-2 black frames between exposure transitions. Frames with $\max(I) < 25$ are identified and discarded.
2. **Smooth Non-Linear Resampling**: Resamples $N_{\text{valid}}$ frames down to $300\text{ frames}$ ($10.0\text{s} @ 30\text{ fps}$) using cubic time-index interpolation.
3. **Sub-Pixel Crescent Tracking**: Applies the multi-threshold oblate estimator to trace the razor-thin solar limb.
4. **Auto-White-Balance & Luminance Continuity**: Telescopes adjusting exposure step-wise introduce visual flicker. A rolling median gain factor $g(t) = \frac{\bar{I}_{\text{target}}}{\bar{I}(t)}$ normalizes frame brightness, while color temperature drift is stabilized in LAB color space.

---

### 3.3 Totality Real-Time 1x Engine (`03_video_realtime`)

Totality is processed at full $1\times$ native speed ($2,931\text{ frames} = 97.7\text{s}$):

1. **Lunar Silhouette Edge Detection**: Applies Canny edge detection ($T_{\text{low}}=15, T_{\text{high}}=45$) combined with a spatial annular mask $r \in [R_{\text{lunar}} - 35, R_{\text{lunar}} + 35]$ around the predicted Moon position.
2. **Direct Lunar Limb Tracking**: Fits the dark lunar disk ($R_{\text{lunar}} = 246.0\text{ px}$) against the luminous inner corona.
3. **Kalman Trajectory Smoothing**: A 2D constant-velocity Kalman filter dampens single-frame atmospheric scintillation while tracking the telescope mount's sidereal tracking rate.
4. **C2/C3 Alignment Offset**: Seamlessly applies the topocentric offset between solar center and lunar center at C2 ($+12.5\text{ px}, +7.5\text{ px}$) to prevent spatial jumping during phase transitions.

---

### 3.4 Sub-Pixel Affine Warping & Centering

For every frame, once the optical center $(c_x, c_y)$ is derived from the active footage:

$$\mathbf{M} = \begin{bmatrix} 1 & 0 & (W_{\text{master}}/2 - c_x) \\ 0 & 1 & (H_{\text{master}}/2 - c_y) \end{bmatrix}$$

where $(W_{\text{master}}/2, H_{\text{master}}/2)$ represents the exact optical center of the destination canvas. Note that these coordinate values depend directly on the master/source resolution being processed:
- **720p Canvas ($1280 \times 720$)**: $(c_{x,\text{target}}, c_{y,\text{target}}) = (\mathbf{640.0}, \mathbf{360.0})$
- **1080p Canvas ($1920 \times 1080$)**: $(c_{x,\text{target}}, c_{y,\text{target}}) = (\mathbf{960.0}, \mathbf{540.0})$
- **4K UHD Canvas ($3840 \times 2160$)**: $(c_{x,\text{target}}, c_{y,\text{target}}) = (\mathbf{1920.0}, \mathbf{1080.0})$
- **8K Canvas ($7680 \times 7680$)**: $(c_{x,\text{target}}, c_{y,\text{target}}) = (\mathbf{3840.0}, \mathbf{3840.0})$

The image is resampled via `cv2.warpAffine` using **Lanczos-4 8-tap sinc interpolation** (`cv2.INTER_LANCZOS4`) with constant black borders (`cv2.BORDER_CONSTANT`), ensuring zero interpolation blurring and pristine high-frequency details on solar prominences and Baily's beads.

---

## 4. NASA JPL Ephemeris & Topocentric Astronomical Geometry

### 4.1 Besselian Polynomials & Topocentric Contact Solver

Rather than storing approximate contact times, `020_src/eclipse_ephemeris_db.py` contains full 8-variable **NASA Besselian Elements**:

$$x(t), y(t), d(t), l_1(t), l_2(t), \mu(t), \tan f_1, \tan f_2$$

Given the observer's geographic position $(\phi, \lambda, h)$ from `000_raw/gps.jpg`:
1. Calculates geocentric coordinates $(\rho \sin \phi', \rho \cos \phi')$ accounting for Earth's WGS84 oblateness.
2. Evaluates the topocentric shadow axis distance $\Delta(t) = \sqrt{(x - \xi)^2 + (y - \eta)^2}$.
3. Numerically solves for exact contact instants:
   - **$C_1$**: Penumbral exterior contact ($\Delta = l_1 - \zeta \tan f_1$)
   - **$C_2$**: Umbral interior contact ($\Delta = l_2 - \zeta \tan f_2$)
   - **$\text{MAX}$**: Minimum shadow axis separation ($\frac{d\Delta}{dt} = 0$)
   - **$C_3$**: Umbral exit contact ($\Delta = l_2 - \zeta \tan f_2$)
   - **$C_4$**: Penumbral exterior exit ($\Delta = l_1 - \zeta \tan f_1$)

---

### 4.2 Optical vs. Theoretical Ephemeris Benchmarking

Run the contact comparator at any time:
```bash
.venv/bin/python3 020_src/compare_contacts_ephemeris.py
```

#### Comparison for August 12, 2026 (Site: 43.2356°N, 7.5583°W, Alt: 438.7m):

| Astronomical Milestone | NASA Ephemeris (Besselian) | Optical Camera Telemetry (RAW) | Residual ($\Delta t$) | Physical Significance |
| :--- | :--- | :--- | :--- | :--- |
| **$C_1$ (First Contact)** | `19:31:18.5 CEST` | `19:35:13 CEST` *(TL start)* | — | Ingress timelapse baseline |
| **$C_2$ (Second Contact)** | `20:27:38.2 CEST` | **`20:27:34.8 CEST`** | **`-3.4 s`** | Extinction of last Baily's bead |
| **$\text{MAX}$ (Deepest Eclipse)**| `20:28:25.0 CEST` | **`20:28:23.3 CEST`** | **`-1.7 s`** | Symmetry midpoint of corona |
| **$C_3$ (Third Contact)** | `20:29:12.2 CEST` | **`20:29:11.8 CEST`** | **`-0.4 s`** | Emergence of first diamond bead |
| **$C_4$ (Fourth Contact)** | `21:21:52.6 CEST` | `21:13:53 CEST` *(TL end)* | — | Egress timelapse completion |
| **Totality Duration** | `1m 33.96s` ($94.0\text{ s}$) | **`1m 36.96s`** ($97.0\text{ s}$) | **`+3.0 s`** | LRO/Kaguya lunar valley delay |

---

## 5. Astronomical Multilingual Subtitle Engine

`020_src/generate_eclipse_subtitles.py` generates frame-accurate subtitles containing live physical telemetry.

### 5.1 Real-Time Telemetry Calculation

Every subtitle entry (every 5.0 seconds) dynamically computes:
- **Local Time**: CEST (UTC+2) formatted as `HH:MM:SS`.
- **UTC Time**: `HH:MM:SS UTC`.
- **Mission Time**: Relative to C2/C3 ($T - 02\text{m } 15\text{s}$ or $T + 00\text{m } 45\text{s}$).
- **Solar Position**: Topocentric Solar Altitude ($h_\odot$) and Azimuth ($A_\odot$) with refraction.
- **Eclipse Phase**: `Partial Ingress`, `Baily's Beads / Diamond Ring`, `Totality (Solar Corona)`, `Partial Egress`.
- **Obscuration & Magnitude**: Real-time lunar disk coverage percentage.

```
Example Subtitle Block (English):
4
00:00:15,000 --> 00:00:20,000
19:55:04 CEST | 17:55:04 UTC | T - 32m 34s
Phase: Partial Ingress | Obscuration: 42.8%
Sun Alt: 14.2° | Az: 281.4° (WNW)
```

```
Example Subtitle Block (Spanish):
4
00:00:15,000 --> 00:00:20,000
19:55:04 CEST | 17:55:04 UTC | T - 32m 34s
Fase: Ingreso Parcial | Oscurecimiento: 42.8%
Alt Solar: 14.2° | Az: 281.4° (ONO)
```

---

### 5.2 Dual QuickTime / Apple TV Subtitle Embedding

FFmpeg embeds both language tracks losslessly into MP4 (`tx3g` / `mov_text` codecs):

```bash
ffmpeg -y -i 040_out/full_eclipse.mp4 \
  -i 040_out/full_eclipse_es.srt \
  -i 040_out/full_eclipse_en.srt \
  -map 0:v -map 1:0 -map 2:0 \
  -c:v copy \
  -c:s mov_text \
  -metadata:s:s:0 language=spa -metadata:s:s:0 title="Español (Hora Local)" \
  -metadata:s:s:1 language=eng -metadata:s:s:1 title="English (Local Time)" \
  040_out/full_eclipse_subtitled.mp4
```

---

### 5.3 YouTube Chapter & Description Export

The pipeline automatically outputs ready-to-paste YouTube metadata and embeds native chapters:
- **`040_out/youtube_chapters.txt`**: Timestamped chapter list for YouTube.
- **`040_out/youtube_description.txt`**: Full astronomical description, observation site coordinates, telescope hardware telemetry, and contact times.
- **`040_out/full_eclipse_subtitled.mp4`**: Contains embedded QuickTime/MP4 chapters (`chpl` / `udta`) and dual multilingual subtitle tracks (`spa` & `eng`) navigable in QuickTime Player, VLC, and Apple TV.

---

## 6. Multi-Layout Eclipse Composite Artwork Generator

`020_src/create_eclipse_composite.py` creates high-resolution sequence mosaics from video and timelapse frames.

```
               [ sample_eclipse_sequence() ]
               Auto-samples partials & totality from DB
                             │
                             ▼
                 [ Layout Curve Geometry ]
      ┌──────────────┬──────────────┬──────────────┐
      │              │              │              │
      ▼              ▼              ▼              ▼
   SPIRAL           ARC          SINUSOID        CIRCLE
(Center Totality) (Vault Arc) (S-Curve Wave) (Orbital Ring)
      │              │              │              │
      └──────────────┴──────┬───────┴──────────────┘
                            │
                            ▼
               [ Auto-Scale Disks & Gaps ]
          Min distance d_min * disk_scale_factor
                            │
                            ▼
          [ Featured Contacts (C2, MAX, C3) ]
           Inner vault placement for arc/circle
                            │
                            ▼
               [ Soft Max Blending Engine ]
         Preserves high-contrast coronal streamers
                            │
                            ▼
             Lossless PNG / 8K / 16K Master Output
```

---

### 6.1 Layout Geometries

1. **`spiral` / `espiral`**:
   - Inward Archimedean/power spiral progression:
     $$r(\theta) = R_{\text{max}} \cdot (1 - t)^\alpha, \quad \theta(t) = 2\pi N_{\text{turns}} t$$
   - Ingress starts on the outermost orbital track; totality sits crowned at the exact center.
2. **`arc`**:
   - Upward vaulted parabola or circular arc mimicking the solar trajectory across the sky.
3. **`sinusoid` / `s-curve`**:
   - Smooth harmonic wave crossing horizontal or vertical axes.
4. **`circle` / `ring` / `ellipse`**:
   - Orbital closed loop with totality contacts displayed inside the central vault.
5. **`horizontal` / `vertical` / `diagonal`**:
   - Linear equidistant alignment across standard print aspects.

---

### 6.2 Featured Contact Alignment (`C2`, `MAX`, `C3`)

In `arc`, `circle`, and `sinusoid` layouts, three high-resolution inner frames are rendered in the central vault:
- **`C2`**: Second Contact (Baily's Beads extinction & Diamond Ring).
- **`MAX`**: Maximum Eclipse coronal crown.
- **`C3`**: Third Contact (Emergence of the trailing Diamond Ring).

---

### 6.3 Consistent Disk Scaling & 8K/16K Ultra-HD Export

To prevent adjacent frames from overlapping regardless of layout density:
$$\text{Disk Diameter} = \min_{i} \left( \|\mathbf{p}_{i+1} - \mathbf{p}_i\| \right) \cdot k_{\text{scale}}$$

Supported render sizes range from 720p up to **8K (7680x7680)** and **16K (15360x8640)** for astronomical gallery printing.

---

## 7. Cinematic Title Card Engine

`020_src/create_title_card.py` creates a 5.0s introduction video (`00_title.mp4`):
- High-DPI 2x supersampled text rasterization downsampled via Lanczos filtering.
- Automatically queries the SQLite/JSON database for topocentric coordinates, totality duration, and contact timestamps.
- Fades smoothly into black before the first ingress timelapse.

---

---

## 8. Cinematic Closing Credits & Production Telemetry Engine

`020_src/create_end_titles.py` generates an anti-aliased closing credits card and video clip (`07_endtitles.mp4`) when a markdown descriptor (e.g., `010_in/07_endtitles.md`) is provided in the input directory:
- **Dynamic Markdown Parsing**: Extends to custom categories (`# Telescope`, `# Cameras`, `# Software`, `# Author`, `# Observation Site`).
- **Software Pipeline Harmonization**: Automatically includes and standardizes pipeline processing attribution (`Processing: eclipse-assembler`) alongside custom post-processing tools (`AI Upscaling: Topaz Video 1.7.0 (Rhea)`).
- **Unified Visual Hierarchy**: 2x supersampled PIL rasterization downsampled with Lanczos filtering on solid black canvas, maintaining identical typography and color palette across all sections.
- **Date Standardization**: Automatically appends the project observation / generation date formatted cleanly as a dedicated metadata block.

---

## 9. Totality HDR Composite Photo & AI Prompt Generator

`020_src/create_totality_hdr.py` extracts the 5 key temporal moments of solar totality, synthesizes an immediate local mathematical HDR photo, and formats a tailored prompt adapted to the detected prominence and Baily bead positions for web-based multi-modal AI generation:

```bash
# Run the automated extraction, local synthesis, and prompt generation:
python 020_src/create_totality_hdr.py
```

### Key Workflow:
1. **5 Temporal Samples (`040_out/hdr_samples/`)**:
   - `1_baily_in.jpg` ($t = 2.0\text{s}$): Ingress Baily's beads & western limb.
   - `2_c2_prom.jpg` ($t = 8.0\text{s}$): Western ruby-red $H\alpha$ prominence loops.
   - `3_mid_corona.jpg` ($t = 51.7\text{s}$): Soft, natural mid-totality solar corona.
   - `4_c3_prom.jpg` ($t = 98.0\text{s}$): Eastern carmine chromospheric spikes.
   - `5_baily_eg.jpg` ($t = 101.2\text{s}$): Egress diamond sparks along southeast limb.
2. **Local Mathematical HDR (`040_out/totality_hdr_local.jpg`)**:
   - 100% offline, deterministic OpenCV composite with sub-pixel ray-tracing alignment and continuous $H\alpha$ spectral feathering.
3. **Web AI Generation (Nano Banana / Gemini / ChatGPT)**:
   - Upload the 5 extracted images from `040_out/hdr_samples/`.
   - Copy & paste the tailored prompt generated in `040_out/hdr_samples/ai_prompt.txt`.
   - Save the returned master image as `040_out/totality_hdr.jpg`.


---

---

## 10. Master Film Concatenation & H.265 Broadcast Encoding

`020_src/build_full_eclipse.py` orchestrates the complete assembly:

1. **Cross-Asset Transition Fades**: Computes smooth 0.5s fade-to-black dips between differing exposure regimes.
2. **Memory-Safe Pipe Streaming**: Passes individual frames directly to FFmpeg via `stdin` Unix pipes to avoid gigabytes of intermediate temporary PNGs.
3. **H.265 / HEVC Broadcast Encoding**:
   ```bash
   ffmpeg -y -f rawvideo -pix_fmt bgr24 -s 1280x720 -r 30 \
     -i - -c:v libx265 -preset fast -crf 16 \
     -tag:v hvc1 -pix_fmt yuv420p 040_out/full_eclipse.mp4
   ```
   The `-tag:v hvc1` flag ensures native hardware decoding on macOS QuickTime Player, iOS Safari, Apple TV, and Windows Movies & TV.

---

## 11. Step-by-Step Recipes & CLI Reference

### Environment Setup
```bash
# Clone repository and enter workspace
cd eclipse26

# Activate virtual environment
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

### Recipe 1: Build the Entire Master Film from Scratch
```bash
.venv/bin/python3 020_src/build_full_eclipse.py --force-all
```
*Processes all raw input clips, stabilizes them, generates title card, composite, subtitles, QuickTime MP4, and YouTube metadata.*

---

### Recipe 2: Quick Incremental Master Rebuild
```bash
.venv/bin/python3 020_src/build_full_eclipse.py
```
*Skips already processed intermediate assets and re-assembles the master film in seconds.*

---

### Recipe 3: Generate 8K Ultra-HD Spiral Composite
```bash
.venv/bin/python3 020_src/create_eclipse_composite.py \
  --layout spiral \
  --size 7680 \
  --frames 20 \
  -o 040_out/eclipse_composite_spiral_8k.png
```

---

### Recipe 4: Generate 4K Arc Composite with Timestamp Labels
```bash
.venv/bin/python3 020_src/create_eclipse_composite.py \
  --layout arc \
  --width 3840 \
  --height 2160 \
  --frames 14 \
  --show-labels \
  -o 040_out/eclipse_composite_arc_labeled_4k.png
```

---

### Recipe 5: Compare Camera Optical Timings vs NASA Ephemeris
```bash
.venv/bin/python3 020_src/compare_contacts_ephemeris.py
```

---

### Recipe 6: Regenerate Multilingual Subtitles Only
```bash
.venv/bin/python3 020_src/generate_eclipse_subtitles.py
```

---

## 10. Troubleshooting & Best Practices

| Symptom | Probable Cause | Recommended Solution |
| :--- | :--- | :--- |
| **Vertical jitter during low-altitude egress** | Atmospheric vertical flattening deceived circular solver | Ensure using the oblate solver (`020_src/build_full_eclipse.py`), which uses the physical ellipse model $R_y(t) = R_x(1.0 - 0.042 \cdot \text{prog})$. |
| **Flicker in pre-totality crescent** | Telescope auto-exposure stepped abruptly | `process_video_slowdown_asset` applies moving-average luminance gain correction automatically. Rebuild with `--force-all`. |
| **Subtitles not appearing in QuickTime** | Track metadata missing `tx3g` flag | The pipeline embeds subtitles with `mov_text` and `-tag:v hvc1`. Open in QuickTime and select *Subtitles $\to$ Español / English*. |
| **Missing observer GPS data** | `gps.jpg` missing from `000_raw/` | Place any smartphone or camera photo taken at the observing site containing EXIF GPS into `000_raw/gps.jpg`. |
| **FFmpeg libx265 error on macOS** | Homebrew FFmpeg missing HEVC encoder | Install full FFmpeg via `brew install ffmpeg`. |

---

*Manual maintained by the Eclipse Assembler Development Team.*
