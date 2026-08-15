# Solar Eclipse 2026: Cinematic Processing & Assembly Pipeline (CoC Multi-Asset Architecture)

An automated, high-precision computer vision pipeline designed to stabilize, accelerate, align, and assemble multi-phase solar eclipse footage, still photographs, and on-the-fly composite artwork into a single, perfectly centered cinematic master film (`full_eclipse.mp4`), as well as an accelerated compact version (`full_eclipse_30s.mp4`) optimized for instant sharing.

---

## 🌒 Overview & CoC Architecture

This pipeline follows **Convention over Configuration (CoC)**. File names placed in `010_in/` define the sequence ordering, asset types, custom durations, and processing pipelines without needing configuration files:

`[INDEX]_[TYPE]_[PARAMS/LAYOUT]_[RES]_[DURATION].[ext]`

### Supported CoC Asset Types in `010_in/`

| Filename Example | Type | Behavior & Stabilization Pipeline | Default Duration |
| :--- | :--- | :--- | :--- |
| **`01_timelapse.mp4`** | `timelapse` | Native 1:1 speed, subpixel solar limb convex-hull stabilization and white balance equalization. | Native clip duration |
| **`02_video_slowdown_10.mp4`** | `video_slowdown` | High-speed intervalometer / burst video. Automatically filters black/corrupt frames, resamples to `duration` (e.g. 10s), smooths camera exposure jumps, and stabilizes solar limb. | 10.0s |
| **`03_video_realtime.mp4`** | `video_realtime` | Continuous 1x real-time video (30 fps) with lunar silhouette, corona, and Baily's beads tracking. | Native clip duration |
| **`04_timelapse.mp4`** | `timelapse` | Native 1:1 speed, solar limb stabilization. | Native clip duration |
| **`05_photo_6.jpg`** | `photo` | Visual still image, scaled preserving aspect ratio with clean black letterbox padding for specified duration (e.g. 6s). No optical tracking. | 10.0s (or `_6` for 6s) |
| **`06_composite_sinusoid_10`** | `composite` | Generates high-resolution composite artwork mosaic on-the-fly, scaled with black padding, held for specified duration (e.g. 10s). | 10.0s |

### Dynamic Resolution Detection
The master project resolution $(W, H)$ and optical center $(W/2, H/2)$ are automatically extracted from the first video/timelapse or photo asset found in `010_in/` (e.g., $1280\times720\text{ px}$, $1920\times1080\text{ px}$, $3840\times2160\text{ px}$, etc.).

---

## 🔬 Algorithmic Pipeline & Features

```
010_in/                                                                    040_out/
┌────────────────────────┐                                        ┌────────────────────────┐
│ 01_timelapse.mp4       │ ───► [Solar Limb Stabilization] ─────► │ 01_timelapse.mp4       │
│ (Ingress Timelapse)    │      (Subpixel Convex Hull)            │ (8.90s @ 30 fps)       │
├────────────────────────┤                                        ├────────────────────────┤
│ 02_video_slowdown_10   │ ───► [Black Frame Filter +          ─► │ 02_video_slowdown_10   │
│ (C2 Approach Burst)    │      10s Resample + Limb Track]        │ (10.00s @ 30 fps)      │
├────────────────────────┤                                        ├────────────────────────┤
│ 03_video_realtime.mp4  │ ───► [Totality Silhouette Track     ─► │ 03_video_realtime.mp4  │
│ (Totality Video)       │      + Corona / Beads Tracking]        │ (96.93s @ 30 fps)      │
├────────────────────────┤                                        ├────────────────────────┤
│ 04_timelapse.mp4       │ ───► [Adaptive Solar Limb Track     ─► │ 04_timelapse.mp4       │
│ (Egress Timelapse)     │      + Color Balance Eq]               │ (8.20s @ 30 fps)       │
├────────────────────────┤                                        ├────────────────────────┤
│ 05_photo_6.jpg         │ ───► [Scale & Pad to Canvas]        ─► │ 05_photo_6.mp4         │
│ (Still Photograph)     │      (Preserve AR + Black Border)      │ (6.00s @ 30 fps)       │
├────────────────────────┤                                        ├────────────────────────┤
│ 06_composite_sinusoid  │ ───► [On-The-Fly Artwork Render]    ─► │ 06_composite_sinusoid  │
│ (Mosaic Directive)     │      (UHD Mosaic + Pad to Canvas)      │ (10.00s @ 30 fps)      │
└────────────────────────┘                                        └───────────┬────────────┘
                                                                              │
                                                                    [Master Assembly]
                                                             (Cinematic Transitions: Freeze,
                                                             Dip to Black / Hard Cut / Crossfade)
                                                                              │
                                                                              ▼
                                                                  ┌────────────────────────┐
                                                                  │ full_eclipse.mp4       │
                                                                  │ (155.03s @ 30 fps)     │
                                                                  └────────────────────────┘
```

### 1. Subpixel Convex Solar Limb Extraction
Standard centroid or contour tracking fails on solar eclipses because the concave lunar bite pulls the centroid away from the true solar center. The pipeline extracts exclusively the **outer convex hull points** of the Sun facing the sky, filtering out:
- The inner concave lunar arc.
- Faint residual corona flares.
- The straight chord bridging the two crescent cusps.

### 2. Dynamic Optical Centering
Every frame is optimized against the calculated solar radius using robust Huber-loss circle fitting, anchoring the geometric solar center to the auto-detected master canvas optical center $(W/2, H/2)$ with subpixel accuracy across all crescent orientations.

### 3. Automatic Black/Corrupted Frame Filtering
High-speed camera intervalometers often produce corrupted or underexposed black frames. During `video_slowdown` processing, the pipeline automatically detects and drops non-illuminated frames before uniformly resampling the sequence to the target duration (default: 10.0s).

### 4. Dynamic White Balance & Exposure Smoothing
Raw DSLR and intervalometer footage frequently experiences sudden camera auto-exposure and white balance shifts as the total solar flux drops. The pipeline incorporates automatic radiometric normalization:
- **Dynamic Chromaticity Equalization**: Automatically detects green/blue channel surges (preventing the sun from turning whitish/washed out) and rescales them to match the physical baseline warmth of the solar filter ($\text{R/B} \approx 1.97$, $\text{G/R} \approx 0.62$).
- **Temporal Luminance Smoothing**: Tracks the 90th percentile intensity of the solar photosphere across all frames and filters out sudden camera exposure dips/jumps via robust median and moving average filters.

### 5. Still Photo & Composite Video Generation (No Optical Tracking)
Still photos (`photo`) and composite mosaics (`composite`) do not undergo optical limb tracking:
- Arbitrary aspect ratios and orientations are preserved without distortion.
- Clean letterbox/pillarbox padding with pure black background (`(0, 0, 0)`) seamlessly aligns with the video aspect ratio.

### 6. Configurable Cinematic Transitions
The assembly supports three distinct transition modes:
- **`fade_to_black` (Default)**:
  1. Freeze on last frame of clip A (`freeze_before`, default `1.0s`).
  2. Smooth fade out to black (`fade_out`, default `0.5s`).
  3. Optional pause in pure black (`black_duration`, default `0.0s`).
  4. Smooth fade in from black to clip B (`fade_in`, default `0.5s`).
  5. Freeze on first frame of clip B (`freeze_after`, default `1.0s`).
- **`hard`**:
  Freeze on clip A $\to$ Instant cut $\to$ Freeze on clip B.
- **`crossfade`**:
  Freeze on clip A $\to$ Smooth cross-dissolve $\to$ Freeze on clip B.
- **Single-Clip Mode**:
  If only a single asset is provided in `010_in/`, it is exported directly without artificial transition padding.

---

## 📦 Installation & Requirements

### Prerequisites
- **Python 3.10+**
- **FFmpeg** with `libx264` and `libx265` HEVC encoder support.

### Setup Virtual Environment
```bash
# Clone and navigate to the repository
git clone https://github.com/nandoide/eclipse_assembler.git
cd eclipse_assembler

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install numpy opencv-python scipy tqdm
```

---

## 🚀 Running the Pipeline

### 1. Default Run (Full Automatic Build)
Discovers all assets in `010_in/`, auto-detects resolution, processes all clips, and generates `040_out/full_eclipse.mp4` with `fade_to_black` transitions:
```bash
python 020_src/build_full_eclipse.py
```

### 2. Customizing Transition Parameters
```bash
# Dip to black with custom timings:
python 020_src/build_full_eclipse.py \
  --transition-type fade_to_black \
  --freeze-before 1.0 \
  --fade-out 0.8 \
  --black-duration 0.2 \
  --fade-in 0.8 \
  --freeze-after 1.0

# Hard cut transitions with 0.5s freezes:
python 020_src/build_full_eclipse.py \
  --transition-type hard \
  --freeze-before 0.5 \
  --freeze-after 0.5

# Crossfade dissolve transitions:
python 020_src/build_full_eclipse.py \
  --transition-type crossfade \
  --fade-duration 1.0
```

### 3. Generating High-Resolution Composite Mosaics (UHD 3840x3840)
To create ultra-high-resolution photographic composite artwork displaying the full chronological progression of the eclipse:
```bash
# Generate all 5 layouts at once (sinusoid, circle, diagonal, horizontal, arc):
python 020_src/create_eclipse_composite.py --layout all

# 1. Sinusoidal S-Curve (default: 3840x3840 UHD squared with totality at center):
python 020_src/create_eclipse_composite.py --layout sinusoid

# 2. Vertical Mobile Format 9:16 (2160x3840 px for smartphones):
python 020_src/create_eclipse_composite.py --layout vertical
python 020_src/create_eclipse_composite.py --layout vertical-s

# 3. Circular wreath progression:
python 020_src/create_eclipse_composite.py --layout circle

# 4. Diagonal progression from bottom-left to top-right:
python 020_src/create_eclipse_composite.py --layout diagonal

# 5. Laser-aligned horizontal progression:
python 020_src/create_eclipse_composite.py --layout horizontal

# 6. Celestial parabolic arc spanning full canvas height:
python 020_src/create_eclipse_composite.py --layout arc

# Custom canvas size or 16:9 desktop widescreen (e.g. 3840x2160):
python 020_src/create_eclipse_composite.py --layout sinusoid --width 3840 --height 2160
```

### 4. Generating a Lightweight 30-Second Compact Video (for Easy Sharing)
To generate an accelerated, universally compatible 30-second version in H.264 (~1.19 MB in 720p HD) for quick distribution via messaging apps (WhatsApp, Telegram), email, or social media:
```bash
python 020_src/create_compact_clip.py
```

### 5. Re-processing from Scratch
If input source videos are modified, force a full re-stabilization pass:
```bash
python 020_src/build_full_eclipse.py --force-all
```

---

## 📂 Project Directory Structure

```
eclipse_assembler/
├── 010_in/                           # Source input assets (CoC naming)
│   ├── 01_timelapse.mp4              # Ingress time-lapse
│   ├── 02_video_slowdown_10.mp4      # Pre-totality burst video (10s resample)
│   ├── 03_video_realtime.mp4         # Totality real-time continuous video
│   ├── 04_timelapse.mp4              # Egress time-lapse
│   ├── 05_photo_6.jpg                # Still photograph (6s hold)
│   └── 06_composite_sinusoid_10      # On-the-fly composite mosaic (10s hold)
│
├── 020_src/                          # Python source code
│   ├── build_full_eclipse.py         # ★ CoC Master pipeline and assembly script
│   ├── create_compact_clip.py        # 30s accelerated lightweight generator
│   ├── create_eclipse_composite.py   # ★ High-resolution UHD composite mosaic generator
│   ├── stabilize_eclipse.py          # Core solar limb subpixel stabilizer
│   └── stabilize_full_film.py        # Global 2-pass master film stabilizer
│
├── 040_out/                          # Generated stabilized output videos and artwork
│   ├── 01_timelapse.mp4              # Stabilized partial ingress (8.90s)
│   ├── 02_video_slowdown_10.mp4      # Filtered & stabilized pre-totality (10.00s)
│   ├── 03_video_realtime.mp4         # Stabilized totality (96.93s)
│   ├── 04_timelapse.mp4              # Stabilized partial egress (8.20s)
│   ├── 05_photo_6.mp4                # Letterboxed photo video (6.00s)
│   ├── 06_composite_sinusoid_10.mp4  # Letterboxed composite video (10.00s)
│   ├── full_eclipse.mp4              # ★ Complete master film (155.03s / 13.34 MB)
│   ├── full_eclipse_30s.mp4          # ⚡ Compact 30s accelerated film
│   └── composite_artwork_*.png       # 🖼️ High-resolution composite artwork
│
└── README.md                         # Project documentation
```

---

## 📊 Summary of Master Output Assets

| Output Asset | Resolution | Frame Rate / Type | Duration / Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`01_timelapse.mp4`** | 1280x720 | 30.0 fps | 8.90s (1.31 MB) | Stabilized partial ingress ($R = 238.5\text{ px}$) |
| **`02_video_slowdown_10.mp4`** | 1280x720 | 30.0 fps | 10.00s (0.69 MB) | Filtered & stabilized pre-totality (10.0s) |
| **`03_video_realtime.mp4`** | 1280x720 | 30.0 fps | 96.93s (13.19 MB) | Totality & corona with lunar silhouette tracking |
| **`04_timelapse.mp4`** | 1280x720 | 30.0 fps | 8.20s (2.66 MB) | Stabilized partial egress with solar limb tracking |
| **`05_photo_6.mp4`** | 1280x720 | 30.0 fps | 6.00s (0.12 MB) | Still photo scaled with black letterboxing |
| **`06_composite_sinusoid_10.mp4`** | 1280x720 | 30.0 fps | 10.00s (0.05 MB) | On-the-fly UHD sinusoidal mosaic clip |
| **`full_eclipse.mp4`** | **1280x720** | **30.0 fps** | **155.03s (13.34 MB)** | 🎬 **Master film with cinematic `fade_to_black` transitions** |

---

## 📄 License & Attribution
Developed for high-precision astronomical astrophotography and eclipse video processing.

- **Project Lead & Domain Guidance**: Fernando ([@nandoide](https://github.com/nandoide))
- **AI Coding Agent & Computer Vision Architecture**: Antigravity (Google Gemini 3.6 Flash High)

Code released under the MIT License.
