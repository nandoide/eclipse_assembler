# Solar Eclipse 2026: Cinematic Processing & Stabilization Pipeline

An automated, high-precision computer vision pipeline designed to stabilize, accelerate, align, and assemble multi-phase solar eclipse footage into a single, perfectly centered cinematic master film (`full_eclipse.mp4`), as well as an accelerated compact version (`full_eclipse_30s.mp4`) optimized for instant sharing.

---

## 🌒 Overview & Astronomical Context

Recording a total solar eclipse in the field involves distinct capture techniques across different eclipse phases:
1. **Partial Ingress (C1 $\to$ C2)**: Captured as time-lapses over tens of minutes.
2. **Pre-totality Approach**: High frame-rate intervalometer or continuous footage of the rapid crescent thin-down leading to the Diamond Ring and Second Contact ($C_2$).
3. **Totality ($C_2 \to C_3$)**: Continuous real-time video showing Baily's Beads, Prominences, and the Solar Corona.
4. **Partial Egress ($C_3 \to C_4$)**: Captured as time-lapses as the Sun emerges from behind the Moon.

Due to telescope mount oscillations, atmospheric seeing, wind jitter, and tracking drift, raw footage is typically jittery and misaligned across transitions. This pipeline applies **subpixel convex limb segmentation**, **fixed-radius non-linear optimization**, **black frame filtering**, **transition boundary center inheritance**, and **cinematic transitions** to create a seamless master film.

---

## 🎬 Workflow & User Starting Point

### 1. Field Recordings
The user starts with **three raw video files** recorded in the field:
- `in.mp4`: Time-lapse recording of the partial eclipse ingress ($C_1 \to C_2$).
- `total.mp4`: Continuous real-time video recorded around totality (capturing pre-totality, totality, and early egress).
- `out.mp4`: Time-lapse recording of the partial eclipse egress ($C_3 \to C_4$).

### 2. Preparing the Input Directory (`010_in/`)
Before executing the pipeline, trim and place the **four required clips** into the `010_in/` folder:

| File Name | Alternative Alias | Astronomical Phase | Description |
| :--- | :--- | :--- | :--- |
| **`partial_ingress.mp4`** | `partial_in.mp4` | Partial Ingress ($C_1 \to C_2$) | Selected time-lapse segment of the initial partial eclipse ingress. |
| **`pre_totality.mp4`** | `pretotal.mp4` | Pre-totality Approach | Continuous sequence cut from `total.mp4` leading to $C_2$. Automatically accelerated to 30.0s and cleaned of black intervalometer frames. |
| **`totality.mp4`** | `total.mp4` | Totality ($C_2 \to C_3$) | Real-time continuous video of totality (Baily's beads, corona, diamond ring). |
| **`partial_egress.mp4`** | `partial_out.mp4` | Partial Egress ($C_3 \to C_4$) | Selected time-lapse segment of the final partial eclipse egress. |

---

## 🔬 Algorithmic Pipeline & Features

```
010_in/                                                                    040_out/
┌────────────────────────┐                                        ┌────────────────────────┐
│ partial_ingress.mp4    │ ───► [Solar Limb Stabilization] ─────► │ partial_ingress.mp4    │
│ (Ingress Timelapse)    │      (R = 238.5 px, Lanczos-4)         │ (8.90s @ 30 fps)       │
├────────────────────────┤                                        ├────────────────────────┤
│ pre_totality.mp4       │ ───► [Black Frame Filter +          ─► │ pre_totality.mp4       │
│ (C2 Approach)          │      30s Resample + Limb Track]        │ (30.00s @ 30 fps)      │
├────────────────────────┤                                        ├────────────────────────┤
│ totality.mp4           │ ───► [Totality Silhouette Track     ─► │ totality.mp4           │
│ (Totality Video)       │      + C2 Transition Offset]           │ (96.93s @ 30 fps)      │
├────────────────────────┤                                        ├────────────────────────┤
│ partial_egress.mp4     │ ───► [Adaptive Solar Limb Track     ─► │ partial_egress.mp4     │
│ (Egress Timelapse)     │      + C3 Transition Offset]           │ (8.17s @ 30 fps)       │
└────────────────────────┘                                        └───────────┬────────────┘
                                                                              │
                                                                    [Master Assembly]
                                                             (Cinematic Transitions: Freeze,
                                                              Dip to Black / Hard Cut / Fade)
                                                                              │
                                                                              ▼
                                                                  ┌────────────────────────┐
                                                                  │ full_eclipse.mp4       │
                                                                  │ (150.03s @ 30 fps)     │
                                                                  └───────────┬────────────┘
                                                                              │
                                                                  [Compact Resampling]
                                                                (5x Speedup, CRF 22 H.264)
                                                                              │
                                                                              ▼
                                                                  ┌────────────────────────┐
                                                                  │ full_eclipse_30s.mp4   │
                                                                  │ (30.00s @ 1.19 MB)     │
                                                                  └────────────────────────┘
```

### 1. Subpixel Convex Solar Limb Extraction
Standard centroid or contour tracking fails on solar eclipses because the concave lunar bite pulls the centroid away from the true solar center. The pipeline extracts exclusively the **outer convex hull points** of the Sun facing the sky, filtering out:
- The inner concave lunar arc.
- Faint residual corona flares.
- The straight chord bridging the two crescent cusps.

### 2. Fixed-Radius Nelder-Mead Optimization
Every frame is optimized against the known solar radius ($R \approx 237.5 - 238.5\text{ px}$) using robust Huber-loss circle fitting:
$$\min_{c_x, c_y} \sum_{i} \rho\left(\|p_i - (c_x, c_y)\| - R_{\text{solar}}\right)$$
This anchors the geometric solar center to the optical center $(640, 360)$ with subpixel accuracy across all crescent orientations.

### 3. Automatic Black/Corrupted Frame Filtering
High-speed camera intervalometers often produce corrupted or underexposed black frames. During pre-totality processing, the pipeline automatically detects and drops non-illuminated frames before uniformly resampling the sequence to the target time-lapse duration (default: 30.0 seconds).

### 4. Dynamic White Balance & Temporal Luminance Smoothing
Raw DSLR and intervalometer footage frequently experiences sudden camera auto-exposure and white balance shifts as the total solar flux drops. The pipeline incorporates automatic radiometric normalization:
- **Dynamic Chromaticity Equalization**: Automatically detects green/blue channel surges (preventing the sun from turning whitish/washed out) and rescales them to match the physical baseline warmth of the solar filter ($\text{R/B} \approx 1.97$, $\text{G/R} \approx 0.62$).
- **Temporal Luminance Smoothing**: Tracks the 90th percentile intensity of the solar photosphere across all frames and filters out sudden camera exposure dips/jumps via robust median and moving average filters, providing seamless, flicker-free brightness continuity throughout the approach to totality.

### 5. Transition Boundary Center Inheritance ($C_2$ & $C_3$)
Because totality tracking is centered on the **lunar silhouette** ($R = 246.0\text{ px}$) rather than the hidden solar center, boundary registration shifts occur at Second Contact ($C_2$) and Third Contact ($C_3$).
- **Transition 1 ($C_2$)**: Totality inherits a $(+12.5\text{ px}, +7.5\text{ px})$ offset to perfectly match the pre-totality crescent.
- **Transition 2 ($C_3$)**: Partial egress inherits a $(+35.0\text{ px}, +24.0\text{ px})$ offset to align with totality's end frame.
- **Result**: Zero jumping, zero ghosting, and razor-sharp boundary transitions.

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
Processes any missing clips in `010_in/` and generates `040_out/full_eclipse.mp4` with `fade_to_black` transitions:
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
  --fade-duration 1.0 \
  --freeze-before 0.5 \
  --freeze-after 0.5
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

#### Composite Key Features:
- **Symmetric Totality Progression**: Captures 6 distinct totality keyframes: Ingress Baily's Beads ($C_2$, frame 30), Ingress Chromosphere $H\alpha$ arc (frame 70), Inner Corona (frame 600), Grand Wide Corona Streamers (frame 1200), Egress Chromosphere arc (frame 2700), and Egress Diamond Ring ($C_3$, frame 2850).
- **Balanced Partial Cresents**: Omits the extreme almost-full solar disks to provide breathing room and crescent symmetry between ingress and egress.
- **Wide Unclipped Corona Streamers**: Extracts full $1280\times1280\text{ px}$ frame areas with smooth elliptical cosine boundary feathering and noise floor clamping, allowing luminous corona streamers to expand organically into the surrounding space without rectangular artifacts.
- **Precision Subpixel Alignment**: Employs the stabilized optical centers $(640, 360)$ directly from the video pipeline, guaranteeing laser-straight geometric alignment across all layouts.
- **Non-Overlapping Spacing**: Uses calibrated relative disk scaling (`0.88`) ensuring clean separation between adjacent disk boundaries.
- **Dual Export**: Automatically saves both lossless `.png` (master quality) and high-Q `.jpg` (sharing/printing).

### 4. Generating a Lightweight 30-Second Compact Video (for Easy Sharing)
To generate an accelerated, universally compatible 30-second version in H.264 (~1.19 MB in 720p HD) for quick distribution via messaging apps (WhatsApp, Telegram), email, or social media:
```bash
# Generate 30s accelerated H.264 version (default: 040_out/full_eclipse_30s.mp4):
python 020_src/create_compact_clip.py

# Custom duration, codec, and quality:
python 020_src/create_compact_clip.py \
  --input 040_out/full_eclipse.mp4 \
  --output 040_out/full_eclipse_30s.mp4 \
  --duration 30.0 \
  --fps 30.0 \
  --codec libx264 \
  --crf 22
```

### 5. Re-processing from Scratch
If input source videos are modified, force a full re-stabilization pass:
```bash
python 020_src/build_full_eclipse.py --force-all
```

### 6. Standalone Video Stabilization
To stabilize any individual solar eclipse video file:
```bash
python 020_src/stabilize_eclipse.py \
  --input 010_in/partial_ingress.mp4 \
  --output 040_out/partial_ingress.mp4 \
  --radius 238.5 \
  --crf 16 \
  --preset fast
```

---

## 📂 Project Directory Structure

```
eclipse_assembler/
├── 010_in/                           # Source input footage (user provided)
│   ├── partial_ingress.mp4           # Partial ingress time-lapse snippet
│   ├── pre_totality.mp4              # Pre-totality continuous approach
│   ├── totality.mp4                  # Totality real-time continuous video
│   └── partial_egress.mp4            # Partial egress time-lapse snippet
│
├── 020_src/                          # Python source code
│   ├── build_full_eclipse.py         # Master pipeline and assembly script
│   ├── create_compact_clip.py        # 30s accelerated lightweight generator
│   ├── create_eclipse_composite.py   # ★ High-resolution UHD composite mosaic generator
│   ├── stabilize_eclipse.py          # Core solar limb subpixel stabilizer
│   └── stabilize_full_film.py        # Global 2-pass master film stabilizer
│
├── 040_out/                          # Generated stabilized output videos and artwork
│   ├── partial_ingress.mp4           # Stabilized partial ingress (8.90s)
│   ├── pre_totality.mp4              # Accelerated & stabilized pre-totality (30.00s)
│   ├── totality.mp4                  # Stabilized totality (96.93s)
│   ├── partial_egress.mp4            # Stabilized partial egress (8.17s)
│   ├── full_eclipse.mp4              # ★ Complete master film (150.03s / 11.34 MB)
│   ├── full_eclipse_30s.mp4          # ⚡ Compact 30s accelerated film (30.00s / 1.19 MB)
│   ├── eclipse_composite_sinusoid_3840p.png # 🖼️ UHD S-Curve composite (3840x3840)
│   ├── eclipse_composite_vertical_2160x3840.png # 📱 UHD Vertical 9:16 linear mobile composite (2160x3840)
│   ├── eclipse_composite_vertical_s_2160x3840.png # 📱 UHD Vertical 9:16 S-curve mobile composite (2160x3840)
│   ├── eclipse_composite_circle_3840p.png   # 🖼️ UHD Circular composite mosaic (3840x3840)
│   ├── eclipse_composite_diagonal_3840p.png # 🖼️ UHD Diagonal composite progression (3840x3840)
│   ├── eclipse_composite_horizontal_3840p.png # 🖼️ UHD Laser-aligned horizontal progression (3840x3840)
│   └── eclipse_composite_arc_3840p.png      # 🖼️ UHD Full-height celestial arc composite (3840x3840)
│
└── README.md                         # Project documentation
```

---

## 📊 Summary of Master Output Assets

| Output Asset | Resolution | Frame Rate / Type | Duration / Size | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`partial_ingress.mp4`** | 1280x720 | 30.0 fps | 8.90s (1.61 MB) | Stabilized partial ingress ($R = 238.5\text{ px}$) |
| **`pre_totality.mp4`** | 1280x720 | 30.0 fps | 30.00s (1.93 MB) | Accelerated 30s pre-totality approach (0 black frames) |
| **`totality.mp4`** | 1280x720 | 30.0 fps | 96.93s (11.82 MB) | Totality & corona with $C_2$ center inheritance |
| **`partial_egress.mp4`** | 1280x720 | 30.1 fps | 8.17s (1.22 MB) | Stabilized partial egress with $C_3$ center inheritance |
| **`full_eclipse.mp4`** | **1280x720** | **30.0 fps** | **150.03s (11.34 MB)** | 🎬 **Master film with cinematic `fade_to_black` transitions** |
| **`full_eclipse_30s.mp4`** | **1280x720** | **30.0 fps** | **30.00s (1.19 MB)** | ⚡ **Accelerated 30s compact video in H.264 for universal sharing** |
| **`eclipse_composite_vertical_2160x3840.png`** | **2160x3840** | **PNG / JPG** | **0.52 MB / 0.23 MB** | 📱 **UHD Vertical 9:16 linear mobile wallpaper format** |
| **`eclipse_composite_vertical_s_2160x3840.png`** | **2160x3840** | **PNG / JPG** | **0.54 MB / 0.23 MB** | 📱 **UHD Vertical 9:16 S-curve mobile wallpaper format** |
| **`eclipse_composite_sinusoid_3840p.png`** | **3840x3840** | **PNG / JPG** | **0.57 MB / 0.33 MB** | 🖼️ **UHD S-Curve wave composite with totality at center** |
| **`eclipse_composite_circle_3840p.png`** | **3840x3840** | **PNG / JPG** | **2.34 MB / 0.60 MB** | 🖼️ **UHD Circular wreath mosaic with corona and beads** |
| **`eclipse_composite_diagonal_3840p.png`**| **3840x3840** | **PNG / JPG** | **0.91 MB / 0.38 MB** | 🖼️ **UHD Diagonal progression (bottom-left to top-right)** |
| **`eclipse_composite_horizontal_3840p.png`**| **3840x3840** | **PNG / JPG** | **0.50 MB / 0.32 MB** | 🖼️ **UHD Laser-straight horizontal midline progression** |
| **`eclipse_composite_arc_3840p.png`** | **3840x3840** | **PNG / JPG** | **0.53 MB / 0.32 MB** | 🖼️ **UHD High celestial parabolic arc spanning full canvas** |

---

## 📄 License & Attribution
Developed for high-precision astronomical astrophotography and eclipse video processing.

- **Project Lead & Domain Guidance**: Fernando ([@nandoide](https://github.com/nandoide))
- **AI Coding Agent & Computer Vision Architecture**: Antigravity (Google Gemini 3.6 Flash High)

Code released under the MIT License.
