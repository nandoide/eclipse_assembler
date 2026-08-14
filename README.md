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

### 4. Transition Boundary Center Inheritance ($C_2$ & $C_3$)
Because totality tracking is centered on the **lunar silhouette** ($R = 246.0\text{ px}$) rather than the hidden solar center, boundary registration shifts occur at Second Contact ($C_2$) and Third Contact ($C_3$).
- **Transition 1 ($C_2$)**: Totality inherits a $(+12.5\text{ px}, +7.5\text{ px})$ offset to perfectly match the pre-totality crescent.
- **Transition 2 ($C_3$)**: Partial egress inherits a $(+35.0\text{ px}, +24.0\text{ px})$ offset to align with totality's end frame.
- **Result**: Zero jumping, zero ghosting, and razor-sharp boundary transitions.

### 5. Configurable Cinematic Transitions
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

### 3. Generating a Lightweight 30-Second Compact Video (for Easy Sharing)
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

### 4. Re-processing from Scratch
If input source videos are modified, force a full re-stabilization pass:
```bash
python 020_src/build_full_eclipse.py --force-all
```

### 5. Standalone Video Stabilization
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
│   ├── stabilize_eclipse.py          # Core solar limb subpixel stabilizer
│   └── stabilize_full_film.py        # Global 2-pass master film stabilizer
│
├── 040_out/                          # Generated stabilized output videos
│   ├── partial_ingress.mp4           # Stabilized partial ingress (8.90s)
│   ├── pre_totality.mp4              # Accelerated & stabilized pre-totality (30.00s)
│   ├── totality.mp4                  # Stabilized totality (96.93s)
│   ├── partial_egress.mp4            # Stabilized partial egress (8.17s)
│   ├── full_eclipse.mp4              # ★ Complete master film (150.03s / 11.34 MB)
│   └── full_eclipse_30s.mp4          # ⚡ Compact 30s accelerated film (30.00s / 1.19 MB)
│
└── README.md                         # Project documentation
```

---

## 📊 Summary of Master Video Outputs

| Output Video | Resolution | Frame Rate | Frames | Duration | File Size | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`partial_ingress.mp4`** | 1280x720 | 30.0 fps | 267 | 8.90s | 1.61 MB | Stabilized partial ingress ($R = 238.5\text{ px}$) |
| **`pre_totality.mp4`** | 1280x720 | 30.0 fps | 900 | 30.00s | 1.93 MB | Accelerated 30s pre-totality approach (0 black frames) |
| **`totality.mp4`** | 1280x720 | 30.0 fps | 2,908 | 96.93s | 11.82 MB | Totality & corona with $C_2$ center inheritance |
| **`partial_egress.mp4`** | 1280x720 | 30.1 fps | 246 | 8.17s | 1.22 MB | Stabilized partial egress with $C_3$ center inheritance |
| **`full_eclipse.mp4`** | **1280x720** | **30.0 fps** | **4,501** | **150.03s** | **11.34 MB** | 🎬 **Master film with cinematic `fade_to_black` transitions** |
| **`full_eclipse_30s.mp4`** | **1280x720** | **30.0 fps** | **900** | **30.00s** | **1.19 MB** | ⚡ **Accelerated 30s compact video in H.264 for universal sharing** |

---

## 📄 License & Attribution
Developed for high-precision astronomical astrophotography and eclipse video processing.

- **Project Lead & Domain Guidance**: Fernando ([@nandoide](https://github.com/nandoide))
- **AI Coding Agent & Computer Vision Architecture**: Antigravity (Google Gemini 3.6 Flash High)

Code released under the MIT License.
