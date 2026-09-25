# Real-Time Hand & Finger Tracker

A webcam app that finds your hands, works out which fingers are extended, and draws the answer back
on top of the video: skeletal mesh, a smoothed dot on each index fingertip, and a telemetry HUD
showing FPS, the total number of extended fingers, per-hand gesture and palm orientation.

Built on **MediaPipe Hands** (21 3D landmarks per hand) and **OpenCV**.

```
FPS: 28
Fingers: 3
Right: 2 (Peace) [Palm]
Left: 1 (Thumbs Up) [Back]
```

## Requirements

- **Python 3.9 – 3.12** (developed and tested on **3.12**; there is no 3.13 wheel for the pinned
  MediaPipe build)
- A webcam for live mode — `--image` mode needs none

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## ⚠️ Do not upgrade MediaPipe

`requirements.txt` pins **`mediapipe==0.10.14`** because this app uses the legacy
`mp.solutions.hands` API, which was **removed** in MediaPipe 0.10.30+. The pin is *load-bearing*:
running `pip install -U mediapipe` will hard-break the app.

You won't get a confusing stack trace if you do — the app checks at import time and exits with a
clear message:

```
RuntimeError: mediapipe>=0.10.30 removed mp.solutions, which this app is built on.
Install the pinned version instead: pip install -r requirements.txt
(do NOT `pip install -U mediapipe`).
```

The other two pins are deliberate as well:

- **`opencv-contrib-python`** instead of `opencv-python` — MediaPipe depends on the contrib build
  anyway; installing both ships two `cv2` modules and causes first-import-wins flakiness.
- **`numpy<2`** — avoids ABI breaks with the 0.10.14 wheels.

## Running

Double-click **`run.bat`** for the no-terminal route: it activates the venv for you, starts the
tracker, and keeps the window open if anything fails so you can read the error. It forwards any
arguments too — `run.bat --camera 1` behaves like the command below.

Or from a terminal:

```bash
python hand_tracker.py                       # live webcam
python hand_tracker.py --camera 1            # second webcam
python hand_tracker.py --image hand.png      # no webcam: process a still, write hand_out.png
```

### Flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--camera N` | `0` | Webcam index. Falls back to the other index if this one won't open. |
| `--width` / `--height` | `640x480` live, native in `--image` | Capture size (image mode resizes only when both are given). |
| `--max-hands N` | `2` | Maximum simultaneous hands. |
| `--no-draw` | off | Hide the skeletal mesh (keeps the dot + HUD). |
| `--thumb-mode {distance,x}` | `distance` | Thumb rule — see below. |
| `--swap-handedness` | off | Swap the `Left`/`Right` labels (display-only). |
| `--static` | off | Run MediaPipe in `static_image_mode` (no temporal tracking). |
| `--image PATH` | — | Run the whole pipeline on one still image and exit. |
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | `INFO` | Logging verbosity. |

### Hotkeys (live mode)

| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit (camera released, windows closed) |
| `s` | Save a screenshot with the HUD burned in to `captures/frame_<timestamp>.png` |

## How the finger logic works

**Index, middle, ring, pinky** — a fingertip counts as extended when it sits strictly above its PIP
knuckle (`y_tip < y_pip`). Image origin is top-left, so smaller `y` means higher on screen.

**Thumb** — the default rule is a distance comparison against the pinky knuckle:

```
extended  ⇔  dist(tip, pinky_MCP) > dist(IP, pinky_MCP)
```

This is robust to handedness and to rotation in the image plane: rotate your wrist or show the back
of your hand and the answer doesn't flip. The spec's original horizontal rule
(`Right` → `x_tip < x_ip`, `Left` → `x_tip > x_ip`) is still available with `--thumb-mode x` for
comparison — it only holds for an upright, palm-facing hand.

**Gestures** are classified from the five finger flags (Open Palm, Fist, Thumbs Up, Peace, Rock On)
and passed through a 5-frame majority debouncer so a single noisy frame can't flicker the label.

**Palm orientation** (`Palm` / `Back`) comes from the z-component of the palm-plane normal, computed
from the wrist, index-MCP and pinky-MCP landmarks. It assumes a mirrored (selfie) view, so in
`--image` mode, which is not mirrored, the label may read inverted.

## Project layout

| File | Purpose |
|------|---------|
| `hand_logic.py` | Pure logic — landmark parsing, finger/thumb rules, palm normal, EMA, gesture classification and debouncing. Standard library only: **no cv2, no MediaPipe**. |
| `hand_tracker.py` | The app — `HandTracker`, capture loop, CLI, HUD, camera fallback, `--image` mode. |
| `tests/test_hand_logic.py` | Offline geometry tests + duck-typed parse tests. No camera needed. |
| `tests/test_hand_tracker.py` | Cache/HUD tests. Skips itself when cv2/mediapipe aren't installed. |
| `tests/fixtures/hand.png` | Synthetic stand-in used by the CI pipeline smoke test. |
| `.github/workflows/ci.yml` | CI: logic tests on 3.11/3.12 + a full-dependency pipeline smoke test. |

## Tests

```bash
pip install pytest
pytest -q
```

The `hand_logic` suite needs no camera, no OpenCV and no MediaPipe, so it runs anywhere Python does.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Could not open a webcam.` | Close other apps using the camera (Zoom/Teams/browser tabs), then try `--camera 1`. |
| `ImportError: libGL.so.1` on Linux | `sudo apt-get install -y libgl1 libglib2.0-0` (non-headless OpenCV wheels need it). |
| `RuntimeError: mediapipe>=0.10.30 removed mp.solutions` | You upgraded MediaPipe. `pip install -r requirements.txt`. |
| Handedness labels look inverted | Expected on a mirrored feed — MediaPipe assumes a flipped selfie input. The label is display-only; relaunch with `--swap-handedness`. |
| Thumb never detects in `--thumb-mode x` | That rule assumes an upright, palm-facing hand. Use the default `distance` mode. |
| FPS is low | Lower `--width`/`--height`, or pass `--no-draw` to skip the mesh. |

## Known limitations

- Tucked "karate-chop" thumbs can read as extended: the distance rule compares the thumb to the
  pinky knuckle only, and an adducted thumb sitting against the index finger is genuinely farther
  from that knuckle than its IP joint is.
- All rules are 2D pixel heuristics. They handle handedness and in-plane rotation, but not extreme
  out-of-plane rotation (a thumb pointing at the camera).
- Per-hand smoothing and gesture history are keyed by handedness label, which is steadier than
  MediaPipe's list order but can still swap if the label flickers.
