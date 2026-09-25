# Real-Time Hand & Finger Tracker — Implementation Plan (v3)

> Revised to incorporate `claude_suggestion.md` and `gemini_suggestion.md`. **Planning stage only —
> no application code yet.** v3 adopts the last deferred item, adds HUD orientation telemetry, and
> adds a full reviewer traceability table plus the beginner study guide deliverable.

## Revision summary

### v1 → v2
| Area | v1 plan | v2 plan | Source |
|------|---------|---------|--------|
| OpenCV dep | pinned `opencv-python==4.10.0.84` | pin **`opencv-contrib-python`** (mediapipe pulls it anyway; avoids two `cv2` installs) | claude #1 |
| `mp.solutions` risk | "may deprecate in future" | **already removed in mediapipe ≥0.10.30**; pin is load-bearing + runtime assertion | claude #2 |
| Python support | "recommend 3.11" | **3.9–3.12, tested on 3.12** | claude #2 |
| Thumb logic | `x_tip` vs `x_pip` by handedness | **distance-based** (rotation/orientation invariant); x-rule kept as fallback mode | claude, gemini #1 |
| Handedness | trust `multi_handedness` | label is display-only + `--swap-handedness`; thumb no longer depends on it | gemini #2 |
| Landmark access | recompute px per method call | **single-pass cache** with norm + px + bbox per hand | gemini #2 |
| Index-tip overlay | raw pixel circle | **EMA smoothing** (α=0.65) | gemini #3 |
| HUD | `putText` on raw frame | **alpha-blended card** behind text | gemini #4 |
| Gesture output | per-frame | **5-frame majority debounce** | gemini #5 |
| Camera open | single `VideoCapture(0)` | **backend + index fallback chain** | claude #4 |
| Tests | pure helpers only | + **duck-typed parse tests** (hand_count/label/positions edge cases) | claude #3 |
| CLI | hardcoded device | **argparse flags** | claude #3 |
| Extras | — | `logging`, screenshot hotkey, GitHub Actions CI | claude |

### v2 → v3
| # | Change | Rationale |
|---|--------|-----------|
| 1 | Add **`--image PATH` headless mode** | Last deferred item from v2. Runs the whole pipeline on a still, writes `<name>_out.png`, exits — verifies and CI-smoke-tests the pipeline with **no webcam**. |
| 2 | Add **HUD orientation readout** (`Palm` / `Back` from `palm_normal_z`) | Surfaces Gemini's palm-normal math as the "diagnostic telemetry" the spec asks for, and makes the thumb rule's invariance visible while running. |
| 3 | Add **reviewer traceability table** (below) | Makes "improve using the suggestions" auditable — every point mapped to a decision and a location. |
| 4 | Add `STUDY_GUIDE.md` deliverable | Beginner-from-zero study guide with full LaTeX math. |
| 5 | Explicitly reaffirm CI / logging / screenshot in scope | Removes the v2 ambiguity in "assumptions". |

---

## Reviewer traceability — every point accounted for

**`claude_suggestion.md`**

| Point | Status | Where it lands |
|-------|--------|----------------|
| Duplicate OpenCV (`mediapipe` pulls `opencv-contrib-python`) | Adopted | `requirements.txt`: pin `opencv-contrib-python`, drop `opencv-python` |
| `mp.solutions` removed ≥0.10.30; the pin is load-bearing | Adopted | README warning + import-time `RuntimeError` guard |
| Install/test on Python 3.12, not just 3.11 | Adopted | Supported 3.9–3.12, tested on 3.12 |
| Thumb logic is orientation-fragile | Adopted | Distance rule is `fingers_state`'s default |
| Make `hand_count`/`hand_label`/`find_positions` mockable & testable | Adopted | `parse_hand_data` + duck-typed fake-`results` tests |
| Add CLI flags | Adopted | `argparse` block |
| Camera-open backend + index fallback | Adopted | `open_camera(...)` chain |
| Define the "safe result" precisely | Adopted | `[False]*5`, asserted by a test |
| Use `logging` over `print` | Adopted | module logger |
| GitHub Actions running `pytest` | Adopted | `.github/workflows/ci.yml` |
| Screenshot hotkey | Adopted | `s` → `captures/` |

**`gemini_suggestion.md`**

| Point | Status | Where it lands |
|-------|--------|----------------|
| Palm-vs-back thumb trap | Adopted | Distance rule is invariant to it; `palm_z` now shown in HUD |
| Handedness inverts after `cv2.flip` | Adopted | Label is display-only + `--swap-handedness` |
| Redundant coordinate conversion | Adopted | Single-pass `hands_data` cache (norm + px + bbox) |
| Landmark-8 jitter | Adopted | EMA, α = 0.65 |
| Illegible HUD on bright backgrounds | Adopted | Alpha-blended ROI card |
| Gesture flicker | Adopted | 5-frame majority debounce (4/5) |
| Headless verification | Adopted (v3) | `--image` mode |

**Declined:** none remaining.

---

## Context & decisions
Greenfield build of the app in `Basic Prompt.md`. The workspace contains the prompt, `PLAN.md`, the
two review docs, and `STUDY_GUIDE.md` — no repo, no code.

Decisions:
- **API:** legacy `mp.solutions.hands.Hands` (spec-faithful), which constrains the mediapipe pin.
- **Layout:** the spec wants a `hand_tracker.py` with a `HandTracker` class; we keep that, but move the
  pure geometry into `hand_logic.py` so it can be tested with **no cv2/mediapipe/camera**. (v1 called
  the helpers "unit-testable without cv2/MediaPipe" while they lived in a module that imports both —
  this split resolves that contradiction and enables a light CI job.)
- **Both hands together:** the loop iterates every detected hand, sums finger counts, and prints one
  HUD line per hand.
- **Robust by default:** thumb direction is orientation-independent; handedness is a label, not a
  dependency.

## Deliverable files
| File | Purpose |
|------|---------|
| `hand_logic.py` | Pure logic: landmark parsing, finger/thumb state, palm normal, bbox, EMA, gesture classify + debounce. Imports only stdlib + math (no cv2/mediapipe). |
| `hand_tracker.py` | The app: `HandTracker` class, capture loop, CLI, HUD drawing, camera fallback, `--image` mode. |
| `requirements.txt` | Corrected pins (below). |
| `README.md` | Setup, run, hotkeys, the mediapipe pin warning, troubleshooting. |
| `STUDY_GUIDE.md` | Beginner-from-zero study guide: Python crash course, ML/CV basics, the full tech stack, the 21 landmarks, coordinates, architecture, and every algorithm with LaTeX + worked examples. |
| `tests/test_hand_logic.py` | Offline tests of geometry + duck-typed parse tests. |
| `.github/workflows/ci.yml` | CI: `pytest -q` on 3.11/3.12 (no camera/mediapipe needed). |
| `.gitignore` | `__pycache__/`, `.venv/`, `*.pyc`, `captures/`. |

---

## `hand_logic.py` — pure, dependency-free

### Landmark index constants
```
WRIST, INDEX_MCP, PINKY_MCP = 0, 5, 17
THUMB_TIP, THUMB_IP = 4, 3
FINGER_PAIRS = [(8,6), (12,10), (16,14), (20,18)]   # tip vs PIP: index, middle, ring, pinky
```

### Functions
- `to_pixel_landmarks(landmarks, w, h) -> list[tuple[int,int]]`
  `[(int(lm.x*w), int(lm.y*h)) for lm in landmarks]`.
- `bbox_from_pixels(px) -> (xmin, ymin, xmax, ymax)` — for HUD/labels.
- `palm_normal_z(px) -> float`
  `Z = (x5-x0)*(y17-y0) - (y5-y0)*(x17-x0)`. Sign ⇒ palm-facing vs back-facing (orientation diagnostic).
- `orientation_from_palm_z(z) -> "Palm" | "Back"` — maps the sign to the HUD label.
- `parse_hand_data(results, w, h, swap_handedness=False) -> list[dict]`
  Duck-typed: only touches `results.multi_hand_landmarks` and `results.multi_handedness`. Returns, per hand:
  ```python
  {"label": "Left"|"Right"|"Unknown", "norm": [(x,y,z)...], "px": [(x,y)...],
   "bbox": (xmin,ymin,xmax,ymax), "palm_z": float, "orientation": "Palm"|"Back"}
  ```
  Returns `[]` for `None`/empty/absent fields — never raises. **This is what makes the
  `hand_count`/`hand_label`/`find_positions` edge cases testable with a fake results object.**
- `fingers_state(px, hand_label="Unknown", palm_z=None, thumb_mode="distance") -> list[bool]`
  Order: `[thumb, index, middle, ring, pinky]`.
  - **Index–pinky:** `px[tip].y < px[pip].y` (y grows downward ⇒ smaller = higher).
  - **Thumb — `"distance"` (default, rotation/orientation invariant):**
    `dist(px[THUMB_TIP], px[PINKY_MCP]) > dist(px[THUMB_IP], px[PINKY_MCP])`.
    Independent of handedness and of palm-vs-back, so wrist rotation cannot invert it.
  - **Thumb — `"x"` (spec's original rule, kept for reference/testing):**
    `Right` → `px[4].x < px[3].x`; `Left` → `px[4].x > px[3].x`.
  - **Safe result:** on a `None`/malformed/short (<21) landmark list returns **`[False]*5`**, never raises.
- `classify_gesture(states) -> str` — `[T,T,T,T,T]`→Open Palm, all-F→Fist, `[T,F,F,F,F]`→Thumbs Up,
  `[F,T,T,F,F]`→Peace, `[T,T,F,F,T]`→Rock On, else `—`.
- `GestureDebouncer(window=5, min_agree=4)` — `deque(maxlen=window)`; `update(g) -> str` returns `g`
  once `>= min_agree` of the window agree, otherwise the last stable label (kills flicker).
- `ema(prev, cur, alpha=0.65) -> (x, y)` — `alpha*cur + (1-alpha)*prev`; `prev=None` ⇒ `cur`.

---

## `hand_tracker.py` — the app

### `HandTracker` (spec method names preserved)
```python
class HandTracker:
    def __init__(self, max_hands=2, mode=False, det_conf=0.7, track_conf=0.6,
                 thumb_mode="distance", swap_handedness=False):
        self.hands = mp_hands.Hands(
            static_image_mode=mode, max_num_hands=max_hands,
            min_detection_confidence=det_conf, min_tracking_confidence=track_conf)
        self.hands_data = []          # single-pass cache, rebuilt each find_hands()
        self._smooth_tip = {}         # per-hand EMA state for the index tip
        self._debounce = {}           # per-hand GestureDebouncer

    def find_hands(self, frame, draw=True) -> frame:
        """BGR->RGB -> process -> draw mesh -> parse + CACHE hand_data once (norm/px/bbox/label/palm_z)."""

    def hand_count(self) -> int:      # len(self.hands_data)
    def hand_label(self, hand_index) -> str:   # cached, else "Unknown"
    def find_positions(self, frame, hand_index=0) -> list[tuple[int,int]]:
        """Returns CACHED px for the hand; `frame` kept only for signature compatibility.
           Out-of-range / no detection -> [] (no IndexError)."""
    def fingers_up(self, landmarks, hand_label) -> list[bool]:  # delegates to fingers_state
    def index_tip(self, hand_index, alpha=0.65) -> tuple[int,int] | None:  # EMA-smoothed landmark 8
    def orientation(self, hand_index) -> str:  # "Palm"/"Back" from the cached palm_z
```

### `--image` headless mode (new in v3)
`--image PATH` runs the **entire pipeline on a single still** instead of the camera:
1. `frame = cv2.imread(path)` — if `None`, log an error and exit non-zero.
2. Optionally resize to `--width`/`--height`.
3. Tracker created with `mode=True` (`static_image_mode`) for stills.
4. Same `find_hands` → per-hand logic → HUD drawing path as the live loop (single pass, no `flip`,
   since a still isn't a selfie feed — or flip only if `--swap-handedness` semantics require it).
5. Write `<stem>_out.png` next to the input and exit 0.
This gives a **webcam-free verification and CI smoke test** of the real drawing/logic path.

### Capture loop (`__main__`) — with CLI
`argparse` flags: `--camera` (int, default 0), `--width`/`--height` (default 640x480),
`--max-hands` (default 2), `--no-draw`, `--thumb-mode {distance,x}`, `--swap-handedness`,
`--static` (static_image_mode), `--image PATH`, `--log-level`.

`open_camera(index, w, h)` fallback chain:
1. Try the requested index across backends in order `[CAP_DSHOW, CAP_MSMF, CAP_ANY]` on Windows,
   `[CAP_ANY]` elsewhere; set width/height via `cap.set(...)`.
2. If still not opened, try the alternate index (1, then 0).
3. If all fail: `log.error("Could not open a webcam. Check it is connected and not in use by another app.")`
   and exit non-zero.

Loop per frame:
1. `ok, frame = cap.read()` — on failure log and break.
2. `frame = cv2.flip(frame, 1)` (mirror view).
3. `frame = tracker.find_hands(frame, draw=not args.no_draw)` — parses/caches once.
4. For each hand `i` in `range(tracker.hand_count())`: `label`, `px = find_positions(frame, i)`,
   `states = fingers_up(px, label)`, `total += sum(states)`,
   draw **EMA-smoothed index-tip circle** at `tracker.index_tip(i)`,
   `gesture = debouncer[i].update(classify_gesture(states))`;
   collect `f"{label}: {sum(states)} ({gesture}) [{tracker.orientation(i)}]"`.
5. **FPS:** `dt = time.time() - t_prev`; instantaneous `1/dt`; smooth with a running average; `t_prev = now`.
6. **HUD:** draw an **alpha-blended translucent card** (ROI -> `cv2.addWeighted(roi, 0.4, overlay, 0.6, 0)`)
   top-left, then white `cv2.putText`: `FPS`, `Fingers: {total}`, one line per hand (with orientation).
7. `cv2.imshow(...)`; key `q`/`ESC` -> break; `s` -> save `captures/frame_<ts>.png` (HUD burned in).
8. **Teardown in `finally`:** `cap.release()`, `cv2.destroyAllWindows()`.

Import-time guard:
```python
if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
    raise RuntimeError("mediapipe>=0.10.30 removed mp.solutions. Install the pinned version: "
                       "pip install -r requirements.txt  (do NOT `pip install -U mediapipe`).")
```
Suppress the `mp.solutions` deprecation warnings for a clean console.

---

## Dependency pinning (corrected) — `requirements.txt`
```
mediapipe==0.10.14
opencv-contrib-python==4.11.0.86
numpy<2
```
README must state plainly:
- **Do not `pip install -U mediapipe`.** `mp.solutions` was removed around 0.10.30; this pin is
  load-bearing and the app hard-breaks above it (the runtime guard turns that into a clear error).
- We pin `opencv-contrib-python` (a superset) instead of `opencv-python` because mediapipe depends on
  it; installing both ships two `cv2` modules and causes first-import-wins flakiness.
- `numpy<2` avoids ABI issues with the 0.10.14 wheels.
- Supported/target: **Python 3.9–3.12, tested on 3.12** (no 3.13 wheel for 0.10.14).
- Setup: `python -m venv .venv` -> activate -> `pip install -r requirements.txt`.

## Tests — `tests/test_hand_logic.py` (no camera, no mediapipe needed)
Geometry:
- all-up / all-down; index-only-up; each single finger in isolation.
- **thumb, distance mode:** extended vs folded synthetic geometry -> `True`/`False`.
- **thumb, x mode:** right-mirrored (`x4 < x3`)->`True`; left-mirrored (`x4 > x3`)->`True`; wrong side->`False`.
- **rotation invariance:** rotate the same open-hand landmark set 90 degrees -> thumb still `True` (locks
  the review's core complaint).
- `palm_normal_z` sign flips between palm- and back-facing synthetic sets; `orientation_from_palm_z` agrees.
- `ema` smoothing; `GestureDebouncer` fires only after `min_agree`; `classify_gesture` for each pose.
- **Guard:** `fingers_state(None/short list)` -> **exactly `[False]*5`**, no raise.
Parse (duck-typed fake `results`):
- `parse_hand_data`: 2 hands, 0 hands, `None`, missing `multi_handedness` -> correct dicts / `[]`.
- `hand_count` reflects parsed length; `find_positions(bad_index)` -> `[]`.

## CI — `.github/workflows/ci.yml`
`ubuntu-latest`, matrix Python `[3.11, 3.12]`, `pip install pytest` then `pytest -q`. Kept light
because `hand_logic.py` has no cv2/mediapipe import; a second job installs the full
`requirements.txt` and runs `python hand_tracker.py --image tests/fixtures/hand.png` as an import +
pipeline smoke test.

## Verification
1. `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`.
2. `pytest -q` -> all logic + parse tests pass (proves finger/thumb math and edge cases with no webcam).
3. `python hand_tracker.py --image <some_hand.png>` -> writes `<some_hand>_out.png` with the skeleton,
   tip dot, and HUD burned in. **Webcam-free pipeline check.**
4. `python hand_tracker.py` -> window opens; test with one hand then both:
   - HUD card shows live FPS (~25–30), correct total, per-hand label, gesture, and Palm/Back.
   - **Rotate the wrist and show the back of the hand** -> thumb state must **not** invert (the v2 fix).
   - Index-tip circle tracks landmark 8 with visibly less jitter.
   - If the *label* reads inverted on your camera, relaunch with `--swap-handedness` (display-only fix).
5. `--camera 1` selects a second webcam; a bogus index falls back then exits with a clear message.
6. Remove hands / cover lens -> no crash, count -> 0, FPS still updates.
7. Press `s` -> file lands in `captures/`; press `q` -> camera released, windows closed, clean exit.
8. `STUDY_GUIDE.md` renders with all LaTeX intact and agrees with this plan on landmark indices,
   thresholds (0.7/0.6), α (0.65), debounce (5/4), and the thumb rule.

## Build order
1. `requirements.txt` + `.gitignore` + venv install.
2. `hand_logic.py` (constants -> parse -> geometry -> EMA/debounce -> gesture -> orientation).
3. `tests/test_hand_logic.py` -> `pytest -q` green.
4. `hand_tracker.py`: HandTracker -> camera fallback + CLI (+ `--image`) -> HUD card -> loop/teardown.
5. `README.md` (incl. the pin warning) -> `.github/workflows/ci.yml`.
6. `STUDY_GUIDE.md` (already drafted) -> cross-check against the code once it exists.
7. Manual pass per Verification steps 3–7.

## Assumptions
- All reviewer file-fixes adopted (contrib pin, 3.12, robust thumb, `--image`).
- Extras **in scope**: screenshot hotkey, `logging`, GitHub Actions CI.
- `thumb_mode="x"` kept available so the spec's literal rule stays reachable.
