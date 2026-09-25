I read through the plan and actually test-drove the dependency pins in a sandbox venv (Python 3.12) rather than just eyeballing them. Overall this is a well-thought-out plan — clean separation of pure logic vs. I/O for testability, sensible confidence defaults, proper teardown handling. But I found a few real issues worth fixing and some gaps worth filling.

## Verified in sandbox — two concrete fixes needed

**1. `opencv-python` + mediapipe pulls in a duplicate OpenCV.**
When I dry-ran `pip install mediapipe==0.10.14 opencv-python==4.10.0.84 "numpy<2"`, pip resolved a plan that installs **both** `opencv-python-4.10.0.84` *and* `opencv-contrib-python-4.11.0.86`, because mediapipe depends on the latter internally. Both packages ship a `cv2` module under the hood — having both installed is a known source of "works on my machine" flakiness (whichever gets imported first wins, and upgrades to one can silently break the other). **Fix:** drop the explicit `opencv-python` pin and either let mediapipe pull its own OpenCV, or pin `opencv-contrib-python` instead (it's a superset, so nothing is lost).

**2. The "mediapipe may deprecate `mp.solutions`" note undersells the actual risk — it already happened.**
I checked and `mp.solutions` isn't just at risk of future deprecation — it's already been **removed entirely** starting around mediapipe 0.10.30+ (confirmed via multiple open GitHub issues where `mediapipe.solutions` throws `AttributeError` on 0.10.31, with maintainers confirming the legacy solutions are deprecated and unmaintained). So `0.10.14` isn't a conservative choice, it's a load-bearing one — the app will hard-break if anyone bumps that pin casually. Worth calling this out explicitly in the README (not just "newest releases *may* drop it") and maybe adding a runtime assertion or comment warning against `pip install -U mediapipe`. Also: 0.10.14 has no Python 3.13 wheel, but it does install cleanly on 3.12, so the README could safely say "3.9–3.12, tested on 3.12" rather than steering everyone to 3.11.

## Design gaps worth addressing

- **Thumb-open logic is orientation-fragile.** The `x_tip` vs `x_pip` horizontal comparison only holds when the palm faces the camera and the hand is roughly upright. Tilt the hand, rotate the wrist, or show the back of the hand, and it flips. A more robust (if slightly more work) approach: compare the thumb tip's distance from the pinky MCP against the thumb IP's distance from the same point — this stays roughly correct across rotation, not just for one canonical pose. Given the plan already flags this as "the one central place to change" if it misbehaves live, it might be worth building the more robust version from the start rather than patching after manual testing surfaces the problem.

- **Test coverage stops at pure logic, but `hand_label()`, `find_positions()`, and `hand_count()` are also mockable.** You don't need a real camera to test these — a fake object standing in for `self.results` (with a `.multi_hand_landmarks` / `.multi_handedness` list) would let you unit-test the out-of-range and "no detection" branches too, which are exactly the kind of edge cases that cause crashes in the wild (e.g., a hand disappearing mid-loop).

- **No CLI flags.** Hardcoding `cv2.VideoCapture(0)` means anyone with a second webcam or an external cam on a different index is stuck editing source. A few `argparse` flags (`--camera`, `--width/--height`, `--no-draw`) are cheap and make this genuinely usable rather than a fixed demo.

- **Camera-open failure handling is single-shot.** On Windows the plan sets `CAP_DSHOW`; worth also falling back to `CAP_MSMF` or default backend if `CAP_DSHOW` fails, and on Linux/Mac trying index 0 then 1 before giving up, since "wrong camera index" is one of the most common real-world failure modes for this kind of script.

## Smaller polish

- Define what "safe result" means in the guard test precisely (e.g., `fingers_state` on a malformed list returns `[False]*5` rather than raising) — right now it's spec'd loosely enough that two implementers could disagree.
- `print()` for the webcam-failure message is fine for a script this size, but if you want it to double as a library import later, swap to `logging`.
- A GitHub Actions workflow running `pytest -q` is nearly free here since the tests need zero camera/GPU access — worth adding given how testable this design already is.
- Optional bonus: a screenshot hotkey (`s` → save current frame with HUD burned in) is a natural, low-cost addition given the gesture-classification bonus is already in scope.

Want me to update `PLAN.md` with these changes, or go ahead and start building `hand_tracker.py` with the fixes (robust thumb check, corrected requirements.txt, CLI flags) folded in?

This v2 is a real upgrade — nearly every point from the last review was incorporated correctly, and the exact `opencv-contrib-python==4.11.0.86` pin matches what I got when I actually resolved the dependency tree in a sandbox, so that fix is verified, not just assumed. A few things still worth tightening before you start building.

## What's solid
- **`opencv-contrib-python` pin** — confirmed correct version.
- **Legacy API guard + explicit "don't upgrade mediapipe" warning** — good, and correctly framed as load-bearing rather than a soft caveat.
- **`hand_logic.py` / `hand_tracker.py` split** — this actually fixes a real contradiction in v1 (helpers described as "no cv2/mediapipe needed" while living in a file that imported both).
- **Rotation-invariance test that encodes the exact complaint** — turning review feedback directly into a regression test is exactly the right move.

## Two remaining design risks

**1. Per-hand state (`_smooth_tip`, `_debounce`) is keyed by list index, but hand order isn't stable.** MediaPipe's `multi_hand_landmarks` list order can change frame-to-frame — there's no persistent track ID, especially when hands cross, one drops out and re-enters, or detection order just shuffles. If `_smooth_tip[0]` and `_debounce[0]` mean "whatever hand mediapipe put first this frame," then on a reorder the EMA will blend two different physical hands' positions for one frame (visible as a jump — the opposite of what EMA is for), and the gesture debouncer's 5-frame window will get contaminated with a different hand's history, briefly showing a stale/wrong gesture. Since you're already treating `hand_label` as a stable-ish display value (with `--swap-handedness` as an escape hatch), it makes sense to key these two caches by label instead of raw index — it's not perfect either (handedness itself can flicker) but it's meaningfully more stable than index and costs nothing extra to implement.

**2. "Orientation-independent" is slightly overclaimed for the thumb heuristic.** The distance-based rule (`dist(tip, pinky_mcp) > dist(ip, pinky_mcp)`) is genuinely robust to *handedness* and to *in-image-plane* rotation — which is exactly what your new test checks (rotating the synthetic landmark set 90° in 2D) and exactly what broke the old rule. But it's still working off 2D pixel positions, so it isn't fully invariant to *out-of-plane* rotation (e.g., the hand tilted so the thumb points toward/away from the camera) — no 2D-only heuristic can be, that needs the landmark z-values or real 3D reasoning, which is out of scope here. Not a bug, just worth softening the claim in the README/comments from "orientation-independent" to something like "robust to handedness and image-plane rotation" so nobody's surprised by an edge case later and files it as a regression.

## Smaller things

- **`GestureDebouncer` needs a defined warm-up behavior.** What does `update()` return before the 5-frame window fills (e.g., on frame 1)? Right now it's ambiguous whether it returns the raw label, `"—"`, or `None`. Pick one and add a test for it — it's the kind of gap that turns into a `KeyError` or a flickering label on startup if left implicit.
- **CI import-smoke job risk: `libGL.so.1`.** If the optional job installs the full `requirements.txt` and does `import cv2` on `ubuntu-latest`, non-headless OpenCV wheels (which `opencv-contrib-python` is) sometimes fail with `ImportError: libGL.so.1: cannot open shared object file` on minimal Linux environments — it's one of the most commonly hit OpenCV/CI gotchas. GitHub's hosted Ubuntu runners are fairly fully-loaded so this may just work, but it's cheap insurance to add `sudo apt-get install -y libgl1` as a step before that job's `pip install`, so a CI failure there doesn't get mistaken for a real code bug.
- **`find_positions(frame, hand_index)` docstring should note the cache dependency explicitly** — since `frame` is now vestigial (kept for signature compatibility), it's worth one line noting the cached px is only valid for the frame size passed to the preceding `find_hands()` call, in case anyone reuses the method against a differently-sized frame later.

None of these are blockers — they're the kind of thing that's much cheaper to bake into the design/tests now than to debug after they show up as an intermittent glitch during the live webcam pass. Want me to fold these into the plan doc, or go straight to building `hand_logic.py` first (it's the piece with no external dependencies, so it's the natural place to start per your build order)?