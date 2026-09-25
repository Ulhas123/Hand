# The Complete Study Guide — Real-Time Hand & Finger Tracker

A from-zero walkthrough of this project. It assumes **no machine-learning knowledge at all** and only
**very basic Python**. Read it top to bottom the first time; every section builds on the one before it.

> This guide describes the project exactly as specified in `PLAN.md`. The file names (`hand_logic.py`,
> `hand_tracker.py`), function names, thresholds, and formulas here match that plan. If the code and
> this guide ever disagree, the code is the source of truth — but they are designed to agree.

---

## 1. How to use this guide

- **Read in order.** Section 2 tells you what we're building, 3–4 give you the two missing
  backgrounds (Python, and ML/CV), 5–8 explain the tools and the data, 9 is the heart (the math),
  and 10–14 are reference material you'll come back to.
- **Don't memorize the math on the first pass.** Every formula in section 9 is followed by a
  plain-English restatement and a worked example with real numbers.
- **Keep `PLAN.md` open beside you.** It's the map; this guide is the territory.
- **You do not need a webcam to study.** Sections 1–12 and most of 13 work on paper. You only need
  hardware when you actually run the app.

---

## 2. What we're building

We are building a program that:

1. Watches your **webcam** live.
2. Finds your **hand(s)** in each video frame.
3. Figures out **where each of your fingers is** — specifically, which fingers are **extended
   (up/open)** and which are **folded (down/closed)**.
4. **Draws** this understanding back on top of the video: a skeleton over your hand, a dot on your
   index fingertip, and a small text box ("HUD") showing frames-per-second, how many fingers are
   up, which hand is which, and a guessed gesture name (like "Peace" or "Fist").

Think of it as a **mirror that understands what your hand is doing** and annotates it in real time.

Two ideas worth separating early, because the whole guide depends on them:

- **Seeing the hand** (finding the landmarks) is done by a **pre-trained machine-learning model**
  that we did *not* train. We just use it. → Sections 4 and 5.
- **Interpreting the hand** (deciding "thumb is up", "3 fingers are extended") is done by **plain
  geometry and arithmetic that we wrote**. → Section 9.

Almost all of the project's own logic is the *second* kind. That's why this is a great first project:
the hard, scary ML part is a black box you call like a function, and the interesting part is simple
math you can fully understand.

### What you'll see when it runs

A window titled **"Hand Tracker"** containing the live mirrored webcam image, with:

- A **skeleton** connecting your hand joints.
- A **colored dot** on each index fingertip.
- A **semi-transparent text card** in the top-left corner reading something like:

```
FPS: 28
Fingers: 3
Right: 2 (Peace) [Palm]
Left: 1 (Thumbs Up) [Back]
```

The trailing `[Palm]` / `[Back]` is the orientation readout: which side of the hand is facing the
camera (see §9.5).

Press **`q`** to quit, **`s`** to save a screenshot.

---

## 3. Python crash course (only what this project uses)

You said you barely know Python. This section teaches *exactly* the subset this project uses. Nothing
more.

### 3.1 Variables and basic types

A variable is a named box holding a value.

```python
fps = 28            # int (whole number)
alpha = 0.65        # float (decimal number)
label = "Right"     # str (text)
running = True      # bool (True/False)
nothing = None      # NoneType (means "no value")
```

`None` matters a lot here: it's how we say "there is no landmark" or "no previous value yet".

### 3.2 Lists and tuples

A **list** is an ordered, changeable collection. A **tuple** is an ordered, *unchangeable* one.

```python
fingers = [True, False, False, False, False]   # list
point = (320, 240)                             # tuple: an (x, y) pair
point[0]      # 320  — indexing starts at 0
len(fingers)  # 5
```

We use lists for "one value per finger" and tuples for "(x, y)" coordinates. You'll see
`list[tuple[int, int]]`, which reads as "a list of (x, y) integer pairs".

### 3.3 Dictionaries

A **dict** maps keys to values — like a labeled filing cabinet.

```python
hand = {
    "label": "Right",
    "px": [(100, 200), (120, 210)],   # list of (x, y)
    "palm_z": -3500.0
}
hand["label"]   # "Right"
```

Our per-hand cache (`hands_data`) is a list of dicts exactly like this.

### 3.4 Functions

A function is a named block of code that takes inputs and returns an output.

```python
def distance(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return (dx * dx + dy * dy) ** 0.5
```

`** 0.5` is "to the power of one half", i.e. the **square root**. The `return` line is what the
function gives back. If a function has no `return`, it returns `None` implicitly.

### 3.5 Classes and objects

A **class** bundles data and the functions that operate on it. An **object** is one concrete
instance made from a class. You'll meet `HandTracker`.

```python
class Dog:
    def __init__(self, name):     # runs when you create the object
        self.name = name          # store data on the object

    def speak(self):              # a "method"
        return self.name + " says woof"

d = Dog("Rex")     # create an object
d.speak()          # "Rex says woof"
```

- `self` means "this particular object".
- `__init__` is the **constructor** — it runs at creation time.
- Methods are just functions that get `self` automatically.

`HandTracker.__init__` creates the MediaPipe model and the caches; its methods do the work.

### 3.6 Imports

Code lives in modules; `import` brings them in.

```python
import cv2                          # OpenCV
import mediapipe as mp              # MediaPipe, nicknamed mp
from collections import deque       # just deque from collections
```

`as mp` is an alias so we can type `mp.solutions...` instead of `mediapipe.solutions...`.

### 3.7 Conditionals, loops, and f-strings

```python
if y_tip < y_pip:
    result = True
elif y_tip == y_pip:
    result = False
else:
    result = False

for i in range(3):        # i = 0, 1, 2
    print(i)

while cap.isOpened():
    ...

text = f"FPS: {fps:.0f}"  # f-string: embed values. :.0f = 0 decimal places
```

### 3.8 List comprehensions

A compact way to build a list from another.

```python
px = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
```

Read it as: "for each landmark, compute a tuple, collect them all into a list `px`".

### 3.9 Truthiness and `any`/`all`/`sum`

```python
sum([True, False, True])   # 2  (True counts as 1)
any([False, True])         # True
all([True, True])          # True
if results.multi_hand_landmarks:   # None or [] is "falsy"
    ...
```

`sum(states)` is how we count extended fingers — nice and short.

### 3.10 Exceptions and `None`-guards

```python
try:
    do_something()
except Exception as e:
    print("failed:", e)
finally:
    cap.release()          # ALWAYS runs, even on error
```

Because "no hand detected" is normal, we check for `None`/empty instead of letting the program
crash: `if self.results is None: return []`.

### 3.11 `if __name__ == "__main__":`

```python
def main():
    ...

if __name__ == "__main__":
    main()
```

This means "only run this when the file is executed directly, not when it's imported by a test."
It's why our tests can import the logic without launching the webcam.

### 3.12 Virtual environments and pip

A **virtual environment** is a private folder of installed packages for this project only, so
different projects don't fight over versions.

```
python -m venv .venv                     # create it
.venv\Scripts\activate                   # Windows — activate it
# source .venv/bin/activate              # macOS/Linux
pip install -r requirements.txt          # install exactly our pinned versions
```

`pip` is Python's package installer. `requirements.txt` is the shopping list.

---

## 4. Machine learning & computer vision, from zero

### 4.1 What "machine learning" actually means here

A **model** is a program whose behavior was learned from examples rather than written by hand.

- **Training** = showing a model millions of examples so it learns patterns. This is the expensive,
  data-hungry part.
- **Inference** = *using* a trained model on a new input to get an answer. This is what our app does.

We do **zero training**. We download a model that someone else trained on millions of hand images
and ask it questions. This is extremely common in real work — most engineers use pretrained models
rather than training their own.

### 4.2 What is computer vision?

Computer vision is teaching computers to get meaning out of images. An image to a computer is just a
**grid of numbers** (pixels). "Understanding" means turning that grid into something useful —
here, a set of coordinates for your hand joints.

### 4.3 Landmarks / keypoints

A **landmark** (also called a **keypoint**) is a specific point on an object that the model has
learned to locate. For hands, MediaPipe finds **21 landmarks per hand** — the wrist, the base of
each finger, each joint, and each fingertip.

So the model's entire output for one hand is basically: *"here are 21 (x, y, z) points."* That's it.
Everything clever we do is on top of those 21 points.

- **x, y** — where on the image, in normalized coordinates (see §7).
- **z** — depth relative to the wrist; roughly "closer/farther", useful but smaller-scale. We mostly
  use x and y.

### 4.4 Confidence

Models don't say "yes" or "no"; they say "how sure am I", a number between 0 and 1. Our app uses two
thresholds:

- `min_detection_confidence = 0.7` — to *find* a new hand, be at least 70% sure.
- `min_tracking_confidence = 0.6` — to *keep following* a hand between frames, be at least 60% sure.

Lower thresholds find more hands but make more mistakes; higher thresholds are stricter. These are
the spec's chosen values.

### 4.5 On-device vs cloud

MediaPipe runs **entirely on your machine** — no internet, no upload, nothing leaves your computer.
That's why it's fast enough for real-time video and why it's a good privacy default.

---

## 5. The tech stack (every tool used, and why)

### 5.1 Python 3.x
The language. Chosen for its huge CV/ML ecosystem and readable syntax. Target **Python 3.9–3.12**
(tested on 3.12).

### 5.2 NumPy
A library for fast numeric arrays. An image in OpenCV **is** a NumPy array of shape
`(height, width, 3)` — three colour channels per pixel. We rarely call NumPy directly; we mostly
benefit from it existing underneath OpenCV. Pinned `<2` because the MediaPipe 0.10.14 wheels are
built against NumPy 1.x.

```python
import numpy as np
a = np.zeros((2, 3, 3), dtype=np.uint8)   # a tiny 2x3 black image
print(a.shape)                            # (2, 3, 3)
```

### 5.3 OpenCV (`opencv-python` / `opencv-contrib-python`), imported as `cv2`
The workhorse of computer vision: reading cameras, colour conversion, drawing, showing windows.

**The one thing to internalize: OpenCV uses BGR, not RGB.** Most libraries (including MediaPipe)
expect RGB. So we convert. Get this wrong and colours/analysis go haywire.

```python
import cv2

cap = cv2.VideoCapture(0)        # open camera #0
ok, frame = cap.read()           # frame is a NumPy array
frame = cv2.flip(frame, 1)       # 1 = mirror horizontally
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cv2.circle(frame, (320, 240), 10, (0, 255, 255), -1)   # colour order is BGR!
cv2.putText(frame, "FPS: 28", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imshow("Hand Tracker", frame)
key = cv2.waitKey(1) & 0xFF
cap.release()
cv2.destroyAllWindows()
```

Key functions we use: `VideoCapture`, `read`, `isOpened`, `flip`, `cvtColor`, `circle`, `putText`,
`imshow`, `waitKey`, `addWeighted`, `rectangle`, `imwrite`.

> **Packaging note (important):** we pin **`opencv-contrib-python`**, not `opencv-python`. MediaPipe
> depends on the contrib build internally; installing *both* puts two different `cv2` modules on
> your machine and whichever imports first wins — a classic "works on my machine" bug.

### 5.4 MediaPipe, imported as `mp`
A library of ready-made, on-device ML pipelines. We use its **Hands** solution.

```python
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,       # False = video mode (tracks between frames, faster)
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,
)
results = hands.process(rgb_frame)
# results.multi_hand_landmarks -> list (one per hand), each with 21 landmarks
# results.multi_handedness     -> list (one per hand), each carrying "Left"/"Right"
```

`mp.solutions.drawing_utils.draw_landmarks(...)` draws the skeleton, using the connection list
`mp_hands.HAND_CONNECTIONS`.

> **Critical version note:** the `mp.solutions` API (`hands`, `drawing_utils`) was **removed in
> MediaPipe ≥ 0.10.30**. We therefore pin **`mediapipe==0.10.14`**. Do **not** run
> `pip install -U mediapipe` — the app will break. Our code also checks at startup and raises a
> clear error if the API is missing, so you get a helpful message instead of a mystery crash.

### 5.5 pytest
The test runner. Tests are functions whose name starts with `test_`; `assert` checks a claim.

```python
def test_fist_has_no_fingers_up():
    assert sum(fingers_state(FIST_LANDMARKS)) == 0
```

Run with `pytest -q`. Our tests need **no camera and no MediaPipe**, because the pure logic lives in
its own file (§8).

### 5.6 Python standard library
No install needed — ships with Python.

- **`argparse`** — command-line flags (`--camera`, `--image`, …).
- **`logging`** — structured messages instead of `print`.
- **`time`** — `time.time()` for the FPS counter.
- **`collections.deque`** — a fixed-length queue; the gesture debouncer uses `deque(maxlen=5)`.

### 5.7 GitHub Actions
A service that runs commands for you when you push code. Our workflow
(`.github/workflows/ci.yml`) runs `pytest -q` on every push across Python 3.11 and 3.12, so a
mistake gets caught automatically. It's cheap because our logic tests need nothing but Python.

---

## 6. The 21 hand landmarks

MediaPipe always returns landmarks in the **same order** for every hand. Memorizing a few indices is
the single most useful thing you can do in this project.

```
                     8   12  16  20        <- fingertips
                     |    |   |   |
                     7   11  15  19        <- DIP joints
                     |    |   |   |
                     6   10  14  18        <- PIP joints
                     |    |   |   |
               4     5    9  13  17        <- MCP knuckles
                \    |    |   |   /
                 3   |    |   |  /
                  \  |    |   | /
             2     \ |    |   |/
              \     \|    |   /
               1     \    |  /
                \     \   | /
                 \     \  |/
                  \     \|/
                   \-----0 (WRIST)
        (thumb chain 1-2-3-4 is off to one side)
```

| # | Landmark | # | Landmark | # | Landmark |
|---|----------|---|----------|---|----------|
| 0 | WRIST | 7 | INDEX_DIP | 14 | RING_PIP |
| 1 | THUMB_CMC | 8 | **INDEX_TIP** | 15 | RING_DIP |
| 2 | THUMB_MCP | 9 | MIDDLE_MCP | 16 | **RING_TIP** |
| 3 | **THUMB_IP** | 10 | MIDDLE_PIP | 17 | **PINKY_MCP** |
| 4 | **THUMB_TIP** | 11 | MIDDLE_DIP | 18 | PINKY_PIP |
| 5 | **INDEX_MCP** | 12 | **MIDDLE_TIP** | 19 | PINKY_DIP |
| 6 | INDEX_PIP | 13 | RING_MCP | 20 | **PINKY_TIP** |

**MCP / PIP / DIP** are just anatomy names: MCP is the knuckle at the base of a finger, PIP is the
middle joint, DIP is the joint nearest the tip.

The indices this project gives names to:

```python
WRIST, INDEX_MCP, PINKY_MCP = 0, 5, 17
THUMB_TIP, THUMB_IP = 4, 3
FINGER_PAIRS = [(8, 6), (12, 10), (16, 14), (20, 18)]   # (tip, pip) for index→pinky
```

---

## 7. Coordinates, explained

### 7.1 Normalized vs pixel coordinates

MediaPipe reports **normalized** coordinates in the range $[0, 1]$ relative to the image size. So a
landmark at the horizontal centre is $x = 0.5$, no matter whether the camera is 640 or 1920 pixels
wide. That's convenient for the model but useless for drawing, which needs **pixels**.

To convert, multiply by width/height and round down:

$$x_{\text{px}} = \lfloor x_{\text{norm}} \cdot W \rfloor, \qquad y_{\text{px}} = \lfloor y_{\text{norm}} \cdot H \rfloor$$

**In plain text:** pixel_x = floor(x_norm × image_width), pixel_y = floor(y_norm × image_height).

*Worked example:* $W = 640$, $H = 480$, landmark $(0.5, 0.25)$ →
$x_{\text{px}} = \lfloor 0.5 \cdot 640 \rfloor = 320$, $y_{\text{px}} = \lfloor 0.25 \cdot 480 \rfloor = 120$.

### 7.2 The origin is top-left, and y grows downward

This trips up *everyone*. In image coordinates:

- $(0, 0)$ is the **top-left** corner.
- **x increases to the right.**
- **y increases downward.** The bottom of the image has the *largest* y.

So "higher up in the image" means a **smaller** y. This is why "the fingertip is above its knuckle"
is written as:

$$y_{\text{tip}} < y_{\text{pip}}$$

*Worked example:* tip at $y = 100$, knuckle at $y = 180$. Since $100 < 180$, the tip is higher on
screen → the finger is extended.

### 7.3 The mirror flip and handedness

A webcam shows you as others see you — raise your right hand and it appears on the *left* of the
screen, which feels wrong. So we mirror the frame:

```python
frame = cv2.flip(frame, 1)   # 1 = flip around the vertical axis
```

Now the image behaves like a mirror. This also affects **handedness**: MediaPipe decides
"Left"/"Right" assuming a mirrored (selfie) image, so mirroring *before* processing gives labels
that match what you see.

Two practical notes:

1. Our **thumb logic does not depend on the handedness label** (see §9.4), so even if a particular
   camera setup mislabels the hand, the *finger count* stays correct. The label is display-only.
2. If the label still looks swapped on your machine, relaunch with `--swap-handedness`. That's a
   one-line display fix, not an algorithm problem.

> A note on the flip itself: flipping changes the *orientation* of the image, so a landmark that was
> on the left moves to the right. Everything downstream simply sees the mirrored coordinates —
> consistent, as long as you flip *before* processing, which is what we do.

---

## 8. Architecture & data flow

### 8.1 Two files, and why

| File | Contains | Imports |
|------|----------|---------|
| `hand_logic.py` | Pure geometry & decisions (finger states, thumb rule, palm normal, EMA, gesture, debounce) | Only the standard library + `math` |
| `hand_tracker.py` | Camera I/O, MediaPipe, drawing, the main loop, CLI | `cv2`, `mediapipe`, and `hand_logic` |

Why split? Because **`hand_logic.py` has no camera and no ML in it**, you can unit-test every
geometric decision by feeding it made-up landmark lists. Tests run in milliseconds, on any machine,
without hardware. That separation is the single biggest design decision in this project.

### 8.2 The pipeline, frame by frame

```
 ┌───────────────────────────────┐
 │ cv2.VideoCapture(0).read()    │   grab one frame (a NumPy BGR array)
 └───────────────┬───────────────┘
                 ▼
        cv2.flip(frame, 1)               mirror view
                 ▼
    cv2.cvtColor(..., BGR2RGB)           OpenCV BGR → MediaPipe RGB
                 ▼
      mp Hands.process(rgb)              ML model runs (inference)
                 ▼
  results.multi_hand_landmarks          21 (x,y,z) points per hand
                 ▼
   hand_logic.parse_hand_data(...)       ONE pass → cache norm + px + bbox + label + palm_z
                 ▼
  fingers_state(px, label, palm_z)       → [thumb, index, middle, ring, pinky] booleans
                 ▼
  classify_gesture(states)               → "Peace", "Fist", ...
                 ▼
  GestureDebouncer.update(g)             → stable gesture (no flicker)
                 ▼
   draw circle (EMA tip) + HUD card      → annotated frame
                 ▼
          cv2.imshow(...)                show it
                 ▼
       cv2.waitKey(1) → 'q'?             quit, else loop again
```

### 8.3 The single-pass cache

Instead of recomputing pixel coordinates every time something needs them, `find_hands()` parses the
results **once** and stores, per hand:

```python
{
  "label":   "Right",                                  # handedness label
  "norm":    [(x, y, z), ...],                         # 21 normalized triples
  "px":      [(px, py), ...],                          # 21 pixel pairs
  "bbox":    (x_min, y_min, x_max, y_max),             # bounding box
  "palm_z":  -3500.0,                                  # palm-normal sign (see §9.5)
}
```

Every later step just reads this cache — no repeated loops over 21 landmarks per frame. That matters
because it's running ~30 times a second, for up to 2 hands.

---

## 9. The algorithms (the heart of the guide)

Each subsection: the idea, the formula, a worked example, and the failure mode it avoids.

### 9.1 Is a finger extended? (index, middle, ring, pinky)

**Idea:** if a finger is extended (pointing up), its **tip** is higher in the image than its **PIP
joint**. Since y grows downward, "higher" means a smaller y.

$$y^{\text{tip}}_i < y^{\text{pip}}_i \quad \Rightarrow \quad \text{finger } i \text{ is extended}$$

**In plain text:** finger i is up if tip_y < pip_y.

We compare four pairs (from §6):

$$\text{index}: (8,6),\quad \text{middle}: (12,10),\quad \text{ring}: (16,14),\quad \text{pinky}: (20,18)$$

*Worked example:* index tip $y = 90$, index PIP $y = 165$. $90 < 165$ → index is up.

*Failure mode avoided:* we compare against the **PIP joint**, not the MCP knuckle. Using the
knuckle is too permissive — a half-curled finger can still have its tip above the knuckle.

*Known limitation:* this only holds when the hand points roughly upward. A sideways or upside-down
hand breaks it. That's acceptable here; the thumb gets the more careful treatment below because it's
the worst offender.

### 9.2 Euclidean distance

Needed by the thumb rule and the EMA examples. The straight-line distance between two points:

$$d(p, q) = \sqrt{(x_p - x_q)^2 + (y_p - y_q)^2}$$

**In plain text:** d = square root of [(xp − xq)² + (yp − yq)²].

*Worked example:* $p = (3, 4)$, $q = (0, 0)$ → $d = \sqrt{9 + 16} = \sqrt{25} = 5$.

In code (avoiding a `sqrt` when we only compare two distances, since comparing squares preserves the
ordering):

```python
def distance(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5
```

### 9.3 The thumb — the naive approach, and why it breaks

The thumb moves **sideways**, not up, so §9.1 doesn't apply. The original spec's rule compares the
thumb tip's x against the thumb IP joint's x, flipped by handedness (because the feed is mirrored):

- Right hand: $x_4 < x_3$ → extended
- Left hand: $x_4 > x_3$ → extended

**In plain text:** for a right hand on a mirrored feed, the thumb is out if the tip is to the left of
the joint; for a left hand, to the right.

*Why this breaks:* it assumes the palm faces the camera and the hand is upright. **Rotate your
wrist, or show the back of your hand, and the thumb's x-relationship inverts** — the app then
reports the opposite of the truth. It also depends entirely on the handedness label being right.

Because of that, this rule is kept as a **non-default fallback** (`--thumb-mode x`) so you can see
the difference yourself, but it is not what we run by default.

### 9.4 The thumb — the robust rule (default)

**Idea:** however the hand is rotated, an extended thumb reaches *away* from the palm, while a
folded thumb is tucked *across* the palm. So measure how far the thumb tip and the thumb joint are
from a fixed palm reference — the **pinky MCP (landmark 17)** — and see which is farther.

$$d(l_4,\ l_{17}) > d(l_3,\ l_{17}) \quad \Rightarrow \quad \text{thumb is extended}$$

**In plain text:** the thumb is out if the thumb tip is farther from the pinky knuckle than the thumb
joint is.

*Worked example* (pixel coordinates):
- $l_{17}$ (pinky MCP) $= (300, 300)$
- $l_3$ (thumb IP) $= (250, 250)$ → $d = \sqrt{50^2 + 50^2} = \sqrt{5000} \approx 70.7$
- $l_4$ (thumb tip) $= (200, 260)$ → $d = \sqrt{100^2 + 40^2} = \sqrt{11600} \approx 107.7$

Since $107.7 > 70.7$, the thumb is extended.

*Why this is better:* it's **robust to handedness and to rotation in the image plane**, so neither
wrist tilt nor the palm/back ambiguity can invert it. This directly addresses the biggest weakness
the reviewers flagged in the original design.

*What it still can't do:* it works on 2D pixel distances, so it is not fully invariant to
*out-of-plane* rotation (a hand tilted so the thumb points at the camera), and a thumb pressed flat
against the index finger in a "karate chop" can genuinely sit farther from the pinky knuckle than
its IP joint does. No 2D-only rule escapes those; fixing them properly needs the landmark `z`
values.

### 9.5 Palm normal — is it the palm or the back of the hand?

We don't need this to count fingers, but it's excellent diagnostics and it's exactly the kind of
"telemetry" the spec asks us to overlay. Take two vectors from the wrist:

$$\vec{v}_1 = l_5 - l_0 \quad (\text{wrist}\to\text{index knuckle}), \qquad \vec{v}_2 = l_{17} - l_0 \quad (\text{wrist}\to\text{pinky knuckle})$$

The 2-D "cross product" (really the determinant) gives a signed area:

$$Z = (x_5 - x_0)(y_{17} - y_0) - (y_5 - y_0)(x_{17} - x_0)$$

**In plain text:** Z = (x5 − x0)×(y17 − y0) − (y5 − y0)×(x17 − x0).

- $Z < 0$ vs $Z > 0$ tells you which way the palm plane is facing → we display **"Palm"** or **"Back"**.
- Because we mirror the frame first, the sign convention is fixed for our setup; the code documents
  which sign means which.

*Vector refresher:* a vector is just a direction with a length, written as (Δx, Δy). The determinant
above is the 2-D analogue of a cross product; its **sign tells you the turn direction** (clockwise
vs counter-clockwise), which here reveals the hand's facing.

*Worked example:* $l_0 = (320, 400)$, $l_5 = (300, 300)$, $l_{17} = (360, 310)$.

$$Z = (300-320)(310-400) - (300-400)(360-320) = (-20)(-90) - (-100)(40) = 1800 + 4000 = 5800$$

$Z > 0$ → by our convention that's one orientation, shown in the HUD.

### 9.6 Bounding box

Just the extremes of the 21 points — handy for labeling a hand in the HUD:

$$(x_{\min},\, y_{\min},\, x_{\max},\, y_{\max}), \qquad x_{\min} = \min_i x_i,\ \text{etc.}$$

**In plain text:** the smallest and largest x and y among the 21 landmarks.

### 9.7 Smoothing the index-tip dot (EMA)

Raw landmark positions jitter by a pixel or two every frame (sensor noise, tiny hand movements), so a
dot drawn at the raw tip visibly shivers. An **Exponential Moving Average** blends the new reading
with the previous smoothed value:

$$P^{(t)} = \alpha \, P^{(t)}_{\text{raw}} + (1 - \alpha) \, P^{(t-1)}_{\text{smooth}}, \qquad \alpha = 0.65$$

**In plain text:** new_smooth = 0.65 × new_raw + 0.35 × previous_smooth.

*Worked example* (x coordinate): previous smoothed $= 300$, new raw $= 340$:
$P = 0.65(340) + 0.35(300) = 221 + 105 = 326$. The dot moves toward the new position but only part
way — that's the smoothing.

*Why α matters:* larger α (closer to 1) tracks faster but jitters more; smaller α is smoother but
laggier. **0.65** is a good middle ground: visibly steady, no perceptible lag. On the very first
frame there's no previous value, so we just use the raw point.

### 9.8 Frames per second

Instantaneous FPS from the time between frames:

$$fps = \frac{1}{\Delta t}, \qquad \Delta t = t_{\text{now}} - t_{\text{prev}}$$

**In plain text:** fps = 1 ÷ seconds since the last frame.

*Worked example:* $\Delta t = 0.035$ s → $fps = 1 / 0.035 \approx 28.6$.

Because a single slow frame would make the display jump, we **smooth** it with the same EMA idea:

$$fps_{\text{smooth}} \leftarrow 0.9 \, fps_{\text{smooth}} + 0.1 \, fps_{\text{inst}}$$

**In plain text:** smoothed = 0.9 × previous smoothed + 0.1 × instantaneous.

### 9.9 The translucent HUD card

White text on a bright webcam image is unreadable. So we darken a rectangle behind the text using
**alpha blending**:

$$I_{\text{out}} = 0.4 \cdot I_{\text{roi}} + 0.6 \cdot I_{\text{overlay}}$$

**In plain text:** output pixel = 0.4 × original region + 0.6 × a solid dark card.

In OpenCV:

```python
roi = frame[y0:y1, x0:x1]
overlay = np.zeros_like(roi)                       # solid black card
blended = cv2.addWeighted(roi, 0.4, overlay, 0.6, 0)
frame[y0:y1, x0:x1] = blended
```

`ROI` means **Region Of Interest** — the rectangular slice we're modifying. Then we `cv2.putText` the
white lines on top.

### 9.10 Gestures, and why they need debouncing

A **gesture** is just a recognizable finger pattern. We map the boolean list to a name:

| States `[T,I,M,R,P]` | Gesture |
|----------------------|---------|
| `[T,T,T,T,T]` | Open Palm |
| `[F,F,F,F,F]` | Fist |
| `[T,F,F,F,F]` | Thumbs Up |
| `[F,T,T,F,F]` | Peace |
| `[T,T,F,F,T]` | Rock On |
| anything else | — |

The problem: frame-to-frame the classifier flickers between names while your hand is in between
poses. So we **debounce** with a rolling window of the last $n = 5$ readings and only report a
gesture once at least $k = 4$ agree:

$$\text{report } g \iff \sum_{i=t-n+1}^{t} \mathbb{1}[g_i = g] \ \ge\ k, \qquad n = 5,\ k = 4$$

**In plain text:** show gesture g only when 4 of the last 5 frames said g; otherwise keep showing the
last stable one.

*Worked example:* history = [Peace, Peace, Fist, Peace, Peace] → Peace appears 4 times → report Peace.
History = [Peace, Fist, Peace, Fist, Peace] → only 3 → keep the previous stable label.

---

## 10. Function reference

### `hand_logic.py` (pure — no camera, no ML)

| Function | Purpose | Inputs | Returns | Edge cases |
|----------|---------|--------|---------|------------|
| `to_pixel_landmarks(landmarks, w, h)` | Normalized → pixel | landmark list, width, height | `list[(x,y)]` | assumes 21 landmarks |
| `bbox_from_pixels(px)` | Bounding box | pixel list | `(xmin,ymin,xmax,ymax)` | — |
| `palm_normal_z(px)` | Palm vs back orientation | pixel list | `float` (sign matters) | — |
| `orientation_from_palm_z(z)` | Sign of the palm normal → HUD label | float | `"Palm"` / `"Back"` | handles `0.0` |
| `parse_hand_data(results, w, h, swap_handedness)` | One-pass cache of everything | MediaPipe results (duck-typed), size | `list[dict]` | `None`/empty/missing → `[]` |
| `fingers_state(px, label, palm_z, thumb_mode)` | Finger booleans | pixel list (+label/orientation) | `[thumb,index,middle,ring,pinky]` | malformed/short → **`[False]*5`** |
| `classify_gesture(states)` | Name a pattern | 5 booleans | `str` | unknown → `"—"` |
| `GestureDebouncer(window, min_agree)` | Kill flicker | gesture strings | `str` | holds last stable until agreement |
| `ema(prev, cur, alpha)` | Smooth a point | prev, cur, α | `(x, y)` | `prev=None` → `cur` |

### `hand_tracker.py` (the app)

| Method | Purpose | Notes |
|--------|---------|-------|
| `HandTracker.__init__(...)` | Create the MediaPipe model + caches | thresholds 0.7 / 0.6 |
| `find_hands(frame, draw=True)` | Process a frame, draw mesh, refresh the cache | the only place the model is called |
| `hand_count()` | How many hands this frame | 0 is normal |
| `hand_label(i)` | "Left"/"Right" for hand `i` | out of range → `"Unknown"` |
| `find_positions(frame, i=0)` | Cached pixel landmarks for hand `i` | out of range → `[]`; `frame` kept for signature compatibility only |
| `fingers_up(landmarks, label)` | Booleans for one hand | delegates to `fingers_state` |
| `index_tip(i, alpha=0.65)` | EMA-smoothed landmark 8 | `None` if no hand |
| `orientation(i)` | `"Palm"` / `"Back"` for hand `i` | out of range → `"—"` |
| `gesture(i, states)` | Debounced gesture for hand `i` | one debouncer per hand, keyed by label |
| `close()` | Release the MediaPipe model | called in teardown |

Module-level: `open_camera(index, w, h)` (backend + index fallback), `hud_lines(...)`,
`draw_hud(frame, lines)`, `annotate_hands(tracker, frame)`, `save_screenshot(frame)`,
`run_live(args)`, `run_image(args)`, `main()`.

---

## 11. Running & testing

### Install

```
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

`requirements.txt`:

```
mediapipe==0.10.14
opencv-contrib-python==4.11.0.86
numpy<2
```

### Run

```
python hand_tracker.py                  # default camera (0), 640x480
python hand_tracker.py --camera 1       # a second webcam
python hand_tracker.py --no-draw        # hide the skeleton, keep the HUD
python hand_tracker.py --swap-handedness
python hand_tracker.py --thumb-mode x   # the old naive thumb rule, for comparison
python hand_tracker.py --image hand.png # headless: annotate a still, save hand_out.png, exit
```

| Flag | Meaning | Default |
|------|---------|---------|
| `--camera` | camera index | `0` |
| `--width` / `--height` | capture size | `640` / `480` |
| `--max-hands` | how many hands to track | `2` |
| `--no-draw` | skip the skeleton overlay | off |
| `--thumb-mode {distance,x}` | which thumb rule | `distance` |
| `--swap-handedness` | flip the Left/Right label | off |
| `--static` | `static_image_mode` (for stills) | off |
| `--image PATH` | process one image, don't open the camera | off |
| `--log-level` | verbosity | `INFO` |

**Hotkeys:** `q` or `Esc` quit · `s` save `captures/frame_<timestamp>.png`.

### Test

```
pytest -q
```

No camera, no MediaPipe needed — that's the payoff of the split design.

---

## 12. Glossary

- **Alpha (α)** — a blending weight between 0 and 1 (used for smoothing and transparency).
- **BGR** — Blue-Green-Red channel order, used by OpenCV. The reverse of the more common RGB.
- **Bounding box (bbox)** — the smallest rectangle containing something.
- **Confidence** — a model's self-reported probability, 0–1.
- **Cross product / determinant (2-D)** — a signed number revealing the turn direction between two
  vectors; we use it to tell the palm from the back of the hand.
- **Debounce** — requiring agreement over several frames before acting, to suppress flicker.
- **DIP / PIP / MCP** — the finger joints: nearest the tip / middle / knuckle at the base.
- **EMA (Exponential Moving Average)** — a smoothed running value giving recent readings more weight.
- **FPS** — frames per second; how many images are processed per second.
- **Handedness** — whether a hand is a left or right hand.
- **HUD** — "heads-up display"; text/graphics overlaid on the video.
- **Inference** — running a trained model to get a prediction.
- **Keypoint** — see *landmark*.
- **Landmark** — a specific located point on an object; here, one of 21 hand joints.
- **Model** — a program whose behavior was learned from data.
- **ndarray** — NumPy's array type. An image is one of these.
- **Normalized coordinates** — positions as fractions of the image size, in `[0, 1]`.
- **Palm normal** — a vector perpendicular to the palm plane; its sign tells you which side faces
  the camera.
- **Pretrained model** — a model someone else already trained; we just use it.
- **ROI** — Region Of Interest; the sub-rectangle of an image you're working on.
- **Threshold** — a cutoff value (e.g. confidence ≥ 0.7).
- **Truthiness** — what Python treats as true/false (`None`, `[]`, `0` are falsy).

---

## 13. Study path & exercises

Work through these in order; each is small and builds confidence.

1. **Run it.** Install, run `python hand_tracker.py`, show one hand then two. Confirm the finger
   count matches reality. *(No code changes.)*
2. **Read `hand_logic.py` top to bottom.** You now understand every function in it (§9, §10). If a
   line is unclear, find it in this guide.
3. **Break the naive rule on purpose.** Run with `--thumb-mode x` and rotate your wrist. Watch the
   thumb reading invert. Then switch back to the default and confirm it doesn't. *You've now
   reproduced the bug the robust rule fixes — the best way to understand why it exists.*
4. **Change a threshold.** In `HandTracker.__init__`, try `min_detection_confidence=0.5`, then `0.9`.
   Describe what changes. *(File: `hand_tracker.py`.)*
5. **Tune the smoothing.** Change `alpha` in `index_tip` from `0.65` to `0.95`, then `0.2`. Which is
   jittery? Which lags? *(File: `hand_tracker.py`.)*
6. **Add a landmark readout.** Print the index fingertip's pixel coordinates in the HUD. *(Hint:
   `tracker.find_positions(frame, i)[8]`. Files: `hand_tracker.py` HUD block.)*
7. **Add a gesture.** Add a case to `classify_gesture` for the "OK" sign (thumb and index touching).
   *(File: `hand_logic.py`.)*
8. **Write a test.** Add a test that a synthetic "Peace" landmark set classifies as `"Peace"`, then
   run `pytest -q`. *(File: `tests/test_hand_logic.py`.)*
9. **Rotate a landmark set.** Take an extended-hand list, rotate every point 90° with a small
   helper, and assert the thumb is *still* extended — proving the invariance from §9.4.
10. **Try the headless mode.** `python hand_tracker.py --image some_hand.png` and inspect the saved
    `some_hand_out.png`. No webcam required.

---

## 14. Appendix

### 14.1 Math refresher

**Vectors.** A 2-D vector is a pair $(\Delta x, \Delta y)$ — a direction and magnitude. From two
points: $\vec{v} = q - p = (x_q - x_p,\ y_q - y_p)$.

**Distance.** $d(p,q) = \|\vec{v}\| = \sqrt{\Delta x^2 + \Delta y^2}$ (Pythagoras).

**Determinant (2-D cross product).** For $\vec{a} = (a_x, a_y)$ and $\vec{b} = (b_x, b_y)$:

$$a_x b_y - a_y b_x$$

The **sign** tells you whether $\vec{b}$ is clockwise or counter-clockwise from $\vec{a}$. In §9.5 we
apply this to the two wrist→knuckle vectors to detect the palm's facing.

**Weighted average.** $\alpha \cdot \text{new} + (1-\alpha) \cdot \text{old}$ is a "blend" between two
values; it appears in both the EMA smoothing and the HUD transparency.

### 14.2 Python cheatsheet used in this project

| Want | Write |
|------|-------|
| Make a list | `[a, b, c]` |
| Make a tuple | `(a, b)` |
| Build a list in a loop | `[f(x) for x in xs]` |
| Access index 8 | `px[8]` |
| Last item | `xs[-1]` |
| Length | `len(xs)` |
| Count truths | `sum(bools)` |
| Square root | `x ** 0.5` or `math.sqrt(x)` |
| Floor an int | `int(x)` (for positives) |
| Format a float, 0 places | `f"{x:.0f}"` |
| Default argument | `def f(a, b=2):` |
| Handle "no value" | `if x is None:` |
| Safe index | `xs[i] if 0 <= i < len(xs) else None` |
| Guard the entry point | `if __name__ == "__main__":` |
| Always clean up | `try: ... finally: cap.release()` |
| Fixed-length queue | `from collections import deque; q = deque(maxlen=5)` |

### 14.3 Repository map

```
Hand/
├── Basic Prompt.md          the original spec
├── PLAN.md                  the implementation plan (v3)
├── STUDY_GUIDE.md           this document
├── claude_suggestion.md     review notes
├── gemini_suggestion.md     review notes
├── hand_logic.py            pure logic (stdlib only)
├── hand_tracker.py          app: camera + MediaPipe + HUD + CLI
├── run.bat                  double-click launcher (CDs to the project, uses .venv)
├── requirements.txt         pinned dependencies
├── README.md                setup + troubleshooting
├── conftest.py              puts the project root on sys.path for pytest
├── tests/
│   ├── test_hand_logic.py   offline geometry + parse tests
│   ├── test_hand_tracker.py cache/HUD tests (skipped without cv2/mediapipe)
│   └── fixtures/hand.png    synthetic stand-in for the CI smoke test
├── .github/workflows/ci.yml CI
└── .gitignore
```

---

**You made it.** You now have the full picture: what the project is, the Python you need, what ML
and computer vision mean here, the entire tech stack, the 21 landmarks, coordinate systems, the
pipeline, and the math behind every decision — including why the robust thumb rule exists and where
the naive one fails. The code in `hand_logic.py` and `hand_tracker.py` is the realization of all of
it: read those two files top to bottom and every section above will line up with a real function.
