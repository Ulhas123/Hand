"""Pure hand geometry, landmark parsing and gesture classification.

This module deliberately imports nothing but the standard library (``math``,
``collections``). That keeps every rule below unit-testable with no camera, no
OpenCV and no MediaPipe, and lets CI run the logic suite in seconds.

All pixel coordinates are ``(x, y)`` with the origin at the **top-left** of the
image, so ``y`` grows *downward*: a smaller ``y`` means "higher on screen".
"""

from __future__ import annotations

import math
from collections import deque

WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17
THUMB_IP = 3
THUMB_TIP = 4
INDEX_TIP = 8

FINGER_PAIRS = [(8, 6), (12, 10), (16, 14), (20, 18)]

EXPECTED_LANDMARKS = 21

THUMB_DISTANCE = "distance"
THUMB_X = "x"

NO_GESTURE = "—"


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def to_pixel_landmarks(landmarks, width, height):
    """Convert normalized landmarks to integer pixel coordinates."""
    return [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]


def bbox_from_pixels(pixel_landmarks):
    """Bounding box ``(xmin, ymin, xmax, ymax)`` of a pixel landmark list."""
    xs = [point[0] for point in pixel_landmarks]
    ys = [point[1] for point in pixel_landmarks]
    return (min(xs), min(ys), max(xs), max(ys))


def palm_normal_z(pixel_landmarks):
    """Z component of the palm-plane normal, from wrist/index-MCP/pinky-MCP.

    ``Z = (x5 - x0)(y17 - y0) - (y5 - y0)(x17 - x0)``

    The sign says which side of the hand faces the camera, which is a useful
    orientation diagnostic. The convention assumes a mirrored (selfie) view:
    ``Z < 0`` means the palm faces the camera, ``Z > 0`` the back of the hand.
    """
    x0, y0 = pixel_landmarks[WRIST][0], pixel_landmarks[WRIST][1]
    x5, y5 = pixel_landmarks[INDEX_MCP][0], pixel_landmarks[INDEX_MCP][1]
    x17, y17 = pixel_landmarks[PINKY_MCP][0], pixel_landmarks[PINKY_MCP][1]
    return float((x5 - x0) * (y17 - y0) - (y5 - y0) * (x17 - x0))


def orientation_from_palm_z(palm_z):
    """Map the palm-normal sign to the HUD label ``"Palm"`` or ``"Back"``."""
    return "Palm" if palm_z < 0 else "Back"


def _handedness_label(handedness, swap_handedness):
    try:
        label = handedness.classification[0].label
    except (AttributeError, IndexError, TypeError):
        return "Unknown"
    if label not in ("Left", "Right"):
        return "Unknown"
    if swap_handedness:
        return "Left" if label == "Right" else "Right"
    return label


def parse_hand_data(results, width, height, swap_handedness=False):
    """Extract everything we need from a MediaPipe result object, once.

    Duck-typed on purpose: it only ever touches ``multi_hand_landmarks`` and
    ``multi_handedness``, so a fake result object can drive it in tests.

    Returns one dict per detected hand::

        {"label": "Left"|"Right"|"Unknown", "norm": [(x, y, z), ...],
         "px": [(x, y), ...], "bbox": (xmin, ymin, xmax, ymax),
         "palm_z": float, "orientation": "Palm"|"Back"}

    ``None``, empty and missing fields all yield ``[]`` — this never raises.
    Hands without a full 21-landmark set are skipped, since no rule below can
    be evaluated for them.
    """
    hand_landmarks = getattr(results, "multi_hand_landmarks", None)
    if not hand_landmarks:
        return []

    handedness = getattr(results, "multi_handedness", None) or []

    hands = []
    for index, hand in enumerate(hand_landmarks):
        landmarks = getattr(hand, "landmark", None)
        if not landmarks or len(landmarks) < EXPECTED_LANDMARKS:
            continue

        norm = [(lm.x, lm.y, lm.z) for lm in landmarks]
        px = to_pixel_landmarks(landmarks, width, height)
        z = palm_normal_z(px)
        label = _handedness_label(handedness[index], swap_handedness) if index < len(handedness) else "Unknown"

        hands.append(
            {
                "label": label,
                "norm": norm,
                "px": px,
                "bbox": bbox_from_pixels(px),
                "palm_z": z,
                "orientation": orientation_from_palm_z(z),
            }
        )
    return hands


def fingers_state(pixel_landmarks, hand_label="Unknown", palm_z=None, thumb_mode=THUMB_DISTANCE):
    """Return ``[thumb, index, middle, ring, pinky]`` extended flags.

    Index–pinky use the vertical rule from the spec: the fingertip is extended
    when it sits strictly above its PIP joint (``y_tip < y_pip``).

    The thumb default is the distance rule, which is robust to handedness and to
    rotation in the image plane: the tip is extended when it moves *away* from
    the pinky knuckle relative to the IP joint. ``thumb_mode="x"`` keeps the
    spec's original horizontal rule (``Right`` -> ``x_tip < x_ip``, ``Left`` ->
    ``x_tip > x_ip``) available for reference; it is only valid for an upright,
    palm-facing hand.

    ``palm_z`` is accepted for API symmetry with the orientation readout; the
    default distance rule does not need it (it is already handedness-agnostic).

    A ``None``, malformed or short landmark list returns exactly ``[False] * 5``
    and never raises.
    """
    if not pixel_landmarks or len(pixel_landmarks) < EXPECTED_LANDMARKS:
        return [False] * 5

    try:
        states = [pixel_landmarks[tip][1] < pixel_landmarks[pip][1] for tip, pip in FINGER_PAIRS]

        if thumb_mode == THUMB_X and hand_label in ("Left", "Right"):
            tip, ip = pixel_landmarks[THUMB_TIP], pixel_landmarks[THUMB_IP]
            thumb = tip[0] < ip[0] if hand_label == "Right" else tip[0] > ip[0]
        else:
            thumb = _distance(pixel_landmarks[THUMB_TIP], pixel_landmarks[PINKY_MCP]) > _distance(
                pixel_landmarks[THUMB_IP], pixel_landmarks[PINKY_MCP]
            )
    except (IndexError, TypeError, ValueError):
        return [False] * 5

    return [thumb] + states


def classify_gesture(states):
    """Name the pose from a 5-element finger-state list."""
    if not states or len(states) < 5:
        return NO_GESTURE

    thumb, index, middle, ring, pinky = (bool(state) for state in states[:5])

    if thumb and index and middle and ring and pinky:
        return "Open Palm"
    if not (thumb or index or middle or ring or pinky):
        return "Fist"
    if thumb and not (index or middle or ring or pinky):
        return "Thumbs Up"
    if not thumb and index and middle and not (ring or pinky):
        return "Peace"
    if thumb and index and not (middle or ring) and pinky:
        return "Rock On"
    return NO_GESTURE


class GestureDebouncer:
    """Rolling majority vote that suppresses single-frame gesture flicker.

    ``update`` returns the new label once at least ``min_agree`` of the last
    ``window`` frames agree on it, and otherwise keeps returning the last stable
    label. Before any majority has ever formed it returns ``"—"``.
    """

    def __init__(self, window=5, min_agree=4):
        self.window = window
        self.min_agree = min_agree
        self._history = deque(maxlen=window)
        self._stable = NO_GESTURE

    @property
    def stable(self):
        return self._stable

    def update(self, gesture):
        self._history.append(gesture)
        if self._history.count(gesture) >= self.min_agree:
            self._stable = gesture
        return self._stable


def ema(previous, current, alpha=0.65):
    """Exponential moving average: ``alpha * current + (1 - alpha) * previous``.

    ``previous is None`` (first sighting) returns ``current`` unchanged.
    """
    if previous is None:
        return (current[0], current[1])
    return (
        alpha * current[0] + (1 - alpha) * previous[0],
        alpha * current[1] + (1 - alpha) * previous[1],
    )
