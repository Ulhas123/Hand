"""Real-time hand and finger tracker.

Captures a webcam feed, extracts 21 3D hand landmarks per hand with MediaPipe,
decides which fingers are extended, and overlays the skeleton, a smoothed index
fingertip marker and a telemetry HUD on the video.

``--image PATH`` runs the exact same pipeline over a single still instead of a
camera, which makes the whole thing verifiable (and CI-testable) without a
webcam.

Run ``python hand_tracker.py --help`` for the full flag list.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import warnings
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from hand_logic import (
    INDEX_TIP,
    GestureDebouncer,
    classify_gesture,
    ema,
    fingers_state,
    parse_hand_data,
)

LOG = logging.getLogger("hand_tracker")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import mediapipe as mp

if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
    raise RuntimeError(
        "mediapipe>=0.10.30 removed mp.solutions, which this app is built on. "
        "Install the pinned version instead: pip install -r requirements.txt "
        "(do NOT `pip install -U mediapipe`)."
    )

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

WINDOW_NAME = "Hand Tracker"
CAPTURES_DIR = "captures"

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SCALE = 0.6
HUD_FONT_THICKNESS = 1
HUD_MARGIN = 10
HUD_PAD = 10
HUD_ALPHA = 0.6

INDEX_TIP_HALO_RADIUS = 14
INDEX_TIP_RADIUS = 10
INDEX_TIP_COLOR = (0, 165, 255)

FPS_WINDOW = 30


class HandTracker:
    """MediaPipe Hands wrapper with a single-pass, per-frame data cache."""

    def __init__(
        self,
        max_hands=2,
        mode=False,
        det_conf=0.7,
        track_conf=0.6,
        thumb_mode="distance",
        swap_handedness=False,
    ):
        self.thumb_mode = thumb_mode
        self.swap_handedness = swap_handedness
        self.hands_data = []
        self._smooth_tip = {}
        self._debounce = {}
        self.hands = mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )

    def find_hands(self, frame, draw=True):
        """Detect hands in a BGR frame, draw the mesh, and rebuild the cache.

        The normalized landmarks, pixel landmarks, bounding box, handedness
        label and palm orientation are computed once here and reused by every
        helper below for the rest of the frame.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True

        height, width = frame.shape[:2]
        self.hands_data = parse_hand_data(results, width, height, self.swap_handedness)

        if draw and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(235, 235, 235), thickness=2),
                )
        return frame

    def hand_count(self):
        """Number of hands seen in the most recent ``find_hands`` call."""
        return len(self.hands_data)

    def hand_label(self, hand_index):
        """Handedness label of a detected hand, or ``"Unknown"``."""
        if 0 <= hand_index < len(self.hands_data):
            return self.hands_data[hand_index]["label"]
        return "Unknown"

    def find_positions(self, frame, hand_index=0):
        """Cached pixel landmarks for one hand, or ``[]`` if there is none.

        ``frame`` is only kept for signature compatibility: the returned points
        were computed inside the last ``find_hands`` call, so they are only
        valid for a frame of that same size.
        """
        if 0 <= hand_index < len(self.hands_data):
            return list(self.hands_data[hand_index]["px"])
        return []

    def fingers_up(self, landmarks, hand_label):
        """``[thumb, index, middle, ring, pinky]`` extended flags for one hand."""
        return fingers_state(landmarks, hand_label, thumb_mode=self.thumb_mode)

    def index_tip(self, hand_index, alpha=0.65):
        """EMA-smoothed pixel position of landmark 8, or ``None``."""
        if not 0 <= hand_index < len(self.hands_data):
            return None

        hand = self.hands_data[hand_index]
        if len(hand["px"]) <= INDEX_TIP:
            return None

        key = self._state_key(hand_index, hand["label"])
        smoothed = ema(self._smooth_tip.get(key), hand["px"][INDEX_TIP], alpha)
        self._smooth_tip[key] = smoothed
        return (int(smoothed[0]), int(smoothed[1]))

    def orientation(self, hand_index):
        """``"Palm"`` or ``"Back"`` for a detected hand, or ``"—"``."""
        if 0 <= hand_index < len(self.hands_data):
            return self.hands_data[hand_index]["orientation"]
        return "—"

    def gesture(self, hand_index, states):
        """Debounced gesture name for the finger states of one hand."""
        key = self._state_key(hand_index, self.hand_label(hand_index))
        debouncer = self._debounce.get(key)
        if debouncer is None:
            debouncer = GestureDebouncer()
            self._debounce[key] = debouncer
        return debouncer.update(classify_gesture(states))

    @staticmethod
    def _state_key(hand_index, label):
        """Key per-hand smoothing/debounce state.

        MediaPipe's hand list order is not stable across frames, so keying by
        list index would blend two different physical hands together when they
        reorder — the label is a much steadier identity.
        """
        return label if label in ("Left", "Right") else f"index:{hand_index}"

    def close(self):
        self.hands.close()


def hud_lines(total_fingers, hand_lines, fps_text):
    """Assemble the HUD text: FPS, total count, then one line per hand."""
    return [f"FPS: {fps_text}", f"Fingers: {total_fingers}"] + list(hand_lines)


def draw_hud(frame, lines):
    """Draw an alpha-blended card behind white telemetry text (top-left)."""
    if not lines:
        return frame

    sizes = [cv2.getTextSize(line, HUD_FONT, HUD_FONT_SCALE, HUD_FONT_THICKNESS)[0] for line in lines]
    text_width = max(width for width, _ in sizes)
    text_height = max(height for _, height in sizes)
    line_height = text_height + 12

    height, width = frame.shape[:2]
    x0, y0 = HUD_MARGIN, HUD_MARGIN
    x1 = min(width, x0 + text_width + 2 * HUD_PAD)
    y1 = min(height, y0 + 2 * HUD_PAD + line_height * len(lines))
    if x1 <= x0 or y1 <= y0:
        return frame

    card = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.addWeighted(card, 1.0 - HUD_ALPHA, np.zeros_like(card), HUD_ALPHA, 0)

    y = y0 + HUD_PAD + text_height
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x0 + HUD_PAD, y),
            HUD_FONT,
            HUD_FONT_SCALE,
            (255, 255, 255),
            HUD_FONT_THICKNESS,
            cv2.LINE_AA,
        )
        y += line_height
    return frame


def annotate_hands(tracker, frame):
    """Run the finger logic and overlays for every detected hand.

    Returns the total number of extended fingers and one HUD line per hand.
    """
    total = 0
    lines = []

    for index in range(tracker.hand_count()):
        label = tracker.hand_label(index)
        landmarks = tracker.find_positions(frame, index)
        states = tracker.fingers_up(landmarks, label)
        count = sum(states)
        total += count
        lines.append(f"{label}: {count} ({tracker.gesture(index, states)}) [{tracker.orientation(index)}]")

        tip = tracker.index_tip(index)
        if tip is not None:
            cv2.circle(frame, tip, INDEX_TIP_HALO_RADIUS, (255, 255, 255), cv2.FILLED)
            cv2.circle(frame, tip, INDEX_TIP_RADIUS, INDEX_TIP_COLOR, cv2.FILLED)

    return total, lines


def save_screenshot(frame):
    """Write the annotated frame (HUD burned in) to ``captures/``."""
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(CAPTURES_DIR, f"frame_{stamp}.png")
    if cv2.imwrite(path, frame):
        LOG.info("Saved screenshot: %s", path)
    else:
        LOG.error("Could not write screenshot: %s", path)


def open_camera(index, width, height):
    """Open a webcam, falling back across backends and camera indices."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]

    for candidate in [index] + [other for other in (1, 0) if other != index]:
        for backend in backends:
            capture = cv2.VideoCapture(candidate, backend)
            if capture.isOpened():
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                LOG.info(
                    "Camera %d opened at %dx%d.",
                    candidate,
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
                return capture
            capture.release()
        LOG.warning("Camera %d did not open.", candidate)

    LOG.error("Could not open a webcam. Check it is connected and not in use by another app.")
    return None


def run_live(args):
    """Webcam capture loop: read, flip, analyse, draw, show."""
    capture = open_camera(args.camera, args.width or DEFAULT_WIDTH, args.height or DEFAULT_HEIGHT)
    if capture is None:
        return 1

    tracker = HandTracker(
        max_hands=args.max_hands,
        mode=args.static,
        thumb_mode=args.thumb_mode,
        swap_handedness=args.swap_handedness,
    )
    frame_times = deque(maxlen=FPS_WINDOW)
    previous_time = time.time()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                LOG.error("Frame capture failed; stopping.")
                break

            frame = cv2.flip(frame, 1)
            frame = tracker.find_hands(frame, draw=not args.no_draw)
            total, hand_lines = annotate_hands(tracker, frame)

            now = time.time()
            delta = now - previous_time
            previous_time = now
            if delta > 0:
                frame_times.append(1.0 / delta)
            fps_text = f"{sum(frame_times) / len(frame_times):.0f}" if frame_times else "0"

            draw_hud(frame, hud_lines(total, hand_lines, fps_text))
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                save_screenshot(frame)
    finally:
        capture.release()
        tracker.close()
        cv2.destroyAllWindows()

    return 0


def run_image(args):
    """Run the full pipeline once over a still image and write ``<stem>_out.png``."""
    frame = cv2.imread(args.image)
    if frame is None:
        LOG.error("Could not read image: %s", args.image)
        return 1

    if args.width and args.height:
        frame = cv2.resize(frame, (args.width, args.height))
        LOG.info("Resized input to %dx%d.", args.width, args.height)

    tracker = HandTracker(
        max_hands=args.max_hands,
        mode=True,
        thumb_mode=args.thumb_mode,
        swap_handedness=args.swap_handedness,
    )
    try:
        frame = tracker.find_hands(frame, draw=not args.no_draw)
        total, hand_lines = annotate_hands(tracker, frame)
        draw_hud(frame, hud_lines(total, hand_lines, "n/a (still)"))
        hands_seen = tracker.hand_count()
    finally:
        tracker.close()

    stem, _ = os.path.splitext(args.image)
    output_path = f"{stem}_out.png"
    if not cv2.imwrite(output_path, frame):
        LOG.error("Could not write output image: %s", output_path)
        return 1

    LOG.info("Detected %d hand(s) with %d extended finger(s); wrote %s", hands_seen, total, output_path)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="hand_tracker.py",
        description="Real-time hand and finger tracker (MediaPipe + OpenCV).",
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument(
        "--width", type=int, default=None, help="capture width (default: 640 live, native in --image)"
    )
    parser.add_argument(
        "--height", type=int, default=None, help="capture height (default: 480 live, native in --image)"
    )
    parser.add_argument("--max-hands", type=int, default=2, help="maximum hands to detect (default: 2)")
    parser.add_argument("--no-draw", action="store_true", help="disable the skeletal mesh overlay")
    parser.add_argument(
        "--thumb-mode",
        choices=("distance", "x"),
        default="distance",
        help="thumb rule: 'distance' (rotation-robust, default) or 'x' (original horizontal rule)",
    )
    parser.add_argument(
        "--swap-handedness",
        action="store_true",
        help="swap Left/Right labels (display-only fix if your mirrored feed reports them inverted)",
    )
    parser.add_argument("--static", action="store_true", help="use MediaPipe static_image_mode (no tracking)")
    parser.add_argument("--image", metavar="PATH", help="process a still image instead of the camera, then exit")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="logging verbosity (default: INFO)",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    if args.image:
        return run_image(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
