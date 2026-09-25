"""Tests for the ``HandTracker`` cache wrapper and the HUD card.

These need cv2/mediapipe, so the module skips itself when they are missing and
the lightweight logic-only CI job stays green.
"""

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from hand_logic import parse_hand_data  # noqa: E402
from hand_tracker import HandTracker, draw_hud, hud_lines  # noqa: E402

from test_hand_logic import (  # noqa: E402
    EXPECTED_LANDMARKS,
    FIST,
    OPEN_HAND,
    FakeClassificationList,
    FakeResults,
    _fake_hand,
)


def make_tracker(hands_data, thumb_mode="distance"):
    """Build a tracker without touching MediaPipe or a camera."""
    tracker = HandTracker.__new__(HandTracker)
    tracker.hands_data = hands_data
    tracker.thumb_mode = thumb_mode
    tracker.swap_handedness = False
    tracker._smooth_tip = {}
    tracker._debounce = {}
    return tracker


def parsed_hands(raw_hands, labels=("Right", "Left")):
    results = FakeResults(
        hands=[_fake_hand(points) for points in raw_hands],
        handedness=[FakeClassificationList(label) for label in labels[: len(raw_hands)]],
    )
    return parse_hand_data(results, 1000, 1000)


def test_hand_count_tracks_parsed_hands():
    assert make_tracker(parsed_hands([])).hand_count() == 0
    assert make_tracker(parsed_hands([OPEN_HAND])).hand_count() == 1
    assert make_tracker(parsed_hands([OPEN_HAND, FIST])).hand_count() == 2


def test_hand_label_for_valid_and_out_of_range_index():
    tracker = make_tracker(parsed_hands([OPEN_HAND, FIST]))
    assert tracker.hand_label(0) == "Right"
    assert tracker.hand_label(1) == "Left"
    assert tracker.hand_label(2) == "Unknown"
    assert tracker.hand_label(-1) == "Unknown"


def test_find_positions_returns_cached_pixels():
    tracker = make_tracker(parsed_hands([OPEN_HAND]))
    positions = tracker.find_positions(None, 0)
    assert len(positions) == EXPECTED_LANDMARKS
    for got, (x, y) in zip(positions, OPEN_HAND):
        assert abs(got[0] - x) <= 1 and abs(got[1] - y) <= 1


def test_find_positions_out_of_range_is_empty():
    tracker = make_tracker(parsed_hands([OPEN_HAND]))
    assert tracker.find_positions(None, 1) == []
    assert tracker.find_positions(None, 99) == []
    assert tracker.find_positions(None, -1) == []
    assert make_tracker([]).find_positions(None, 0) == []


def test_fingers_up_and_orientation():
    tracker = make_tracker(parsed_hands([OPEN_HAND, FIST]))
    assert tracker.fingers_up(tracker.find_positions(None, 0), "Right") == [True] * 5
    assert tracker.fingers_up(tracker.find_positions(None, 1), "Left") == [False] * 5
    assert tracker.orientation(0) == "Palm"
    assert tracker.orientation(5) == "—"


def test_index_tip_smooths_towards_the_target():
    tracker = make_tracker(parsed_hands([OPEN_HAND]))
    raw = tracker.find_positions(None, 0)[8]

    first = tracker.index_tip(0)
    assert first == raw

    tracker.hands_data[0]["px"][8] = (raw[0] + 100, raw[1] + 100)
    second = tracker.index_tip(0)
    assert raw[0] < second[0] < raw[0] + 100
    assert (second[0], second[1]) != (raw[0] + 100, raw[1] + 100)


def test_index_tip_is_none_without_a_hand():
    tracker = make_tracker(parsed_hands([]))
    assert tracker.index_tip(0) is None
    assert tracker.index_tip(7) is None


def test_gesture_is_debounced_per_hand():
    tracker = make_tracker(parsed_hands([OPEN_HAND, FIST]))
    open_states = [True] * 5
    fist_states = [False] * 5

    assert [tracker.gesture(0, open_states) for _ in range(3)] == ["—", "—", "—"]
    assert tracker.gesture(0, open_states) == "Open Palm"
    assert [tracker.gesture(1, fist_states) for _ in range(4)] == ["—", "—", "—", "Fist"]
    assert tracker.gesture(0, open_states) == "Open Palm"


def test_gesture_state_is_keyed_by_label_not_index():
    tracker = make_tracker(parsed_hands([OPEN_HAND, FIST]))
    for _ in range(5):
        tracker.gesture(0, [True] * 5)
    assert tracker.gesture(0, [True] * 5) == "Open Palm"

    tracker.hands_data.reverse()  # same two hands, mediapipe shuffled the order
    assert tracker.gesture(1, [True] * 5) == "Open Palm"


def test_hud_lines_formatting():
    assert hud_lines(3, ["Right: 2 (Peace) [Palm]"], "30") == [
        "FPS: 30",
        "Fingers: 3",
        "Right: 2 (Peace) [Palm]",
    ]
    assert hud_lines(0, [], "0") == ["FPS: 0", "Fingers: 0"]


def test_draw_hud_blends_a_card_and_writes_text():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = 40
    before = frame[10:60, 10:60].copy()

    draw_hud(frame, hud_lines(2, ["Right: 2 (Peace) [Palm]"], "30"))

    assert not np.array_equal(frame[10:60, 10:60], before)
    assert frame[10:60, 10:60].max() > 200


def test_draw_hud_survives_a_frame_smaller_than_the_card():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    assert draw_hud(frame, hud_lines(1, [], "30")).shape == (20, 20, 3)
    assert draw_hud(frame, []).shape == (20, 20, 3)
