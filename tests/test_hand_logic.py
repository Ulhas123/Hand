"""Offline tests for the pure logic in ``hand_logic.py``.

No camera, no OpenCV, no MediaPipe: the synthetic hands below are plain
coordinate lists, so this file runs in a bare Python environment.
"""

import math

from hand_logic import (
    EXPECTED_LANDMARKS,
    GestureDebouncer,
    NO_GESTURE,
    THUMB_X,
    bbox_from_pixels,
    classify_gesture,
    ema,
    fingers_state,
    orientation_from_palm_z,
    palm_normal_z,
    parse_hand_data,
    to_pixel_landmarks,
)

# --------------------------------------------------------------------------
# Synthetic hands. Coordinates are pixels in a mirrored (selfie) view with the
# origin at the top-left, so y grows downward. Index 0 = wrist.
# --------------------------------------------------------------------------

#: Right hand, palm facing the camera, fingers up, thumb extended sideways.
OPEN_HAND = [
    (320, 420),  # 0  wrist
    (365, 400),  # 1  thumb CMC
    (395, 370),  # 2  thumb MCP
    (415, 345),  # 3  thumb IP
    (430, 320),  # 4  thumb TIP  (far from the pinky knuckle -> extended)
    (355, 320),  # 5  index MCP
    (350, 280),  # 6  index PIP
    (348, 260),  # 7  index DIP
    (346, 235),  # 8  index TIP  (above its PIP -> extended)
    (320, 315),  # 9  middle MCP
    (318, 270),  # 10 middle PIP
    (317, 245),  # 11 middle DIP
    (316, 215),  # 12 middle TIP
    (290, 320),  # 13 ring MCP
    (285, 280),  # 14 ring PIP
    (283, 258),  # 15 ring DIP
    (281, 230),  # 16 ring TIP
    (262, 335),  # 17 pinky MCP
    (252, 305),  # 18 pinky PIP
    (247, 285),  # 19 pinky DIP
    (243, 262),  # 20 pinky TIP
]

#: Closed fist: every tip curled below its PIP, thumb folded across the palm.
FIST = [
    (320, 420),  # 0  wrist
    (355, 395),  # 1  thumb CMC
    (375, 375),  # 2  thumb MCP
    (395, 355),  # 3  thumb IP
    (375, 350),  # 4  thumb TIP  (closer to the pinky knuckle -> folded)
    (355, 320),  # 5  index MCP
    (350, 280),  # 6  index PIP
    (350, 297),  # 7
    (352, 305),  # 8  index TIP
    (320, 315),  # 9  middle MCP
    (318, 270),  # 10 middle PIP
    (319, 288),  # 11
    (320, 296),  # 12 middle TIP
    (290, 320),  # 13 ring MCP
    (285, 280),  # 14 ring PIP
    (286, 298),  # 15
    (287, 306),  # 16 ring TIP
    (262, 335),  # 17 pinky MCP
    (252, 305),  # 18 pinky PIP
    (253, 322),  # 19
    (254, 330),  # 20 pinky TIP
]


def _with_finger(points, tip_index, tip_position):
    updated = list(points)
    updated[tip_index] = tip_position
    return updated


def _with_thumb(points, cmc, mcp, ip, tip):
    updated = list(points)
    updated[1], updated[2], updated[3], updated[4] = cmc, mcp, ip, tip
    return updated


def _rotate(points, degrees):
    """Rotate every landmark around the set's centroid."""
    radians = math.radians(degrees)
    cos_a, sin_a = math.cos(radians), math.sin(radians)
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return [
        (
            round(cx + cos_a * (x - cx) - sin_a * (y - cy)),
            round(cy + sin_a * (x - cx) + cos_a * (y - cy)),
        )
        for x, y in points
    ]


def _mirror_x(points):
    return [(-x, y) for x, y in points]


# --------------------------------------------------------------------------
# Finger state
# --------------------------------------------------------------------------


def test_all_fingers_extended():
    assert fingers_state(OPEN_HAND) == [True, True, True, True, True]


def test_all_fingers_folded():
    assert fingers_state(FIST) == [False, False, False, False, False]


def test_index_only_extended():
    hand = _with_finger(FIST, 8, (346, 235))
    assert fingers_state(hand) == [False, True, False, False, False]


def test_each_finger_in_isolation():
    for tip, position, expected_index in (
        (12, (316, 215), 2),
        (16, (281, 230), 3),
        (20, (243, 262), 4),
    ):
        states = fingers_state(_with_finger(FIST, tip, position))
        expected = [False] * 5
        expected[expected_index] = True
        assert states == expected, f"finger {tip} not detected in isolation"


def test_thumb_only_extended():
    hand = _with_thumb(FIST, (365, 400), (395, 370), (415, 345), (430, 320))
    assert fingers_state(hand) == [True, False, False, False, False]


def test_thumb_distance_mode_extended_and_folded():
    assert fingers_state(OPEN_HAND)[0] is True
    assert fingers_state(FIST)[0] is False


def test_thumb_x_mode_right_hand():
    hand = _with_thumb(FIST, (0, 0), (0, 0), (200, 100), (100, 100))
    assert fingers_state(hand, "Right", thumb_mode=THUMB_X)[0] is True


def test_thumb_x_mode_left_hand():
    hand = _with_thumb(FIST, (0, 0), (0, 0), (200, 100), (300, 100))
    assert fingers_state(hand, "Left", thumb_mode=THUMB_X)[0] is True


def test_thumb_x_mode_wrong_side_is_folded():
    hand = _with_thumb(FIST, (0, 0), (0, 0), (200, 100), (300, 100))
    assert fingers_state(hand, "Right", thumb_mode=THUMB_X)[0] is False


def test_thumb_x_mode_unknown_label_falls_back_to_distance():
    # OPEN_HAND's thumb sits at x4=430 > x3=415, so the horizontal rule folds it
    # for a "Right" label while the distance rule extends it.
    assert fingers_state(OPEN_HAND, "Right", thumb_mode=THUMB_X)[0] is False
    assert fingers_state(OPEN_HAND, "Unknown", thumb_mode=THUMB_X)[0] is True


def test_thumb_survives_image_plane_rotation():
    """The regression test for the old x-only rule: rotate the hand 90 degrees."""
    for degrees in (90, 37, 180, -120):
        rotated = _rotate(OPEN_HAND, degrees)
        assert fingers_state(rotated)[0] is True, f"thumb lost at {degrees} degrees"


def test_thumb_state_is_handedness_agnostic():
    for label in ("Left", "Right", "Unknown"):
        assert fingers_state(OPEN_HAND, label)[0] is True
        assert fingers_state(FIST, label)[0] is False


def test_malformed_landmarks_return_safe_result():
    safe = [False] * 5
    assert fingers_state(None) == safe
    assert fingers_state([]) == safe
    assert fingers_state([(0, 0)] * (EXPECTED_LANDMARKS - 1)) == safe
    assert fingers_state([None] * EXPECTED_LANDMARKS) == safe
    assert fingers_state(OPEN_HAND)[0] is True  # unchanged by the guard tests


# --------------------------------------------------------------------------
# Palm geometry
# --------------------------------------------------------------------------


def test_palm_normal_sign_flips_between_palm_and_back():
    palm_facing = palm_normal_z(OPEN_HAND)
    back_facing = palm_normal_z(_mirror_x(OPEN_HAND))
    assert palm_facing < 0 < back_facing


def test_orientation_from_palm_z():
    assert orientation_from_palm_z(palm_normal_z(OPEN_HAND)) == "Palm"
    assert orientation_from_palm_z(palm_normal_z(_mirror_x(OPEN_HAND))) == "Back"
    assert orientation_from_palm_z(0.0) == "Back"


def test_bbox_from_pixels():
    assert bbox_from_pixels(OPEN_HAND) == (243, 215, 430, 420)


def test_to_pixel_landmarks_scales_and_truncates():
    class Landmark:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    landmarks = [Landmark(0.25, 0.5, 0.0), Landmark(0.999, 0.001, -0.2)]
    assert to_pixel_landmarks(landmarks, 640, 480) == [(160, 240), (639, 0)]


# --------------------------------------------------------------------------
# Gesture classification
# --------------------------------------------------------------------------


def test_classify_gesture_known_poses():
    assert classify_gesture([True] * 5) == "Open Palm"
    assert classify_gesture([False] * 5) == "Fist"
    assert classify_gesture([True, False, False, False, False]) == "Thumbs Up"
    assert classify_gesture([False, True, True, False, False]) == "Peace"
    assert classify_gesture([True, True, False, False, True]) == "Rock On"


def test_classify_gesture_unknown_and_malformed():
    assert classify_gesture([False, False, True, True, True]) == NO_GESTURE
    assert classify_gesture([]) == NO_GESTURE
    assert classify_gesture(None) == NO_GESTURE


def test_classify_gesture_from_real_landmarks():
    assert classify_gesture(fingers_state(OPEN_HAND)) == "Open Palm"
    assert classify_gesture(fingers_state(FIST)) == "Fist"


# --------------------------------------------------------------------------
# Debouncer and EMA
# --------------------------------------------------------------------------


def test_debouncer_waits_for_min_agree():
    debouncer = GestureDebouncer()
    assert [debouncer.update("Peace") for _ in range(3)] == [NO_GESTURE] * 3
    assert debouncer.update("Peace") == "Peace"


def test_debouncer_ignores_single_frame_blip():
    debouncer = GestureDebouncer()
    for _ in range(5):
        debouncer.update("Fist")
    assert debouncer.update("Open Palm") == "Fist"
    assert debouncer.update("Fist") == "Fist"


def test_debouncer_switches_after_sustained_change():
    debouncer = GestureDebouncer()
    for _ in range(5):
        debouncer.update("Peace")
    for _ in range(3):
        assert debouncer.update("Fist") == "Peace"
    assert debouncer.update("Fist") == "Fist"


def test_debouncer_window_is_bounded():
    debouncer = GestureDebouncer(window=3, min_agree=2)
    assert debouncer.update("Peace") == NO_GESTURE
    assert debouncer.update("Peace") == "Peace"
    assert len(debouncer._history) <= 3


def test_ema_first_sample_passes_through():
    assert ema(None, (10, 20)) == (10, 20)


def test_ema_weights_current_sample():
    assert ema((0, 0), (100, 100), 0.65) == (65.0, 65.0)
    assert ema((65, 65), (100, 100), 0.65) == (87.75, 87.75)


# --------------------------------------------------------------------------
# Parsing (duck-typed fake MediaPipe results)
# --------------------------------------------------------------------------


class FakeLandmark:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


class FakeHand:
    def __init__(self, landmarks):
        self.landmark = landmarks


class FakeClassification:
    def __init__(self, label, score=0.9):
        self.label, self.score = label, score


class FakeClassificationList:
    def __init__(self, label):
        self.classification = [FakeClassification(label)]


class FakeResults:
    def __init__(self, hands=(), handedness=None, include_handedness=True):
        if hands is not None:
            self.multi_hand_landmarks = list(hands)
        if include_handedness:
            self.multi_handedness = list(handedness or [])


WIDTH = HEIGHT = 1000


def _fake_hand(pixel_points):
    return FakeHand([FakeLandmark(x / WIDTH, y / HEIGHT, 0.0) for x, y in pixel_points])


def test_parse_two_hands():
    results = FakeResults(
        hands=[_fake_hand(OPEN_HAND), _fake_hand(FIST)],
        handedness=[FakeClassificationList("Right"), FakeClassificationList("Left")],
    )
    hands = parse_hand_data(results, WIDTH, HEIGHT)

    assert len(hands) == 2
    assert [hand["label"] for hand in hands] == ["Right", "Left"]
    assert [len(hand["px"]) for hand in hands] == [EXPECTED_LANDMARKS] * 2
    assert len(hands[0]["norm"][0]) == 3

    assert hands[0]["palm_z"] < 0
    assert hands[0]["orientation"] == "Palm"

    for hand, source in zip(hands, (OPEN_HAND, FIST)):
        for got, want in zip(hand["px"], source):
            assert abs(got[0] - want[0]) <= 1 and abs(got[1] - want[1]) <= 1
        xmin, ymin, xmax, ymax = hand["bbox"]
        assert xmin == min(p[0] for p in source) or abs(xmin - min(p[0] for p in source)) <= 1
        assert ymax <= max(p[1] for p in source) + 1


def test_parse_zero_hands():
    assert parse_hand_data(FakeResults(hands=[], handedness=[]), WIDTH, HEIGHT) == []


def test_parse_none_and_empty_objects():
    assert parse_hand_data(None, WIDTH, HEIGHT) == []
    assert parse_hand_data(object(), WIDTH, HEIGHT) == []
    assert parse_hand_data(FakeResults(hands=None, include_handedness=False), WIDTH, HEIGHT) == []


def test_parse_missing_handedness_is_unknown():
    results = FakeResults(hands=[_fake_hand(OPEN_HAND)], include_handedness=False)
    hands = parse_hand_data(results, WIDTH, HEIGHT)
    assert len(hands) == 1
    assert hands[0]["label"] == "Unknown"


def test_parse_short_handedness_list_is_unknown():
    results = FakeResults(hands=[_fake_hand(OPEN_HAND), _fake_hand(FIST)], handedness=[])
    assert [hand["label"] for hand in parse_hand_data(results, WIDTH, HEIGHT)] == ["Unknown", "Unknown"]


def test_parse_swap_handedness():
    results = FakeResults(
        hands=[_fake_hand(OPEN_HAND), _fake_hand(FIST)],
        handedness=[FakeClassificationList("Right"), FakeClassificationList("Left")],
    )
    hands = parse_hand_data(results, WIDTH, HEIGHT, swap_handedness=True)
    assert [hand["label"] for hand in hands] == ["Left", "Right"]


def test_parse_skips_incomplete_hand():
    results = FakeResults(
        hands=[_fake_hand(OPEN_HAND[:10]), _fake_hand(FIST)],
        handedness=[FakeClassificationList("Right"), FakeClassificationList("Left")],
    )
    hands = parse_hand_data(results, WIDTH, HEIGHT)
    assert len(hands) == 1
    assert hands[0]["label"] == "Left"


def test_parse_returns_mirrored_and_back_facing_cases():
    palm = parse_hand_data(FakeResults(hands=[_fake_hand(OPEN_HAND)]), WIDTH, HEIGHT)[0]
    back = parse_hand_data(FakeResults(hands=[_fake_hand(_mirror_x(OPEN_HAND))]), WIDTH, HEIGHT)[0]
    assert (palm["orientation"], back["orientation"]) == ("Palm", "Back")
