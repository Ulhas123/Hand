Build a real-time computer vision application in Python that tracks hands and individual finger movements from a live webcam feed. 

### Objective
Create a clean, modular script (e.g., `hand_tracker.py`) that captures live video, extracts 3D hand landmarks, determines which fingers are extended, and overlays diagnostic visual telemetry onto the video feed.

### Core Tech Stack
- Python (3.9+)
- OpenCV (`opencv-python`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)

### Architectural & Functional Requirements

1. **Webcam Pipeline & Pre-Processing:**
   - Initialize video capture from default webcam (`cv2.VideoCapture(0)`). Include an explicit check and descriptive error message if the camera cannot be opened.
   - Flip each incoming frame horizontally (`cv2.flip(frame, 1)`) to provide a natural mirror-view experience.
   - Convert frames from BGR to RGB before feeding them to MediaPipe.
   - Maintain a stable target frame rate and implement an FPS counter using frame delta times (`time.time()`).

2. **MediaPipe Tracking Setup:**
   - Initialize `mp.solutions.hands.Hands` with:
     - `static_image_mode=False`
     - `max_num_hands=2`
     - `min_detection_confidence=0.7`
     - `min_tracking_confidence=0.6`
   - Handle empty detection cases gracefully without throwing `IndexError` or `NoneType` exceptions.

3. **Finger State Detection Logic:**
   - Account for the top-left origin $(0, 0)$ coordinate system where vertical values decrease upward.
   - **4 Fingers (Index, Middle, Ring, Pinky):** Check if the fingertip landmark is strictly above its respective PIP knuckle joint ($y_{\text{tip}} < y_{\text{pip}}$).
   - **Thumb:** Check lateral horizontal displacement rather than vertical height. Determine handedness (`Left` vs `Right`) using MediaPipe's classification output:
     - On a mirrored feed for a Right hand, thumb is extended if $x_{\text{tip}} < x_{\text{pip}}$.
     - For a Left hand, thumb is extended if $x_{\text{tip}} > x_{\text{pip}}$.
   - Compute and track the total count of raised fingers across all visible hands.

4. **Visual Overlay & UI:**
   - Draw the standard hand skeletal mesh and joint landmarks using `mp.solutions.drawing_utils`.
   - Draw an accentuated colored circle over the tip of the index finger (`landmark 8`) mapped to actual pixel coordinates ($x \times \text{width}$, $y \times \text{height}$).
   - Overlay a clean HUD in the top-left corner using `cv2.putText` displaying:
     - Current FPS (e.g., `FPS: 30`)
     - Total extended finger count (e.g., `Fingers: 3`)
     - Detected handedness label per hand

5. **Code Structure & Teardown:**
   - Encapsulate the hand detection logic in a reusable class `HandTracker` with methods:
     - `find_hands(frame, draw=True)`
     - `find_positions(frame, hand_index=0)`
     - `fingers_up(landmarks, hand_label)`
   - Include a standard `if __name__ == "__main__":` entry point running the capture loop.
   - Listen for the `'q'` key to break the loop.
   - Properly release camera hardware and close all OpenCV windows upon termination.