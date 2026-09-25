You are an expert Computer Vision engineer and a master technical educator. Write a complete, production-grade LaTeX document (.tex) that serves as an exhaustive, beautifully designed Study Handbook for a real-time computer vision project: "Real-Time Hand & Finger Tracker with MediaPipe and OpenCV".

### Target Audience & Pedagogical Goal
- The reader has ALMOST ZERO knowledge of Python and ABSOLUTE ZERO knowledge of Machine Learning.
- Do not skip steps or hand-wave concepts. Start from first principles using vivid physical analogies (e.g., comparing frames to flipbooks, tensors to multi-layer grids, neural nets to feature detectors) before diving into exact technical formulations.
- The document must be mathematically and architecturally deep, covering every single decision, algorithm, edge case, and system design pattern specified below.

### Visual Design & Styling Instructions (Prevent Boredom)
Use a modern, professional visual style:
- Document Class: `report` or `scrreprt` (A4 paper, 1-inch margins via `geometry`).
- Typography & Layout: Clean modern fonts, active headers/footers via `fancyhdr`, customized table of contents.
- Color Palette (`xcolor`): Define a cohesive modern palette:
  - Deep Navy / Midnight Blue (`#1E293B`) for primary headers and titles.
  - Electric Cyan / Teal (`#0EA5E9` / `#0D9488`) for accents and key highlights.
  - Coral / Amber (`#F59E0B` / `#EF4444`) for warnings, traps, and false-positive notes.
  - Soft Slate Grey (`#F8FAFC` / `#E2E8F0`) for box backgrounds and table striping.
- Custom Environments (`tcolorbox`):
  - `pythonconcept`: Soft blue card for teaching Python language fundamentals from scratch.
  - `mlconcept`: Soft purple card for demystifying Computer Vision and Machine Learning.
  - `mathdeepdive`: Framed teal card for step-by-step vector and geometric derivations.
  - `edgetrap`: Amber/Red warning card highlighting physical failure modes (e.g., palm inversion, knife-hand false positives, MediaPipe package breaks).
- Diagrams (`tikz`): Include native TikZ diagrams:
  1. System Architecture: Clean block diagram showing the decoupling between `hand_tracker.py` (I/O, OpenCV, camera loop) and `hand_logic.py` (pure mathematical functions, zero dependencies).
  2. Frame Coordinate System: Comparing Cartesian coordinates ($y$ grows upward) with Screen/OpenCV coordinates ($y=0$ at top-left, growing downward).
  3. MediaPipe 21 Hand Landmarks: Visual skeletal hand topology showing the 21 keypoints, specifically labeling Wrist (0), Thumb Tip (4), Index MCP (5), Index Tip (8), and Pinky MCP (17).
  4. Palm Normal Vector: Visualizing the vectors $\vec{v}_1 (0 \to 5)$ and $\vec{v}_2 (0 \to 17)$ and their 3D cross-product normal vector determining Palm vs. Back orientation.
  5. Pipeline Data Flow: Sequence diagram from raw camera frame $\to$ flip $\to$ BGR-to-RGB $\to$ BlazePalm $\to$ Landmark regression $\to$ single-pass cache $\to$ geometry $\to$ debouncer $\to$ alpha HUD $\to$ screen.
- Code Blocks: Use `listings` (or `minted`) styled with dark backgrounds, syntax highlighting, and explanatory callouts.

---

### Core Curriculum & Technical Topics to Cover in Full Detail

#### Chapter 1: Foundations for Absolute Beginners
1. Python from Scratch for CV:
   - What is an interpreter? How Python runs scripts line-by-line.
   - Variables, dynamic typing, lists vs. tuples vs. dictionaries.
   - Functions, arguments, return values, and type hinting (`list[bool]`, `tuple[int, int]`).
   - Object-Oriented Programming (OOP) simplified: What are Classes, Objects, `self`, and `__init__`?
   - Special constructs: `collections.deque(maxlen=N)`, list comprehensions, and what `if __name__ == "__main__":` actually does.
2. How Digital Images Work:
   - Pixels, resolution, and color channels.
   - Why OpenCV uses BGR (Blue-Green-Red) instead of standard RGB, and why converting is essential.
   - How a 640x480 video frame is stored as a 3D NumPy array of shape $(480, 640, 3)$ with values in $[0, 255]$.
3. Machine Learning Demystified (Without Jargon):
   - Traditional programming (Rules + Data = Answers) vs. Machine Learning (Data + Answers = Rules).
   - What is a Neural Network? Weights, layers, and feature extraction.
   - Training vs. Inference: Why this project uses inference-only pre-trained models.
   - Normalized Coordinates: Why MediaPipe outputs coordinates in $[0.0, 1.0]$ and the math to convert them to monitor pixels ($x_{\text{px}} = x \cdot W$, $y_{\text{px}} = y \cdot H$).

#### Chapter 2: The Under-the-Hood ML Engine (MediaPipe Hands)
1. The Two-Stage Pipeline Architecture:
   - Stage 1: BlazePalm Detector (Single Shot Detector scanning the full frame for palm bounds).
   - Stage 2: Hand Landmark Model (Cropping to the palm bounding box and predicting 21 3D landmarks).
   - Why this two-stage approach provides 30+ FPS performance on CPU without running full-frame neural nets every tick.
2. The Dependency Matrix & Package Traps:
   - The fragile wheel ecosystem: Why `opencv-contrib-python==4.11.0.86` is pinned instead of `opencv-python` (preventing dual `cv2` binary conflicts).
   - Why `numpy<2` is required for binary ABI stability with MediaPipe 0.10.x wheels.
   - The breaking change in MediaPipe $\ge 0.10.30$: Why `mp.solutions` was removed, why `mediapipe==0.10.14` is strictly pinned, and the implementation of an import-time `RuntimeError` guard.

#### Chapter 3: Mathematical & Geometric Algorithms
1. The Screen Coordinate System & Inverted Y-Axis:
   - Why $y_{\text{tip}} < y_{\text{pip}}$ indicates an extended finger for the 4 long fingers (Index, Middle, Ring, Pinky).
2. The Thumb Mechanics & Invariance:
   - Why horizontal $X$-comparison fails when hands rotate or turn backwards.
   - The Distance-Based Invariant Rule:
     $$\text{dist}(P_{\text{thumb\_tip}}, P_{\text{pinky\_mcp}}) > \text{dist}(P_{\text{thumb\_ip}}, P_{\text{pinky\_mcp}})$$
   - The "Knife Hand" (Adducted Thumb) edge case and the secondary clearance gate against Index MCP:
     $$\text{dist}(P_{\text{thumb\_tip}}, P_{\text{index\_mcp}}) > 1.2 \cdot \text{dist}(P_{\text{thumb\_ip}}, P_{\text{index\_mcp}})$$
3. Palm Normal Vector & Hand Orientation:
   - Defining vectors $\vec{v}_1 = \mathbf{P}_5 - \mathbf{P}_0$ and $\vec{v}_2 = \mathbf{P}_{17} - \mathbf{P}_0$.
   - Computing the 2D cross-product $Z$-component:
     $$Z_{\text{normal}} = (x_5 - x_0)(y_{17} - y_0) - (y_5 - y_0)(x_{17} - x_0)$$
   - Deriving whether the hand presents its Palm ($Z < 0$) or Back ($Z > 0$) to the camera.
4. Signal Processing & Smoothing:
   - Sensor noise and sub-pixel coordinate jitter.
   - Exponential Moving Average (EMA) formulation for Landmark 8 (Index Tip):
     $$\mathbf{P}_{\text{smooth}}^{(t)} = \alpha \mathbf{P}_{\text{raw}}^{(t)} + (1 - \alpha) \mathbf{P}_{\text{smooth}}^{(t-1)}, \quad \alpha = 0.65$$
   - Rolling Window Majority Debouncing: Using a 5-frame queue to eliminate gesture flickering, requiring a $\ge 4/5$ consensus before updating state.

#### Chapter 4: System Design & Production Architecture
1. Architectural Separation:
   - Why `hand_logic.py` imports ONLY standard library and math (Zero dependencies: no OpenCV, no MediaPipe, no camera hardware).
   - How `hand_tracker.py` encapsulates hardware access, GUI, telemetry, and fallback chains.
2. Single-Pass Telemetry Caching:
   - Why repeated coordinate conversion creates bottlenecks.
   - Caching normalized coordinates, pixel coordinates, bounding boxes, and palm normal in a structured dictionary per frame.
3. Camera Fallback Chain & Hardware Resiliency:
   - Handling Windows DirectShow (`CAP_DSHOW`), Media Foundation (`CAP_MSMF`), and `CAP_ANY`.
   - Dynamic camera index hunting (fallback from 0 to 1).
4. Headless Mode (`--image PATH`) for CI & Verification:
   - Running the entire detection, geometry, and rendering pipeline on still images without physical webcams.
   - Setting up GitHub Actions CI using lightweight Python matrix runners without graphical display servers.
5. Alpha-Blended Telemetry HUD:
   - Extracting Region of Interest (ROI), creating an overlay buffer, and blending using `cv2.addWeighted`.
   - Dynamically resizing the card height to prevent multi-hand telemetry text overflow.

#### Chapter 5: Complete Walkthrough of the Source Code
- Include the fully implemented, clean code for:
  1. `hand_logic.py` (with full docstrings and annotations).
  2. `hand_tracker.py` (complete capture loop, CLI arguments, and rendering).
  3. `tests/test_hand_logic.py` (offline unit tests testing synthetic coordinates with zero hardware).
- Provide a line-by-line explanation of every complex block, explaining what each variable stores and why each condition exists.

---

### Output Requirements
- Output pure, valid, compilable LaTeX code inside a single markdown code fence (` ```latex ... ``` `).
- Ensure all necessary LaTeX packages are imported in the preamble (`geometry`, `xcolor`, `tcolorbox`, `tikz`, `amsmath`, `amssymb`, `listings`, `booktabs`, `hyperref`, `fancyhdr`).
- Ensure no placeholders, no `TODO` comments, and no truncated chapters. Write out the explanations thoroughly and completely.