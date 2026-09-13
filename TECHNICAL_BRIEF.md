# Technical Architecture Brief: Real-Time Monocular 3D Relighting & Spatial Perception Engine
**Event:** National Robotics Week (NRW) 8th Edition – AI & Vision Challenge  
**Organizers:** IEEE INSAT Student Branch & IEEE RAS INSAT Chapter  

---

## 1. Executive Summary & Architectural Overview
Our pipeline transforms a single 2D monocular RGB stream into a real-time, mathematically rigorous 3D Augmented Reality relighting engine without dedicated hardware depth sensors (LiDAR/ToF).

The architecture completely decouples neural inference from the display loop:
1. **Asynchronous Depth Perception:** A background daemon thread continuously executes *Depth Anything V2 Small* (`vits`) in half-precision (FP16) on CUDA Tensor Cores.
2. **Low-Latency Spatial Hand Engine:** MediaPipe tracking extracts sub-pixel fingertip coordinates and computes rotation-invariant 3D proximity.
3. **Physically-Based Radiance Shading:** Rendering operates in linear photometric space with Blinn-Phong reflections, radial occlusion shadows, and ACES photographic tone mapping, sustaining **35–50+ FPS** on consumer laptop GPUs.

---

## 2. Mathematical Formulations & Algorithms

### 2.1 3D Surface Normal Geometry Extraction (Level 01)
From the continuous normalized depth field $Z(x, y) \in [0, 1]$:
1. Spatial orthogonal depth gradients are extracted via central 2D Sobel differential operators:
   $$\frac{\partial Z}{\partial x} \approx \text{Sobel}_x(Z), \quad \frac{\partial Z}{\partial y} \approx \text{Sobel}_y(Z)$$
2. Surface tangent basis vectors are constructed:
   $$\vec{T}_x = \left[1, \; 0, \; \frac{\partial Z}{\partial x}\right]^T, \quad \vec{T}_y = \left[0, \; 1, \; \frac{\partial Z}{\partial y}\right]^T$$
3. The continuous 3D surface normal field $\vec{N}(x, y)$ is calculated via cross-product orthogonalization and unit normalization:
   $$\vec{N} = \frac{\left[-\frac{\partial Z}{\partial x}, \; -\frac{\partial Z}{\partial y}, \; 1\right]^T}{\sqrt{\left(\frac{\partial Z}{\partial x}\right)^2 + \left(\frac{\partial Z}{\partial y}\right)^2 + 1}}$$

---

### 2.2 Linear Radiance Space & Blinn-Phong Relighting (Level 02)
To prevent highlight clipping and color bleaching, all optical calculations execute in **linear radiometric space**:
1. **Gamma Expansion (sRGB to Linear):**
   $$I_{\text{linear}} = \left(\frac{I_{\text{sRGB}}}{255.0}\right)^{2.2}$$
2. **Photometric Inverse-Square Distance Attenuation:**
   $$A(d) = \frac{I_{\text{light}}}{1.0 + 2.5 d + 7.5 d^2}, \quad \text{where } d = \|\vec{P}_{\text{light}} - \vec{P}_{\text{surface}}\|_2$$
3. **Lambertian Diffuse Reflection:**
   $$I_{\text{diff}} = \max(0, \; \vec{N} \cdot \vec{L}), \quad \vec{L} = \frac{\vec{P}_{\text{light}} - \vec{P}_{\text{surface}}}{d}$$
4. **Blinn-Phong Specular Highlight:**
   $$I_{\text{spec}} = \max(0, \; \vec{N} \cdot \vec{H})^\alpha, \quad \vec{H} = \frac{\vec{L} + \vec{V}}{\|\vec{L} + \vec{V}\|_2} \quad (\alpha = 32.0)$$
5. **Photographic ACES Tone Mapping:**
   Linear radiance is compressed using the cinematic ACES curve before display:
   $$f(x) = \frac{x (2.51 x + 0.03)}{x (2.43 x + 0.59) + 0.14}, \quad I_{\text{display}} = \left[\text{clip}(f(x), 0, 1)\right]^{\frac{1}{2.2}} \times 255$$

---

### 2.3 Robust 3D Spatial Tracking & Adaptive Calibration (Level 03)

#### A. Directional Sub-Pixel Fingertip Extrapolation
Instead of placing the virtual bulb over the finger pad, the system projects the bulb onto the fingernail tip along the terminal phalangeal vector:
$$\vec{P}_{\text{tip}} = \vec{L}_8 + \min\left(0.35 \cdot \|\vec{L}_8 - \vec{L}_7\|, \; 0.025\right) \cdot \frac{\vec{L}_8 - \vec{L}_7}{\|\vec{L}_8 - \vec{L}_7\|}$$
When pinching, the light source snaps to the centroid: $\vec{P}_{\text{pinch}} = \frac{1}{2}(\vec{L}_8 + \vec{L}_4)$.

#### B. Rotation-Invariant Palm Area Proximity ($Z$)
Single bone lengths shrink when the hand pitches or yaws, creating false depth movement. Our engine solves this by calculating the **Shoelace Polygon Area** across five structural joints (Wrist $\vec{L}_0$, Index MCP $\vec{L}_5$, Middle MCP $\vec{L}_9$, Ring MCP $\vec{L}_{13}$, Pinky MCP $\vec{L}_{17}$):
$$\text{Area} = \frac{1}{2} \left| \sum_{i=0}^{4} \left(x_i y_{i+1} - x_{i+1} y_i\right) \right|$$
$$\text{Hand Scale Metric } S = \sqrt{\max(\text{Area}, \; 10^{-8})}$$
Because rotation shrinks one axis while expanding another, the enclosed 2D area remains stable under hand tilt, isolating genuine camera distance changes.

#### C. Adaptive Moving-Window Depth Calibrator
A self-normalizing runtime class maps hand scale $S \in [S_{\min}, S_{\max}] \rightarrow Z \in [0.10, 0.65]$. A slow range decay coefficient ($\lambda = 0.001$) relaxes boundaries over time, letting the system auto-calibrate for different users or desk distances. A manual reset hook (`Key 'r'`) instantly resets bounds for new participants.

---

### 2.4 Bounded Radial Perspective Occlusion Shadows (Level 04)
Unlike flat affine translations that cast parallel shadows across the whole image, our engine computes individual **radial perspective rays** for every pixel:
1. **Radial Unit Vector:**
   $$\vec{u}_{\text{dir}}(x, y) = \frac{\vec{P}(x, y) - \vec{L}_{xy}}{\|\vec{P}(x, y) - \vec{L}_{xy}\|_2}$$
2. **Perspective Depth Scaling:**
   $$\text{Scale}_Z = \text{clip}\left(\frac{0.30}{\max(Z_{\text{light}}, 0.10)}, \; 0.7, \; 1.85\right)$$
3. **Distance-Constrained Remapping:**
   Displacement magnitude is clamped to prevent unnatural silhouette distortion:
   $$\Delta \vec{P} = \vec{u}_{\text{dir}} \cdot \min\left(0.16 \cdot d_{\text{light}} \cdot \text{Scale}_Z, \; 18.0 \cdot \text{Scale}_Z \cdot k_{\text{burst}}\right)$$
4. **Differential Depth Occlusion Gate:**
   $$\text{Shadow}_{\text{raw}} = \text{clip}\left(\left[\mathcal{W}(\text{Mask}_{\text{fg}}) - \text{Mask}_{\text{fg}}\right] \times \max(Z - 0.35, 0) \times 3.5, \; 0.0, \; 1.0\right)$$
   This guarantees shadows only fall onto deeper background geometry, never on the foreground user.

---

### 2.5 Multi-Light Independent Tracking & Atmospheric Volumetrics (Level 05)
1. **Spatial Biometric Disambiguation:** Detections with wrist distance $\|\vec{W}_A - \vec{W}_B\|_2 < 0.22$ are merged to reject single-hand ghost duplicates. When two distinct hands are confirmed, left-to-right sorting assigns independent control of Light 1 (Warm Gold Key) and Light 2 (Ice Blue Fill).
2. **Atmospheric God Rays (Volumetric Haze):**
   Radial atmospheric scattering is evaluated in screen space:
   $$H(x, y) = \exp\left(-\frac{\|\vec{P}(x, y) - \vec{L}_{xy}\|^2}{0.012}\right) \times I_{\text{light}} \times k_{\text{haze}}$$
   Scattering is attenuated by the occlusion shadow mask, casting visible volumetric light rays into the room.

---

## 3. Frame-Rate (FPS) & Latency Engineering
* **Non-Blocking Threaded I/O:** `CameraStream` runs on DirectShow with MJPG negotiation at 60 Hz, decoupling camera shutter latency from rendering.
* **Asynchronous Neural Offload:** Depth inference runs in `BackgroundDepthWorker` on a dedicated thread, eliminating ViT forward pass stalls from the main loop.
* **Multi-Scale Decoupled Shading:** Albedo is sampled at full native resolution ($640 \times 480$), while geometry and ray-marched shading execute over a lightweight coordinate buffer ($240 \times 180$) with bilinear reconstruction, saving ~70% computational overhead.
* **Token-Optimized Tensor Size:** Depth input is resized to $182 \times 182$ (exactly $13 \times 13$ ViT patches), maximizing throughput on NVIDIA RTX Tensor Cores.
* **Temporal EMA Filtering:** An Exponential Moving Average ($\alpha = 0.65$) eliminates neural depth flicker without motion ghosting.