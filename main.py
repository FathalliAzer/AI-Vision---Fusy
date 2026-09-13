"""AI & Vision Challenge — Level 05 Multi-Light Relighting Engine.

Controls:
    - Hand 1: Controls Light 1 (Solar Gold)
    - Hand 2: Controls Light 2 (Ice Blue)
    - Pinch (Thumb + Index): Supernova Burst & High-Contrast Shadow
    - 'b': Toggle Cinematic Blackout / Horror Chamber Mode (Level Infinity)
    - 'f': 3D Spatial World Anchoring (Pin / Unpin Light 1 in mid-air)
    - 'c': Toggle color rig
    - 'r': Recalibrate depth (Z) tracking — use if a new person steps in
    - Keys 1-6: Jury verification views
    - 'q' or ESC: Quit
"""

from __future__ import annotations

import os
import sys
import logging
import warnings
import io
import contextlib

# 1. Silence all Python-level warnings and third-party library logging
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

# 2. Silence TensorFlow / MediaPipe low-level backend logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOG_LEVEL"] = "error"

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# 3. Clean imports without console output noise
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        import mediapipe.python.solutions.hands as mp_hands
    except ImportError:
        import mediapipe as mp
        mp_hands = mp.solutions.hands

    from depth_anything_v2.dpt import DepthAnythingV2


CAMERA_W, CAMERA_H = 640, 480
RENDER_W, RENDER_H = 240, 180       # Fast shading grid for >= 30 FPS
DEPTH_INPUT_SIZE = 182             # Optimal ViT patch alignment
MAX_LIGHTS = 2


# ============================================================
# OFFICIAL DEPTH ANYTHING V2 SPECTRAL_R COLORMAP TABLE
# ============================================================
def build_depth_anything_lut() -> np.ndarray:
    """Generates the official Depth Anything V2 'Spectral_r' 256-color BGR lookup table."""
    try:
        import matplotlib
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255)[:, ::-1].astype(np.uint8)
        return lut
    except Exception:
        dummy = np.arange(256, dtype=np.uint8)[:, None]
        return cv2.applyColorMap(dummy, cv2.COLORMAP_TURBO).reshape(256, 3)

SPECTRAL_LUT = build_depth_anything_lut()


def make_depth_visualization(
    smooth_depth: np.ndarray,
    camera_width: int,
    camera_height: int,
) -> np.ndarray:
    """Official Depth Anything V2 Spectral Map: Near = Yellow/Red, Far = Indigo/Purple."""
    depth_vis_uint8 = np.clip((1.0 - smooth_depth) * 255.0, 0, 255).astype(np.uint8)
    spectral_depth = SPECTRAL_LUT[depth_vis_uint8]
    return cv2.resize(spectral_depth, (camera_width, camera_height), interpolation=cv2.INTER_LINEAR)


def normalize_vectors(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def srgb_to_linear(image_bgr: np.ndarray) -> np.ndarray:
    return np.power(image_bgr.astype(np.float32) / 255.0, 2.2)


def aces_tonemap(linear_bgr: np.ndarray) -> np.ndarray:
    x = np.maximum(linear_bgr, 0.0)
    mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return np.clip(np.power(np.clip(mapped, 0.0, 1.0), 1.0 / 2.2) * 255.0, 0, 255).astype(np.uint8)


# ============================================================
# 1. NON-BLOCKING CAMERA STREAM
# ============================================================
class CameraStream:
    def __init__(self, src: int = 0, width: int = CAMERA_W, height: int = CAMERA_H):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open camera.")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.004)
                continue
            with self._lock:
                self._frame = cv2.flip(frame, 1)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._running = False
        self.cap.release()


# ============================================================
# 2. BACKGROUND DEPTH WORKER
# ============================================================
class BackgroundDepthWorker:
    def __init__(self, model: DepthAnythingV2, device: str):
        self.model = model
        self.device = device
        self._frame: Optional[np.ndarray] = None
        self._depth = np.full((RENDER_H, RENDER_W), 0.50, np.float32)
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _clean_depth(raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        d_min, d_max = raw.min(), raw.max()
        if d_max - d_min > 1e-5:
            norm = 1.0 - ((raw - d_min) / (d_max - d_min))
        else:
            norm = np.zeros_like(raw)
        return norm.astype(np.float32)

    def push(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame.copy()

    def pull(self) -> np.ndarray:
        with self._lock:
            return self._depth.copy()

    def _run(self) -> None:
        while self._running:
            with self._lock:
                frame = self._frame
                self._frame = None
            if frame is None:
                time.sleep(0.002)
                continue

            with torch.inference_mode():
                if self.device == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        raw = self.model.infer_image(frame, input_size=DEPTH_INPUT_SIZE)
                else:
                    raw = self.model.infer_image(frame, input_size=DEPTH_INPUT_SIZE)

            depth = self._clean_depth(raw)
            depth = cv2.resize(depth, (RENDER_W, RENDER_H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            with self._lock:
                self._depth = (self._depth * 0.30 + depth * 0.70).astype(np.float32)

    def stop(self) -> None:
        self._running = False


# ============================================================
# 3. SPATIAL ANCHORING & PERSISTENT MULTI-HAND TRACKING
# ============================================================
@dataclass
class HandTarget:
    u: float
    v: float
    z: float
    pinching: bool


@dataclass
class LightState:
    u: float
    v: float
    z: float
    intensity: float = 0.0
    seen_at: float = -100.0
    pinching: bool = False
    
    # LEVEL INFINITY: Spatial World Anchoring
    is_anchored: bool = False
    anchor_intensity: float = 1.35

    def update(self, target: HandTarget, now: float) -> None:
        if self.is_anchored:
            self.seen_at = now
            return

        self.u = self.u * 0.15 + target.u * 0.85
        self.v = self.v * 0.15 + target.v * 0.85
        self.z = self.z * 0.55 + target.z * 0.45
        desired = 2.60 if target.pinching else 1.25
        self.intensity = self.intensity * 0.30 + desired * 0.70
        self.pinching = target.pinching
        self.seen_at = now

    def fade(self, now: float) -> None:
        if self.is_anchored:
            self.intensity = self.anchor_intensity
            self.pinching = False
            return
            
        if now - self.seen_at > 0.15:
            self.intensity *= 0.75
            self.pinching = False

    def active(self) -> bool:
        return self.is_anchored or (self.intensity > 0.05)


def compute_hand_scale(lm) -> float:
    """Shoelace formula palm polygon area for rotation-invariant distance estimation."""
    pts = np.array(
        [
            [lm[0].x, lm[0].y],
            [lm[5].x, lm[5].y],
            [lm[9].x, lm[9].y],
            [lm[13].x, lm[13].y],
            [lm[17].x, lm[17].y],
        ],
        dtype=np.float32,
    )
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return float(np.sqrt(max(area, 1e-8)))


class DepthCalibrator:
    def __init__(self, warmup_frames: int = 30, range_decay: float = 0.001):
        self.warmup_frames = warmup_frames
        self.range_decay = range_decay
        self.frame_count = 0
        self.scale_min: Optional[float] = None
        self.scale_max: Optional[float] = None

    def update_range(self, scale: float) -> None:
        self.scale_min = scale if self.scale_min is None else min(self.scale_min, scale)
        self.scale_max = scale if self.scale_max is None else max(self.scale_max, scale)
        self.frame_count += 1
        self.scale_min += (scale - self.scale_min) * self.range_decay
        self.scale_max += (scale - self.scale_max) * self.range_decay

    def normalize(self, scale: float) -> float:
        if self.frame_count < self.warmup_frames or self.scale_max is None or (self.scale_max - self.scale_min) < 1e-5:
            return float(np.clip((scale - 0.05) / (0.16 - 0.05), 0.0, 1.0))
        return float(np.clip((scale - self.scale_min) / (self.scale_max - self.scale_min), 0.0, 1.0))

    def reset(self) -> None:
        self.frame_count = 0
        self.scale_min = None
        self.scale_max = None


def match_hands_to_lights(
    results,
    lights: List[LightState],
    calibrators: List[DepthCalibrator],
    now: float,
) -> None:
    if not results.multi_hand_landmarks:
        for l in lights:
            l.fade(now)
        return

    candidates: List[Tuple[float, float, bool, np.ndarray, float]] = []

    for hand in results.multi_hand_landmarks:
        lm = hand.landmark
        idx_tip = np.array([lm[8].x, lm[8].y])
        idx_dip = np.array([lm[7].x, lm[7].y])
        thumb_tip = np.array([lm[4].x, lm[4].y])
        wrist = np.array([lm[0].x, lm[0].y])

        pinch_dist = float(np.linalg.norm(idx_tip - thumb_tip))
        is_pinching = pinch_dist < 0.082

        if is_pinching:
            target_pos = (idx_tip + thumb_tip) * 0.5
        else:
            point_dir = idx_tip - idx_dip
            dir_len = np.linalg.norm(point_dir)
            if dir_len > 1e-4:
                target_pos = idx_tip + (point_dir / dir_len) * min(dir_len * 0.35, 0.025)
            else:
                target_pos = idx_tip

        raw_scale = compute_hand_scale(lm)
        candidates.append((float(np.clip(target_pos[0], 0.01, 0.99)), float(np.clip(target_pos[1], 0.01, 0.99)), is_pinching, wrist, raw_scale))

    filtered: List[Tuple[float, float, bool, float]] = []
    kept_wrists: List[np.ndarray] = []

    for u, v, pinching, wrist, raw_scale in candidates:
        is_dup = any(np.linalg.norm(wrist - kw) < 0.22 for kw in kept_wrists)
        if not is_dup:
            is_dup = any(np.hypot(u - fu, v - fv) < 0.18 for fu, fv, _, _ in filtered)
        if not is_dup:
            filtered.append((u, v, pinching, raw_scale))
            kept_wrists.append(wrist)

    def make_target(entry: Tuple[float, float, bool, float], calibrator: DepthCalibrator) -> HandTarget:
        u, v, pinching, raw_scale = entry
        calibrator.update_range(raw_scale)
        z_norm = calibrator.normalize(raw_scale)
        z = float(np.clip(0.65 - z_norm * 0.55, 0.10, 0.65))
        return HandTarget(u=u, v=v, z=z, pinching=pinching)

    if lights[0].is_anchored:
        if len(filtered) >= 1:
            lights[1].update(make_target(filtered[0], calibrators[1]), now)
        else:
            lights[1].fade(now)
    else:
        if len(filtered) == 1:
            lights[0].update(make_target(filtered[0], calibrators[0]), now)
            lights[1].fade(now)
        elif len(filtered) >= 2:
            sorted_entries = sorted(filtered[:2], key=lambda e: e[0])
            lights[0].update(make_target(sorted_entries[0], calibrators[0]), now)
            lights[1].update(make_target(sorted_entries[1], calibrators[1]), now)


# ============================================================
# 4. GEOMETRY & PHYSICAL SHADOW CALCULATIONS
# ============================================================
def build_geometry(depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    depth_f32 = depth.astype(np.float32)
    h, w = depth_f32.shape
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
    uu, vv = np.meshgrid(xs, ys)
    points = np.dstack((uu, vv, depth_f32)).astype(np.float32)

    # 3D Surface Normals (Level 01)
    dzdx = cv2.Sobel(depth_f32, cv2.CV_32F, 1, 0, ksize=3) * 4.0
    dzdy = cv2.Sobel(depth_f32, cv2.CV_32F, 0, 1, ksize=3) * 4.0
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(depth_f32, dtype=np.float32)
    normals = normalize_vectors(np.dstack((nx, ny, nz)))
    return points, normals


def compute_geometry_shadow(depth: np.ndarray, light: LightState, blackout_mode: bool = False) -> np.ndarray:
    """
    Level 04: Highly Dynamic Point-Light Occlusion Shadows.
    Moves visibly and dramatically across X, Y, and Z as the hand moves.
    Deepens into pitch-black silhouette in Blackout Mode.
    """
    depth_f32 = depth.astype(np.float32)
    h, w = depth_f32.shape
    
    fg_mask = (depth_f32 < 0.62).astype(np.float32)

    clamped_z = max(float(light.z), 0.08)
    z_stretch = float(np.clip(0.32 / clamped_z, 0.75, 2.2))
    burst_mult = 1.35 if light.pinching else 1.0

    shift_scale = 52.0 * z_stretch * burst_mult
    dx = (0.5 - light.u) * shift_scale
    dy = (0.5 - light.v) * (shift_scale * 0.70)

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    projected = cv2.warpAffine(fg_mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    shadow_on_bg = np.clip(projected - fg_mask, 0.0, 1.0)

    blur_k = (19, 19) if (light.pinching or z_stretch > 1.5) else (11, 11)
    soft_shadow = cv2.GaussianBlur(shadow_on_bg, blur_k, 0)
    
    # In Blackout mode, shadows become near pitch-black (96% darkening)
    if blackout_mode:
        shadow_depth = 0.96
        floor_val = 0.04
    else:
        shadow_depth = 0.88 if light.pinching else 0.72
        floor_val = 0.12
        
    return np.clip(1.0 - soft_shadow * shadow_depth, floor_val, 1.0).astype(np.float32)


# ============================================================
# 5. SHADING ENGINE (WITH CINEMATIC BLACKOUT MODE)
# ============================================================
LIGHT_RIGS = [
    ("Warm Gold + Ice Blue", (0.35, 0.85, 1.00), (1.00, 0.85, 0.55)),
    ("Neon Purple + Cyan",   (0.95, 0.40, 1.00), (0.30, 0.95, 1.00)),
    ("Studio Neutral",       (0.90, 0.95, 1.00), (1.00, 0.95, 0.90)),
]


def shade_scene(
    frame: np.ndarray,
    depth: np.ndarray,
    lights: List[LightState],
    rig_index: int,
    blackout_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    depth_f32 = depth.astype(np.float32)
    small = cv2.resize(frame, (RENDER_W, RENDER_H), interpolation=cv2.INTER_AREA)
    albedo = srgb_to_linear(small)
    
    # In Blackout Mode, background camera albedo is dimmed to create true darkness
    if blackout_mode:
        albedo = albedo * 0.75

    points, normals = build_geometry(depth_f32)
    view = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    h, w = depth_f32.shape
    uu, vv = np.meshgrid(np.linspace(0.0, 1.0, w, dtype=np.float32), np.linspace(0.0, 1.0, h, dtype=np.float32))

    direct_bgr = np.zeros((h, w, 3), np.float32)
    haze_bgr = np.zeros_like(direct_bgr)
    combined_shadow = np.ones((h, w), np.float32)
    light_energy = np.zeros((h, w), np.float32)
    _, color_0, color_1 = LIGHT_RIGS[rig_index]

    any_pinching = any(l.active() and l.pinching for l in lights)

    for index, light in enumerate(lights):
        if not light.active():
            continue
        color = np.asarray(color_0 if index == 0 else color_1, dtype=np.float32)
        light_pos = np.array([light.u, light.v, light.z], dtype=np.float32)
        to_light = light_pos - points
        distance = np.maximum(np.linalg.norm(to_light, axis=-1), 0.035)
        direction = to_light / distance[..., None]

        # Diffuse
        ndotl = np.maximum(np.sum(normals * direction, axis=-1), 0.0)
        
        # Specular
        half_vector = normalize_vectors(direction + view)
        ndoth = np.maximum(np.sum(normals * half_vector, axis=-1), 0.0)
        specular = (ndoth ** 32.0) * (0.95 if light.pinching else 0.45)
        
        # Attenuation
        attenuation = (light.intensity * 1.65) / (1.0 + 2.5 * distance + 7.5 * (distance**2))
        
        # Dynamic Occlusion Shadow
        shadow = compute_geometry_shadow(depth_f32, light, blackout_mode=blackout_mode)

        diffuse_boost = 1.85 if light.pinching else 1.25
        direct = (ndotl * diffuse_boost + specular * 1.70) * attenuation * shadow
        direct_bgr += direct[..., None] * color
        light_energy += direct
        combined_shadow = np.minimum(combined_shadow, shadow)

        # Volumetric haze
        dist_2d = np.sqrt((uu - light.u) ** 2 + (vv - light.v) ** 2)
        haze = np.exp(- (dist_2d**2) / 0.012) * (0.24 if light.pinching else 0.09) * light.intensity
        haze_bgr += (haze * shadow)[..., None] * color

    # LEVEL INFINITY: Ambient light in Blackout mode plunges to near-zero (0.04)
    if blackout_mode:
        ambient_base = 0.03
    else:
        ambient_base = 0.12 if any_pinching else 0.22

    ambient_field = ambient_base * combined_shadow
    lit = albedo * (ambient_field[..., None] + direct_bgr) + haze_bgr
    final_small = aces_tonemap(lit)
    final = cv2.resize(final_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return final, combined_shadow, light_energy, normals


def draw_light_orbs(image: np.ndarray, lights: List[LightState], rig_index: int) -> None:
    _, color_0, color_1 = LIGHT_RIGS[rig_index]
    h, w = image.shape[:2]
    for i, light in enumerate(lights):
        if not light.active():
            continue
        bgr = tuple(int(v * 255) for v in (color_0 if i == 0 else color_1))
        x, y = int(light.u * (w - 1)), int(light.v * (h - 1))
        
        if light.is_anchored:
            radius = 20
            cv2.circle(image, (x, y), radius + 8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(image, (x, y), radius, bgr, -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(image, f"L{i+1}:PINNED", (x + 16, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            radius = 28 if light.pinching else 15
            orb_fill = (255, 100, 255) if light.pinching else bgr
            glow = image.copy()
            cv2.circle(glow, (x, y), radius * 2, orb_fill, -1, cv2.LINE_AA)
            cv2.addWeighted(glow, 0.25 if light.pinching else 0.15, image, 0.75 if light.pinching else 0.85, 0, dst=image)
            cv2.circle(image, (x, y), radius, orb_fill, 2, cv2.LINE_AA)
            cv2.circle(image, (x, y), 6, (255, 255, 255), -1, cv2.LINE_AA)
            txt = f"L{i+1}:BURST" if light.pinching else f"L{i+1}:ON"
            cv2.putText(image, txt, (x + 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, orb_fill, 2, cv2.LINE_AA)


def add_hud(image: np.ndarray, fps: float, lights: List[LightState], label: str, mode: str, blackout_mode: bool = False) -> None:
    h = image.shape[0]
    active = []
    for i, l in enumerate(lights):
        if l.is_anchored:
            active.append(f"L{i+1}:PINNED")
        elif l.active():
            active.append(f"L{i+1}:{'BURST' if l.pinching else 'LOCKED'}")
        else:
            active.append(f"L{i+1}:OFF")
            
    blackout_tag = " | [BLACKOUT]" if blackout_mode else ""
    cv2.rectangle(image, (10, 10), (430, 91), (5, 5, 5), -1)
    cv2.rectangle(image, (10, 10), (430, 91), (175, 175, 175), 1)
    cv2.putText(image, f"{fps:5.1f} FPS | {'  '.join(active)}{blackout_tag}", (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 255, 190), 2, cv2.LINE_AA)
    cv2.putText(image, label, (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(image, mode, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 255, 225), 2, cv2.LINE_AA)


def load_depth_model(device: str) -> DepthAnythingV2:
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    candidates = [
        os.path.join(os.getcwd(), "weights", "depth_anything_v2_vits.pth"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "depth_anything_v2_vits.pth"),
    ]
    weights = next((p for p in candidates if os.path.isfile(p)), None)
    if weights is None:
        raise FileNotFoundError("Missing weights/depth_anything_v2_vits.pth.")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    print("=" * 65)
    print(" AI & Vision Challenge — Level 05 Multi-Light Production Engine")
    print(f" Device: {device.upper()} | Renderer: {RENDER_W}x{RENDER_H} | Camera: {CAMERA_W}x{CAMERA_H}")
    print("=" * 65)

    model = load_depth_model(device)
    camera = CameraStream()
    worker = BackgroundDepthWorker(model, device)
    
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_LIGHTS,
            model_complexity=0,
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
        )
    
    lights = [LightState(0.35, 0.45, 0.35), LightState(0.65, 0.45, 0.35)]
    calibrators = [DepthCalibrator(), DepthCalibrator()]
    smooth_depth = np.full((RENDER_H, RENDER_W), 0.50, np.float32)
    rig_index, view_mode, hand_skip = 0, 1, 0
    blackout_mode = False
    last_time, fps_ema = time.perf_counter(), 0.0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            worker.push(frame)
            now = time.perf_counter()
            hand_skip += 1

            if hand_skip % 2 == 0:
                thumbnail = cv2.resize(frame, (240, 180), interpolation=cv2.INTER_AREA)
                result = hands.process(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
                match_hands_to_lights(result, lights, calibrators, now)

            fresh_depth = worker.pull()
            smooth_depth = (smooth_depth * 0.35 + fresh_depth * 0.65).astype(np.float32)
            
            final, shadow, direct, normals = shade_scene(
                frame, smooth_depth, lights, rig_index, blackout_mode=blackout_mode
            )
            draw_light_orbs(final, lights, rig_index)

            # View Selector for Jury
            if view_mode == 1:
                display, mode = final, "Level 05 — Multi-Light Relighting + Volumetric Scattering"
            elif view_mode == 2:
                display = cv2.resize(cv2.applyColorMap((shadow * 255).astype(np.uint8), cv2.COLORMAP_BONE), (CAMERA_W, CAMERA_H))
                mode = "Level 04 — Geometry-Aware Dynamic Occlusion Shadows"
            elif view_mode == 3:
                display = cv2.resize(cv2.applyColorMap(np.clip(direct * 145.0, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO), (CAMERA_W, CAMERA_H))
                mode = "Level 02 — Inverse-Square Diffuse and Specular Field"
            elif view_mode == 4:
                norm_rgb = (normals * 0.5 + 0.5) * 255.0
                norm_bgr = cv2.cvtColor(norm_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                display = cv2.resize(norm_bgr, (CAMERA_W, CAMERA_H), interpolation=cv2.INTER_LINEAR)
                mode = "Level 01 — Depth-Derived 3D Surface Normals"
            elif view_mode == 5:
                display = np.hstack((frame, final))
                mode = "Live Verification — Original Camera vs Relit Reality"
            elif view_mode == 6:
                display = make_depth_visualization(smooth_depth, CAMERA_W, CAMERA_H)
                mode = "Depth Anything V2 — Official Spectral Depth Map"
            else:
                display, mode = final, "Level 05 — Multi-Light Relighting + Volumetric Scattering"

            elapsed = max(time.perf_counter() - last_time, 1e-5)
            last_time = time.perf_counter()
            instant_fps = 1.0 / elapsed
            fps_ema = instant_fps if fps_ema == 0 else fps_ema * 0.88 + instant_fps * 0.12
            add_hud(display, fps_ema, lights, LIGHT_RIGS[rig_index][0], mode, blackout_mode=blackout_mode)
            cv2.imshow("AI & Vision Challenge | Realistic Level 05", display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                rig_index = (rig_index + 1) % len(LIGHT_RIGS)
            if key == ord("r"):
                for c in calibrators:
                    c.reset()
            # LEVEL INFINITY: Cinematic Blackout Toggle
            if key == ord("b"):
                blackout_mode = not blackout_mode
            # LEVEL INFINITY: Spatial World Anchoring
            if key == ord("f"):
                lights[0].is_anchored = not lights[0].is_anchored
                if lights[0].is_anchored:
                    lights[0].anchor_intensity = max(lights[0].intensity, 1.30)
            if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
                view_mode = int(chr(key))
    finally:
        hands.close()
        worker.stop()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()