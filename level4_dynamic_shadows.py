import cv2
import torch
import numpy as np
import time
try:
    import mediapipe.python.solutions.hands as mp_hands
    import mediapipe.python.solutions.drawing_utils as mp_drawing
except ImportError:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

from depth_anything_v2.dpt import DepthAnythingV2

def compute_fast_normals(depth_map, scale=4.0):
    """Computes normalized surface normals from depth map"""
    dzdx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3) * scale
    dzdy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3) * scale
    
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(depth_map, dtype=np.float32)
    
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    norm = np.maximum(norm, 1e-6)
    return nx / norm, ny / norm, nz / norm

def compute_dynamic_shadows(depth_map, light_x, light_y, light_z):
    """
    Level 04 Engine: Computes geometry-aware dynamic occlusion shadows.
    depth_map: [0.0 (near) to 1.0 (far)]
    light_x, light_y: [0.0, 1.0] light position in screen space
    """
    h, w = depth_map.shape
    
    # 1. Identify foreground mask (subjects closer to camera)
    # Median or relative threshold separates foreground user from background
    fg_mask = (depth_map < 0.65).astype(np.float32)
    
    # 2. Compute shadow projection displacement vector
    # Shadow moves AWAY from the light
    dx = (0.5 - light_x) * 60.0  # Max shadow shift in pixels
    dy = (0.5 - light_y) * 40.0
    
    # Affine transformation matrix to shift and scale the shadow
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    projected_shadow = cv2.warpAffine(fg_mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # 3. Geometry-awareness: Shadows can ONLY fall on deeper surfaces (the background)
    # The shadow should not fall on the foreground caster itself
    shadow_on_bg = np.clip(projected_shadow - fg_mask, 0.0, 1.0)
    
    # 4. Penumbra (Soft shadow blur)
    soft_shadow = cv2.GaussianBlur(shadow_on_bg, (25, 25), 0)
    
    # Multiplier: 1.0 = full light, 0.35 = deep shadow
    shadow_factor = 1.0 - (soft_shadow * 0.65)
    return shadow_factor

def apply_relighting_and_shadows(frame, depth_map, light_pos, light_color=(1.0, 0.95, 0.82)):
    """
    Applies Diffuse + Specular + Dynamic Shadows
    """
    h, w = depth_map.shape
    lx, ly, lz = light_pos
    
    # 1. Compute Normals
    nx, ny, nz = compute_fast_normals(depth_map, scale=3.5)
    
    # 2. Pixel coordinates
    xs = np.linspace(0, 1, w, dtype=np.float32)
    ys = np.linspace(0, 1, h, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    
    # 3. Vector to light
    dx = lx - gx
    dy = ly - gy
    dz = lz - depth_map
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    dist = np.maximum(dist, 1e-4)
    
    lx_unit = dx / dist
    ly_unit = dy / dist
    lz_unit = dz / dist
    
    # Distance attenuation (smooth falloff)
    attenuation = 1.0 / (1.0 + 2.0 * dist + 8.0 * (dist**2))
    
    # 4. Lambertian Diffuse: max(0, N . L)
    diffuse = np.maximum(nx * lx_unit + ny * ly_unit + nz * lz_unit, 0.0)
    
    # 5. Specular (Blinn-Phong)
    # View direction V = [0, 0, 1]
    hx = lx_unit
    hy = ly_unit
    hz = lz_unit + 1.0
    h_norm = np.sqrt(hx**2 + hy**2 + hz**2) + 1e-5
    hx /= h_norm
    hy /= h_norm
    hz /= h_norm
    
    specular = np.maximum(nx * hx + ny * hy + nz * hz, 0.0) ** 24.0
    
    # 6. Dynamic Occlusion Shadows (Level 04)
    shadow_map = compute_dynamic_shadows(depth_map, lx, ly, lz)
    
    # 7. Final Composite
    img_f = frame.astype(np.float32)
    
    # Base ambient + diffuse modulated by the shadow map
    ambient = 0.35
    total_diffuse = ambient + (diffuse * attenuation * 1.5 * shadow_map)
    
    # Specular shine
    spec_highlight = (specular * attenuation * 180.0 * shadow_map)[:, :, None] * np.array(light_color)
    
    relit = (img_f * total_diffuse[:, :, None]) + spec_highlight
    relit = np.clip(relit, 0, 255).astype(np.uint8)
    
    return relit, shadow_map

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
    print(f"Loading Depth Model on {device}...")
    depth_model = DepthAnythingV2(**model_configs['vits'])
    depth_model.load_state_dict(torch.load('weights/depth_anything_v2_vits.pth', map_location='cpu', weights_only=True))
    depth_model = depth_model.to(device).eval()

    W, H = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    alpha = 0.75
    smooth_depth = None
    light_pos = [0.5, 0.4, 0.3] # X, Y, Z
    prev_time = time.time()
    fps_history = []
    
    # Frame skip counter to keep pipeline responsive
    frame_count = 0
    cached_depth = None

    print("\n" + "="*55)
    print("LEVEL 04: DYNAMIC SHADOWS + SPATIAL GESTURES")
    print(" -> Raise your hand to move the virtual lightbulb!")
    print(" -> Move left/right to watch dynamic shadows cast behind you")
    print(" -> Pinch thumb & index for electric purple light!")
    print(" -> Press 'q' to quit")
    print("="*55 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_count += 1

        # 1. MediaPipe Hand Tracking
        rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(rgb_small)
        
        hand_detected = False
        is_pinching = False

        if hand_results.multi_hand_landmarks:
            hand_detected = True
            lm = hand_results.multi_hand_landmarks[0].landmark
            
            # Target Light position follows fingertip
            target_x = lm[8].x
            target_y = lm[8].y
            
            # Estimate Z depth from hand scale (wrist to middle knuckle)
            hand_span = np.sqrt((lm[0].x - lm[9].x)**2 + (lm[0].y - lm[9].y)**2)
            # Map span to [0.15, 0.65] depth
            target_z = float(np.clip(0.65 - (hand_span * 1.8), 0.15, 0.65))

            # Smooth light movement
            light_pos[0] = light_pos[0] * 0.4 + target_x * 0.6
            light_pos[1] = light_pos[1] * 0.4 + target_y * 0.6
            light_pos[2] = light_pos[2] * 0.6 + target_z * 0.4

            # Pinch check (Thumb tip #4 and Index tip #8)
            pinch_gap = np.sqrt((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2)
            if pinch_gap < 0.055:
                is_pinching = True

        # 2. Depth Inference (Runs every frame with AMP for smoothness)
        with torch.inference_mode():
            if device == 'cuda':
                with torch.autocast('cuda', dtype=torch.float16):
                    raw_depth = depth_model.infer_image(frame, input_size=252)
            else:
                raw_depth = depth_model.infer_image(frame, input_size=252)

        # Normalize Depth (0 = Near, 1 = Far)
        d_min, d_max = raw_depth.min(), raw_depth.max()
        if d_max - d_min > 1e-5:
            norm_depth = 1.0 - ((raw_depth - d_min) / (d_max - d_min))
        else:
            norm_depth = np.zeros_like(raw_depth)

        # Temporal EMA filter
        if smooth_depth is None:
            smooth_depth = norm_depth.copy()
        else:
            smooth_depth = alpha * norm_depth + (1.0 - alpha) * smooth_depth

        # 3. Dynamic Relighting + Level 04 Dynamic Shadows
        light_col = (1.0, 0.4, 0.95) if is_pinching else (1.0, 0.92, 0.8)
        relit_frame, shadow_map = apply_relighting_and_shadows(
            frame, smooth_depth, light_pos, light_color=light_col
        )

        # 4. Render glowing light orb
        lx = int(np.clip(light_pos[0] * w, 0, w - 1))
        ly = int(np.clip(light_pos[1] * h, 0, h - 1))
        orb_col = (255, 120, 255) if is_pinching else (100, 225, 255)
        cv2.circle(relit_frame, (lx, ly), 14, (255, 255, 255), -1)
        cv2.circle(relit_frame, (lx, ly), 22, orb_col, 3)

        # FPS Calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # Overlays
        cv2.putText(relit_frame, f"FPS: {avg_fps:.1f}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        status = "Hand Locked" if hand_detected else "No Hand (Auto Center)"
        cv2.putText(relit_frame, f"Status: {status}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255) if hand_detected else (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(relit_frame, f"Light 3D: X={light_pos[0]:.2f} Y={light_pos[1]:.2f} Z={light_pos[2]:.2f}", 
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(relit_frame, "Level 04: Dynamic Occlusion Shadows Active", (20, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

        cv2.imshow("Level 04 - Dynamic Shadows & Gestures", relit_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()