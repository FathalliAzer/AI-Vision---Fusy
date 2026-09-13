import cv2
import torch
import numpy as np
import time
from depth_anything_v2.dpt import DepthAnythingV2

# Global interactive light state (X, Y, Z)
# Normalized coordinates: X in [0, 1], Y in [0, 1], Z in [0, 1]
light_pos = [0.5, 0.4, 0.35]  
is_dragging = False

def mouse_callback(event, x, y, flags, param):
    global light_pos, is_dragging
    w, h = param['width'], param['height']
    
    # If mouse is on the left half of the display or single window
    if event == cv2.EVENT_LBUTTONDOWN:
        is_dragging = True
        light_pos[0] = np.clip(x / w, 0.0, 1.0)
        light_pos[1] = np.clip(y / h, 0.0, 1.0)
    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        light_pos[0] = np.clip(x / w, 0.0, 1.0)
        light_pos[1] = np.clip(y / h, 0.0, 1.0)
    elif event == cv2.EVENT_LBUTTONUP:
        is_dragging = False
    elif event == cv2.EVENT_MOUSEWHEEL:
        # Mouse wheel adjusts Z (depth) of the light
        delta = 0.05 if flags > 0 else -0.05
        light_pos[2] = float(np.clip(light_pos[2] + delta, 0.05, 1.0))

def compute_normals_fast(depth_map, scale=4.0):
    """Vectorized calculation of 3D surface normals"""
    # Sobel kernels for gradient
    dzdx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3) * scale
    dzdy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3) * scale

    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(depth_map, dtype=np.float32)

    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    norm = np.maximum(norm, 1e-6)
    
    nx /= norm
    ny /= norm
    nz /= norm

    return np.stack((nx, ny, nz), axis=-1)

def apply_blinn_phong_relighting(frame, depth_map, normals, light_pos, 
                                 ambient=0.25, diffuse_intensity=1.1, 
                                 specular_intensity=0.8, shininess=32.0):
    h, w = depth_map.shape
    
    # 1. Construct 3D pixel grid (X, Y, Z) normalized to [0, 1]
    xs = np.linspace(0, 1, w, dtype=np.float32)
    ys = np.linspace(0, 1, h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    
    # 3D surface point for every pixel
    P_surf = np.stack((grid_x, grid_y, depth_map), axis=-1)

    # 2. Light Vector L = Light_Pos - P_surf
    L = np.array(light_pos, dtype=np.float32) - P_surf
    dist = np.sqrt(np.sum(L**2, axis=-1, keepdims=True))
    dist = np.maximum(dist, 1e-5)
    L_unit = L / dist

    # Distance attenuation: light falls off smoothly
    # radius scale can be tweaked for spread
    attenuation = 1.0 / (1.0 + 3.0 * dist + 12.0 * (dist**2))
    attenuation = np.squeeze(attenuation, axis=-1)

    # 3. Lambertian Diffuse = max(0, N . L)
    # Normals: (H, W, 3), L_unit: (H, W, 3)
    N_dot_L = np.sum(normals * L_unit, axis=-1)
    diffuse = np.maximum(N_dot_L, 0.0)

    # 4. Blinn-Phong Specular Highlight
    # View direction V is facing the screen (0, 0, 1)
    # Halfway vector H = (L + V) / ||L + V||
    V = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    H = L_unit + V
    H_norm = np.sqrt(np.sum(H**2, axis=-1, keepdims=True))
    H_unit = H / np.maximum(H_norm, 1e-5)

    N_dot_H = np.sum(normals * H_unit, axis=-1)
    specular = np.maximum(N_dot_H, 0.0) ** shininess

    # 5. Composite illumination
    # Normalize original image to [0, 1]
    img_float = frame.astype(np.float32) / 255.0

    # Total diffuse & ambient multiplier per pixel
    lighting_factor = ambient + (diffuse * diffuse_intensity * attenuation)
    lighting_factor = np.expand_dims(lighting_factor, axis=-1)

    # Specular term (pure white/yellow highlight)
    specular_highlight = (specular * specular_intensity * attenuation)
    specular_highlight = np.expand_dims(specular_highlight, axis=-1)

    # Apply lighting equation
    light_color = np.array([1.0, 0.95, 0.85], dtype=np.float32) # warm white point light
    relit = (img_float * lighting_factor) + (specular_highlight * light_color)
    relit = np.clip(relit * 255.0, 0.0, 255.0).astype(np.uint8)

    # Debug shading view (just the lighting without texture)
    pure_shading = np.clip((diffuse * attenuation)[:, :, None] * 255.0 + specular_highlight * 255.0, 0, 255).astype(np.uint8)
    pure_shading = cv2.applyColorMap(pure_shading, cv2.COLORMAP_BONE)

    return relit, pure_shading

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    
    print("Loading Depth Anything V2...")
    model = DepthAnythingV2(**model_configs['vits'])
    model.load_state_dict(torch.load('weights/depth_anything_v2_vits.pth', map_location='cpu', weights_only=True))
    model = model.to(device).eval()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("Level 02 - Dynamic Relighting")
    cv2.setMouseCallback("Level 02 - Dynamic Relighting", mouse_callback, {'width': 640, 'height': 480})

    alpha = 0.7
    smooth_depth = None
    prev_time = time.time()
    fps_history = []
    view_mode = 1 # 1: Relit Scene, 2: Shading Only, 3: Normals, 4: Split View

    print("\n" + "="*50)
    print("LEVEL 02: DYNAMIC RELIGHTING READY")
    print(" -> CLICK & DRAG mouse to move the light in 2D (X, Y)")
    print(" -> SCROLL MOUSE WHEEL or press 'w' / 's' to move light in Depth (Z)")
    print(" -> Press '1', '2', '3', '4' to switch viewing modes")
    print(" -> Press 'q' to quit")
    print("="*50 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 1. Fast Depth Inference
        with torch.inference_mode():
            if device == 'cuda':
                with torch.autocast('cuda', dtype=torch.float16):
                    raw_depth = model.infer_image(frame, input_size=266)
            else:
                raw_depth = model.infer_image(frame, input_size=266)

        # 2. Normalize Depth (Inverted: closer objects have lower Z, farther have higher Z for 3D physics)
        d_min, d_max = raw_depth.min(), raw_depth.max()
        if d_max - d_min > 1e-5:
            # High depth model values = near. Invert so near = small Z, far = large Z
            norm_depth = 1.0 - ((raw_depth - d_min) / (d_max - d_min))
        else:
            norm_depth = np.zeros_like(raw_depth)

        # 3. Temporal EMA
        if smooth_depth is None:
            smooth_depth = norm_depth.copy()
        else:
            smooth_depth = alpha * norm_depth + (1.0 - alpha) * smooth_depth

        # 4. Compute Surface Normals
        normals = compute_normals_fast(smooth_depth, scale=4.0)

        # 5. Apply Lambertian + Specular Relighting
        relit_frame, pure_shading = apply_blinn_phong_relighting(
            frame, smooth_depth, normals, light_pos,
            ambient=0.2, diffuse_intensity=1.2, specular_intensity=0.9, shininess=24.0
        )

        # Draw a visual glowing orb at the virtual light position (X, Y)
        lx, ly = int(light_pos[0] * w), int(light_pos[1] * h)
        cv2.circle(relit_frame, (lx, ly), 12, (255, 255, 255), -1)
        cv2.circle(relit_frame, (lx, ly), 18, (120, 220, 255), 2)

        # FPS Calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # Normal map visualization
        normal_vis = ((normals * 0.5 + 0.5) * 255.0).astype(np.uint8)
        normal_vis_bgr = cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR)

        # Display selection
        if view_mode == 1:
            display = relit_frame
            title = "Level 02: Relit Scene (Lambertian + Specular)"
        elif view_mode == 2:
            display = pure_shading
            title = "Level 02: Pure Shading / Illumination Field"
        elif view_mode == 3:
            display = normal_vis_bgr
            title = "Level 01: Surface Normal Map"
        else:
            display = np.hstack((relit_frame, normal_vis_bgr))
            title = "Relit Feed (Left) | Normal Map (Right)"

        # Info overlays
        cv2.putText(display, f"FPS: {avg_fps:.1f}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(display, f"Light 3D: X={light_pos[0]:.2f} Y={light_pos[1]:.2f} Z={light_pos[2]:.2f}", 
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, title, (20, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Level 02 - Dynamic Relighting", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('1'):
            view_mode = 1
        elif key == ord('2'):
            view_mode = 2
        elif key == ord('3'):
            view_mode = 3
        elif key == ord('4'):
            view_mode = 4
        elif key == ord('w'): # Move light closer
            light_pos[2] = float(np.clip(light_pos[2] - 0.05, 0.05, 1.0))
        elif key == ord('s'): # Move light further
            light_pos[2] = float(np.clip(light_pos[2] + 0.05, 0.05, 1.0))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()