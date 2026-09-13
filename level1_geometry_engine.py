import cv2
import torch
import numpy as np
import time
from depth_anything_v2.dpt import DepthAnythingV2

def compute_surface_normals(depth_map, smooth_ksize=5, gradient_scale=5.0):
    """
    Computes continuous 3D surface normal vectors N(x, y) from depth gradients.
    
    depth_map: 2D numpy array [0.0, 1.0]
    gradient_scale: scales the steepness of normals (higher = more pronounced edges)
    """
    # 1. Subtle blur to eliminate micro-noise on depth gradients
    if smooth_ksize > 1:
        smoothed_depth = cv2.GaussianBlur(depth_map, (smooth_ksize, smooth_ksize), 0)
    else:
        smoothed_depth = depth_map

    # 2. Compute spatial gradients along X and Y using Sobel operators
    # CV_32F allows precise gradient calculation
    dzdx = cv2.Sobel(smoothed_depth, cv2.CV_32F, 1, 0, ksize=3) * gradient_scale
    dzdy = cv2.Sobel(smoothed_depth, cv2.CV_32F, 0, 1, ksize=3) * gradient_scale

    # 3. Construct 3D Normal Vector: N = [-dZ/dx, -dZ/dy, 1.0]
    # (Note: In image space, Y goes down, so we adjust signs for camera space)
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(depth_map, dtype=np.float32)

    # 4. Normalize normal vectors: N / ||N||
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    norm = np.maximum(norm, 1e-6) # Avoid division by zero
    
    nx /= norm
    ny /= norm
    nz /= norm

    # Stack into an (H, W, 3) normal vector field
    normals = np.stack((nx, ny, nz), axis=-1)

    # 5. Map from [-1.0, 1.0] to [0, 255] RGB color for visual benchmark
    # R: X-tilt, G: Y-tilt, B: pointing at camera (Z)
    normal_vis = ((normals * 0.5 + 0.5) * 255.0).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV display
    normal_vis_bgr = cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR)

    return normals, normal_vis_bgr

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True # Accelerate fixed-size GPU convolutions

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    
    print(f"Loading Depth Anything V2 on {device}...")
    model = DepthAnythingV2(**model_configs['vits'])
    state_dict = torch.load('weights/depth_anything_v2_vits.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print("Geometry Engine Initialized!")

    cap = cv2.VideoCapture(0)
    # Set to 640x480 resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Camera access failed.")
        return

    alpha = 0.75 # Temporal EMA smoothing factor
    smooth_depth = None
    prev_time = time.time()
    fps_history = []

    print("\n[LEVEL 01: GEOMETRY ENGINE RUNNING]")
    print("Press 'q' or ESC to exit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # 1. Faster Inference: input_size=266 (Multiple of 14, ~2x speedup)
        with torch.inference_mode():
            if device == 'cuda':
                with torch.autocast('cuda', dtype=torch.float16):
                    raw_depth = model.infer_image(frame, input_size=266)
            else:
                raw_depth = model.infer_image(frame, input_size=266)

        # 2. Normalize Depth
        d_min, d_max = raw_depth.min(), raw_depth.max()
        if d_max - d_min > 1e-5:
            norm_depth = (raw_depth - d_min) / (d_max - d_min)
        else:
            norm_depth = np.zeros_like(raw_depth)

        # 3. Temporal EMA Filter
        if smooth_depth is None:
            smooth_depth = norm_depth.copy()
        else:
            smooth_depth = alpha * norm_depth + (1.0 - alpha) * smooth_depth

        # 4. Level 01 Core: Compute Surface Normals N(x, y)
        normals, normal_map_vis = compute_surface_normals(smooth_depth, smooth_ksize=5, gradient_scale=4.0)

        # 5. FPS Meter
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # Visual overlays
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Raw Camera Feed", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(normal_map_vis, "Level 01: 3D Surface Normals", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Display side by side: [Live RGB | 3D Normal Map]
        display = np.hstack((frame, normal_map_vis))
        cv2.imshow("National Robotics Week - Level 01 Geometry Engine", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()