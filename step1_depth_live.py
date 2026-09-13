import cv2
import torch
import numpy as np
import time
from depth_anything_v2.dpt import DepthAnythingV2

def main():
    # 1. Device Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    
    print(f"Loading Depth Anything V2 (Small) on {device}...")
    model = DepthAnythingV2(**model_configs['vits'])
    
    # Safe weight loading
    state_dict = torch.load('weights/depth_anything_v2_vits.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print("Model loaded successfully!")

    # 2. Camera Setup (640x480 gives optimal speed & aspect ratio)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not access camera.")
        return

    # Temporal Smoothing parameter (EMA Filter)
    alpha = 0.7 
    smooth_depth = None

    prev_time = time.time()
    fps_history = []

    print("\nLive stream active! Press 'q' or ESC to exit.\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # 3. Fast Inference with AMP (Automatic Mixed Precision)
        with torch.inference_mode():
            if device == 'cuda':
                with torch.autocast('cuda', dtype=torch.float16):
                    raw_depth = model.infer_image(frame, input_size=392)
            else:
                raw_depth = model.infer_image(frame, input_size=392)

        # 4. Normalize Depth to [0.0, 1.0]
        d_min, d_max = raw_depth.min(), raw_depth.max()
        if d_max - d_min > 1e-5:
            norm_depth = (raw_depth - d_min) / (d_max - d_min)
        else:
            norm_depth = np.zeros_like(raw_depth)

        # 5. Temporal Filtering (EMA) to prevent flicker
        if smooth_depth is None:
            smooth_depth = norm_depth.copy()
        else:
            smooth_depth = alpha * norm_depth + (1.0 - alpha) * smooth_depth

        # 6. Colormapped Depth for Visualization
        depth_uint8 = (smooth_depth * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

        # 7. Real-time FPS Calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)

        # Draw overlays
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(depth_color, "Depth Map", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show side-by-side
        combined = np.hstack((frame, depth_color))
        cv2.imshow("Hackathon AI Vision - Step 1 Depth Backbone", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()