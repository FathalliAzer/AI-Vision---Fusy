import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time
import mediapipe as mp
from depth_anything_v2.dpt import DepthAnythingV2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class GPUShaderEngine:
    """Performs all normal map and Blinn-Phong shading directly on the RTX GPU."""
    def __init__(self, device='cuda', height=480, width=640):
        self.device = device
        self.h = height
        self.w = width
        
        # Precompute coordinate grid on GPU
        ys, xs = torch.meshgrid(
            torch.linspace(0, 1, height, device=device, dtype=torch.float32),
            torch.linspace(0, 1, width, device=device, dtype=torch.float32),
            indexing='ij'
        )
        self.grid_xy = torch.stack((xs, ys), dim=-1) # (H, W, 2)
        
        # Sobel kernels for normal extraction on GPU
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        self.sobel_x = sobel_x
        self.sobel_y = sobel_y

    def compute_normals_gpu(self, depth_tensor, scale=6.0):
        """Depth tensor shape: (1, 1, H, W) in range [0, 1]"""
        # Smooth depth slightly with 3x3 average filter
        smoothed = F.avg_pool2d(depth_tensor, kernel_size=3, stride=1, padding=1)
        
        dzdx = F.conv2d(smoothed, self.sobel_x, padding=1) * scale
        dzdy = F.conv2d(smoothed, self.sobel_y, padding=1) * scale
        
        nx = -dzdx.squeeze()
        ny = -dzdy.squeeze()
        nz = torch.ones_like(nx)
        
        norm = torch.sqrt(nx**2 + ny**2 + nz**2).clamp(min=1e-6)
        normals = torch.stack((nx / norm, ny / norm, nz / norm), dim=-1) # (H, W, 3)
        return normals

    def render_relighting_gpu(self, frame_bgr, depth_tensor, normals, light_pos, light_color=(1.0, 0.95, 0.8)):
        """
        Runs Blinn-Phong shading on CUDA in ~1-2 ms.
        light_pos: [X, Y, Z]
        """
        # Frame to GPU float tensor
        img_gpu = torch.from_numpy(frame_bgr).to(self.device).float() / 255.0
        
        # Construct 3D surface points: (H, W, 3)
        P_surf = torch.cat((self.grid_xy, depth_tensor.squeeze().unsqueeze(-1)), dim=-1)
        
        # Light Vector L
        light_p = torch.tensor(light_pos, device=self.device, dtype=torch.float32)
        L = light_p - P_surf
        dist = torch.norm(L, dim=-1, keepdim=True).clamp(min=1e-5)
        L_unit = L / dist
        
        # Attenuation
        attenuation = 1.0 / (1.0 + 3.5 * dist + 15.0 * (dist**2))
        
        # Diffuse (Lambertian)
        N_dot_L = torch.sum(normals * L_unit, dim=-1, keepdim=True).clamp(min=0.0)
        diffuse = N_dot_L * 1.3
        
        # Specular (Blinn-Phong)
        V = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)
        H = L_unit + V
        H_unit = H / torch.norm(H, dim=-1, keepdim=True).clamp(min=1e-5)
        N_dot_H = torch.sum(normals * H_unit, dim=-1, keepdim=True).clamp(min=0.0)
        specular = (N_dot_H ** 28.0) * 0.9
        
        # Composite
        ambient = 0.22
        lighting_mult = ambient + (diffuse * attenuation)
        l_col = torch.tensor(light_color, device=self.device, dtype=torch.float32)
        
        # (B, G, R) relit frame
        relit = (img_gpu * lighting_mult) + (specular * attenuation * l_col * 255.0)
        relit = relit.clamp(0.0, 255.0).byte().cpu().numpy()
        
        return relit

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    # 1. Load Depth Model
    model_configs = {'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}}
    print(f"Loading Depth Model on {device}...")
    depth_model = DepthAnythingV2(**model_configs['vits'])
    depth_model.load_state_dict(torch.load('weights/depth_anything_v2_vits.pth', map_location='cpu', weights_only=True))
    depth_model = depth_model.to(device).eval()

    # 2. Initialize GPU Shader & MediaPipe
    W, H = 640, 480
    gpu_engine = GPUShaderEngine(device=device, height=H, width=W)
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    alpha = 0.75
    smooth_depth = None
    light_pos = [0.5, 0.5, 0.3] # Default center
    prev_time = time.time()
    fps_history = []

    print("\n" + "="*55)
    print("LEVEL 03: SPATIAL GESTURE RELIGHTING")
    print(" -> Raise your hand to control the 3D Light!")
    print(" -> Fingertip controls X and Y")
    print(" -> Move hand closer / further to change Light Z depth")
    print(" -> Pinch (thumb+index) turns light super-bright!")
    print("="*55 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Hand Tracking (Fast CPU/MediaPipe)
        hand_results = hands.process(rgb_frame)
        hand_detected = False
        is_pinching = False

        if hand_results.multi_hand_landmarks:
            hand_detected = True
            landmarks = hand_results.multi_hand_landmarks[0].landmark
            
            # Index Finger Tip (Landmark 8) controls X and Y
            idx_x = landmarks[8].x
            idx_y = landmarks[8].y
            
            # Estimate Z from hand size: distance between wrist (0) and middle MCP (9)
            wrist = np.array([landmarks[0].x, landmarks[0].y])
            knuckle = np.array([landmarks[9].x, landmarks[9].y])
            hand_span = np.linalg.norm(wrist - knuckle)
            
            # Map hand size to depth: closer hand (larger span) -> small Z (close light)
            # Far hand (smaller span) -> larger Z (deeper light)
            target_z = float(np.clip(1.0 - (hand_span * 3.5), 0.05, 0.85))
            
            # Smooth gesture tracking
            light_pos[0] = light_pos[0] * 0.5 + idx_x * 0.5
            light_pos[1] = light_pos[1] * 0.5 + idx_y * 0.5
            light_pos[2] = light_pos[2] * 0.7 + target_z * 0.3

            # Check Pinch (Landmark 4: Thumb Tip, Landmark 8: Index Tip)
            thumb = np.array([landmarks[4].x, landmarks[4].y])
            pinch_dist = np.linalg.norm(np.array([idx_x, idx_y]) - thumb)
            if pinch_dist < 0.05:
                is_pinching = True

        # 2. Depth Inference (Fast 266 input)
        with torch.inference_mode():
            if device == 'cuda':
                with torch.autocast('cuda', dtype=torch.float16):
                    raw_depth = depth_model.infer_image(frame, input_size=266)
            else:
                raw_depth = depth_model.infer_image(frame, input_size=266)

        # Invert depth: close=small Z, far=large Z
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

        # 3. GPU Normal and Shading Pipeline
        depth_tensor = torch.from_numpy(smooth_depth).unsqueeze(0).unsqueeze(0).to(device).float()
        normals_gpu = gpu_engine.compute_normals_gpu(depth_tensor, scale=6.0)

        # Light Color: Golden warm, or Electric Cyan if pinching!
        light_col = (1.0, 0.4, 1.0) if is_pinching else (1.0, 0.95, 0.8)

        relit_frame = gpu_engine.render_relighting_gpu(
            frame, depth_tensor, normals_gpu, light_pos, light_color=light_col
        )

        # Draw glowing orb at fingertip / light position
        lx, ly = int(light_pos[0] * W), int(light_pos[1] * H)
        orb_color = (255, 100, 255) if is_pinching else (100, 230, 255)
        cv2.circle(relit_frame, (lx, ly), 14, (255, 255, 255), -1)
        cv2.circle(relit_frame, (lx, ly), 22, orb_color, 3)

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
        status_text = "Hand Locked (Fingertip Light)" if hand_detected else "No Hand Detected (Default Pos)"
        cv2.putText(relit_frame, status_text, (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255) if hand_detected else (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(relit_frame, f"Light 3D: X={light_pos[0]:.2f} Y={light_pos[1]:.2f} Z={light_pos[2]:.2f}", 
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Level 03 - Spatial Gesture Relighting", relit_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()