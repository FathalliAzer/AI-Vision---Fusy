import torch
import cv2

print("=" * 40)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
    print(f"VRAM Available:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: Running on CPU! We need CUDA for 60 FPS.")
print(f"OpenCV Version:  {cv2.__version__}")
print("=" * 40)