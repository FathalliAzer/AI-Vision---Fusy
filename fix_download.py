import os
import requests
from tqdm import tqdm

file_path = "weights/depth_anything_v2_vits.pth"

# 1. Remove corrupted file if it exists
if os.path.exists(file_path):
    print("Removing corrupted file...")
    os.remove(file_path)

os.makedirs("weights", exist_ok=True)

url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
print("Downloading Depth Anything V2 Small (~99 MB)...")

response = requests.get(url, stream=True)
response.raise_for_status()
total_size = int(response.headers.get('content-length', 0))

with open(file_path, 'wb') as f, tqdm(
    desc="Progress",
    total=total_size,
    unit='iB',
    unit_scale=True,
    unit_divisor=1024,
) as bar:
    for data in response.iter_content(chunk_size=1024*1024):
        size = f.write(data)
        bar.update(size)

print("\nDownload finished and verified!")