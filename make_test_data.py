import os
from PIL import Image
import random

os.makedirs("test_data/hr", exist_ok=True)
os.makedirs("test_data/lr", exist_ok=True)

for i in range(10):
    name = f"test_{i:04d}.png"
    Image.new("RGB", (1920, 1440), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255))).save(f"test_data/hr/{name}")
    Image.new("RGB", (960, 720),   color=(random.randint(0,255), random.randint(0,255), random.randint(0,255))).save(f"test_data/lr/{name}")

print("Done")