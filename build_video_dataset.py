import argparse
import os
import cv2
from PIL import Image
from tqdm import tqdm

def process_video(video_path, out_root, scene_name, fps_extract=2, lr_scale=2):
    """
    Extracts frames from a video and saves perfectly aligned HR and LR (Bicubic) pairs.
    Folder structure matches the GameIR dataset nested loader.
    """
    # Create GameIR-style nested directories: root/scene/sequence/resolution/
    seq_name = "00"
    hr_dir = os.path.join(out_root, scene_name, seq_name, '1440p')
    lr_dir = os.path.join(out_root, scene_name, seq_name, '720p')
    
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps_extract) if video_fps > 0 else 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {video_fps:.2f} | Extracting 1 frame every {frame_interval} frames")
    
    frame_count = 0
    saved_count = 0

    pbar = tqdm(total=total_frames)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # OpenCV loads as BGR, convert to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hr_img = Image.fromarray(frame_rgb)
            
            # Calculate LR size
            w, h = hr_img.size
            lr_size = (w // lr_scale, h // lr_scale)
            
            # Downsample using anti-aliased Bicubic
            lr_img = hr_img.resize(lr_size, Image.Resampling.BICUBIC)
            
            filename = f"{saved_count:08d}.rgb.png"
            
            hr_img.save(os.path.join(hr_dir, filename))
            lr_img.save(os.path.join(lr_dir, filename))
            
            saved_count += 1
            
        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    print(f"\nDone! Extracted {saved_count} perfect HR/LR pairs to:")
    print(f" - HR: {hr_dir}")
    print(f" - LR: {lr_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build perfectly aligned SR dataset from video")
    parser.add_argument('--video', type=str, required=True, help="Path to high-bitrate gameplay video")
    parser.add_argument('--out_root', type=str, default="custom_dataset", help="Output root directory")
    parser.add_argument('--scene_name', type=str, default="my_game", help="Name of the game/scene")
    parser.add_argument('--fps', type=int, default=2, help="How many frames to extract per second of video")
    parser.add_argument('--scale', type=int, default=2, help="Downscale factor for LR (e.g. 2 means 1440p -> 720p)")
    
    args = parser.parse_args()
    process_video(args.video, args.out_root, args.scene_name, args.fps, args.scale)
