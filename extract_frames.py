import cv2
import os
import json

# Load evaluation samples
with open("data/evaluation_samples.json", "r") as file:
    samples = json.load(file)

# Create the frames folder
os.makedirs("frames", exist_ok=True)

# Extract frames from each video
for i, sample in enumerate(samples):
    video_path = sample["video"]

    # Create a separate folder for each video
    video_frames_dir = f"frames/video_{i + 1}"
    os.makedirs(video_frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Extract one frame every 1 second
    interval = int(fps)

    frame_number = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % interval == 0:
            frame_path = f"{video_frames_dir}/frame_{saved_count}.jpg"
            cv2.imwrite(frame_path, frame)

            saved_count += 1

        frame_number += 1

    cap.release()

    print(f"Video {i + 1}: saved {saved_count} frames.")