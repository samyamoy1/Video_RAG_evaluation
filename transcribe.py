import whisper
import json
import os

# Load the Whisper model
model = whisper.load_model("base")

# Load evaluation samples
with open("data/evaluation_samples.json", "r") as file:
    samples = json.load(file)

print(f"Loaded {len(samples)} evaluation samples.")

# Transcribe each video
for i, sample in enumerate(samples):
    video_path = sample["video"]

    print(f"\nTranscribing {video_path}...")

    result = model.transcribe(video_path)

    transcript = result["text"]

    print("Transcript:", transcript)