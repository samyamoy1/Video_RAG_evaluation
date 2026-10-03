from datasets import load_dataset
import os
import json


def load_video_dataset():
    dataset = load_dataset(
        "weikaih/TaskMeAnything-v1-videoqa-random"
    )

    print("Dataset loaded successfully.")
    return dataset


def save_evaluation_samples(dataset):
    # Create data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Use the first dataset split
    split_name = list(dataset.keys())[0]
    split_data = dataset[split_name]

    samples = []

    # Select 3 samples
    for i in range(3):
        sample = split_data[i]

        # Save the video
        video_path = f"data/video_{i + 1}.mp4"

        with open(video_path, "wb") as video_file:
            video_file.write(sample["video"])

        # Store evaluation information
        samples.append({
            "video": video_path,
            "question": sample["question"],
            "options": sample["options"],
            "answer": sample["answer"]
        })

        print(f"Saved video {i + 1}: {video_path}")

    # Save questions and answers
    with open("data/evaluation_samples.json", "w") as file:
        json.dump(samples, file, indent=4)

    print("\nEvaluation samples saved successfully.")


if __name__ == "__main__":
    dataset = load_video_dataset()
    save_evaluation_samples(dataset)