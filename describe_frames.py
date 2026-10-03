import os
import json
import torch

from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Store all visual descriptions
all_descriptions = {}

# Load the vision-language model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load the processor
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct"
)

# Process frames from each video
for video_number in range(1, 4):

    frames_dir = f"frames/video_{video_number}"

    print(f"\nProcessing video {video_number}...")

    # Store descriptions for this video
    video_descriptions = {}

    for frame_file in sorted(os.listdir(frames_dir)):

        frame_path = os.path.join(frames_dir, frame_file)

        # Open the frame
        image = Image.open(frame_path)

        # Create the prompt
        prompt = "Describe what is happening in this image."

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        # Convert the conversation into Qwen3-VL's expected text format
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Convert image and text into model inputs
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        )

        # Move inputs to the model device
        inputs = {
            key: value.to(model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        # Generate the description
        output = model.generate(
            **inputs,
            max_new_tokens=100
        )

        # Remove the original prompt from the output
        generated_ids = output[:, inputs["input_ids"].shape[1]:]

        # Convert output into text
        description = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        # Save the description for this frame
        video_descriptions[frame_file] = description

        print(f"{frame_file}: {description}")

    # Save descriptions for this video
    all_descriptions[f"video_{video_number}"] = video_descriptions

# Save all visual descriptions
with open("visual_descriptions.json", "w") as file:
    json.dump(all_descriptions, file, indent=4)

print("\nVisual descriptions saved successfully.")