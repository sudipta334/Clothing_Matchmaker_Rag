import os, json
from PIL import Image
from io import BytesIO
import base64
from tqdm import tqdm
from src.utils import client, vision_model

# Prompt Template

vision_prompt = """
You are a strict clothing image classifier assistant.

Extract the following attributes from this image:
- Type (Shirt, Pants, Dress, etc.)
- Color
- Style (Casual, Formal, Sporty)
- Pattern (Solid, Striped, Floral)
- Sleeve length
- Season

Respond ONLY with a valid JSON object matching EXACTLY the structure below. 
DO NOT return any commentary, explanation, or text outside this JSON object. 
If you cannot extract attributes, return empty strings for those fields.

{
  "type": "",
  "color": "",
  "style": "",
  "pattern": "",
  "sleeve_length": "",
  "season": ""
}
"""


import sys
import openai
from PIL import Image
import io

def extract_metadata(image_path):
    try:
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
            img = Image.open(io.BytesIO(img_bytes))
            img.verify()
    except Exception as e:
        print(f"Skipping {image_path}: invalid image file. Error: {str(e)}")
        return None

    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    image_size_mb = (len(img_base64) * 3 / 4) / (1024 * 1024)
    if image_size_mb > 19.5:
        print(f"Skipping {image_path}: image size {image_size_mb:.2f}MB exceeds limit.")
        return None

    try:
        response = client.chat.completions.create(
            model=vision_model,
            messages=[
                {"role": "system", "content": vision_prompt},
                {"role": "user", "content": [
                    {
                        "type": "image_url",
                        "image_url": { "url": f"data:image/jpeg;base64,{img_base64}" }
                    }
                ]}
            ]
        )

        result = response.choices[0].message.content
        print(f"\nRaw GPT Vision output for {image_path}:\n{result}\n")

        metadata = json.loads(result)
        return metadata

    except openai.BadRequestError as e:
        if 'content_filter' in str(e).lower():
            print(f"Skipping {image_path}: content filter triggered.")
        else:
            print(f"Skipping {image_path}: OpenAI BadRequestError: {str(e)}")
        return None

    except Exception as e:
        print(f"Unexpected error for {image_path}: {str(e)}")
        return None
























def process_folder(folder_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for filename in tqdm(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        metadata = extract_metadata(img_path)
        if metadata:  # only save valid ones
            out_file = os.path.join(output_path, f"{filename}.json")
            with open(out_file, "w") as f:
                json.dump(metadata, f, indent=2)

# Usage:
# process_folder("data/raw_images", "data/extracted_metadata")



