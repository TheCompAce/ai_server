import requests
import torch
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image



# Function to load images from URLs
def load_images_from_urls(image_urls):
    images = []
    for url in image_urls:
        image = load_image(url)
        images.append(image)
    return images


# Function to check if images are URLs or raw data
def resolve_images(image_inputs):
    images = []
    for item in image_inputs:
        if isinstance(item, str):  # Check if it's a URL
            image = load_image(item)
            images.append(image)
        elif isinstance(item, Image.Image):  # Check if it's a PIL Image
            images.append(item)
    return images


# Function that takes a list of images or URLs and a text prompt
def process_images_with_prompt(image_inputs, text_prompt):    
    # Define the device (using GPU if available)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        DEVICE = "cuda:0"
    else:
        DEVICE = "cpu"

    # Initialize the Idefics2 processor and model
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b-base")
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceM4/idefics2-8b-base"
    ).to(DEVICE)


    # Resolve the input to PIL images
    images = resolve_images(image_inputs)

    # Prepare the inputs for the model
    inputs = processor(images=images, text=text_prompt, return_tensors="pt").to(DEVICE)
    

    # Generate outputs from the model
    outputs = model.generate(**inputs, do_sample=True, max_length=2048)

    # Convert the output tensor to text
    decoded_texts = processor.batch_decode(outputs, skip_special_tokens=True)

    return decoded_texts


# Example usage (for testing purposes, comment out or remove in production)
# mixed_inputs = [
#     "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
#     Image.open("path/to/your/image.jpg"),  # Example of raw PIL Image
# ]

# prompt = "Describe the content of these images."
# results = process_images_with_prompt(mixed_inputs, prompt)

# print("Results:", results)
