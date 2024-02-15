from flask import jsonify, request
import torch
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def image_to_depth_dpt(image_file):
    # Initialize the DPT model
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

    image = Image.open(image_file.stream).convert("RGB")  # Ensure image is in RGB

    # Prepare image for the model
    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size and visualize the prediction
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    # Normalize the depth map for visualization
    formatted = (prediction * 255 / np.max(prediction)).astype("uint8")
    depth_image = Image.fromarray(formatted)
    
    return depth_image
