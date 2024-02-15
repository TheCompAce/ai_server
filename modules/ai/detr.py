from flask import jsonify
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image


def detect_image_detr(file):
    # Initialize the DETR model and processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    image = Image.open(file.stream)

    # Prepare image for the model
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Convert outputs to COCO API and keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        detections.append({
            "label": model.config.id2label[label.item()],
            "score": round(score.item(), 3),
            "box": box
        })

    return {"detections": detections}