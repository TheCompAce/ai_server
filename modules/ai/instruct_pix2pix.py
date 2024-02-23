import PIL
import requests
import torch
from PIL import ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

def transform_image_with_prompt(image, prompt):
    
    # Initialize the pipeline
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")  # Use GPU for computation. Change to "cpu" if GPU is not available.
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # Generate the image based on the prompt
    images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1).images
    
    return images[0]  # Return the first image in the list
