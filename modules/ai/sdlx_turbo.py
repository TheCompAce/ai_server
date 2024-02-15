# Initialize the text-to-image pipeline
from datetime import datetime
import io
import os
import torch

from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionXLImg2ImgPipeline
from flask import request, send_file

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

def text_to_image_sdxl(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    image_pipe.to(device)
    image = image_pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    return image

    
